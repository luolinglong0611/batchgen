"""Opt-in contiguous-expert SiLU/FP8 candidate with explicit BF16 boundary.

The intermediate BF16 rounding step is retained in registers, before per128
amax/UE8M0 quantization. Qualification must compare both FP8 values and packed
scale bytes against the installed stock SGLang two-kernel path before use.
This module is imported only by the selected prefill candidate.
"""
import os


ENV = "BATCHGEN_R1_PREFILL_FUSED_SILU_QUANT"


def requested():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be0 or1")
    return value == "1"


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    @triton.jit(do_not_specialize=["ALIGNED_ROWS"])
    def fused(
        X, Y, PACKED,
        ALIGNED_ROWS, K: tl.constexpr,
        GROUPS: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK_K)
        gate = tl.load(X + row * (2 * K) + column, column < K, other=0).to(tl.float32)
        up = tl.load(X + row * (2 * K) + K + column, column < K, other=0).to(tl.float32)
        # Match the stock separate path's BF16 materialization before amax.
        # No FMA contraction; the GPU byte gate decides whether the installed
        # SiLU exp/div implementation is equivalent at BF16 boundaries.
        product = (tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
        blocks = tl.reshape(product, (GROUPS, 128))
        amax = tl.maximum(tl.max(tl.abs(blocks), axis=1), 1.0e-10)
        # Stock SGLang0.5.5 has two lower bounds: the amax reduction starts
        # at eps, then UE8M0 applies max(amax/448,1e-10) before exponent
        # rounding (csrc/gemm/per_token_group_quant_8bit.cu:78-80).
        # Omitting this second floor changes zero/tiny-group scale bytes.
        raw_scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0e-10)
        scale_bits = raw_scale.to(tl.int32, bitcast=True)
        exponent = (scale_bits + 0x7fffff) >> 23
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        quant = (blocks / scale[:, None]).to(tl.float8e4nv)
        tl.store(Y + row * K + column, tl.reshape(quant, (BLOCK_K,)), column < K)
        shifts = tl.arange(0, 4) * 8
        packed = tl.sum(tl.reshape(exponent, (GROUPS // 4, 4)) << shifts[None, :], axis=1).to(tl.int32)
        groups4 = tl.arange(0, GROUPS // 4)
        tl.store(PACKED + groups4 * ALIGNED_ROWS + row, packed)

    _KERNEL = fused
    return fused


def fused_silu_quantize(gateup):
    import torch
    if (gateup.device.type != "cuda" or gateup.dtype != torch.bfloat16
            or gateup.ndim != 2 or not gateup.is_contiguous()
            or gateup.shape[1] != 4096):
        raise ValueError("R1 fused prefill SiLU/quant requires contiguous CUDA BF16 [M,4096]")
    import triton
    rows, k = gateup.shape[0], gateup.shape[1] // 2
    aligned_rows = (rows + 3) // 4 * 4
    q = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=gateup.device)
    # Native DeepGEMM's MN-major int32 packs four consecutive UE8M0 scales.
    scales = torch.empty((k // 512, aligned_rows), dtype=torch.int32,
                         device=gateup.device).t()[:rows]
    if rows:
        _kernel()[(rows,)](gateup, q, scales, K=k,
                            ALIGNED_ROWS=aligned_rows, GROUPS=k // 128,
                            BLOCK_K=triton.next_power_of_2(k), num_warps=8,
                            enable_fp_fusion=False)
    return q, scales
