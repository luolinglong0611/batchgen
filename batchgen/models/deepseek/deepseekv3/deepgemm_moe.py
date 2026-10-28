"""Opt-in R1 Blackwell FP8 MoE math shared by prefill and decode.

This module is inert until selected with ``BATCHGEN_R1_MOE_BACKEND=deep_gemm``.
It calls the installed DeepGEMM and stock SGLang activation/quantization kernels;
it neither patches SGLang nor falls back to the old Triton MMA2 expert kernels.
Weight conversion follows stock SGLang's Blackwell checkpoint conversion:
FP8 + FP32 block scales -> BF16 -> FP8 with rounded UE8M0 scales.  It is a
numerical-policy change, not a byte-preserving repack of the original weights.

Build each layer's bank once, directly from Host tensors, before GPU KV sizing.
Keep that bank across prefill/decode phases.  Materializing it alongside the old
resident expert allocation would duplicate approximately 76 GiB per R1 rank.
"""
from dataclasses import dataclass
import time
from types import SimpleNamespace

import torch


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _load_runtime(device):
    if _capturing():
        raise RuntimeError("Initialize the DeepGEMM runtime before CUDA Graph capture")
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("R1 UE8M0 DeepGEMM requires a Blackwell SM10x CUDA device")
    import deep_gemm
    from sgl_kernel import silu_and_mul
    from sglang.srt.layers.moe.ep_moe.kernels import (
        silu_and_mul_masked_post_quant_fwd,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    for name in ("fp8_m_grouped_gemm_nt_masked", "m_grouped_fp8_gemm_nt_contiguous",
                 "get_mn_major_tma_aligned_packed_ue8m0_tensor"):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError("Installed DeepGEMM is missing " + name)
    return SimpleNamespace(
        dg=deep_gemm, silu_and_mul=silu_and_mul,
        silu_masked=silu_and_mul_masked_post_quant_fwd,
        quantize=sglang_per_token_group_quant_fp8,
    )


def _validate_sources(sources):
    """Validate all Host descriptors before allocating the destination bank."""
    sources = list(sources)
    if not sources:
        raise ValueError("DeepGEMM requires at least one local expert")
    first = sources[0]
    if first["gate"].ndim != 2:
        raise ValueError("Expert weights must be two dimensional")
    intermediate, hidden = first["gate"].shape
    if min(hidden, intermediate) <= 0 or hidden % 128 or intermediate % 128:
        raise ValueError("R1 expert dimensions must be positive multiples of 128")
    expected = {
        "gate": (intermediate, hidden), "up": (intermediate, hidden),
        "down": (hidden, intermediate),
        "gate_scale": (intermediate // 128, hidden // 128),
        "up_scale": (intermediate // 128, hidden // 128),
        "down_scale": (hidden // 128, intermediate // 128),
    }
    for source in sources:
        for key, shape in expected.items():
            tensor = source[key]
            dtype = torch.float32 if key.endswith("scale") else torch.float8_e4m3fn
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"Invalid original expert {key}: {tensor.shape}, {tensor.dtype}")
    return sources, hidden, intermediate


def _requant_weight_ue8m0(weight, scale, dg):
    """Stock SG FP32 block-dequant/BF16/pow2-FP8 semantics, one expert at a time.

    Same operations as SGLang 0.5.5.post3 fp8_utils._requant_weight_ue8m0,
    quant_weight_ue8m0 and transform_scale_ue8m0, specialized to 128-aligned R1.
    Host checkpoint tensors are never changed in place.
    """
    n, k = weight.shape
    repeated = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    dequant = (weight.float() * repeated).to(torch.bfloat16)
    blocks = dequant.view(n // 128, 128, k // 128, 128)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    converted = (blocks * (1.0 / sf)).to(torch.float8_e4m3fn).view(n, k).contiguous()
    sf = sf.view(n // 128, k // 128).repeat_interleave(128, dim=0)
    packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return converted, packed


@dataclass
class R1DeepGemmWeightBank:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    def tensors(self):
        return self.w13, self.w13_scale, self.w2, self.w2_scale

    @property
    def resident_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass
class R1DeepGemmMaskedWorkspace:
    gateup: torch.Tensor
    down_fp8: torch.Tensor
    down_scale: torch.Tensor

    @classmethod
    def for_pool(cls, pool):
        existing = getattr(pool, "_r1_deepgemm_workspace", None)
        if existing is not None:
            return existing
        if _capturing():
            raise RuntimeError("Prepare the DeepGEMM workspace before CUDA Graph capture")
        e, m, n = pool.num_local_experts, pool.stride, pool.intermediate
        workspace = cls(
            torch.empty((e, m, 2 * n), dtype=torch.bfloat16, device=pool.x.device),
            pool.intermediate_fp8, pool.intermediate_scale,
        )
        pool._r1_deepgemm_workspace = workspace
        return workspace


class R1DeepGemmExpertMath:
    """One immutable layer bank; masked decode and contiguous prefill share it."""
    def __init__(self, bank, runtime, *, source_identity=None, build_stats=None):
        self.bank = self.weights = bank
        self.runtime = runtime
        self.source_identity = source_identity
        self.build_stats = build_stats or {}
        self.num_local_experts, twice_intermediate, self.hidden = bank.w13.shape
        self.intermediate = twice_intermediate // 2
        self.resident_bytes = bank.resident_bytes

    @classmethod
    def from_expert_sources(cls, sources, *, device, source_identity=None, runtime=None):
        """Consume CPU expert descriptors and upload only one expert temporarily.

        Keys: gate/up/down (FP8), gate_scale/up_scale/down_scale (FP32).
        The caller memoizes the returned object by layer and Host source identity.
        ``runtime`` is injectable for CPU contract tests; those are not GPU proof.
        """
        if _capturing():
            raise RuntimeError("Do not convert expert weights during CUDA Graph capture")
        sources, h, n = _validate_sources(sources)
        if h % 512 or n % 512:
            raise ValueError("Blackwell packed UE8M0 weights require K multiples of 512")
        runtime = _load_runtime(device) if runtime is None else runtime
        device = torch.device(device)
        e = len(sources)
        started = time.monotonic()
        # Packed scales must be MN-major, not torch.stack's row-major layout.
        # PSM configuration is often under inference_mode.  Graph lifetime
        # guards need real version counters on this immutable long-lived bank.
        with torch.inference_mode(False):
            bank = R1DeepGemmWeightBank(
                torch.empty((e, 2 * n, h), dtype=torch.float8_e4m3fn, device=device),
                torch.empty((e, h // 512, 2 * n), dtype=torch.int32, device=device).transpose(1, 2),
                torch.empty((e, h, n), dtype=torch.float8_e4m3fn, device=device),
                torch.empty((e, n // 512, h), dtype=torch.int32, device=device).transpose(1, 2),
            )
        for idx, source in enumerate(sources):
            # Concatenation order is exactly SG w13: gate first, up second.
            # copy_ supports FP8 byte transfers even on runtimes whose cat
            # kernels do not implement FP8; no full-bank temporary is made.
            w13 = torch.empty((2 * n, h), dtype=torch.float8_e4m3fn, device=device)
            s13 = torch.empty((2 * n // 128, h // 128), dtype=torch.float32, device=device)
            w13[:n].copy_(source["gate"])
            w13[n:].copy_(source["up"])
            s13[:n // 128].copy_(source["gate_scale"])
            s13[n // 128:].copy_(source["up_scale"])
            converted, packed = _requant_weight_ue8m0(w13, s13, runtime.dg)
            bank.w13[idx].copy_(converted)
            bank.w13_scale[idx].copy_(packed)
            del w13, s13, converted, packed
            down = source["down"].to(device)
            scale = source["down_scale"].to(device)
            converted, packed = _requant_weight_ue8m0(down, scale, runtime.dg)
            bank.w2[idx].copy_(converted)
            bank.w2_scale[idx].copy_(packed)
            del down, scale, converted, packed
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats = {"build_wall_seconds": time.monotonic() - started,
                 "build_synchronized": device.type == "cuda",
                 "resident_bytes": bank.resident_bytes,
                 "local_experts": e, "hidden": h, "intermediate": n,
                 "source_policy": "original_fp8_fp32_block_scales_to_bf16_then_ue8m0_fp8",
                 "host_sources_modified": False, "temporary_upload_max_experts": 1}
        return cls(bank, runtime, source_identity=source_identity, build_stats=stats)

    def graph_weight_tensors(self):
        return self.bank.tensors()

    def _packed_scales(self, scales):
        if scales.dtype == torch.int32:
            return scales
        if scales.dtype != torch.float32:
            raise ValueError("DeepGEMM scales must be FP32 UE8M0 values or packed int32")
        return self.runtime.dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(scales)

    def masked(self, x_fp8, scales, counts, expected_m, *, workspace, out):
        if (x_fp8.ndim != 3 or x_fp8.dtype != torch.float8_e4m3fn or
                x_fp8.shape[0] != self.num_local_experts or x_fp8.shape[-1] != self.hidden):
            raise ValueError("Masked DeepGEMM requires FP8 [local_experts, stride, hidden]")
        e, m, _ = x_fp8.shape
        if counts.dtype != torch.int32 or tuple(counts.shape) != (e,):
            raise ValueError("Masked counts must be device int32 [local_experts]")
        if not isinstance(expected_m, int) or expected_m <= 0:
            raise ValueError("expected_m must be a positive CPU integer")
        if (tuple(workspace.gateup.shape) != (e, m, 2 * self.intermediate) or
                tuple(workspace.down_fp8.shape) != (e, m, self.intermediate) or
                tuple(workspace.down_scale.shape) != (e, m, self.intermediate // 128) or
                tuple(out.shape) != (e, m, self.hidden)):
            raise ValueError("DeepGEMM masked workspace/output geometry mismatch")
        dg, bank = self.runtime.dg, self.bank
        dg.fp8_m_grouped_gemm_nt_masked(
            (x_fp8, self._packed_scales(scales)), (bank.w13, bank.w13_scale),
            workspace.gateup, counts, expected_m)
        self.runtime.silu_masked(workspace.gateup, workspace.down_fp8,
                                 workspace.down_scale, 128, counts, scale_ue8m0=True)
        dg.fp8_m_grouped_gemm_nt_masked(
            (workspace.down_fp8, self._packed_scales(workspace.down_scale)),
            (bank.w2, bank.w2_scale), out, counts, expected_m)
        return out

    def contiguous(self, x_fp8, scales, m_indices, *, out=None):
        """Expert-sorted padded rows; m_indices uses local IDs and -1 for padding.

        Matches SG C1's normal DeepEP path, including native SiLU/BF16 and
        UE8M0 per-token-group quantization.  The routing/permutation stays with
        the prefill dispatcher, not this layer-local math object.
        """
        if x_fp8.ndim != 2 or x_fp8.dtype != torch.float8_e4m3fn or x_fp8.shape[1] != self.hidden:
            raise ValueError("Contiguous DeepGEMM requires FP8 [sorted_rows, hidden]")
        rows = x_fp8.shape[0]
        if m_indices.dtype != torch.int32 or tuple(m_indices.shape) != (rows,):
            raise ValueError("m_indices must be device int32 [sorted_rows]")
        if out is None:
            out = torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=x_fp8.device)
        if tuple(out.shape) != (rows, self.hidden) or out.dtype != torch.bfloat16:
            raise ValueError("Contiguous DeepGEMM output geometry/dtype mismatch")
        if rows == 0:
            return out
        dg, bank = self.runtime.dg, self.bank
        gateup = torch.empty((rows, 2 * self.intermediate), dtype=torch.bfloat16, device=x_fp8.device)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (x_fp8, self._packed_scales(scales)), (bank.w13, bank.w13_scale), gateup, m_indices)
        down_bf16 = torch.empty((rows, self.intermediate), dtype=torch.bfloat16, device=x_fp8.device)
        self.runtime.silu_and_mul(gateup, down_bf16)
        down_fp8, down_scale = self.runtime.quantize(
            down_bf16, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (down_fp8, self._packed_scales(down_scale)), (bank.w2, bank.w2_scale), out, m_indices)
        return out


class R1DeepGemmDecodeAdapter:
    """Fixed-capacity LL collectives, native FP8 GEMMs, dynamic live-row mask."""
    def __init__(self, moe, pool, *, layer_idx, expert_math):
        self.moe, self.pool = moe, pool
        self.layer_idx = int(layer_idx)
        self.expert_math = expert_math
        if (not 3 <= self.layer_idx < pool.layer_counters.shape[0] or
                expert_math.num_local_experts != pool.num_local_experts or
                expert_math.hidden != pool.hidden or expert_math.intermediate != pool.intermediate):
            raise ValueError("DeepGEMM adapter layer/pool/weight geometry mismatch")
        self.workspace = R1DeepGemmMaskedWorkspace.for_pool(pool)

    def graph_weight_tensors(self):
        return self.expert_math.graph_weight_tensors()

    def warmup_local_only(self, *, rows=None):
        """Compile native compute without an EP collective or old MMA2 kernels."""
        p = self.pool
        counts = torch.ones(p.num_local_experts, dtype=torch.int32, device=p.x.device)
        p.x_fp8[:, 0, :].zero_()
        self.expert_math.masked(
            p.x_fp8, p.x_scale.fill_(1.0), counts, self._expected_m(rows),
            workspace=self.workspace, out=p.expert_out)

    def _expected_m(self, rows=None):
        p = self.pool
        # Stock SG uses actual local rows for this tuning hint, not LL capacity.
        # It never bounds valid expert rows: counts remains authoritative.
        rows = p.capacity if rows is None else rows
        return (rows * p.world_size * p.topk + p.num_experts) // p.num_experts

    def __call__(self, x):
        p, moe = self.pool, self.moe
        rows, hidden = x.shape
        if rows > p.capacity or hidden != p.hidden or x.dtype != torch.bfloat16:
            raise ValueError("R1 DeepGEMM LL requires BF16 [rows<=fixed_capacity, hidden]")
        if _capturing() and not p.external_valid_rows:
            raise RuntimeError("Wire external valid_rows before capturing R1 DeepGEMM LL")
        if not p.external_valid_rows:
            p.set_valid_rows(rows)
        p.x.zero_()
        p.x[:rows].copy_(x)
        idx, weights = moe.gate.moe_gate_forward_hybrid(p.x[:, None, :])
        valid = (p.row_ids < p.valid_rows).unsqueeze(1)
        p.topk_idx.copy_(torch.where(valid, idx, -1))
        p.topk_weights.copy_(torch.where(valid, weights, 0.0))
        recv, counts, handle, _event, _hook = p.exchange.buffer.low_latency_dispatch(
            p.x, p.topk_idx, p.capacity, p.num_experts, use_fp8=True,
            round_scale=True, use_ue8m0=True, async_finish=False, return_recv_hook=False)
        if not isinstance(recv, (tuple, list)) or len(recv) != 2:
            raise RuntimeError("DeepEP FP8 dispatch did not return values and UE8M0 scales")
        recv_fp8, recv_scale = recv
        p.layer_counters[self.layer_idx, 0].add_(1)
        p.layer_counters[self.layer_idx, 2].add_(p.valid_rows[0])
        p.layer_counters[self.layer_idx, 3].add_(counts.sum(dtype=torch.int64))
        self.expert_math.masked(recv_fp8, recv_scale, counts, self._expected_m(rows),
                                workspace=self.workspace, out=p.expert_out)
        p.exchange.combine(p.expert_out, p.topk_idx, p.topk_weights, handle, out=p.local_out)
        p.layer_counters[self.layer_idx, 1].add_(1)
        return p.local_out[:rows]
