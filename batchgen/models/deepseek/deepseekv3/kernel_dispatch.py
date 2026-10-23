"""R1 projection/attention dispatch without importing legacy attention at startup.

SM103 is an explicit experiment until full-layer/model qualification passes.
Other devices retain the existing implementation and import it only when used.
This module does not bypass the separate server runtime preflight contract.
"""

import os
import types

import torch


def select_sm103_experiment(device):
    """Select exact SM103 or reject an unrequested migration before imports."""
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3):
        return False
    if os.environ.get("BATCHGEN_R1_SM103_EXPERIMENTAL") != "1":
        raise RuntimeError(
            "R1 SM103 is not qualified for default dispatch. The explicit "
            "BATCHGEN_R1_SM103_EXPERIMENTAL=1 experiment is required; "
            "legacy FA3/FlashMLA will not be selected on this device."
        )
    return True


def w8a16_gemm(weight_data_fp8, weight_scale_inv_fp32, activation_bf16):
    if select_sm103_experiment(activation_bf16.device):
        if os.environ.get("DISABLE_MMA_V5") != "1":
            raise RuntimeError(
                "The R1 SM103 Triton 3.5 projection experiment requires "
                "DISABLE_MMA_V5=1 to avoid the unqualified MMA v5 compiler path"
            )
        from batchgen.attention.mla.sm103_backend import w8a16_gemm as implementation
    else:
        from batchgen.attention.mla.fa3_backend import w8a16_gemm as implementation
    return implementation(weight_data_fp8, weight_scale_inv_fp32, activation_bf16)


def act_quant(x, block_size=128, eps=1e-12, num_valid_tokens=None, scale_tma_aligned=False):
    """Retain the R1 BF16 block128 quantization contract without a FA3 import."""
    if not select_sm103_experiment(x.device):
        from batchgen.attention.mla.fa3_backend import act_quant as implementation
        return implementation(x, block_size=block_size, eps=eps,
                              num_valid_tokens=num_valid_tokens, scale_tma_aligned=scale_tma_aligned)
    if x.dtype != torch.bfloat16 or block_size != 128 or eps != 1e-12:
        raise ValueError("R1 SM103 quantization requires BF16, block128, default epsilon")
    if x.ndim not in (2, 3) or not x.is_contiguous():
        raise ValueError("R1 SM103 quantization requires a contiguous 2D/3D input")
    if scale_tma_aligned and x.ndim != 2:
        raise ValueError("TMA-aligned scales require a 2D input")
    if num_valid_tokens is not None:
        if num_valid_tokens.device != x.device or num_valid_tokens.dtype != torch.int32 or num_valid_tokens.numel() != 1:
            raise ValueError("num_valid_tokens must be a device-local int32 scalar")
    shape = x.shape
    flat = x.view(-1, shape[-1])
    if flat.shape[0] == 0:
        # No experts may receive tokens on a rank. Do not launch a zero-grid
        # Triton kernel or import the legacy fallback for that legitimate case.
        return torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty(
            (*shape[:-1], (shape[-1] + 127) // 128), dtype=torch.float32, device=x.device,
        )
    from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d
    values, scales = per_token_blocked_quantize_bf16_to_fp8_1d(
        flat, block_size=128, num_valid_tokens=num_valid_tokens, scale_tma_aligned=scale_tma_aligned,
    )
    return values.view(*shape), scales if scale_tma_aligned else scales.view(*shape[:-1], scales.shape[-1])


def _unsupported_legacy_attention(self, *args, **kwargs):
    raise RuntimeError(
        "The R1 SM103 experiment only supports FP8-weight prefill and "
        "BF16 paged one-token decode; a GPU paged KV manager is required"
    )


def bind_sm103_attention(module):
    """Bind only wrapper-supported methods; do not enable a new primitive here."""
    from batchgen.attention.mla.sm103_backend import (
        mla_prefill_sm103_w8a16,
        mla_prefill_sm103_w8a16_prepacked,
        mla_decoding_sm103_bf16_with_pagekv,
    )
    bindings = {
        "prefill_attn_w8a16": mla_prefill_sm103_w8a16,
        "prefill_attn_w8a16_prepacked": mla_prefill_sm103_w8a16_prepacked,
        "decoding_attn_mode_3_bf16": mla_decoding_sm103_bf16_with_pagekv,
        "prefill_attn": _unsupported_legacy_attention,
        "prefill_attn_prepacked": _unsupported_legacy_attention,
        "decoding_attn": _unsupported_legacy_attention,
        "decoding_attn_mode_3_fp8": _unsupported_legacy_attention,
        "decoding_attn_mode_3_dequant_fusion": _unsupported_legacy_attention,
        "fused_get_query_states_triton": _unsupported_legacy_attention,
    }
    for name, implementation in bindings.items():
        setattr(module, name, types.MethodType(implementation, module))
    module._batchgen_r1_sm103_experimental = True
    # Selection of _r1_attention_implementation remains an independent, explicit
    # qualification decision. Binding does not promote the high-precision path.
