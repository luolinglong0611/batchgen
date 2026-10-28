"""Opt-in routed-only R1 block-FP8 -> official FlashInfer MXFP8 MegaMoE.

SGLang source contract: baa9befa381511cd4d56f83980951e5a45fffc6d (v0.5.22).
The checkpoint/model-wide quantization stays fp8. Only Fp8MoEMethod's routed
expert GPU tensors are prepared; no dense/attention/shared-expert changes.
This is an experimental integration, not unmodified stock SGLang.
"""
from __future__ import annotations

import logging
import json
import os
import time

ENVIRONMENT = "SGLANG_R1_FP8_MEGAMOE_BRIDGE"
UPSTREAM_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"
POLICY = "C1_block128_FP8_FP32scale_to_BF16_then_pow2_FP8; lossless_scale_expand_to_row_K32"
logger = logging.getLogger(__name__)


def requested() -> bool:
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(ENVIRONMENT + " must be explicitly 0 or 1")
    return value == "1"


def allow_standard_fp8_model(hf_config, quantization: str) -> bool:
    """Pure configuration gate; existing SM-family/EP/attention-DP guards stay."""
    if not requested():
        return False
    expected = {"hidden_size": 7168, "moe_intermediate_size": 2048,
                "n_routed_experts": 256, "num_experts_per_tok": 8,
                "num_hidden_layers": 61, "first_k_dense_replace": 3}
    if quantization != "fp8" or list(getattr(hf_config, "architectures", ())) != ["DeepseekV3ForCausalLM"]:
        raise ValueError("The FP8 MegaMoE bridge is restricted to the original R1 FP8 topology")
    for name, value in expected.items():
        if getattr(hf_config, name, None) != value:
            raise ValueError("Unsupported R1 FP8 MegaMoE bridge model field " + name)
    return True


def c1_expert_to_mxfp8(weight, block_scale):
    """One expert: exact old-C1 BF16/pow2 block128 preparation, then re-layout.

    The second stage does NOT re-quantize weights: each pow2 block scale is
    repeated over its128 rows and four32-column groups. Return FP8 values and
    canonical uint8 biased-exponent scales [N,K/32]. Host sources are not mutated.
    """
    import torch
    if (weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn or
            block_scale.dtype != torch.float32):
        raise ValueError("Bridge source must be two-dimensional E4M3 FP8 with FP32 block scales")
    n, k = weight.shape
    if n % 128 or k % 128 or tuple(block_scale.shape) != (n // 128, k // 128):
        raise ValueError("Bridge source must use full128x128 weight blocks")
    # Startup only; reject poisoned checkpoint metadata before altering GPU weights.
    if not bool(torch.isfinite(block_scale).all() and (block_scale > 0).all()):
        raise ValueError("Bridge scales must be positive and finite")
    scale = block_scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    bf16 = (weight.float() * scale).to(torch.bfloat16)
    blocks = bf16.view(n // 128, 128, k // 128, 128)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks * (1.0 / sf)).to(torch.float8_e4m3fn).reshape(n, k).contiguous()
    exponent = ((sf.reshape(n // 128, k // 128).view(torch.int32) >> 23) & 255).to(torch.uint8)
    per32 = exponent.repeat_interleave(128, -2).repeat_interleave(4, -1).contiguous()
    return q, per32


def _runtime():
    import torch
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.moe.flashinfer_megamoe import prepare_mxfp8_moe_weights_for_flashinfer_megamoe
    return torch, get_moe_a2a_backend(), get_moe_runner_backend(), prepare_mxfp8_moe_weights_for_flashinfer_megamoe


def _validate_routed(method, layer, torch):
    if (getattr(method, "is_fp4_expert", False) or getattr(method, "use_mxfp8", False) or
            not getattr(method, "block_quant", False) or
            list(getattr(method.quant_config, "weight_block_size", ())) != [128, 128] or
            not getattr(method.quant_config, "is_checkpoint_fp8_serialized", False)):
        raise ValueError("Bridge requires original serialized block128 FP8 routed experts")
    if (getattr(layer, "hidden_size", None) != 7168 or
            getattr(layer, "intermediate_size_per_partition", None) != 2048 or
            getattr(layer, "num_experts", None) != 256 or
            getattr(layer, "num_local_experts", None) != 32 or
            getattr(layer, "moe_ep_size", None) != 8 or
            getattr(layer, "top_k", None) != 8):
        raise ValueError("Bridge requires R1 EP8/top8/32 local experts and separate shared expert")
    for name, shape in (("w13_weight", (32, 4096, 7168)), ("w2_weight", (32, 7168, 2048))):
        value = getattr(layer, name)
        if tuple(value.shape) != shape or value.dtype != torch.float8_e4m3fn:
            raise ValueError("Wrong original R1 expert weight " + name)
        scale = getattr(layer, name + "_scale_inv")
        if (scale.dtype != torch.float32 or tuple(scale.shape) != (32, shape[1] // 128, shape[2] // 128)):
            raise ValueError("Wrong original R1 block128 scale " + name)
        if value.device.type != "cuda" or torch.cuda.get_device_capability(value.device) != (10, 3):
            raise ValueError("The experimental R1 FP8 MegaMoE bridge is qualified only for SM103")


def maybe_prepare_routed(method, layer) -> bool:
    """Called before stock block-quant processing; no action without opt-in.

    On failure the process must abort: partial in-place conversion is not a
    fallback format. The original checkpoint on disk is never modified.
    """
    if not requested():
        return False
    torch, a2a, runner, prepare = _runtime()
    if not a2a.is_flashinfer_megamoe() or not runner.is_flashinfer_megamoe():
        raise ValueError("Bridge requires both FlashInfer MegaMoE communication and runner")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare the R1 FP8 bridge outside CUDA Graph capture")
    if getattr(layer, "_r1_fp8_megamoe_bridge", None) is not None:
        raise RuntimeError("Do not apply the R1 FP8 bridge twice or hot-reload its weights")
    _validate_routed(method, layer, torch)
    started = time.monotonic()
    weight_storage = {}
    canonical_bytes = 0
    source_fp8_bytes = sum(getattr(layer, name).numel() for name in ("w13_weight", "w2_weight"))
    with torch.inference_mode():
        for weight_name in ("w13_weight", "w2_weight"):
            parameter = getattr(layer, weight_name)
            scale_parameter = getattr(layer, weight_name + "_scale_inv")
            count, n, k = parameter.shape
            weight_storage[weight_name] = parameter.data_ptr()
            scales = torch.empty((count, n, k // 32), dtype=torch.uint8, device=parameter.device)
            for expert in range(count):
                q, per32 = c1_expert_to_mxfp8(parameter.data[expert], scale_parameter.data[expert])
                # Reuse the loaded FP8 allocation; do not hold a second full model bank.
                parameter.data[expert].copy_(q)
                scales[expert].copy_(per32)
                del q, per32
            scale_parameter.data = scales
            scale_parameter.format_ue8m0 = True
            canonical_bytes += parameter.numel() + scales.numel()
        # Only this layer-local quant method changes. The model-wide
        # Fp8Config and dense/shared/attention quantizers remain original FP8.
        method.use_mxfp8 = True
        method.convert_mxfp8_to_block = False
        prepare(layer)
        torch.cuda.synchronize(layer.w13_weight.device)
    report = {"policy": POLICY, "upstream_commit": UPSTREAM_COMMIT,
              "layer_id": getattr(layer, "layer_id", -1),
              "ep_rank": getattr(layer, "moe_ep_rank", -1),
              "source_checkpoint_changed": False, "routed_only": True,
              "weight_bits": 8, "source_scale_block": [128, 128],
              "effective_scale_block": [1, 32], "scale_dtype": "uint8_UE8M0",
              "C1_to_per32_further_weight_requantization": False,
              "max_temporary_experts": 1, "canonical_bytes": canonical_bytes,
              "source_fp8_weight_bytes": source_fp8_bytes,
              "transformed_tensor_bytes": sum(getattr(layer, name).numel() * getattr(layer, name).element_size()
                  for name in ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")),
              "original_loaded_FP8_storage": weight_storage,
              "setup_seconds": time.monotonic() - started,
              "communication_compute_graph": "official_FlashInfer_MXFP8_MegaMoE",
              "activation_policy_change": "per128_FP8_to_per32_MXFP8",
              "topk_weight_policy_change": "official_FI_applies_topk_after_SwiGLU_before_MXFP8_requant_and_fc2",
              "full_model_SG_precision_qualified": False}
    layer._r1_fp8_megamoe_bridge = report
    logger.info("[R1_FP8_MEGAMOE_BRIDGE] %s", json.dumps(report, sort_keys=True))
    return True


def report_apply_success(layer, dispatch_output, result) -> None:
    """Once per eager/capture mode, after the real native runner returns.

    No device value or synchronization is read. A captured Python call is not a
    CUDA-graph replay counter; the actual returned output is still authoritative.
    """
    import torch
    capture = torch.cuda.is_current_stream_capturing()
    seen = layer.__dict__.setdefault("_r1_fp8_megamoe_apply_seen", set())
    if capture in seen:
        return
    mega = layer._flashinfer_megamoe_layer
    x = dispatch_output.hidden_states
    payload = {"layer_id": layer.layer_id, "ep_rank": layer.moe_ep_rank,
               "rows": int(x.shape[0]), "capturing": capture,
               "native_runner_returned": True,
               "native_class": type(mega).__module__ + "." + type(mega).__name__,
               "max_tokens_per_rank": mega._fleet_params.max_tokens_per_rank,
               "observation_is_replay_counter": False}
    logger.info("[R1_FP8_MEGAMOE_APPLY] %s", json.dumps(payload, sort_keys=True))
    seen.add(capture)
