"""Opt-in Blackwell ordinary FP8 GEMMs, prepared before KV allocation.

Unlike the legacy SM103 fallback, this uses native DeepGEMM. Checkpoint
FP8/FP32 block scales are requantized with stock SGLang's UE8M0 policy once
per model lifetime. Original tensors remain unchanged. This is a numerical
policy change, not a lossless scale-layout conversion.
"""
import importlib
import logging
import os
import weakref
from dataclasses import dataclass


ENV = "BATCHGEN_R1_ORDINARY_GEMM"
SHARED_INPUT_QUANT_ENV = "BATCHGEN_R1_SHARED_INPUT_QUANT_REUSE"
_LOOKUP = weakref.WeakValueDictionary()
_RUNTIME = None


def enabled():
    value = os.environ.get(ENV, "legacy")
    if value not in ("legacy", "deep_gemm"):
        raise ValueError(f"{ENV} must be legacy or deep_gemm")
    return value == "deep_gemm"


def shared_input_quant_reuse_enabled():
    """Reuse only a single forward's common input; never retain activations.

    The legacy GEMM path is unchanged even if the new flag is present. Weight
    banks, quantization policy and the two individual native GEMMs stay the
    same; this only avoids quantizing their identical BF16 input twice.
    """
    value = os.environ.get(SHARED_INPUT_QUANT_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{SHARED_INPUT_QUANT_ENV} must be 0 or 1")
    return value == "1" and enabled()


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare native ordinary GEMMs before graph capture")
        from sglang.srt.layers.quantization.fp8_utils import _requant_weight_ue8m0
        from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
        _RUNTIME = (importlib.import_module("deep_gemm"), _requant_weight_ue8m0,
                    sglang_per_token_group_quant_fp8)
    return _RUNTIME


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _key(weight, scale):
    return (str(weight.device), weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
            scale.data_ptr(), tuple(scale.shape), tuple(scale.stride()))


@dataclass
class PreparedWeight:
    source: object
    source_scale: object
    source_key: tuple
    versions: tuple
    weight: object
    scale: object

    def valid(self):
        return (_key(self.source, self.source_scale) == self.source_key and
                (_version(self.source), _version(self.source_scale)) == self.versions)


def prepare_weight(weight, scale):
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Weight conversion is forbidden during CUDA Graph capture")
    if (weight.device.type != "cuda" or torch.cuda.get_device_capability(weight.device)[0] != 10
            or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2
            or scale.dtype != torch.float32 or weight.shape[1] % 512
            or scale.device != weight.device or not weight.is_contiguous()
            or not scale.is_contiguous() or not weight.shape[0] or not weight.shape[1]):
        raise ValueError("Native ordinary GEMM requires Blackwell FP8 weights and FP32 block128 scales")
    n, k = weight.shape
    if tuple(scale.shape) != ((n + 127) // 128, k // 128):
        raise ValueError("Invalid original ordinary GEMM scale geometry")
    key = _key(weight, scale)
    previous = _LOOKUP.get(key)
    if previous is not None and previous.valid():
        return previous
    _, requant, _ = _runtime()
    # Real version counters allow graph lifetime guards to detect mutation.
    with torch.inference_mode(False), torch.no_grad():
        converted, packed = requant(weight, scale, [128, 128])
    if (_key(weight, scale) != key or converted.dtype != torch.float8_e4m3fn
            or converted.shape != weight.shape or packed.dtype != torch.int32
            or converted.device != weight.device or packed.device != weight.device):
        raise RuntimeError("Native ordinary preparation returned an invalid weight/scale contract")
    prepared = PreparedWeight(weight, scale, key, (_version(weight), _version(scale)), converted, packed)
    _LOOKUP[key] = prepared
    return prepared


def prepare_model(model):
    """Keep the prepared banks on their actual owning modules, never globally."""
    if not enabled():
        return
    import torch
    if os.environ.get("BATCHGEN_R1_MOE_BACKEND") != "deep_gemm":
        raise ValueError("Native ordinary GEMMs require the native resident R1 MoE profile")
    count = size = 0
    for layer in model.model.layers:
        owners = [(layer.self_attn, ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"))]
        if hasattr(layer.mlp, "shared_experts"):
            owners.append((layer.mlp.shared_experts, ("gate_proj", "up_proj", "down_proj")))
        # The first three dense MLPs already contain BF16 weights and use
        # torch Linear; they do not enter the legacy ordinary FP8 fallback.
        for owner, names in owners:
            module = getattr(owner, "module", owner)
            if not getattr(owner, "persistent", False):
                raise ValueError("Native ordinary GEMMs require persistent attention/shared weights")
            scales = owner.weight_dequant_scale
            bank = [prepare_weight(getattr(module, name).weight, scales[name + ".weight_scale_inv"])
                    for name in names]
            module._r1_native_gemm_weights = bank
            count += len(bank)
            size += sum(t.numel() * t.element_size() for item in bank for t in (item.weight, item.scale))
    torch.cuda.synchronize()
    logging.info("[R1-NATIVE-GEMM] prepared=%d resident_bytes=%d backend=DeepGEMM "
                 "policy=stock_SG_UE8M0 source_unchanged=true conversion_in_forward=false", count, size)


def graph_tensors(module):
    return tuple(t for item in getattr(module, "_r1_native_gemm_weights", ())
                 for t in (item.weight, item.scale))


def quantize(x):
    """Quantize BF16 once with power-of-two scales, then pack for native GEMM."""
    import torch
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Native ordinary quantization requires contiguous BF16 [M,K]")
    if x.shape[-1] % 128:
        raise ValueError("Native ordinary quantization requires K divisible by128")
    dg, _, quant = _runtime()
    if not x.shape[0]:
        return (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                torch.empty((0, x.shape[-1] // 512), dtype=torch.int32, device=x.device))
    a, scales = quant(x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
    if scales.dtype != torch.int32:
        scales = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(scales)
    return a, scales


def gemm(a, scales, weight, weight_scale, out=None):
    import torch
    entry = _LOOKUP.get(_key(weight, weight_scale))
    if entry is None or not entry.valid():
        raise RuntimeError("Native ordinary weight is absent/stale; prepare the model before forward")
    if (a.ndim != 2 or a.dtype != torch.float8_e4m3fn or not a.is_contiguous()
            or a.shape[-1] != weight.shape[-1] or a.device != weight.device
            or scales.device != a.device):
        raise ValueError("Native ordinary GEMM requires contiguous FP8 [M,K] on the weight device")
    if scales.dtype != torch.int32:
        raise ValueError("Native ordinary GEMM requires packed activation scales from its matching quantizer")
    if out is None:
        out = torch.empty((a.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=a.device)
    if (tuple(out.shape) != (a.shape[0], weight.shape[0]) or out.dtype != torch.bfloat16
            or out.device != a.device or not out.is_contiguous()):
        raise ValueError("Native ordinary GEMM output geometry/dtype mismatch")
    if a.shape[0]:
        _runtime()[0].fp8_gemm_nt((a, scales), (entry.weight, entry.scale), out)
    return out


def linear(weight, weight_scale, x):
    shape = x.shape
    a, scales = quantize(x.view(-1, shape[-1]))
    return gemm(a, scales, weight, weight_scale).view(*shape[:-1], weight.shape[0])


def linear_pair(weight_a, scale_a, weight_b, scale_b, x):
    """Two original projections with one local FP8 activation materialization.

    This neither fuses/reorders the GEMMs nor creates a combined weight bank.
    Both calls retain ``gemm``'s complete source lifetime and mutation checks.
    During graph capture the quantization remains a graph operation, so a
    replay consumes the current input rather than a cached activation value.
    """
    shape = x.shape
    a, scales = quantize(x.view(-1, shape[-1]))
    first = gemm(a, scales, weight_a, scale_a).view(*shape[:-1], weight_a.shape[0])
    second = gemm(a, scales, weight_b, scale_b).view(*shape[:-1], weight_b.shape[0])
    return first, second
