"""Explicit R1 SM103 FP8 GEMM compatibility, preserving FP32 block scales.

Importing this module neither imports accelerator packages nor initializes CUDA.
The default remains DeepGEMM.  The opt-in path calls main's existing W8A8
wrapper with its four tensor arguments and unmodified launch defaults.
"""
import importlib
import logging
import os


COMPAT_ENV = "BATCHGEN_R1_SM103_FP8_COMPAT"
DEFAULT_BACKEND = "deep_gemm.fp8_gemm_nt"
COMPAT_BACKEND = "batchgen.gemm.w8a8.w8a8_gemm/default_launch/fp32_scales"
QUANT_BACKEND = "batchgen_kernels.triton.fp8_quantize.per_token_blocked_quantize_bf16_to_fp8_1d/default_block128"
_ACTUAL_BACKENDS = set()


def sm103_fp8_compat_enabled():
    value = os.environ.get(COMPAT_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{COMPAT_ENV} must be explicitly 0 or 1, got {value!r}")
    return value == "1"


def backend_receipt():
    """Report selection and dispatched functions without loading GPU libraries.

    This receipt is evidence of dispatch selection, not a CUDA synchronization
    or an arithmetic/full-model qualification result.
    """
    return {
        "selection_environment": {COMPAT_ENV: os.environ.get(COMPAT_ENV, "0")},
        "requested_backend": COMPAT_BACKEND if sm103_fp8_compat_enabled() else DEFAULT_BACKEND,
        "actual_backends": sorted(_ACTUAL_BACKENDS),
        "scope": "ordinary_fp8_nt_gemm_only_NOT_attention_or_grouped_moe",
    }


def load_deep_gemm():
    """Load the original dependency only when its path is actually selected."""
    return importlib.import_module("deep_gemm")


def _record_backend(backend):
    if backend not in _ACTUAL_BACKENDS:
        logging.getLogger(__name__).warning("R1 FP8 GEMM actual backend: %s", backend)
        _ACTUAL_BACKENDS.add(backend)


def _validate_sm103(torch, device):
    if device.type != "cuda":
        raise ValueError("SM103 FP8 compatibility requires a CUDA device")
    if os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("SM103 FP8 compatibility requires explicit DISABLE_MMA_V5=1")
    if tuple(torch.cuda.get_device_capability(device)) != (10, 3):
        raise RuntimeError("this explicit compatibility path is restricted to SM103")


def _validate_compat_contract(torch, lhs, rhs, out):
    a, a_scale = lhs
    w, w_scale = rhs
    if a.dim() != 2 or w.dim() != 2:
        raise ValueError("SM103 FP8 compatibility requires rank-2 A[M,K] and W[N,K]")
    m, k = a.shape
    n, k_w = w.shape
    if m < 0 or n <= 0 or k <= 0 or k != k_w:
        raise ValueError(f"invalid SM103 FP8 GEMM dimensions: A={tuple(a.shape)}, W={tuple(w.shape)}")
    if a.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise TypeError("SM103 compatibility requires original float8_e4m3fn A and W")
    if a_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise TypeError("SM103 compatibility requires FP32 scales; packed UE8M0 is not accepted")
    if tuple(a_scale.shape) != (m, (k + 127) // 128):
        raise ValueError("A scales must be [M,ceil(K/128)]")
    if tuple(w_scale.shape) != ((n + 127) // 128, (k + 127) // 128):
        raise ValueError("W scales must be [ceil(N/128),ceil(K/128)]")
    if a.device.type != "cuda" or any(t.device != a.device for t in (w, a_scale, w_scale)):
        raise ValueError("SM103 FP8 compatibility requires all tensors on the same CUDA device")
    if out is not None:
        if tuple(out.shape) != (m, n) or out.dtype != torch.bfloat16 or out.device != a.device:
            raise ValueError("SM103 FP8 compatibility output must be BF16 [M,N] on A's device")
        if not out.is_contiguous():
            raise ValueError("SM103 FP8 compatibility output must be contiguous")
    _validate_sm103(torch, a.device)
    return m, n


def w8a8_gemm(a, a_scale, w, w_scale, out=None):
    """SM103-only common four-tensor API, returning BF16 [M,N].

    Optional ``out`` preserves DeepGEMM's caller-owned output convention with
    a copy.  Without it the native W8A8 allocation is returned directly.
    """
    if not sm103_fp8_compat_enabled():
        raise RuntimeError(f"common SM103 W8A8 requires {COMPAT_ENV}=1")
    torch = importlib.import_module("torch")
    m, n = _validate_compat_contract(torch, (a, a_scale), (w, w_scale), out)
    if not m:
        result = out if out is not None else torch.empty((0, n), device=a.device, dtype=torch.bfloat16)
    else:
        kernel = importlib.import_module("batchgen.gemm.w8a8").w8a8_gemm
        # Preserve data, scales and strides by identity; use main's defaults.
        result = kernel(a, a_scale, w, w_scale)
        if (tuple(result.shape) != (m, n) or result.dtype != torch.bfloat16
                or result.device != a.device):
            raise RuntimeError("main W8A8 wrapper returned an invalid output contract")
        if out is not None:
            out.copy_(result)
            result = out
        _record_backend(COMPAT_BACKEND)
    return result


def w8a16_gemm(weight_data_fp8, weight_scale_inv_fp32, activation_bf16):
    """Common R1 BF16-activation entry: main block-128 quantizer + W8A8.

    The historical name is retained: activations are quantized to FP8, just as
    main's ``fa3_backend.w8a16_gemm`` already does.  This is not BF16 GEMM and
    does not accept BF16 checkpoint weights or add a bias.
    """
    if os.environ.get("BATCHGEN_R1_ORDINARY_GEMM", "legacy") != "legacy":
        from . import r1_native_gemm
        r1_native_gemm.enabled()  # Validate explicit selection before dispatch.
        result = r1_native_gemm.linear(weight_data_fp8, weight_scale_inv_fp32, activation_bf16)
        _record_backend("deep_gemm.fp8_gemm_nt/stock_SG_UE8M0/prepared_ordinary_weights")
        return result
    if not sm103_fp8_compat_enabled():
        raise RuntimeError(f"common SM103 W8A16 requires {COMPAT_ENV}=1")
    torch = importlib.import_module("torch")
    if activation_bf16.dtype != torch.bfloat16 or activation_bf16.dim() not in (2, 3):
        raise TypeError("common R1 GEMM requires rank-2 or rank-3 BF16 activations")
    if not activation_bf16.is_contiguous():
        raise ValueError("common R1 GEMM requires contiguous BF16 activations, as main act_quant does")
    if weight_data_fp8.dim() != 2 or weight_data_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("common R1 GEMM requires rank-2 original FP8 weights")
    n, k = weight_data_fp8.shape
    if n <= 0 or k <= 0 or activation_bf16.shape[-1] != k:
        raise ValueError("common R1 GEMM activation and weight dimensions do not match")
    if (weight_scale_inv_fp32.dtype != torch.float32
            or tuple(weight_scale_inv_fp32.shape) != ((n + 127) // 128, (k + 127) // 128)):
        raise ValueError("common R1 GEMM requires original FP32 [ceil(N/128),ceil(K/128)] weight scales")
    if any(t.device != activation_bf16.device for t in (weight_data_fp8, weight_scale_inv_fp32)):
        raise ValueError("common R1 GEMM requires all inputs on the same device")
    _validate_sm103(torch, activation_bf16.device)
    x = activation_bf16.view(-1, activation_bf16.shape[-1])
    if x.shape[0]:
        quantize = importlib.import_module("batchgen_kernels.triton.fp8_quantize").per_token_blocked_quantize_bf16_to_fp8_1d
        a, a_scale = quantize(x)
        _record_backend(QUANT_BACKEND)
    else:
        k = x.shape[1]
        a = torch.empty((0, k), device=x.device, dtype=torch.float8_e4m3fn)
        a_scale = torch.empty((0, (k + 127) // 128), device=x.device, dtype=torch.float32)
    result = w8a8_gemm(a, a_scale, weight_data_fp8, weight_scale_inv_fp32)
    return result.view(*activation_bf16.shape[:-1], weight_data_fp8.shape[0])


def fp8_gemm_nt(lhs, rhs, out, **kwargs):
    """DeepGEMM-style output API with explicit SM103 dispatch.

    FP8 data and FP32 scales are forwarded by identity.  No scale rounding,
    checkpoint conversion, tile override, autotuning, or input sanitization is
    performed.  This ordinary GEMM path does not implement accumulation,
    recipes, masked grouped GEMM, or quantization of BF16 activations.
    """
    if not sm103_fp8_compat_enabled():
        result = load_deep_gemm().fp8_gemm_nt(lhs, rhs, out, **kwargs)
        _record_backend(DEFAULT_BACKEND)
        return result

    unknown = set(kwargs) - {"disable_ue8m0_cast", "recipe"}
    if unknown or kwargs.get("recipe") is not None:
        raise ValueError("SM103 FP8 compatibility does not support recipes or additional DeepGEMM options")
    if kwargs.get("disable_ue8m0_cast", True) is not True:
        raise ValueError("SM103 FP8 compatibility preserves FP32 scales and cannot request UE8M0 casting")
    w8a8_gemm(lhs[0], lhs[1], rhs[0], rhs[1], out=out)
    return None
