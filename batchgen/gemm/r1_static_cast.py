"""Inference-only reuse of immutable R1 router and rotary conversions.

The original source tensors remain authoritative. This caches a dtype conversion,
not model activations or a lower-precision router. A normal source write/rebind
invalidates it; CUDA Graph owners still guard the original weights/buffers.
"""
import os

import torch


ENV = "BATCHGEN_R1_STATIC_CAST_REUSE"
_ATTRIBUTE = "_batchgen_r1_static_casts"


def enabled():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(ENV + " must be 0 or 1")
    return value == "1"


def _key(tensor, *, require_version):
    try:
        version = tensor._version
    except RuntimeError:
        if require_version:
            raise RuntimeError("R1 cached router casts require versioned source weights")
        # Native rotary caches can be inference tensors. They are read-only
        # constants, replaced by the embedding when the context cap grows.
        version = None
    return (id(tensor), tensor.data_ptr(), tensor.untyped_storage().data_ptr(),
            version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device)


def _capturing(tensor):
    return tensor.device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def cached_cast(tensor, dtype, *, require_version=True):
    """Cache one static conversion on its source, shared by all source users.

    Eager warmup must populate it before capture. Inference-tensor source
    mutation through unversioned/external writers is prohibited, as it is for
    the existing RoPE graph inputs. Source lifetime owns the cache; there is
    no process-global map retaining old model weights.
    """
    if tensor.dtype == dtype:
        return tensor
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise RuntimeError("R1 static casts are inference-only")
    source_key = _key(tensor, require_version=require_version)
    entries = getattr(tensor, _ATTRIBUTE, None)
    item = None if entries is None else entries.get(dtype)
    if (item is None or item[0] != source_key
            or item[2] != _key(item[1], require_version=True)):
        if _capturing(tensor):
            raise RuntimeError("R1 static casts must be warmed outside CUDA Graph capture")
        with torch.inference_mode(False), torch.no_grad():
            value = tensor.to(dtype=dtype)
        item = (source_key, value, _key(value, require_version=True))
        if entries is None:
            entries = {}
            setattr(tensor, _ATTRIBUTE, entries)
        entries[dtype] = item
    value = item[1]
    if _capturing(tensor):
        # A graph can outlive a module's replaced Parameter. Retain its exact
        # converted allocation until that graph has been invalidated/freed.
        from batchgen.attention.mla.r1_decode_cuda_graph import _retain_capture_inputs
        _retain_capture_inputs(value)
    return value


def router_weight_fp32(weight):
    return cached_cast(weight, torch.float32) if enabled() else weight.float()


def rotary_values(cos, sin, x, seq_len):
    if not enabled():
        return cos[:seq_len].to(dtype=x.dtype), sin[:seq_len].to(dtype=x.dtype)
    if (cos.device != x.device or sin.device != x.device
            or cos.shape != sin.shape or seq_len < 0 or seq_len > cos.shape[0]):
        raise ValueError("R1 static RoPE cache device/shape/context differs")
    # Convert the full source cache once, then slice. Slicing first would
    # create a new source object on every forward and defeat safe reuse.
    return (cached_cast(cos, x.dtype, require_version=False)[:seq_len],
            cached_cast(sin, x.dtype, require_version=False)[:seq_len])
