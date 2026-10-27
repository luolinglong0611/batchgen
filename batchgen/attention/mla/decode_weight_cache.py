"""Bounded BF16 MLA decode-weight reuse for explicitly resident attention.

This cache reuses the original dequantizer's complete, unmodified matrix; the
caller still owns all views, slices and projection operations. It is disabled
unless the weight owner explicitly enables it after loading resident weights.

Normal PyTorch in-place updates are detected through version counters. Unsafe
updates through ``tensor.data`` or an external writer can bypass those counters:
the owner MUST call ``clear_mla_decode_weight_cache`` before such an update.
The load/unload and device-move hooks also clear the cache. This is inference
reuse, not a transparent cache for arbitrary parameter mutation or training.
"""

import torch


_ENABLED = "_mla_decode_weight_cache_enabled"
_ENTRY = "_mla_decode_weight_cache_entry"
_SCALE = "kv_b_proj.weight_scale_inv"
_GEOMETRY = ("num_heads", "qk_nope_head_dim", "v_head_dim", "kv_lora_rank")


def clear_mla_decode_weight_cache(module):
    """Release the cached matrix and source aliases without changing opt-in."""
    setattr(module, _ENTRY, None)


def configure_mla_decode_weight_cache(module, enabled=True):
    """Opt in only after loading resident weights; reset any previous entry."""
    clear_mla_decode_weight_cache(module)
    setattr(module, _ENABLED, bool(enabled))


def _tensor_key(tensor):
    # Use the actual Parameter, not a fresh .data wrapper: its version tracks
    # normal in-place parameter updates. Inference tensors raise at _version.
    return (
        id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
        tuple(tensor.stride()), tensor.dtype, tensor.device,
        tensor.storage_offset(), tensor.ndim,
    )


def get_mla_decode_weight(module, scales, dequantize):
    """Return the original BF16 matrix, reusing it only while sources are stable.

    There is at most one entry per attention module. A different CUDA stream
    rebuilds the result on that stream instead of consuming a result whose
    producer has not completed. Callers must consume it on the current stream,
    as in the original inline dequantization path; cross-stream consumers need
    their own stream ordering and tensor lifetime handling.

    Unversioned inference tensors, training/grad-enabled execution, non-CUDA
    inputs and modules without explicit opt-in always execute the original
    dequantizer and clear any old entry. No tensor contents are read on the CPU.
    """
    weight = module.kv_b_proj.weight
    scale = scales[_SCALE]
    key = None
    if (
        getattr(module, _ENABLED, False)
        and not module.training
        and not torch.is_grad_enabled()
        and weight.device.type == "cuda"
        and scale.device == weight.device
    ):
        try:
            key = (
                _tensor_key(weight), _tensor_key(scale),
                tuple(getattr(module, name, None) for name in _GEOMETRY),
                torch.cuda.current_stream(weight.device).cuda_stream,
                id(dequantize),
            )
        except RuntimeError:
            # Inference tensors have no usable mutation counter. Recomputing
            # is safer than treating an unavailable counter as a stable value.
            key = None

    entry = getattr(module, _ENTRY, None)
    if key is not None and entry is not None and entry[0] == key:
        return entry[-1]

    # Drop old-device/source/stream references before allocating a replacement.
    clear_mla_decode_weight_cache(module)
    entry = None
    result = dequantize(weight.data, scale)
    if (
        key is not None
        and result.dtype == torch.bfloat16
        and result.ndim == 2
        and tuple(result.shape) == tuple(weight.shape)
    ):
        # Retain actual objects for identity checks and detached storage aliases
        # so a .data rebind cannot free/reuse the previous storage address.
        setattr(module, _ENTRY, (
            key, weight, scale, weight.detach(), scale.detach(), dequantize, result,
        ))
    return result
