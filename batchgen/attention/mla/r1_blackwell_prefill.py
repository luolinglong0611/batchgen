"""Opt-in R1 fresh prefill through FlashInfer 0.5.2's Blackwell CUTLASS FMHA.

Only the materialized attention operation changes. The caller still projects
Q/K192 and V128, applies RoPE, and stores normalized latent512 + rotated key64
in Host KV. This module neither reads nor writes that compressed KV cache.

There is one retained plan and 32 MiB workspace per calling CPU thread, shared
by every layer on the same CUDA stream. Metadata values, geometry and stream
are part of the cache key; Q/K/V addresses are not. No tensor values or device
metadata are copied to the CPU. Decode continues to use its existing backend.
"""
from __future__ import annotations

from functools import cache
import importlib.metadata
import logging
import os
from threading import local
from types import SimpleNamespace

from .common_sdpa_backend import _scale, plan_packed_prefill


ENV = "BATCHGEN_R1_PREFILL_ATTENTION_BACKEND"
BACKEND_ID = "r1-flashinfer-0.5.2-cutlass-sm100-bf16-fresh-v1"
FLASHINFER_VERSION = "0.5.2"
WORKSPACE_BYTES = 32 * 1024**2
PLAN_CAPACITY = 131072
Q_TILE_SIZE = 256
_STATE = local()


def enabled():
    value = os.environ.get(ENV, "legacy")
    if value not in ("legacy", "cutlass-sm100"):
        raise ValueError(f"{ENV} must be legacy or cutlass-sm100")
    return value == "cutlass-sm100"


def packed_contract(q_shape, k_shape, v_shape, cu_q, cu_k, max_q, max_k, scale):
    """Validate fresh-sequence math and native scheduler capacity on the CPU."""
    scale = _scale(scale)
    spans = plan_packed_prefill(q_shape, k_shape, v_shape, cu_q, max_q)
    plan_packed_prefill(q_shape, k_shape, v_shape, cu_k, max_k)
    if tuple(cu_q) != tuple(cu_k) or max_q != max_k:
        raise ValueError("CUTLASS R1 prefill requires identical fresh Q/K boundaries")
    if not spans:
        raise ValueError("CUTLASS R1 prefill requires at least one nonempty sequence")
    if any(d >= 2**31 for d in (*q_shape, *k_shape, *v_shape)):
        raise ValueError("CUTLASS prefill dimensions must fit native int32")
    if cu_q[-1] >= 2**31:
        raise ValueError("CUTLASS cumulative token offsets must fit int32")
    work_items = sum((end - start + Q_TILE_SIZE - 1) // Q_TILE_SIZE
                     for start, end in spans) * q_shape[1]
    if work_items > PLAN_CAPACITY:
        raise ValueError("CUTLASS prefill exceeds the native 131072 work-item capacity")
    return tuple(cu_q), scale, work_items


def clear_plan_cache():
    """Release this CPU thread's retained buffers; native JIT code stays cached."""
    _STATE.entry = None
    _STATE.plans = 0
    _STATE.hits = 0


def cache_stats():
    entry = getattr(_STATE, "entry", None)
    return {
        "backend_id": BACKEND_ID,
        "plans": getattr(_STATE, "plans", 0),
        "hits": getattr(_STATE, "hits", 0),
        "retained_entries": int(entry is not None),
        "workspace_bytes": WORKSPACE_BYTES if entry is not None else 0,
        "plan_storage_bytes": entry.plan_storage_bytes if entry is not None else 0,
        "retained_gpu_bytes": (WORKSPACE_BYTES + entry.plan_storage_bytes
                               if entry is not None else 0),
        "capacity_work_items": PLAN_CAPACITY,
        "native_module_file": getattr(entry.module, "__file__", None) if entry else None,
        "cu_seqlens": list(entry.cu_values) if entry else None,
        "graph_scope": "eager-prefill-only; existing-decode-graphs-unchanged",
    }


def _torch():
    import torch
    return torch


@cache
def _flashinfer_prefill():
    actual = importlib.metadata.version("flashinfer-python")
    if actual != FLASHINFER_VERSION:
        raise RuntimeError(f"CUTLASS R1 prefill requires flashinfer-python={FLASHINFER_VERSION}; got {actual}")
    from flashinfer import prefill
    return prefill


def _native_module(torch, device):
    # This loader already caches the compiled module in FlashInfer. Keeping
    # its 0.5.2 ABI avoids a package upgrade changing the existing MLA decoder.
    return _flashinfer_prefill().get_fmha_module(
        torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32,
        192, 128, 0, False, False, device,
    )


def _host_cu(value, name, torch):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if (not isinstance(value, torch.Tensor) or value.ndim != 1
            or value.device.type != "cpu" or value.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f"{name} must be a CPU int32/int64 vector; CUDA metadata would cause layer-wise synchronization")
    return tuple(value.tolist())


def _geometry(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _prepare(q, k, v, cu_values, max_seqlen, scale, torch):
    device = q.device
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device, stream, _geometry(q), _geometry(k), _geometry(v),
           cu_values, max_seqlen, scale)
    old = getattr(_STATE, "entry", None)
    if old is not None and old.key == key:
        _STATE.hits = getattr(_STATE, "hits", 0) + 1
        return old

    # Invalidate before replanning: a failed native plan must never leave an
    # apparently reusable old key pointing at partially overwritten buffers.
    _STATE.entry = None
    if old is not None and old.device == device and old.stream == stream:
        entry = old
    else:
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability not in ((10, 0), (10, 3)):
            raise ValueError("This CUTLASS candidate supports only qualified SM100/SM103 GPUs")
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # Native 0.5.2 FwdRunner queries device0's SM count internally. Reject
        # heterogeneous counts instead of silently constructing a wrong plan.
        if torch.cuda.get_device_properties(0).multi_processor_count != num_sms:
            raise ValueError("FlashInfer 0.5.2 CUTLASS requires device0 and target to have equal SM counts")
        entry = SimpleNamespace(
            key=None, device=device, stream=stream, num_sms=num_sms,
            module=_native_module(torch, device),
            workspace=torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            work_indptr=torch.empty(num_sms + 1, dtype=torch.int32, device=device),
            qo_tiles=torch.empty(PLAN_CAPACITY, dtype=torch.int32, device=device),
            heads=torch.empty(PLAN_CAPACITY, dtype=torch.int32, device=device),
            batches=torch.empty(PLAN_CAPACITY, dtype=torch.int32, device=device),
        )
    entry.cu = torch.tensor(cu_values, dtype=torch.int32, device=device)
    entry.cu_values = cu_values
    entry.module.plan(
        entry.cu, entry.cu, entry.work_indptr, entry.qo_tiles, entry.heads,
        entry.batches, Q_TILE_SIZE, q.shape[1], entry.num_sms, True,
    )
    entry.plan_storage_bytes = 4 * (3 * PLAN_CAPACITY + entry.num_sms + 1 + len(cu_values))
    entry.key = key
    _STATE.entry = entry
    _STATE.plans = getattr(_STATE, "plans", 0) + 1
    logging.info("[R1-PREFILL-CUTLASS] plan=%d hits=%d rows=%d tokens=%d "
                 "workspace_bytes=%d plan_storage_bytes=%d same_stream_reuse=true",
                 _STATE.plans, getattr(_STATE, "hits", 0), len(cu_values) - 1,
                 q.shape[0], WORKSPACE_BYTES, entry.plan_storage_bytes)
    return entry


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False,
    **kwargs,
):
    """Return BF16 NHD output for complete causal sequences, with no KV mutation.

    Prefix/append attention, asymmetric lengths, custom masks/windows/bias,
    dropout, LSE/probability output, FP8 KV and CUDA Graph prefill are rejected.
    The 0.5.2 native ABI is deliberately not used with a newer FlashInfer.
    """
    if (kwargs or dropout_p != 0 or causal is not True or window_size != (-1, -1)
            or softcap != 0 or alibi_slopes is not None or deterministic or return_attn_probs):
        raise ValueError("CUTLASS R1 prefill supports only fresh causal attention without extra options")
    torch = _torch()
    if any(not isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise ValueError("CUTLASS R1 prefill requires Torch tensor Q/K/V")
    device = q.device
    if device.type != "cuda" or any(x.device != device or x.dtype != torch.bfloat16
                                    or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("CUTLASS R1 prefill requires contiguous BF16 NHD Q/K/V on one CUDA device")
    cu_q = _host_cu(cu_seqlens_q, "cu_seqlens_q", torch)
    cu_k = _host_cu(cu_seqlens_k, "cu_seqlens_k", torch)
    cu, scale, _ = packed_contract(tuple(q.shape), tuple(k.shape), tuple(v.shape),
                                   cu_q, cu_k, max_seqlen_q, max_seqlen_k, softmax_scale)
    with torch.inference_mode(), torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUTLASS R1 prefill planning and execution require eager execution")
        entry = _prepare(q, k, v, cu, max_seqlen_q, scale, torch)
        # Supplying out also avoids FI's convenience allocation of a hidden
        # max-sequence-sized output prefix. This matches its ragged wrapper.
        out = torch.empty(tuple(v.shape), dtype=torch.bfloat16, device=device)
        try:
            entry.module.run(
                entry.workspace, q, k, v, entry.cu, entry.cu, entry.work_indptr,
                entry.qo_tiles, entry.heads, entry.batches, out, None, 1, scale,
                q.shape[1], k.shape[1], q.shape[2], v.shape[2], max_seqlen_q,
            )
        except Exception:
            _STATE.entry = None
            raise
        return out
