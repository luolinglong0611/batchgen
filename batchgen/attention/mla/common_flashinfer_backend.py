"""Explicit BF16 R1 compatibility shims using one pinned FlashInfer fa2 API.

These names can replace only main's native attention imports under an explicit
selector. Projections, scales, RoPE, absorption and cache append stay with the
caller. This module selects no runtime itself and imports Torch/FlashInfer only
inside an actual call. Explicit eager decode-forward scopes share only the
compatibility adapter's preparation across layers. No state survives a forward;
there is no graph, weight/phase cache or old-branch GEMM tuning.

Decode reinterprets native page64 cache as page1 token views and supplies only
valid token addresses. No KV content, including valid NaNs, is written or
sanitized. The shared pure metadata contracts also serve the SDPA oracle path.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .common_sdpa_backend import (
    LATENT_DIM, ROPE_DIM, _cpu_ints, _integer, _scale,
    plan_packed_prefill, plan_paged_decode, token_page_indices,
)


EXPECTED_FLASHINFER_VERSION = "0.5.2"
BACKEND_ID = "r1-common-flashinfer-0.5.2-fa2-bf16-v1"
WORKSPACE_BYTES = 128 * 1024**2
_DECODE_FORWARD_STATE = ContextVar("r1_fa2_decode_forward_state", default=None)


def _tensor_geometry(tensor):
    # Q/KV addresses and values deliberately do not belong to the plan key:
    # every layer supplies different data with the same physical layout.
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.layout,
            tensor.dtype, tensor.device)


def _metadata_geometry(tensor):
    return _tensor_geometry(tensor) + (tensor.storage_offset(),)


def _tensor_version(tensor, name, *, allow_inference=False):
    try:
        return tensor._version
    except RuntimeError as exc:
        if allow_inference and "inference" in str(exc).lower():
            return None
        raise RuntimeError(
            f"Scoped FA2 requires versioned {name}; use a normal tensor or the unscoped path"
        ) from exc


def _metadata_payload(lengths, num_heads):
    # Return independent mutable containers for the existing public API. The
    # scope's authoritative snapshot is a tuple, never a caller-owned dict/list.
    return {"schema": "r1-common-fi-mla-metadata-v1", "backend_id": BACKEND_ID,
            "lengths": list(lengths), "num_heads": num_heads, "num_kv_heads": 1}


class _DecodeForwardState:
    def __init__(self, manager):
        if not callable(getattr(manager, "get_page_table_version", None)):
            raise RuntimeError("Scoped FA2 requires a versioned GPU page-table manager")
        if not hasattr(getattr(manager, "_gpu_page_table_manager", None), "gpu_table"):
            raise RuntimeError("Scoped FA2 requires the manager's active GPU page table")
        self.manager = manager
        self.manager_version = manager.get_page_table_version()
        self.stream_key = None
        self.length_tensor = None
        self.length_signature = None
        self.lengths = None
        self.num_heads = None
        self.table = None
        self.table_signature = None
        self.attention_signature = None
        self.prepared = None
        self.planned = False

    def clear(self):
        # A caller or exception traceback may retain the yielded state after
        # exit. Release its workspace and tensor/manager references regardless.
        self.prepared = None
        self.length_tensor = None
        self.table = None
        self.manager = None
        self.lengths = None
        self.length_signature = None
        self.table_signature = None
        self.attention_signature = None
        self.stream_key = None
        self.num_heads = None
        self.manager_version = None
        self.planned = False

    def check_manager(self):
        if self.manager.get_page_table_version() != self.manager_version:
            raise ValueError("GPU page-table generation changed inside one FA2 decode forward")

    def check_stream(self, device):
        torch = _torch()
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("Scoped FA2 preparation requires eager CUDA execution")
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        if self.stream_key is None:
            self.stream_key = key
        elif self.stream_key != key:
            raise ValueError("CUDA device or stream changed inside one FA2 decode forward")

    def bind_lengths(self, tensor, num_heads):
        torch = _torch()
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache_seqlens must be an int32/int64 tensor of rank 1")
        self.check_manager()
        self.check_stream(tensor.device)
        signature = (_metadata_geometry(tensor), _tensor_version(tensor, "cache_seqlens"))
        if self.length_tensor is None:
            lengths = _cpu_ints(tensor, "cache_seqlens", 1)
            for length in lengths:
                _integer(length, "inclusive cache length", minimum=1)
            self.length_tensor = tensor
            self.length_signature = signature
            self.lengths = tuple(lengths)
            self.num_heads = num_heads
        elif (tensor is not self.length_tensor or signature != self.length_signature
              or num_heads != self.num_heads):
            raise ValueError("cache_seqlens or query-head count changed inside one FA2 decode forward")
        return self.lengths

    def bind_attention(self, q, k_cache, block_table, scale, causal):
        torch = _torch()
        self.check_manager()
        self.check_stream(q.device)
        if (not isinstance(block_table, torch.Tensor) or block_table.ndim != 2
                or block_table.dtype not in (torch.int32, torch.int64)
                or block_table.device != q.device):
            raise ValueError("Scoped FA2 block_table must be an integer matrix on the query CUDA device")
        if block_table is not self.manager._gpu_page_table_manager.gpu_table:
            raise ValueError("FA2 block_table does not belong to the scoped GPU manager")
        signature = (_metadata_geometry(block_table),
                     _tensor_version(block_table, "block_table", allow_inference=True))
        attention_signature = (_tensor_geometry(q), _tensor_geometry(k_cache), scale, causal)
        if self.table is None:
            self.table = block_table
            self.table_signature = signature
            self.attention_signature = attention_signature
        elif (block_table is not self.table or signature != self.table_signature
              or attention_signature != self.attention_signature):
            raise ValueError("Attention layout, page table, scale or causal mode changed inside one FA2 decode forward")


@contextmanager
def decode_forward_scope(manager):
    """Share FA2 preparation only within one serial eager model decode forward.

    The owner must keep ordered rows/lengths/page mappings fixed until return.
    Normal length tensors are required so in-place changes are detectable. An
    inference-mode page table may lack a tensor version: its manager's existing
    rebuild/clear generation plus ownership/identity/geometry guard it instead.
    Each entry creates fresh state; pointer reuse across steps is never a cache
    key. The same-stream workspace is private to this scope, including on error.
    """
    if _DECODE_FORWARD_STATE.get() is not None:
        raise RuntimeError("Nested FA2 decode-forward scopes are unsupported")
    state = _DecodeForwardState(manager)
    token = _DECODE_FORWARD_STATE.set(state)
    try:
        yield state
    finally:
        _DECODE_FORWARD_STATE.reset(token)
        state.clear()


def _torch():
    import torch
    return torch


def _flashinfer():
    version = importlib.metadata.version("flashinfer-python")
    if version != EXPECTED_FLASHINFER_VERSION:
        raise RuntimeError(f"Common attention requires flashinfer-python={EXPECTED_FLASHINFER_VERSION}; got {version}")
    import flashinfer
    return flashinfer


def _inputs(*tensors):
    torch = _torch()
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("Common FI attention requires Torch tensor inputs")
    device = tensors[0].device
    if device.type != "cuda" or any(
        tensor.dtype != torch.bfloat16 or tensor.device != device for tensor in tensors
    ):
        raise ValueError("Common FI attention requires BF16 tensors on one CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("This compatibility candidate requires Graph OFF")
    return torch, device


def _ragged_wrapper(workspace, flashinfer):
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=False, backend="fa2",
    )
    if getattr(wrapper, "_backend", None) != "fa2":
        raise RuntimeError("FlashInfer did not retain the explicit fa2 prefill backend")
    return wrapper


def _mla_wrapper(workspace, flashinfer):
    wrapper = flashinfer.BatchMLAPagedAttentionWrapper(
        workspace, use_cuda_graph=False, backend="fa2",
    )
    if getattr(wrapper, "_backend", None) != "fa2":
        raise RuntimeError("FlashInfer did not retain the explicit fa2 MLA backend")
    return wrapper


def runtime_identity():
    """Actual imported files and hashes, not a model/primitive pass receipt."""
    torch, flashinfer = _torch(), _flashinfer()
    files = {"shim": str(Path(__file__).resolve())}
    for name in ("BatchPrefillWithRaggedKVCacheWrapper", "BatchMLAPagedAttentionWrapper"):
        files[name] = inspect.getsourcefile(getattr(flashinfer, name))
    return {"backend_id": BACKEND_ID, "flashinfer_version": importlib.metadata.version("flashinfer-python"),
            "torch_version": torch.__version__, "graph_scope": "off",
            "kernel_backend": "fa2", "allocator_page_size": 64, "kernel_decode_page_size": 1,
            "decode_preparation_reuse": "explicit-eager-forward-scope-v1",
            "decode_preparation_lifetime": "one-forward; no cross-step reuse; unscoped calls remain fresh",
            "files": {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                      for name, path in files.items()},
            "status": "runtime_identity_NOT_qualification"}


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False,
    **kwargs,
):
    """Main-compatible fresh prefill call; output [T,H,128] in BF16.

    Cached prefixes, asymmetric Q/K, dropout, custom windows/bias, softcaps and
    attention-probability output are intentionally unsupported and rejected.
    The caller supplies already projected and RoPE-transformed Q/K192, V128.
    """
    if kwargs or dropout_p != 0 or window_size != (-1, -1) or softcap != 0:
        raise ValueError("Unsupported attention option in common fresh-prefill shim")
    if alibi_slopes is not None or return_attn_probs or deterministic:
        raise ValueError("Unsupported bias/probability/determinism option in common prefill shim")
    if causal is not True:
        raise ValueError("Common fresh-prefill shim requires causal=True")
    scale = _scale(softmax_scale)
    torch, device = _inputs(q, k, v)
    cu_q = _cpu_ints(cu_seqlens_q, "cu_seqlens_q", 1)
    cu_k = _cpu_ints(cu_seqlens_k, "cu_seqlens_k", 1)
    if cu_q != cu_k or max_seqlen_q != max_seqlen_k:
        raise ValueError("Common fresh-prefill shim requires identical full-sequence Q/K boundaries")
    spans = plan_packed_prefill(tuple(q.shape), tuple(k.shape), tuple(v.shape), cu_q, max_seqlen_q)
    if not spans:
        return v.new_empty(v.shape)
    flashinfer = _flashinfer()
    with torch.inference_mode():
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        wrapper = _ragged_wrapper(workspace, flashinfer)
        cu = torch.tensor(cu_q, dtype=torch.int32, device=device)
        wrapper.plan(
            qo_indptr=cu, kv_indptr=cu,
            num_qo_heads=q.shape[1], num_kv_heads=k.shape[1],
            head_dim_qk=q.shape[2], head_dim_vo=v.shape[2],
            causal=True, sm_scale=scale, q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16, pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
        )
        output = wrapper.run(q, k, v)
        if tuple(output.shape) != tuple(v.shape) or output.dtype != torch.bfloat16 or output.device != device:
            raise RuntimeError("FlashInfer prefill returned an unexpected shape/dtype/device")
        return output


def get_mla_metadata(cache_seqlens, num_q_heads_per_kv_head, num_kv_heads=1, **kwargs):
    """Checked CPU metadata snapshot for main's existing FlashMLA call shape.

    This is shim metadata, not native FlashMLA scheduler output. It is consumed
    only by the matching function below and never passed to another binary.
    """
    if kwargs:
        raise ValueError("Unsupported native scheduler option in common MLA shim")
    _integer(num_q_heads_per_kv_head, "query heads", minimum=1)
    if type(num_kv_heads) is not int or num_kv_heads != 1:
        raise ValueError("Original R1 MLA has one latent KV head")
    scope = _DECODE_FORWARD_STATE.get()
    if scope is None:
        lengths = _cpu_ints(cache_seqlens, "cache_seqlens", 1)
        for length in lengths:
            _integer(length, "inclusive cache length", minimum=1)
    else:
        lengths = scope.bind_lengths(cache_seqlens, num_q_heads_per_kv_head)
    return _metadata_payload(lengths, num_q_heads_per_kv_head), None


def _token_page_indices_array(decode_plan):
    """Vectorize the validated page64 -> page1 metadata on CPU.

    This changes only host metadata construction. Every call owns fresh arrays;
    no page table, wrapper, layer or step is cached. ``plan_paged_decode`` has
    already validated nonnegative page ids, sequence bounds and page coverage.
    The int32 overflow check operates on used physical pages before expansion,
    preserving the reference check for every addressed token, including partial
    final pages. Unused page-table tails are never addressed or checked here.
    """
    import numpy as np

    int32_max = 2**31 - 1
    indptr = [0]
    used_rows = []
    lengths = []
    for record in decode_plan:
        length = record["length"]
        pages = record["physical_pages"][: (length + 63) // 64]
        total = indptr[-1] + length
        # For nonnegative page ids, page*64+offset exceeds INT32_MAX iff
        # page > INT32_MAX//64, for every valid offset in [0,63].
        if total > int32_max or (pages and max(pages) > int32_max // 64):
            raise ValueError("FlashInfer int32 token metadata would overflow")
        indptr.append(total)
        lengths.append(length)
        used_rows.append(pages)

    indices = np.empty(indptr[-1], dtype=np.int32)
    offsets = np.arange(64, dtype=np.int32)
    for row, (length, pages) in enumerate(zip(lengths, used_rows)):
        if length:
            page_ids = np.asarray(pages, dtype=np.int32)
            expanded = page_ids[:, None] * np.int32(64) + offsets
            indices[indptr[row]:indptr[row + 1]] = expanded.reshape(-1)[:length]
    return {"query_indptr": np.arange(len(decode_plan) + 1, dtype=np.int32),
            "kv_indptr": np.asarray(indptr, dtype=np.int32),
            "kv_indices": indices, "kv_lengths": np.asarray(lengths, dtype=np.int32),
            "kernel_page_size": 1, "allocator_page_size": 64}


def _prepare_mla_decode(torch, device, query_shape, cache_shape, table, lengths, scale):
    maximum = max(lengths, default=1)
    plan = plan_paged_decode(query_shape, cache_shape, table, lengths, maximum)
    if not plan:
        return None
    metadata = _token_page_indices_array(plan)
    flashinfer = _flashinfer()
    with torch.inference_mode():
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        wrapper = _mla_wrapper(workspace, flashinfer)
        q_indptr = torch.tensor(metadata["query_indptr"], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(metadata["kv_indptr"], dtype=torch.int32, device=device)
        kv_indices = torch.tensor(metadata["kv_indices"], dtype=torch.int32, device=device)
        kv_lengths = torch.tensor(metadata["kv_lengths"], dtype=torch.int32, device=device)
        wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lengths, query_shape[2],
                     LATENT_DIM, ROPE_DIM, 1, False, scale, torch.bfloat16, torch.bfloat16)
    return wrapper


def flash_mla_with_kvcache(
    q, k_cache, block_table, cache_seqlens, head_dim_v,
    tile_scheduler_metadata=None, num_splits=None, softmax_scale=None,
    causal=True, descale_q=None, descale_k=None, **kwargs,
):
    """Main-compatible BF16 dense paged single-last-token MLA call.

    Input query [B,1,H,576], native cache [P,64,1,576]. Returns output
    [B,1,H,512] and LSE [B,H,1]. Caller must already have appended the current
    token and must retain stable metadata/current-stream KV readiness.
    Native causal=True for this one last query maps to FI causal=False over
    the precisely gathered valid history; no future token is addressed.
    """
    if kwargs or descale_q is not None or descale_k is not None:
        raise ValueError("Only original BF16 MLA without descale is supported")
    if type(causal) is not bool or head_dim_v != LATENT_DIM or type(head_dim_v) is not int:
        raise ValueError("Expected an explicit causal bool and latent value dimension512")
    if num_splits is not None:
        raise ValueError("Native FlashMLA split metadata cannot enter the common FI shim")
    scale = _scale(softmax_scale)
    torch, device = _inputs(q, k_cache)
    if not k_cache.is_contiguous():
        raise ValueError("Native MLA cache must be contiguous before its page1 view")
    if q.ndim != 4:
        raise ValueError("Decode query must have four dimensions")
    scope = _DECODE_FORWARD_STATE.get()
    if scope is None:
        lengths = _cpu_ints(cache_seqlens, "cache_seqlens", 1)
    else:
        lengths = scope.bind_lengths(cache_seqlens, q.shape[2])
        scope.bind_attention(q, k_cache, block_table, scale, causal)
    expected = _metadata_payload(lengths, q.shape[2])
    if tile_scheduler_metadata != expected:
        raise ValueError("MLA metadata is missing, stale, or belongs to a different backend/shape")
    if scope is None or not scope.planned:
        table = _cpu_ints(block_table, "block_table", 2)
        wrapper = _prepare_mla_decode(torch, device, tuple(q.shape), tuple(k_cache.shape), table, lengths, scale)
        if scope is not None:
            scope.prepared = wrapper
            scope.planned = True
    else:
        wrapper = scope.prepared
    if wrapper is None:
        return q.new_empty((*q.shape[:3], LATENT_DIM)), q.new_empty((0, q.shape[2], 1), dtype=torch.float32)
    with torch.inference_mode():
        query = q[:, 0]
        cache = k_cache.view(-1, 1, LATENT_DIM + ROPE_DIM)
        # Keep the native576 row stride; neither full cache nor valid values
        # are copied or rewritten. SG's FI MLA uses these same feature views.
        output, lse = wrapper.run(query[..., :LATENT_DIM], query[..., LATENT_DIM:],
                                  cache[..., :LATENT_DIM], cache[..., LATENT_DIM:],
                                  return_lse=True)
        if tuple(output.shape) != (q.shape[0], q.shape[2], LATENT_DIM) or output.dtype != torch.bfloat16 or output.device != device:
            raise RuntimeError("FlashInfer MLA returned an unexpected output shape/dtype/device")
        if tuple(lse.shape) != (q.shape[0], q.shape[2]) or lse.dtype != torch.float32 or lse.device != device:
            raise RuntimeError("FlashInfer MLA returned unexpected LSE")
        return output.unsqueeze(1), lse.unsqueeze(-1)
