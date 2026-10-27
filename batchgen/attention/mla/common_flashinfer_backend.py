"""Explicit BF16 R1 compatibility shims using one pinned FlashInfer fa2 API.

These names can replace only main's native attention imports under an explicit
selector. Projections, scales, RoPE, absorption and cache append stay with the
caller. This module selects no runtime itself and imports Torch/FlashInfer only
inside an actual call. It contains no graph, metadata fusion, weight/phase cache,
JIT specialization policy or GEMM tuning from the old performance branch.

Decode reinterprets native page64 cache as page1 token views and supplies only
valid token addresses. No KV content, including valid NaNs, is written or
sanitized. The shared pure metadata contracts also serve the SDPA oracle path.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

from .common_sdpa_backend import (
    LATENT_DIM, ROPE_DIM, _cpu_ints, _integer, _scale,
    plan_packed_prefill, plan_paged_decode, token_page_indices,
)


EXPECTED_FLASHINFER_VERSION = "0.5.2"
BACKEND_ID = "r1-common-flashinfer-0.5.2-fa2-bf16-v1"
WORKSPACE_BYTES = 128 * 1024**2


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
    lengths = _cpu_ints(cache_seqlens, "cache_seqlens", 1)
    for length in lengths:
        _integer(length, "inclusive cache length", minimum=1)
    return {"schema": "r1-common-fi-mla-metadata-v1", "backend_id": BACKEND_ID,
            "lengths": lengths, "num_heads": num_q_heads_per_kv_head,
            "num_kv_heads": 1}, None


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
    lengths = _cpu_ints(cache_seqlens, "cache_seqlens", 1)
    table = _cpu_ints(block_table, "block_table", 2)
    maximum = max(lengths, default=1)
    plan = plan_paged_decode(tuple(q.shape), tuple(k_cache.shape), table, lengths, maximum)
    expected = {"schema": "r1-common-fi-mla-metadata-v1", "backend_id": BACKEND_ID,
                "lengths": lengths, "num_heads": q.shape[2], "num_kv_heads": 1}
    if tile_scheduler_metadata != expected:
        raise ValueError("MLA metadata is missing, stale, or belongs to a different backend/shape")
    if not plan:
        return q.new_empty((*q.shape[:3], LATENT_DIM)), q.new_empty((0, q.shape[2], 1), dtype=torch.float32)
    metadata = token_page_indices(plan)
    flashinfer = _flashinfer()
    with torch.inference_mode():
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        wrapper = _mla_wrapper(workspace, flashinfer)
        q_indptr = torch.tensor(metadata["query_indptr"], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(metadata["kv_indptr"], dtype=torch.int32, device=device)
        kv_indices = torch.tensor(metadata["kv_indices"], dtype=torch.int32, device=device)
        kv_lengths = torch.tensor(metadata["kv_lengths"], dtype=torch.int32, device=device)
        wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lengths, q.shape[2],
                     LATENT_DIM, ROPE_DIM, 1, False, scale, torch.bfloat16, torch.bfloat16)
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
