"""Explicit common BF16 attention primitives for correctness on SM103.

This independent module is not selected by main's model, preflight or dispatch.
It imports neither FA3, FlashMLA, FlashInfer, DeepGEMM nor Torch at module load.
Both frameworks may call these exact primitives after their projection/RoPE
boundary. They do not implement projection, normalization, RoPE, KV append,
weight absorption, offload, or scheduling, and therefore do not qualify a model.

The explicit MATH path computes on FP32 Q/K/V and returns BF16. That is a new
correctness compatibility implementation, not the paper's native attention
kernel. There are no graphs, cached weights, fused metadata or autotuned tiles.
Packed prefill handles complete fresh sequences only, with absolute causal
masking and bounded query chunks. Decode gathers only valid logical KV tokens:
unused physical page tails and unrelated pages never enter attention arithmetic,
and the source cache is not written or sanitized.
"""
from __future__ import annotations

import math


BACKEND_ID = "r1-common-sdpa-math-fp32-bf16-v1"
PAGE_SIZE = 64
LATENT_DIM = 512
ROPE_DIM = 64
PREFILL_QK_DIM = 192
PREFILL_V_DIM = 128
QUERY_CHUNK = 32


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _shape(value, ndim, name):
    if not isinstance(value, (list, tuple)) or len(value) != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return tuple(_integer(dimension, f"{name} dimension") for dimension in value)


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Pass the model's positive finite softmax_scale explicitly")
    return float(value)


def plan_packed_prefill(q_shape, k_shape, v_shape, cu_seqlens, max_seqlen):
    """CPU layout contract for post-RoPE R1 Q/K192 and V128 in NHD layout."""
    q, k, v = (_shape(shape, 3, name) for shape, name in
               ((q_shape, "query"), (k_shape, "key"), (v_shape, "value")))
    if q != k or q[:2] != v[:2] or q[1] <= 0:
        raise ValueError("Prefill requires equal token/head counts and matching Q/K shapes")
    if q[2] != PREFILL_QK_DIM or v[2] != PREFILL_V_DIM:
        raise ValueError("R1 materialized prefill requires Q/K192 and V128")
    _integer(max_seqlen, "max_seqlen", minimum=1)
    if not isinstance(cu_seqlens, (list, tuple)) or not cu_seqlens:
        raise ValueError("cu_seqlens must be a nonempty CPU sequence")
    cu = [_integer(value, "cumulative length") for value in cu_seqlens]
    if cu[0] != 0 or cu[-1] != q[0]:
        raise ValueError("cu_seqlens must start at zero and cover all packed tokens")
    spans = []
    for start, stop in zip(cu, cu[1:]):
        if not 0 < stop - start <= max_seqlen:
            raise ValueError("Packed sequence lengths must be positive and within max_seqlen")
        spans.append((start, stop))
    return spans


def causal_query_chunks(length, query_chunk_size=QUERY_CHUNK):
    """Return absolute query spans; caller masks key_position <= query_position."""
    _integer(length, "sequence length", minimum=1)
    _integer(query_chunk_size, "query_chunk_size", minimum=1)
    return [(first, min(length, first + query_chunk_size))
            for first in range(0, length, query_chunk_size)]


def plan_paged_decode(query_shape, cache_shape, table_rows, lengths, max_seqlen):
    """Validate CPU metadata and return only pages belonging to valid KV tokens.

    Query is [B,1,H,576], matching main's latent query after weight absorption.
    Cache is [P,64,1,576], containing normalized latent512 plus rotated key64.
    The single query is the last token already appended to each valid length.
    Unused table entries may be negative sentinels and are never dereferenced.
    Read-only shared pages across sequences are supported; repeated physical
    pages within one sequence are rejected as an invalid allocation contract.
    """
    q = _shape(query_shape, 4, "query")
    cache = _shape(cache_shape, 4, "cache")
    if q[1] != 1 or q[2] <= 0 or q[3] != LATENT_DIM + ROPE_DIM:
        raise ValueError("Decode query must be [B,1,H,576]")
    if cache[0] <= 0 or cache[1:] != (PAGE_SIZE, 1, LATENT_DIM + ROPE_DIM):
        raise ValueError("Decode cache must be [P,64,1,576] with P>0")
    _integer(max_seqlen, "max_seqlen", minimum=1)
    if not isinstance(lengths, (list, tuple)) or len(lengths) != q[0]:
        raise ValueError("Length count must match query batch")
    if not isinstance(table_rows, (list, tuple)) or len(table_rows) != q[0]:
        raise ValueError("Table row count must match query batch")
    result = []
    for row, (length, table) in enumerate(zip(lengths, table_rows)):
        _integer(length, "sequence length", minimum=1)
        if length > max_seqlen:
            raise ValueError("Sequence length exceeds the exact runtime bound")
        if not isinstance(table, (list, tuple)):
            raise ValueError("Table rows must contain CPU page ids")
        count = (length + PAGE_SIZE - 1) // PAGE_SIZE
        if len(table) < count:
            raise ValueError("Page table does not cover the valid sequence length")
        pages = [_integer(page, "physical page") for page in table[:count]]
        if any(page >= cache[0] for page in pages):
            raise ValueError("Physical page is out of bounds")
        if len(set(pages)) != len(pages):
            raise ValueError("A sequence repeats a physical page")
        result.append({"row": row, "length": length, "physical_pages": pages})
    return result


def token_page_indices(decode_plan):
    """Page64 -> page1 metadata for a potential shared FlashInfer MLA wrapper.

    No cache value is touched. This is the SG FlashInfer MLA token-page layout:
    valid physical token id = physical_page * 64 + offset. Unused last-page
    slots are excluded, so their NaNs cannot become an addressed KV token.
    This helper does not select, instantiate, or qualify a FlashInfer backend.
    """
    indices, indptr = [], [0]
    for record in decode_plan:
        length, pages = record["length"], record["physical_pages"]
        indices.extend(pages[position // PAGE_SIZE] * PAGE_SIZE + position % PAGE_SIZE
                       for position in range(length))
        indptr.append(len(indices))
    if len(indices) > 2**31 - 1 or any(index > 2**31 - 1 for index in indices):
        raise ValueError("FlashInfer int32 token metadata would overflow")
    return {"query_indptr": list(range(len(decode_plan) + 1)),
            "kv_indptr": indptr, "kv_indices": indices,
            "kv_lengths": [record["length"] for record in decode_plan],
            "kernel_page_size": 1, "allocator_page_size": PAGE_SIZE}


def _torch():
    import torch
    return torch


def _bf16_inputs(*tensors):
    torch = _torch()
    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("Attention inputs must be Torch tensors")
    device = tensors[0].device
    if device.type not in ("cpu", "cuda") or any(
        value.dtype != torch.bfloat16 or value.device != device for value in tensors
    ):
        raise ValueError("Attention inputs must be BF16 on one CPU/CUDA device")
    if device.type == "cuda":
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("The correctness SDPA adapter is eager-only")
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Common FP32 SDPA requires TF32 disabled explicitly")
    return torch, device


def _cpu_ints(tensor, name, ndim):
    torch = _torch()
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim or tensor.dtype not in (
        torch.int32, torch.int64
    ):
        raise ValueError(f"{name} must be an int32/int64 tensor of rank {ndim}")
    return tensor.detach().cpu().tolist()


def packed_prefill_bf16(query, key, value, cu_seqlens, max_seqlen,
                        *, softmax_scale, query_chunk_size=QUERY_CHUNK):
    """Fresh full-sequence causal prefill; returns [T,H,128] BF16.

    Q/K are already projected and identically RoPE-transformed by the caller.
    An explicit absolute mask is required for query chunks; SDPA's upper-left
    is_causal alignment would be wrong for chunks starting after position zero.
    No cached prefix, asymmetric Q/K lengths, GQA or positional encoding is
    inferred here. Those cases require a separate validated adapter.
    """
    torch, device = _bf16_inputs(query, key, value)
    scale = _scale(softmax_scale)
    spans = plan_packed_prefill(tuple(query.shape), tuple(key.shape), tuple(value.shape),
                                _cpu_ints(cu_seqlens, "cu_seqlens", 1), max_seqlen)
    _integer(query_chunk_size, "query_chunk_size", minimum=1)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = value.new_empty(value.shape)
        for first, stop in spans:
            length = stop - first
            k = key[first:stop].transpose(0, 1).unsqueeze(0).float()
            v = value[first:stop].transpose(0, 1).unsqueeze(0).float()
            key_positions = torch.arange(length, device=device)
            for begin, end in causal_query_chunks(length, query_chunk_size):
                q = query[first + begin:first + end].transpose(0, 1).unsqueeze(0).float()
                query_positions = torch.arange(begin, end, device=device)
                visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    attended = scaled_dot_product_attention(
                        q, k, v, attn_mask=visible, dropout_p=0.0,
                        is_causal=False, scale=scale,
                    )
                output[first + begin:first + end] = attended[0].transpose(0, 1).to(torch.bfloat16)
        return output


def paged_decode_bf16(query, cache, block_table, cache_seqlens, max_seqlen,
                      *, softmax_scale):
    """Last-token latent MLA; returns [B,1,H,512] BF16 without touching cache.

    The caller must append the current token first and pass its inclusive
    cache length. No causal triangle is needed for a single last-token query:
    every gathered token is valid history including the current token. Using
    SDPA is_causal=True here would incorrectly expose only the first key.
    """
    torch, device = _bf16_inputs(query, cache)
    scale = _scale(softmax_scale)
    if not cache.is_contiguous():
        raise ValueError("Native page64 cache must be contiguous")
    plan = plan_paged_decode(tuple(query.shape), tuple(cache.shape),
                             _cpu_ints(block_table, "block_table", 2),
                             _cpu_ints(cache_seqlens, "cache_seqlens", 1), max_seqlen)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = query.new_empty((*query.shape[:3], LATENT_DIM))
        for record in plan:
            row, length = record["row"], record["length"]
            positions = torch.arange(length, device=device)
            table = torch.tensor(record["physical_pages"], dtype=torch.int64, device=device)
            # Advanced-index by valid positions before loading any cache data.
            history = cache[table[positions // PAGE_SIZE], positions % PAGE_SIZE, 0]
            q = query[row:row + 1].transpose(1, 2).float()
            history_fp32 = history.float()
            heads = query.shape[2]
            # Explicit read-only head views avoid relying on an implicit GQA
            # option or physically duplicating the latent cache.
            k = history_fp32.unsqueeze(0).unsqueeze(0).expand(1, heads, length, LATENT_DIM + ROPE_DIM)
            v = history_fp32[:, :LATENT_DIM].unsqueeze(0).unsqueeze(0).expand(1, heads, length, LATENT_DIM)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                attended = scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False, scale=scale,
                )
            output[row:row + 1] = attended.transpose(1, 2).to(torch.bfloat16)
        return output


def contract():
    """Serializable description only; no pass or runtime claim is implied."""
    return {
        "backend_id": BACKEND_ID, "status": "explicit_compatibility_candidate",
        "input_dtype": "bfloat16", "attention_compute_dtype": "float32",
        "output_dtype": "bfloat16", "torch_sdpa_backend": "MATH",
        "graph_scope": "off", "weight_or_projection_changes": False,
        "prefill": {"layout": "NHD", "qk_dim": PREFILL_QK_DIM,
                    "v_dim": PREFILL_V_DIM, "causal": True,
                    "query_chunk_size": QUERY_CHUNK, "cached_prefix_supported": False},
        "decode": {"query": "B,1,H,576", "cache": "P,64,1,576",
                   "output": "B,1,H,512", "valid_tokens_only": True,
                   "cache_writes": False, "single_last_token_query": True},
        "rope": "Caller applies identical original model RoPE before this boundary",
        "softmax_scale": "Explicit original model scale, including YaRN correction",
        "head_mapping": "Head h attends matching materialized head h; latent KV is shared by heads",
        "qualification_scope": "primitive_candidate_NOT_full_model_or_paper_performance",
    }
