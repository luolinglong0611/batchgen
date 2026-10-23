"""Experimental eager R1 MLA primitives with IEEE probability/value accumulation.

Kernel math derived without modification from the local, user-owned file
batchgen/models/moonshotai/kimi_linear/mla_triton.py, SHA256:
8d5beee05c932a89e4bec916d8f9a98cc0528f36e7d46daea20c5453d044403e
(snapshot 2026-10-23). This module is self-contained; it neither imports nor
modifies that file. Its own source hash must be qualified independently.

BF16 Q/K/V and outputs, FP32 QK and IEEE FP32 P@V; prefill uses online FP32
softmax, decode uses two passes with FP64 exp/denominator. Decode masks loads
of unused cache, including values, so NaN padding cannot enter P@V. This is a
correctness candidate, not the paper's kernel or a throughput claim. Metadata
validation synchronizes the GPU. Only SM103 eager execution is qualified in
scope; no CUDA graphs, prefix/chunked prefill, or FP8 cache support.
"""

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _prefill(
    Q, K, V, CU, O,
    qs0, qs1, qs2, ks0, ks1, ks2, vs0, vs1, vs2,
    H: tl.constexpr, SCALE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    begin = tl.load(CU + batch)
    length = tl.load(CU + batch + 1) - begin
    rows = block * BM + tl.arange(0, BM)
    dq = tl.arange(0, 256)
    dv = tl.arange(0, 128)
    q = tl.load(
        Q + (begin + rows[:, None]) * qs0 + head * qs1 + dq[None, :] * qs2,
        (rows[:, None] < length) & (dq[None, :] < 192), other=0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, 128), tl.float32)
    # No later key tile can contribute to this causal query tile.
    for start in range(0, tl.minimum(length, (block + 1) * BM), BN):
        keys = start + tl.arange(0, BN)
        k = tl.load(
            K + (begin + keys[None, :]) * ks0 + head * ks1 + dq[:, None] * ks2,
            (keys[None, :] < length) & (dq[:, None] < 192), other=0,
        )
        scores = tl.dot(q, k) * SCALE
        valid = (rows[:, None] < length) & (keys[None, :] < length)
        valid = valid & (keys[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # Never subtract -inf from -inf, including padded / empty query rows.
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum[:, None])
        v = tl.load(
            V + (begin + keys[:, None]) * vs0 + head * vs1 + dv[None, :] * vs2,
            keys[:, None] < length, other=0,
        ).to(tl.float32)
        accumulator = accumulator * correction[:, None]
        # tf32x3 rounding can amplify past the BF16 error envelope after o_proj.
        accumulator = tl.dot(probabilities, v, accumulator, input_precision="ieee")
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = new_maximum
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        O + (begin + rows[:, None]) * H * 128 + head * 128 + dv[None, :],
        result.to(tl.bfloat16), rows[:, None] < length,
    )


@triton.jit
def _decode(
    Q, CACHE, TABLE, LENGTHS, O,
    qs0, qs1, qs2, cs0, cs1, cs3, ts0, ts1,
    H: tl.constexpr, PAGES: tl.constexpr, MAX_PAGES: tl.constexpr,
    SCALE: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    length = tl.load(LENGTHS + batch)
    dv = tl.arange(0, 512)
    dr = tl.arange(0, 64)
    q_value = tl.load(
        Q + batch * qs0 + heads[:, None] * qs1 + dv[None, :] * qs2,
        heads[:, None] < H, other=0,
    )
    q_rope = tl.load(
        Q + batch * qs0 + heads[:, None] * qs1 + (512 + dr[None, :]) * qs2,
        heads[:, None] < H, other=0,
    )
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    # FP32 online rescaling errors can survive BF16 rounding and amplify through out_absorb.
    denominator = tl.zeros((BH,), tl.float64)
    for start in range(0, length, BN):
        keys = start + tl.arange(0, BN)
        logical_pages = keys // 64
        in_table = (keys < length) & (logical_pages < MAX_PAGES)
        physical_pages = tl.load(
            TABLE + batch * ts0 + logical_pages * ts1, in_table, other=-1,
        )
        valid_keys = in_table & (physical_pages >= 0) & (physical_pages < PAGES)
        safe_pages = tl.where(valid_keys, physical_pages, 0).to(tl.int64)
        addresses = safe_pages * cs0 + (keys % 64) * cs1
        k_value = tl.load(
            CACHE + addresses[None, :] + dv[:, None] * cs3,
            valid_keys[None, :], other=0,
        )
        k_rope = tl.load(
            CACHE + addresses[None, :] + (512 + dr[:, None]) * cs3,
            valid_keys[None, :], other=0,
        )
        scores = tl.dot(q_value, k_value)
        scores = tl.dot(q_rope, k_rope, scores) * SCALE
        valid = (heads[:, None] < H) & valid_keys[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        correction = libdevice.exp(maximum.to(tl.float64) - safe_maximum.to(tl.float64))
        probabilities = libdevice.exp(
            scores.to(tl.float64) - safe_maximum[:, None].to(tl.float64),
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = new_maximum
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    denominator = tl.where(denominator > 0, denominator, 1.0)
    accumulator = tl.zeros((BH, 512), tl.float32)
    for start in range(0, length, BN):
        keys = start + tl.arange(0, BN)
        logical_pages = keys // 64
        in_table = (keys < length) & (logical_pages < MAX_PAGES)
        physical_pages = tl.load(
            TABLE + batch * ts0 + logical_pages * ts1, in_table, other=-1,
        )
        valid_keys = in_table & (physical_pages >= 0) & (physical_pages < PAGES)
        safe_pages = tl.where(valid_keys, physical_pages, 0).to(tl.int64)
        addresses = safe_pages * cs0 + (keys % 64) * cs1
        k_value = tl.load(
            CACHE + addresses[None, :] + dv[:, None] * cs3,
            valid_keys[None, :], other=0,
        )
        k_rope = tl.load(
            CACHE + addresses[None, :] + (512 + dr[:, None]) * cs3,
            valid_keys[None, :], other=0,
        )
        scores = tl.dot(q_value, k_value)
        scores = tl.dot(q_rope, k_rope, scores) * SCALE
        valid = (heads[:, None] < H) & valid_keys[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        probabilities = (
            libdevice.exp(scores.to(tl.float64) - maximum[:, None].to(tl.float64))
            / denominator[:, None]
        ).to(tl.float32)
        accumulator = tl.dot(
            probabilities, tl.trans(k_value).to(tl.float32), accumulator,
            input_precision="ieee",
        )
    tl.store(
        O + batch * H * 512 + heads[:, None] * 512 + dv[None, :],
        accumulator.to(tl.bfloat16), heads[:, None] < H,
    )


SOURCE_PROVENANCE_SHA256 = "8d5beee05c932a89e4bec916d8f9a98cc0528f36e7d46daea20c5453d044403e"


def _validate_runtime(device, scale):
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("This experimental candidate is restricted to SM103")
    if os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("Set DISABLE_MMA_V5=1 before the process starts on SM103")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Precision qualification is eager-only")


def _validate_boundaries(boundaries, tokens, max_seqlen):
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != tokens:
        raise ValueError("cu_seqlens must start at zero and end at the packed token count")
    lengths = [end - begin for begin, end in zip(boundaries, boundaries[1:])]
    if any(length < 0 for length in lengths) or max(lengths, default=0) > max_seqlen:
        raise ValueError("cu_seqlens must be monotonic and covered by max_seqlen")


def _validate_page_contents(table_rows, lengths, pages, width):
    for row, length in zip(table_rows, lengths):
        if length < 1 or length > width * 64:
            raise ValueError("Decode lengths must be positive and covered by the page table")
        active = row[:(length + 63) // 64]
        if any(page < 0 or page >= pages for page in active):
            raise ValueError("Active page table contains an invalid physical page")


def _validate_float_inputs(*tensors):
    device = tensors[0].device
    for tensor in tensors:
        if tensor.device != device or not tensor.is_cuda or tensor.dtype != torch.bfloat16:
            raise ValueError("inputs must be BF16 tensors on one CUDA device")


def _validate_metadata(tensor, device):
    if tensor.device != device or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError("metadata must be int32/int64 on the input CUDA device")


def varlen_prefill(
    q, k, v, cu_seqlens, max_seqlen, *, softmax_scale, block_m=16, block_n=32,
):
    """Causal Q/K [T,H,192], V [T,H,128]; validates CU contents eagerly."""
    _validate_float_inputs(q, k, v)
    _validate_runtime(q.device, softmax_scale)
    _validate_metadata(cu_seqlens, q.device)
    if q.ndim != 3 or q.shape[-1] != 192 or k.shape != q.shape:
        raise ValueError("Q/K must have identical [T,H,192] shapes")
    if v.shape != (*q.shape[:2], 128) or q.shape[1] < 1:
        raise ValueError("V must have shape [T,H,128] and H must be positive")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous vector with at least two entries")
    if block_m not in (16, 32) or block_n not in (32, 64):
        raise ValueError("supported tiles: rows 16/32, keys 32/64")
    if not isinstance(max_seqlen, int) or max_seqlen < (1 if q.shape[0] else 0):
        raise ValueError("max_seqlen must be a nonnegative host integer covering the batch")
    _validate_boundaries(cu_seqlens.tolist(), q.shape[0], max_seqlen)
    output = torch.empty((*q.shape[:2], 128), dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return output
    with torch.cuda.device(q.device):
        _prefill[(triton.cdiv(max_seqlen, block_m), q.shape[1], cu_seqlens.numel() - 1)](
            q, k, v, cu_seqlens, output,
            *q.stride(), *k.stride(), *v.stride(),
            H=q.shape[1], SCALE=softmax_scale, BM=block_m, BN=block_n,
            num_warps=4, num_stages=1,
        )
    return output


def paged_mla_decode(
    q, cache, block_table, lengths, *, softmax_scale, block_h=16, block_n=32,
):
    """Q [B,H,576], cache [P,64,1,576] -> [B,H,512]; validates active pages."""
    _validate_float_inputs(q, cache)
    _validate_runtime(q.device, softmax_scale)
    _validate_metadata(block_table, q.device)
    _validate_metadata(lengths, q.device)
    if q.ndim != 3 or q.shape[-1] != 576 or q.shape[1] < 1:
        raise ValueError("Q must have shape [B,H,576] with positive H")
    if cache.ndim != 4 or cache.shape[1:] != (64, 1, 576) or cache.shape[0] < 1:
        raise ValueError("cache must have shape [P,64,1,576] with positive P")
    if block_table.ndim != 2 or block_table.shape[0] != q.shape[0]:
        raise ValueError("block_table must have shape [B,max_pages]")
    if lengths.shape != (q.shape[0],) or not lengths.is_contiguous():
        raise ValueError("lengths must be a contiguous [B] vector")
    if block_h not in (16, 32) or block_n not in (32, 64):
        raise ValueError("supported tiles: heads 16/32, keys 32/64")
    _validate_page_contents(block_table.tolist(), lengths.tolist(), cache.shape[0], block_table.shape[1])
    output = torch.empty((*q.shape[:2], 512), dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return output
    with torch.cuda.device(q.device):
        _decode[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
            q, cache, block_table, lengths, output,
            *q.stride(), cache.stride(0), cache.stride(1), cache.stride(3),
            *block_table.stride(), H=q.shape[1], PAGES=cache.shape[0],
            MAX_PAGES=block_table.shape[1], SCALE=softmax_scale,
            BH=block_h, BN=block_n, num_warps=8, num_stages=1,
        )
    return output
