"""Explicit, bounded last-page initialization for MLA qualification only.

No production backend imports this module. The caller supplies a complete CPU
allocator snapshot and its current table-row ordering. Shared or unknown tail
ownership is rejected before any cache write. Only unused tokens in the last
logical page are initialized; valid KV values, including NaNs, are untouched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def plan_tail_initialization(*, sequence_ids, lengths, table_rows,
                             all_sequence_pages, page_refcounts, page_count,
                             page_size=64, features=576):
    """Validate a complete allocator snapshot and return exact write ranges.

    ``all_sequence_pages`` covers every live sequence, not just this microbatch.
    Its reference counts must exactly match ``page_refcounts``. The caller owns
    that snapshot and must exclude concurrent allocation/release until writes
    have been enqueued on the cache's consuming stream.
    """
    _integer(page_count, "page_count", minimum=1)
    _integer(page_size, "page_size", minimum=1)
    _integer(features, "features", minimum=1)
    if len(sequence_ids) != len(lengths) or len(lengths) != len(table_rows):
        raise ValueError("sequence ids, lengths and table rows must align")
    if any(type(s) is not int for s in sequence_ids):
        raise ValueError("sequence ids must be integers")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("active sequence ids must be unique")
    if not isinstance(all_sequence_pages, dict) or not isinstance(page_refcounts, dict):
        raise ValueError("allocator pages and reference counts must be dictionaries")

    observed = Counter()
    for sequence, pages in all_sequence_pages.items():
        if type(sequence) is not int:
            raise ValueError("allocator sequence ids must be integers")
        if not isinstance(pages, (list, tuple)) or not pages:
            raise ValueError("every live sequence requires allocated pages")
        if len(set(pages)) != len(pages):
            raise ValueError("a sequence cannot reference the same physical page twice")
        for page in pages:
            _integer(page, "allocated page")
            if page >= page_count:
                raise ValueError("allocated physical page is out of bounds")
            observed[page] += 1
    for page, count in page_refcounts.items():
        _integer(page, "reference-count page")
        _integer(count, "reference count", minimum=1)
        if page >= page_count:
            raise ValueError("reference-count page is out of bounds")
    if dict(observed) != page_refcounts:
        raise ValueError("complete allocator ownership and reference counts disagree")

    ranges = []
    for row, (sequence, length, table) in enumerate(zip(sequence_ids, lengths, table_rows)):
        _integer(length, "sequence length", minimum=1)
        if sequence not in all_sequence_pages:
            raise ValueError("active sequence has unknown ownership")
        if not isinstance(table, (list, tuple)):
            raise ValueError("table rows must contain CPU page ids")
        used_pages = (length + page_size - 1) // page_size
        if len(table) < used_pages or len(all_sequence_pages[sequence]) < used_pages:
            raise ValueError("page table or allocator does not cover sequence length")
        for page in table[:used_pages]:
            _integer(page, "active table page")
        if list(table[:used_pages]) != list(all_sequence_pages[sequence][:used_pages]):
            raise ValueError("page table does not match its sequence's allocation")
        start = length % page_size
        if not start:
            continue
        physical = table[used_pages - 1]
        if page_refcounts[physical] != 1:
            raise ValueError("refuse to initialize a shared last page")
        ranges.append({"row": row, "sequence_id": sequence,
                       "physical_page": physical, "start_token": start,
                       "stop_token": page_size})

    elements = sum((r["stop_token"] - r["start_token"]) * features for r in ranges)
    if elements > len(sequence_ids) * (page_size - 1) * features:
        raise AssertionError("last-page write bound exceeded")
    ownership = {"sequences": sorted((s, list(p)) for s, p in all_sequence_pages.items()),
                 "reference_counts": sorted(page_refcounts.items()),
                 "sequence_ids": list(sequence_ids), "lengths": list(lengths),
                 "table_rows": [list(r) for r in table_rows]}
    return {"schema": "r1-mla-exclusive-tail-plan-v1", "ranges": ranges,
            "page_count": page_count, "page_size": page_size, "features": features,
            "batch": len(sequence_ids), "elements_written": elements,
            "ownership_sha256": hashlib.sha256(json.dumps(ownership, sort_keys=True).encode()).hexdigest()}


def initialize_owned_tails_(cache, **ownership):
    """Zero validated unused tail slots in place on the current Torch stream.

    This qualification helper launches at most one small fill per partial
    sequence. It does not inspect, clone, copy or sanitize valid cache values.
    CPU metadata validation can be expensive; this is not a performance change.
    """
    import torch

    if cache.ndim != 4 or tuple(cache.shape[1:]) != (64, 1, 576):
        raise ValueError("expected MLA cache[pages,64,1,576]")
    if cache.dtype != torch.bfloat16 or not cache.is_contiguous():
        raise ValueError("expected contiguous BF16 cache")
    if ownership.get("page_count") != cache.shape[0]:
        raise ValueError("allocator page count does not match cache")
    if ownership.get("page_size", 64) != 64 or ownership.get("features", 576) != 576:
        raise ValueError("this qualification supports only page64 MLA576")
    plan = plan_tail_initialization(**ownership)
    for write in plan["ranges"]:
        cache[write["physical_page"], write["start_token"]:write["stop_token"], 0, :].zero_()
    return plan
