"""Explicit metadata ownership for intra-layer MLA decode microbatches.

This adds no scheduler, KV capacity change or cross-forward cache. The existing
worker owns forward-wide position/FA2 scopes; each declared row slice has stable
metadata and its own FA2 plan within that lifetime. Native attention also works
without an FA2 scope. Original full-batch paths do not call this helper.
"""
from contextlib import contextmanager

from .common_flashinfer_backend import decode_forward_batch_slice_scope
from .decode_position_validation import decode_position_slice_scope


@contextmanager
def decode_attention_microbatch_scope(manager, full_positions, full_lengths, start, end):
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError("Microbatch requires a nonempty integer row range")
    if (full_positions.ndim != 2 or full_positions.shape[1] != 1
            or full_lengths.ndim != 1
            or full_positions.shape[0] != full_lengths.shape[0]
            or end > full_lengths.shape[0]
            or full_positions.device != full_lengths.device):
        raise ValueError("Microbatch position/length rows, geometry or device do not match")
    with decode_position_slice_scope(full_positions, start, end) as positions:
        with decode_forward_batch_slice_scope(manager, full_lengths, start, end) as lengths:
            yield positions, lengths
