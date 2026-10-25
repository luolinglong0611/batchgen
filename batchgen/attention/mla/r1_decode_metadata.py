"""One checked GPU operation for unchanged R1 decode metadata.

Validate every call, including unused page-table tails and the exact sequence
bound. No host scalar reads, cross-step cache, or relaxed bounds are used.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["MAX_LENGTH", "TABLE_ROW", "WIDTH", "OUTPUT_WIDTH", "PAGES"])
def _prepare(lengths, positions, table, safe, write_positions,
             POSITION_STRIDE: tl.constexpr, TABLE_ROW,
             TABLE_COL: tl.constexpr, WIDTH,
             OUTPUT_WIDTH, PAGES,
             MAX_LENGTH, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    position = tl.load(positions + row * POSITION_STRIDE)
    tl.device_assert((length > 0) & (length <= MAX_LENGTH) & (position == length.to(tl.int64) - 1),
                     "Invalid R1 decode sequence lengths/positions")
    pages = (length.to(tl.int64) + 63) // 64
    tl.device_assert(pages <= WIDTH, "Invalid R1 decode page-table coverage")
    col = tl.arange(0, BLOCK)
    present = col < WIDTH
    value = tl.load(table + row * TABLE_ROW + col * TABLE_COL, present, other=0)
    used = present & (col < pages)
    valid = ((value >= 0) & (value < PAGES)) | ~used
    tl.device_assert(tl.sum((~valid).to(tl.int32), 0) == 0,
                     "Invalid R1 decode physical page table")
    tl.store(safe + row * OUTPUT_WIDTH + col, tl.where(used, value, 0), col < OUTPUT_WIDTH)
    tl.store(write_positions + row, position.to(tl.int32))


def prepare(cache, table, lengths, positions, max_seqlen):
    """Caller has checked the same cache/table/tensor layouts as the baseline."""
    if type(max_seqlen) is not int or not 0 < max_seqlen <= 2**31 - 1:
        raise ValueError('R1 decode requires a positive int32 sequence bound')
    # A zero-width table is invalid for any nonempty decode. Reject before a
    # kernel could form a zero-size launch; zero-batch calls exit in the caller.
    batch, width = table.shape
    if width == 0:
        raise ValueError('R1 decode page table has no coverage')
    if not lengths.is_contiguous():
        lengths = lengths.contiguous()
    output_width = width + width % 2
    safe = torch.empty((batch, output_width), dtype=torch.int32, device=table.device)
    write_positions = torch.empty((batch,), dtype=torch.int32, device=table.device)
    _prepare[(batch,)](
        lengths, positions, table, safe, write_positions,
        positions.stride(0), table.stride(0), table.stride(1), width,
        output_width, cache.shape[0], max_seqlen,
        triton.next_power_of_2(output_width), debug=True,
    )
    return safe, write_positions
