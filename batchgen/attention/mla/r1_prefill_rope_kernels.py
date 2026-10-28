"""CUDA-only kernels imported lazily by the R1 prefill helper."""
import triton
import triton.language as tl


@triton.jit
def rotate_query(Q, COS, SIN, POS, OUT, PAIRS,
                 HEADS: tl.constexpr, COS_ROWS: tl.constexpr,
                 COS_ROW_STRIDE: tl.constexpr, COS_COL_STRIDE: tl.constexpr,
                 SIN_ROW_STRIDE: tl.constexpr, SIN_COL_STRIDE: tl.constexpr,
                 POS_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair < PAIRS
    channel_pair = pair % 96
    token = pair // (HEADS * 96)
    rope = channel_pair >= 64
    position = tl.load(POS + token * POS_STRIDE, mask=valid & rope, other=0)
    position = tl.where(position < 0, position + COS_ROWS, position)
    frequency = tl.maximum(channel_pair - 64, 0)
    c = tl.load(COS + position * COS_ROW_STRIDE + frequency * COS_COL_STRIDE,
                mask=valid & rope, other=1).to(tl.float32)
    s = tl.load(SIN + position * SIN_ROW_STRIDE + frequency * SIN_COL_STRIDE,
                mask=valid & rope, other=0).to(tl.float32)
    even = tl.load(Q + pair * 2, mask=valid, other=0).to(tl.float32)
    odd = tl.load(Q + pair * 2 + 1, mask=valid, other=0).to(tl.float32)
    out_even = tl.where(rope, even * c - odd * s, even)
    out_odd = tl.where(rope, even * s + odd * c, odd)
    tl.store(OUT + pair * 2, out_even, mask=valid)
    tl.store(OUT + pair * 2 + 1, out_odd, mask=valid)


@triton.jit
def rotate_k_offload(LATENT, K, COS, SIN, POS, OUT, PAIRS,
                     K_TOKEN_STRIDE: tl.constexpr, COS_ROWS: tl.constexpr,
                     COS_ROW_STRIDE: tl.constexpr, COS_COL_STRIDE: tl.constexpr,
                     SIN_ROW_STRIDE: tl.constexpr, SIN_COL_STRIDE: tl.constexpr,
                     POS_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair < PAIRS
    token = pair // 288
    channel_pair = pair % 288
    rope = channel_pair >= 256
    latent_channel = channel_pair * 2
    frequency = tl.maximum(channel_pair - 256, 0)
    even_latent = tl.load(LATENT + token * 512 + latent_channel,
                         mask=valid & ~rope, other=0)
    odd_latent = tl.load(LATENT + token * 512 + latent_channel + 1,
                        mask=valid & ~rope, other=0)
    even = tl.load(K + token * K_TOKEN_STRIDE + frequency * 2,
                   mask=valid & rope, other=0).to(tl.float32)
    odd = tl.load(K + token * K_TOKEN_STRIDE + frequency * 2 + 1,
                  mask=valid & rope, other=0).to(tl.float32)
    position = tl.load(POS + token * POS_STRIDE, mask=valid & rope, other=0)
    position = tl.where(position < 0, position + COS_ROWS, position)
    c = tl.load(COS + position * COS_ROW_STRIDE + frequency * COS_COL_STRIDE,
                mask=valid & rope, other=1).to(tl.float32)
    s = tl.load(SIN + position * SIN_ROW_STRIDE + frequency * SIN_COL_STRIDE,
                mask=valid & rope, other=0).to(tl.float32)
    out_even = tl.where(rope, even * c - odd * s, even_latent.to(tl.float32))
    out_odd = tl.where(rope, even * s + odd * c, odd_latent.to(tl.float32))
    tl.store(OUT + pair * 2, out_even, mask=valid)
    tl.store(OUT + pair * 2 + 1, out_odd, mask=valid)


@triton.jit
def pack_key_value(EXPANDED, OFFLOAD, KEY, VALUE, ELEMENTS,
                   HEADS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ELEMENTS
    channel = index % 192
    head_row = index // 192
    token = head_row // HEADS
    nope = channel < 128
    k_nope = tl.load(EXPANDED + head_row * 256 + channel,
                     mask=valid & nope, other=0)
    k_rope = tl.load(OFFLOAD + token * 576 + 512 + tl.maximum(channel - 128, 0),
                     mask=valid & ~nope, other=0)
    value = tl.load(EXPANDED + head_row * 256 + 128 + channel,
                    mask=valid & nope, other=0)
    tl.store(KEY + index, tl.where(nope, k_nope, k_rope), mask=valid)
    tl.store(VALUE + head_row * 128 + channel, value, mask=valid & nope)
