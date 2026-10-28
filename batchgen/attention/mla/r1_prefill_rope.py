"""Opt-in R1 prefill RoPE and materialization, with no persistent buffers.

All rotations use the original interleaved cos/sin values and separate FP32
multiply/add operations before BF16 storage. Preparation owns only this call's
Q/K/V/offload outputs, on the caller's current stream. Native attention choice,
RoPE cache generation, positions and weight math remain with the caller.
"""
import os
import logging

ENV = 'BATCHGEN_R1_PREFILL_FUSED_ROPE'
_ANNOUNCED = False


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{ENV} must be explicitly 0 or 1')
    return value == '1'


def _require_tensor(torch, value, shape, device, *, name, contiguous=True):
    if (not isinstance(value, torch.Tensor) or value.device != device
            or value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(shape)):
        raise ValueError(f'{name} must be BF16 {tuple(shape)} on the query CUDA device')
    if contiguous and not value.is_contiguous():
        raise ValueError(f'{name} must be contiguous')


def prepare_query_and_offload(query, normed_kv, k_pe, cos, sin, positions):
    """[T,H,192] Q and [T,576] offload, without materialized RoPE temporaries.

    Positions retain Torch indexing semantics for valid negative indices. As
    with the native rotary kernel, the caller must provide in-bounds positions;
    no device-to-host index-value read is introduced in a per-layer hot path.
    """
    import torch
    if (not isinstance(query, torch.Tensor) or query.ndim != 3
            or query.device.type != 'cuda' or query.shape[1] <= 0 or query.shape[2] != 192):
        raise ValueError('R1 fused prefill expects CUDA query [T,H,192]')
    tokens, heads, _ = query.shape
    if tokens * heads * 256 >= 2**31:
        raise ValueError('R1 fused prefill exceeds int32-safe materialization geometry')
    device = query.device
    _require_tensor(torch, query, query.shape, device, name='query')
    _require_tensor(torch, normed_kv, (tokens, 512), device, name='normalized latent KV')
    _require_tensor(torch, k_pe, (tokens, 1, 64), device, name='raw K rope', contiguous=False)
    if k_pe.stride(-1) != 1 or k_pe.stride(0) <= 0:
        raise ValueError('R1 fused raw K rope needs positive token stride and unit channel stride')
    for name, cache in (('cos', cos), ('sin', sin)):
        if (not isinstance(cache, torch.Tensor) or cache.ndim != 2 or cache.shape[0] <= 0
                or cache.shape[1] != 64 or cache.device != device
                or cache.dtype not in (torch.bfloat16, torch.float32)
                or cache.stride(0) <= 0 or cache.stride(1) <= 0):
            raise ValueError(f'R1 fused {name} must be CUDA BF16/FP32 [positions,64]')
    if cos.shape != sin.shape:
        raise ValueError('R1 fused cos/sin shapes differ')
    if (not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (tokens,)
            or positions.device != device or positions.dtype not in (torch.int32, torch.int64)
            or positions.stride(0) <= 0):
        raise ValueError('R1 fused positions must be CUDA int32/int64 [T] with positive stride')
    q_out = torch.empty_like(query)
    offload = torch.empty((tokens, 576), dtype=torch.bfloat16, device=device)
    if tokens:
        from .r1_prefill_rope_kernels import rotate_query, rotate_k_offload
        # Fusion is disabled at the compiler level: Torch evaluates the two
        # products and then their sum/subtraction, without an implicit FMA.
        rotate_query[( (tokens * heads * 96 + 2047) // 2048, )](
            query, cos, sin, positions, q_out, tokens * heads * 96,
            HEADS=heads, COS_ROWS=cos.shape[0], COS_ROW_STRIDE=cos.stride(0),
            COS_COL_STRIDE=cos.stride(1), SIN_ROW_STRIDE=sin.stride(0),
            SIN_COL_STRIDE=sin.stride(1), POS_STRIDE=positions.stride(0),
            BLOCK=2048, num_warps=8, enable_fp_fusion=False,
        )
        rotate_k_offload[((tokens * 288 + 1023) // 1024,)](
            normed_kv, k_pe, cos, sin, positions, offload, tokens * 288,
            K_TOKEN_STRIDE=k_pe.stride(0), COS_ROWS=cos.shape[0],
            COS_ROW_STRIDE=cos.stride(0), COS_COL_STRIDE=cos.stride(1),
            SIN_ROW_STRIDE=sin.stride(0), SIN_COL_STRIDE=sin.stride(1),
            POS_STRIDE=positions.stride(0), BLOCK=1024,
            num_warps=4, enable_fp_fusion=False,
        )
        global _ANNOUNCED
        if not _ANNOUNCED:
            logging.info('[R1-PREFILL-ROPE] fused=true backend=triton interleaved=true '
                         'compute=FP32 output=BF16 fp_fusion=false persistent_bytes=0 '
                         'tokens=%d heads=%d', tokens, heads)
            _ANNOUNCED = True
    return q_out, offload


def materialize_key_value(expanded_kv, offload):
    """Copy latent-expanded K/V and broadcast rotated K into contiguous outputs."""
    import torch
    if (not isinstance(expanded_kv, torch.Tensor) or expanded_kv.ndim != 3
            or expanded_kv.device.type != 'cuda' or expanded_kv.shape[1] <= 0
            or expanded_kv.shape[2] != 256):
        raise ValueError('R1 fused expanded KV must be CUDA [T,H,256]')
    tokens, heads, _ = expanded_kv.shape
    if tokens * heads * 256 >= 2**31:
        raise ValueError('R1 fused prefill exceeds int32-safe materialization geometry')
    device = expanded_kv.device
    _require_tensor(torch, expanded_kv, expanded_kv.shape, device, name='expanded KV')
    _require_tensor(torch, offload, (tokens, 576), device, name='offload KV')
    key = torch.empty((tokens, heads, 192), dtype=torch.bfloat16, device=device)
    value = torch.empty((tokens, heads, 128), dtype=torch.bfloat16, device=device)
    if tokens:
        from .r1_prefill_rope_kernels import pack_key_value
        pack_key_value[((tokens * heads * 192 + 4095) // 4096,)](
            expanded_kv, offload, key, value, tokens * heads * 192,
            HEADS=heads, BLOCK=4096, num_warps=8,
        )
    return key, value
