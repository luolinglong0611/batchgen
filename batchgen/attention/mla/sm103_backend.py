"""Experimental R1 SM103 MLA backend; full-layer qualification is required.

Candidate dependencies: flash-attn-4[cu13]==4.0.0b36,
flashinfer-python[cu13]==0.6.13 and flashinfer-cubin==0.6.13, plus the repository's
Triton blockscale GEMM and batchgen_kernels. Run attention_qualification.py first, then a
complete-layer prefill/append/decode comparison before enabling this module.
The independent precision candidate is selected only by setting the layer's
_r1_attention_implementation to "triton_precision"; the default remains
"fa4_flashinfer". The precision path imports neither FA4 nor FlashInfer.
All candidates require DISABLE_MMA_V5=1 before launch for SM103 Triton W8A8.

This initial implementation is eager-only, with deliberately synchronous input
checks. It is not ready for throughput comparisons or CUDA graph capture.
Only full-sequence prefill and one-token BF16 paged decode are implemented.
Both stages use native-interleaved RoPE: never mix their KV with the legacy
split-half decoder or change backends during a live session.

No FA3 or FlashMLA module is imported here. Existing model/dispatch imports
still need a separate integration change before this backend can be selected.
"""

from __future__ import annotations

import os

import torch


_R1_DIMS = {
    "num_heads": 128,
    "q_head_dim": 192,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
}
_WORKSPACE_BYTES = 128 * 1024**2


def _check_compiler_environment():
    if os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("Set DISABLE_MMA_V5=1 before the process starts: SM103 W8A8 requires the MMA v3 path")


def _attention_implementation(module):
    implementation = getattr(module, "_r1_attention_implementation", "fa4_flashinfer")
    if implementation not in ("fa4_flashinfer", "triton_precision"):
        raise ValueError(f"Unknown R1 attention implementation: {implementation}")
    return implementation


def _check_module(module, hidden_states):
    _check_compiler_environment()
    _attention_implementation(module)
    for name, expected in _R1_DIMS.items():
        if getattr(module, name, None) != expected:
            raise ValueError(f"R1 SM103 requires {name}={expected}")
    if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.bfloat16:
        raise ValueError("R1 SM103 requires CUDA BF16 hidden states")
    if torch.cuda.get_device_capability(hidden_states.device) != (10, 3):
        raise ValueError("This experimental backend is restricted to SM103")
    with torch.cuda.device(hidden_states.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("R1 SM103 qualification backend is eager-only")
    if os.environ.get("BATCHGEN_W8A16_DEQUANT", "0") != "0":
        raise ValueError("R1 SM103 qualification requires the default FP8 activation GEMM path")


def _quantize(x):
    # Same BF16/block128 path as fa3_backend.act_quant, without importing FA3.
    from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d

    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.shape[0]:
        raise ValueError("Projection quantization requires a nonempty 2D BF16 input")
    return per_token_blocked_quantize_bf16_to_fp8_1d(x.contiguous(), block_size=128)


def w8a16_gemm(weight_data_fp8, weight_scale_inv_fp32, activation_bf16):
    """Preserve original FP32 block scales using the portable Triton FP8 GEMM.

    DeepGEMM fa3a5ca's SM10 path requires power-of-two UE8M0 scales, while
    original R1 checkpoint scales and BatchGen activation scales are general
    FP32. Packing them as UE8M0 either asserts or changes their values. This
    migration path retains original weights/scales and FP8 activations instead
    of silently requantizing the model. It is not the paper's DeepGEMM backend.
    """
    _check_compiler_environment()
    from batchgen.gemm.w8a8 import w8a8_gemm

    if activation_bf16.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D activation")
    x = activation_bf16.reshape(-1, activation_bf16.shape[-1])
    out = w8a8_gemm(*_quantize(x), weight_data_fp8, weight_scale_inv_fp32)
    return out.view(*activation_bf16.shape[:-1], out.shape[-1])


def _project_prefill(module, name, x, scales):
    return w8a16_gemm(getattr(module, name).weight, scales[f"{name}.weight_scale_inv"], x)


def _project_decode(module, name, quantized, scales):
    from batchgen.gemm.w8a8 import w8a8_gemm

    return w8a8_gemm(*quantized, getattr(module, name).weight, scales[f"{name}.weight_scale_inv"])


def _validate_full_prefill_layout(boundaries, positions, tokens, sequences, max_seqlen):
    """CPU-side validation: this entry point does not accept prefix chunks."""
    if len(boundaries) != sequences + 1 or not boundaries or boundaries[0] != 0 or boundaries[-1] != tokens:
        raise ValueError("Invalid cumulative prefill lengths")
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if sequences < 1 or any(length <= 0 for length in lengths) or max(lengths) != max_seqlen:
        raise ValueError("Prefill requires nonempty sequences and the exact maximum length")
    if len(positions) != tokens:
        raise ValueError("Position count must equal packed token count")
    for start, end in zip(boundaries, boundaries[1:]):
        if positions[start:end] != list(range(end - start)):
            raise ValueError("Only full prefill with positions reset to zero per sequence is supported")


def _attend_prefill(module, queries, keys, values, cu_seqlens, max_seqlen):
    if _attention_implementation(module) == "triton_precision":
        from .triton_precision import varlen_prefill

        return varlen_prefill(queries, keys, values, cu_seqlens, max_seqlen,
                              softmax_scale=module.softmax_scale)
    from flash_attn.cute import flash_attn_varlen_func

    # FA4's fourth positional argument is qv, NOT cu_seqlens_q.
    output = flash_attn_varlen_func(
        queries, keys, values,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=module.softmax_scale, causal=True,
    )
    # FA4 b36 returns (output, lse), including when return_lse=False.
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("Expected FA4 b36's (output, lse) return contract")
    return output[0]


def _attend_decode(module, query, cache, table, lengths, max_seqlen):
    if _attention_implementation(module) == "triton_precision":
        from .triton_precision import paged_mla_decode

        return paged_mla_decode(query.squeeze(1), cache, table, lengths,
                                softmax_scale=module.softmax_scale).unsqueeze(1)
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    return trtllm_batch_decode_with_kv_cache_mla(
        query=query, kv_cache=cache.squeeze(2), workspace_buffer=_decode_workspace(module, query.device),
        qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
        block_tables=table, seq_lens=lengths, max_seq_len=max_seqlen,
        sparse_mla_top_k=0, bmm1_scale=module.softmax_scale, bmm2_scale=1.0,
        backend="trtllm-gen",
    )


@torch.inference_mode()
def mla_prefill_sm103_w8a16_prepacked(
    self, hidden_states, position_ids, cu_seqlens, max_seqlen, num_sequences, weight_scale,
):
    """R1 packed wrapper contract: [T,D] -> ([T,D], native KV[T,576])."""
    _check_module(self, hidden_states)
    if hidden_states.ndim != 2 or position_ids.shape != (hidden_states.shape[0],):
        raise ValueError("Expected hidden_states[T,D] and position_ids[T]")
    if position_ids.dtype not in (torch.int32, torch.int64) or position_ids.device != hidden_states.device:
        raise ValueError("Position IDs must be integer tensors on the activation device")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32 or cu_seqlens.device != hidden_states.device:
        raise ValueError("cu_seqlens must be CUDA int32 on the activation device")
    _validate_full_prefill_layout(
        cu_seqlens.tolist(), position_ids.tolist(), hidden_states.shape[0], num_sequences, max_seqlen,
    )
    if num_sequences == 1 and hidden_states.shape[0] > 262_144 and getattr(self, "single_sequence_prefill_chunk_size", 0):
        raise NotImplementedError("Chunked single-sequence prefill needs a separate FA4 integration")

    from batchgen.attention.mla.rotary_embedding import rotary_pos_emb_interleaved_native

    tokens = hidden_states.shape[0]
    q = _project_prefill(self, "q_a_proj", hidden_states, weight_scale)
    q = _project_prefill(self, "q_b_proj", self.q_a_layernorm(q), weight_scale)
    q_nope, q_pe = q.view(tokens, 128, 192).split((128, 64), dim=-1)
    kv = _project_prefill(self, "kv_a_proj_with_mqa", hidden_states, weight_scale)
    latent, k_pe = kv.split((512, 64), dim=-1)
    latent = self.kv_a_layernorm(latent)
    k_pe = k_pe.view(tokens, 1, 64)
    cos, sin = self.rotary_emb(q_pe.unsqueeze(0), seq_len=max_seqlen)
    q_pe = rotary_pos_emb_interleaved_native(q_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)
    k_pe = rotary_pos_emb_interleaved_native(k_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)
    offload_kv = torch.cat((latent, k_pe.reshape(tokens, 64)), dim=-1)
    expanded = _project_prefill(self, "kv_b_proj", latent, weight_scale).view(tokens, 128, 256)
    k_nope, values = expanded.split((128, 128), dim=-1)
    queries = torch.cat((q_nope, q_pe), dim=-1).contiguous()
    keys = torch.cat((k_nope, k_pe.expand(-1, 128, -1)), dim=-1).contiguous()
    output = _attend_prefill(self, queries, keys, values.contiguous(), cu_seqlens, max_seqlen)
    if not isinstance(output, torch.Tensor) or output.shape != (tokens, 128, 128) or output.dtype != torch.bfloat16:
        raise RuntimeError("Selected kernel returned an unexpected R1 attention tensor")
    return _project_prefill(self, "o_proj", output.reshape(tokens, 128 * 128), weight_scale), offload_kv


@torch.inference_mode()
def mla_prefill_sm103_w8a16(self, hidden_states, attention_mask, position_ids, weight_scale):
    """Padded wrapper contract; padded output and KV slots are explicitly zero."""
    _check_module(self, hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError("Expected hidden_states[B,S,D]")
    batch, sequence, _ = hidden_states.shape
    if attention_mask.shape != (batch, sequence) or position_ids.shape != (batch, sequence):
        raise ValueError("Expected attention_mask[B,S] and position_ids[B,S]")
    if attention_mask.device != hidden_states.device or position_ids.device != hidden_states.device:
        raise ValueError("Prefill metadata must be on the activation device")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("Expected a binary padding mask")
    valid = attention_mask.bool()
    lengths = valid.sum(dim=1, dtype=torch.int32)
    if batch == 0 or not bool((lengths > 0).all()):
        raise ValueError("Padded prefill requires nonempty sequences")
    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    packed, packed_kv = mla_prefill_sm103_w8a16_prepacked(
        self, hidden_states[valid], position_ids[valid], cu, int(lengths.max()), batch, weight_scale,
    )
    output = hidden_states.new_zeros((batch, sequence, packed.shape[-1]))
    offload_kv = hidden_states.new_zeros((batch, sequence, 576))
    output[valid], offload_kv[valid] = packed, packed_kv
    return output, offload_kv


def _decode_workspace(module, device):
    # No sharing across layers or CUDA streams, including concurrent microbatches.
    # The initial implementation retains these buffers for the layer's lifetime.
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    workspaces = getattr(module, "_r1_sm103_decode_workspaces", None)
    if workspaces is None:
        workspaces = {}
        module._r1_sm103_decode_workspaces = workspaces
    if key not in workspaces:
        workspaces[key] = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return workspaces[key]


def _validated_block_table(cache, table, lengths, batch, device):
    if cache.ndim != 4 or cache.shape[1:] != (64, 1, 576):
        raise ValueError("Expected MLA cache[physical_pages,64,1,576]")
    if cache.dtype != torch.bfloat16 or cache.device != device or not cache.is_contiguous():
        raise ValueError("MLA cache must be contiguous BF16 on the activation device")
    if table.ndim != 2 or table.shape[0] != batch or table.dtype != torch.int32 or table.device != device:
        raise ValueError("Page table must be int32[B,max_pages] on the activation device")
    pages_per_sequence = (lengths + 63) // 64
    if int(pages_per_sequence.max()) > table.shape[1]:
        raise ValueError("Page table does not cover the current decode token")
    used = torch.arange(table.shape[1], device=device)[None, :] < pages_per_sequence[:, None]
    if not bool(((table[used] >= 0) & (table[used] < cache.shape[0])).all()):
        raise ValueError("Active page table contains an invalid physical page")
    # TRTLLM 0.6.13 requires width divisible by 128/page_size. Invalid unused
    # sentinels are replaced only in this private view; manager storage is intact.
    safe = torch.where(used, table, 0)
    if safe.shape[1] % 2:
        safe = torch.cat((safe, safe.new_zeros((batch, 1))), dim=1)
    return safe.contiguous()


@torch.inference_mode()
def mla_decoding_sm103_bf16_with_pagekv(
    self, hidden_states, q_position_ids, cache_seqlens, max_seqlen,
    weight_scale=None, gpu_paged_kv_manager=None, layer_idx=0, batch_slice=None,
):
    """Paged wrapper contract: returns output[B,1,D], appended KV[B,1,1,576]."""
    _check_module(self, hidden_states)
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("Decode requires hidden_states[B,1,D]")
    batch = hidden_states.shape[0]
    if batch == 0:
        return hidden_states, hidden_states.new_empty((0, 1, 1, 576))
    if gpu_paged_kv_manager is None or weight_scale is None:
        raise ValueError("Decode requires a paged KV manager and FP8 scales")
    device = hidden_states.device
    if q_position_ids.shape != (batch, 1) or q_position_ids.dtype not in (torch.int32, torch.int64) or q_position_ids.device != device:
        raise ValueError("Expected integer q_position_ids[B,1] on the activation device")
    if cache_seqlens.shape != (batch,) or cache_seqlens.dtype != torch.int32 or cache_seqlens.device != device:
        raise ValueError("Expected int32 cache_seqlens[B] on the activation device")
    if not bool((cache_seqlens == q_position_ids[:, 0] + 1).all()) or not bool((cache_seqlens > 0).all()):
        raise ValueError("cache_seqlens must include the current token; write positions are zero-based")
    if max_seqlen < int(cache_seqlens.max()):
        raise ValueError("max_seqlen is smaller than a sequence length")

    from batchgen_kernels.triton.fp8_quantize import deepseek_v3_dequantization
    from batchgen.attention.mla.rotary_embedding import rotary_pos_emb_interleaved_native

    cache, _, table = gpu_paged_kv_manager.get_layer_kv_with_page_table(layer_idx=layer_idx)
    if batch_slice is not None:
        start, end = batch_slice
        if start < 0 or end - start != batch or end > table.shape[0]:
            raise ValueError("Invalid decode microbatch slice")
        table = table[start:end]
    table = _validated_block_table(cache, table, cache_seqlens, batch, device)
    x = _quantize(hidden_states.squeeze(1))
    q = _project_decode(self, "q_a_proj", x, weight_scale)
    new_kv = _project_decode(self, "kv_a_proj_with_mqa", x, weight_scale).view(batch, 1, 576)
    q = _project_decode(self, "q_b_proj", _quantize(self.q_a_layernorm(q)), weight_scale)
    q_nope, q_pe = q.view(batch, 128, 1, 192).split((128, 64), dim=-1)
    q_pe = q_pe.contiguous()
    cos, sin = self.rotary_emb(q_pe, seq_len=max_seqlen)
    if cos.shape[0] < max_seqlen or sin.shape != cos.shape:
        raise ValueError("RoPE cache does not cover the requested sequence length")
    # Use the same FP32 rotation boundary as prefill. The legacy fused kernel
    # performs BF16 products before addition when its cos/sin cache is BF16;
    # that is a different rounding contract and changes the appended cache.
    q_pe = rotary_pos_emb_interleaved_native(q_pe, cos, sin, q_position_ids, 1)
    latent = self.kv_a_layernorm(new_kv[..., :512])
    k_pe = rotary_pos_emb_interleaved_native(
        new_kv[..., 512:].unsqueeze(1), cos, sin, q_position_ids, 1,
    ).squeeze(1)
    offload_kv = torch.cat((latent, k_pe), dim=-1)
    k_tensor = offload_kv.view(batch, 1, 1, 576)
    gpu_paged_kv_manager.update_layer_decode_new_token(
        k_tensor=k_tensor, v_tensor=None,
        sequence_lengths=q_position_ids[:, 0].to(torch.int32).contiguous(),
        layer_idx=layer_idx, batch_slice=batch_slice,
    )
    # Do not persist expanded weights: BatchGen can replace a layer's parameter
    # storage while streaming. kv_b has both dimensions divisible by 128.
    absorb = deepseek_v3_dequantization(
        self.kv_b_proj.weight, weight_scale["kv_b_proj.weight_scale_inv"],
    ).view(128, 256, 512)
    latent_query = torch.einsum("bhd,hdc->bhc", q_nope.squeeze(2), absorb[:, :128])
    query = torch.cat((latent_query, q_pe.squeeze(2)), dim=-1).unsqueeze(1).contiguous()
    attn_out = _attend_decode(self, query, cache, table, cache_seqlens.contiguous(), max_seqlen)
    if attn_out.shape != (batch, 1, 128, 512) or attn_out.dtype != torch.bfloat16:
        raise RuntimeError("Selected kernel returned an unexpected R1 latent attention tensor")
    output = torch.einsum("bqhc,hdc->bqhd", attn_out, absorb[:, 128:]).reshape(batch, 128 * 128)
    output = _project_decode(self, "o_proj", _quantize(output), weight_scale)
    return output.view(batch, 1, -1), k_tensor
