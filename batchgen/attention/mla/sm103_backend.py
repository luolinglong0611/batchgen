"""Experimental R1 SM103 MLA backend; full-layer qualification is required.

Candidate dependencies: flash-attn-4[cu13]==4.0.0b36,
flashinfer-python[cu13]==0.6.13 and flashinfer-cubin==0.6.13, plus the repository's
Triton blockscale GEMM and batchgen_kernels. Run attention_qualification.py first, then a
complete-layer prefill/append/decode comparison before enabling this module.
The independent precision candidate is selected only by setting the layer's
_r1_attention_implementation to "triton_precision"; the default remains
"fa4_flashinfer". The precision path imports neither FA4 nor FlashInfer.
"torch_fp64_reference" is a separate diagnostic fallback: Triton prefill and
bounded PyTorch FP64 decode. Production dispatch must never select it implicitly.
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
import logging
import time
from types import SimpleNamespace

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
    if implementation not in ("fa4_flashinfer", "triton_precision", "torch_fp64_reference"):
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
            if not (_decode_optimization_enabled("BATCHGEN_R1_SM103_ALLOW_CAPTURE")
                    and _decode_optimization_enabled("BATCHGEN_R1_SM103_ASYNC_VALIDATION")
                    and _decode_optimization_enabled("BATCHGEN_R1_SM103_CACHE_ABSORB")):
                raise RuntimeError("R1 SM103 capture requires explicit checked decode optimizations")
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
    if activation_bf16.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D activation")
    if activation_bf16.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 activation")
    if weight_data_fp8.ndim != 2 or weight_data_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected 2D E4M3FN FP8 weight")
    n, k = weight_data_fp8.shape
    if n <= 0 or k <= 0 or activation_bf16.shape[-1] != k:
        raise ValueError("Expected positive weight dimensions and matching activation width")
    if (weight_scale_inv_fp32.dtype != torch.float32
            or weight_scale_inv_fp32.shape != ((n + 127) // 128, (k + 127) // 128)):
        raise ValueError("Expected FP32 weight scales for 128x128 blocks")
    device = activation_bf16.device
    if (device.type != "cuda" or weight_data_fp8.device != device
            or weight_scale_inv_fp32.device != device):
        raise ValueError("Activation, weight and scales must share a CUDA device")
    # Empty local decode ranks still participate in routed MoE collectives.
    # Their shared expert has no rows; skip only its local GEMM, after checking
    # the same tensor metadata as nonempty inputs. No quantizer/kernel import.
    if activation_bf16.numel() == 0:
        return activation_bf16.new_empty((*activation_bf16.shape[:-1], n))

    from batchgen.gemm.w8a8 import w8a8_gemm

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
    if _attention_implementation(module) in ("triton_precision", "torch_fp64_reference"):
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


def _decode_torch_fp64_reference(query, cache, table, lengths, scale):
    """Diagnostic online FP64 decode with bounded scratch, one BF16 output cast.

    The production wrapper has validated metadata. Head/key tiles bound scratch
    independently of context length. This deliberately slow fallback helps
    distinguish numerical errors from physical cache layout errors; it is not a
    performance backend and never substitutes values into the independent oracle.
    """
    batch, _, heads, _ = query.shape
    output = torch.empty((batch, 1, heads, 512), dtype=torch.bfloat16, device=query.device)
    for row, length in enumerate(lengths.tolist()):
        for head in range(0, heads, 16):
            q = query[row, 0, head:head+16].double()
            maximum = torch.full((q.shape[0], 1), -torch.inf, dtype=torch.float64, device=q.device)
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros((q.shape[0], 512), dtype=torch.float64, device=q.device)
            for start in range(0, length, 256):
                positions = torch.arange(start, min(start+256, length), device=q.device)
                keys = cache[table[row, positions // 64].long(), positions % 64, 0].double()
                scores = (q @ keys.T) * scale
                new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                correction = torch.exp(maximum - new_maximum)
                probabilities = torch.exp(scores - new_maximum)
                denominator = denominator * correction + probabilities.sum(-1, keepdim=True)
                accumulator = accumulator * correction + probabilities @ keys[:, :512]
                maximum = new_maximum
            output[row, 0, head:head+16] = (accumulator / denominator).bfloat16()
    return output


def _attend_decode(module, query, cache, table, lengths, max_seqlen):
    if _attention_implementation(module) == "torch_fp64_reference":
        return _decode_torch_fp64_reference(query, cache, table, lengths, module.softmax_scale)
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


def _decode_optimization_enabled(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _validate_decode_lengths(lengths, positions, max_seqlen, *, async_validation=False):
    if async_validation:
        valid = ((lengths == positions[:, 0] + 1)
                 & (lengths > 0) & (lengths <= max_seqlen)).all()
        torch._assert_async(valid, "Invalid R1 decode sequence lengths/positions")
    else:
        if not bool((lengths == positions[:, 0] + 1).all()) or not bool((lengths > 0).all()):
            raise ValueError("cache_seqlens must include the current token; write positions are zero-based")
        if max_seqlen < int(lengths.max()):
            raise ValueError("max_seqlen is smaller than a sequence length")


def _check_block_table_layout(cache, table, batch, device):
    if cache.ndim != 4 or cache.shape[1:] != (64, 1, 576):
        raise ValueError("Expected MLA cache[physical_pages,64,1,576]")
    if cache.dtype != torch.bfloat16 or cache.device != device or not cache.is_contiguous():
        raise ValueError("MLA cache must be contiguous BF16 on the activation device")
    if table.ndim != 2 or table.shape[0] != batch or table.dtype != torch.int32 or table.device != device:
        raise ValueError("Page table must be int32[B,max_pages] on the activation device")


def _validated_block_table(cache, table, lengths, batch, device, *, async_validation=False):
    _check_block_table_layout(cache, table, batch, device)
    pages_per_sequence = (lengths + 63) // 64
    used = torch.arange(table.shape[1], device=device)[None, :] < pages_per_sequence[:, None]
    if async_validation:
        # Boolean indexing creates a dynamically sized CUDA result and forces
        # a host synchronization even without an explicit .item(). Keep every
        # value check, but assert on the producing stream before cache access.
        # Invalid metadata invalidates the CUDA context; the worker's existing
        # error/cleanup path must terminate the run, never return a response.
        covered = (pages_per_sequence <= table.shape[1]).all()
        valid_pages = (((table >= 0) & (table < cache.shape[0])) | ~used).all()
        torch._assert_async(covered & valid_pages, "Invalid R1 decode physical page table")
    else:
        if int(pages_per_sequence.max()) > table.shape[1]:
            raise ValueError("Page table does not cover the current decode token")
        if not bool(((table[used] >= 0) & (table[used] < cache.shape[0])).all()):
            raise ValueError("Active page table contains an invalid physical page")
    # TRTLLM 0.6.13 requires width divisible by 128/page_size. Invalid unused
    # sentinels are replaced only in this private view; manager storage is intact.
    safe = torch.where(used, table, 0)
    if safe.shape[1] % 2:
        safe = torch.cat((safe, safe.new_zeros((batch, 1))), dim=1)
    return safe.contiguous()


def _decode_absorbed_weights(module, scales, *, cache_enabled=False):
    from batchgen_kernels.triton.fp8_quantize import deepseek_v3_dequantization

    weight = module.kv_b_proj.weight
    scale = scales["kv_b_proj.weight_scale_inv"]
    key = None
    if cache_enabled:
        try:
            # A strong detached alias below keeps the old storage alive, so a
            # replacement cannot reuse its address and match a stale entry.
            # Unversioned inference tensors cannot support the mutation check.
            key = tuple((t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                        for t in (weight, scale))
        except RuntimeError:
            key = None
    if key is not None:
        stream = torch.cuda.current_stream(weight.device).cuda_stream
        entries = getattr(module, "_r1_sm103_absorbed_weight_cache", None)
        if entries is None:
            entries = {}
            module._r1_sm103_absorbed_weight_cache = entries
        entry = entries.get(stream)
        if entry is not None and entry[0] == key:
            return entry[3]
    # Same kernel, dtype and layout as the original per-token computation.
    absorb = deepseek_v3_dequantization(weight, scale).view(128, 256, 512)
    if key is not None:
        entries[stream] = (key, weight.detach(), scale.detach(), absorb)
    return absorb


def _decode_query_kv(module, hidden_states, positions, scales, cos, sin, absorb):
    from batchgen.attention.mla.rotary_embedding import rotary_pos_emb_interleaved_native

    batch = hidden_states.shape[0]
    x = _quantize(hidden_states.squeeze(1))
    q = _project_decode(module, "q_a_proj", x, scales)
    new_kv = _project_decode(module, "kv_a_proj_with_mqa", x, scales).view(batch, 1, 576)
    q = _project_decode(module, "q_b_proj", _quantize(module.q_a_layernorm(q)), scales)
    q_nope, q_pe = q.view(batch, 128, 1, 192).split((128, 64), dim=-1)
    q_pe = rotary_pos_emb_interleaved_native(q_pe.contiguous(), cos, sin, positions, 1)
    latent = module.kv_a_layernorm(new_kv[..., :512])
    k_pe = rotary_pos_emb_interleaved_native(
        new_kv[..., 512:].unsqueeze(1), cos, sin, positions, 1,
    ).squeeze(1)
    k_tensor = torch.cat((latent, k_pe), dim=-1).view(batch, 1, 1, 576)
    latent_query = torch.einsum("bhd,hdc->bhc", q_nope.squeeze(2), absorb[:, :128])
    query = torch.cat((latent_query, q_pe.squeeze(2)), dim=-1).unsqueeze(1).contiguous()
    return query, k_tensor


def _decode_output(module, attn_out, scales, absorb):
    batch = attn_out.shape[0]
    output = torch.einsum("bqhc,hdc->bqhd", attn_out, absorb[:, 128:]).reshape(batch, 128 * 128)
    output = _project_decode(module, "o_proj", _quantize(output), scales)
    return output.view(batch, 1, -1)


def _projection_graph_entry(module, hidden_states, positions, scales):
    """Capture only projections; keep the actual attention launch unchanged.

    One shape per layer bounds graph-pool growth as continuous batches shrink.
    Static outputs are never returned directly: asynchronous host KV offload
    must retain a distinct allocation while later tokens replay these graphs.
    """
    source_tensors = [getattr(module, name).weight for name in
                      ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                       "q_a_layernorm", "kv_a_layernorm")]
    source_tensors += [scales[name] for name in sorted(scales)]
    source_tensors += [module.rotary_emb.cos_cached, module.rotary_emb.sin_cached]
    try:
        sources = tuple((t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                        for t in source_tensors)
    except RuntimeError as error:
        raise RuntimeError("R1 projection graphs require versioned immutable decode weights") from error
    current = torch.cuda.current_stream(hidden_states.device)
    key = (tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device,
           positions.dtype, current.cuda_stream, sources)
    entry = getattr(module, "_r1_sm103_projection_graph_entry", None)
    if entry is not None and entry.key == key:
        return entry
    started = time.perf_counter()
    entry = SimpleNamespace(key=key, aliases=[t.detach() for t in source_tensors],
                            hidden=torch.empty_like(hidden_states), positions=torch.empty_like(positions),
                            attn=torch.zeros((hidden_states.shape[0],1,128,512),
                                             device=hidden_states.device,dtype=torch.bfloat16))
    # The fixed RoPE table supplies identical per-position values. Crucially,
    # FlashInfer still receives each call's exact max_seqlen outside the graph;
    # rounding that bound changed a tested output and is deliberately avoided.
    entry.cos, entry.sin = module.rotary_emb(hidden_states, seq_len=32768)
    entry.absorb = _decode_absorbed_weights(module, scales, cache_enabled=True)
    entry.hidden.copy_(hidden_states)
    entry.positions.copy_(positions)
    stream = getattr(module, "_r1_sm103_projection_capture_stream", None)
    if stream is None:
        stream = torch.cuda.Stream(device=hidden_states.device)
        module._r1_sm103_projection_capture_stream = stream
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            _decode_query_kv(module,entry.hidden,entry.positions,scales,entry.cos,entry.sin,entry.absorb)
            _decode_output(module,entry.attn,scales,entry.absorb)
    current.wait_stream(stream)
    entry.pre, entry.post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(entry.pre,stream=stream):
        entry.query, entry.kv = _decode_query_kv(
            module,entry.hidden,entry.positions,scales,entry.cos,entry.sin,entry.absorb)
    with torch.cuda.graph(entry.post,stream=stream):
        entry.output = _decode_output(module,entry.attn,scales,entry.absorb)
    entry.capture_seconds = time.perf_counter()-started
    module._r1_sm103_projection_graph_entry = entry
    stats = getattr(module,"_r1_sm103_projection_graph_stats",None)
    if stats is None:
        stats = {"captures":0,"capture_seconds":0.,"replays":0}
        module._r1_sm103_projection_graph_stats = stats
    stats["captures"] += 1
    stats["capture_seconds"] += entry.capture_seconds
    return entry


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
    async_validation = _decode_optimization_enabled("BATCHGEN_R1_SM103_ASYNC_VALIDATION")
    fused_metadata = _decode_optimization_enabled("BATCHGEN_R1_SM103_FUSED_METADATA")
    if fused_metadata and not async_validation:
        raise ValueError("Fused R1 metadata requires async validation")
    if not fused_metadata:
        _validate_decode_lengths(cache_seqlens, q_position_ids, max_seqlen,
                                 async_validation=async_validation)

    cache, _, table = gpu_paged_kv_manager.get_layer_kv_with_page_table(layer_idx=layer_idx)
    if batch_slice is not None:
        start, end = batch_slice
        if start < 0 or end - start != batch or end > table.shape[0]:
            raise ValueError("Invalid decode microbatch slice")
        table = table[start:end]
    write_positions = None
    if fused_metadata:
        _check_block_table_layout(cache, table, batch, device)
        from .r1_decode_metadata import prepare
        table, write_positions = prepare(cache, table, cache_seqlens, q_position_ids, max_seqlen)
    else:
        table = _validated_block_table(cache, table, cache_seqlens, batch, device,
                                       async_validation=async_validation)
    projection_graphs = _decode_optimization_enabled("BATCHGEN_R1_SM103_PROJECTION_GRAPHS")
    entry = None
    if projection_graphs:
        if not async_validation or not _decode_optimization_enabled("BATCHGEN_R1_SM103_CACHE_ABSORB"):
            raise ValueError("R1 projection graphs require async validation and guarded absorbed-weight caching")
        if max_seqlen <= 32768:
            previous = getattr(self,"_r1_sm103_projection_graph_entry",None)
            entry = _projection_graph_entry(self,hidden_states,q_position_ids,weight_scale)
            if layer_idx == 0 and entry is not previous:
                logging.info("R1 projection graphs captured layer0 batch=%s seconds=%.3f",
                             batch,entry.capture_seconds)
    absorb = _decode_absorbed_weights(
        self,weight_scale,cache_enabled=_decode_optimization_enabled("BATCHGEN_R1_SM103_CACHE_ABSORB"))
    if entry is not None:
        entry.hidden.copy_(hidden_states)
        entry.positions.copy_(q_position_ids)
        entry.pre.replay()
        query,k_tensor = entry.query,entry.kv
    else:
        cos,sin = self.rotary_emb(hidden_states,seq_len=max_seqlen)
        if cos.shape[0] < max_seqlen or sin.shape != cos.shape:
            raise ValueError("RoPE cache does not cover the requested sequence length")
        query,k_tensor = _decode_query_kv(self,hidden_states,q_position_ids,weight_scale,cos,sin,absorb)
    gpu_paged_kv_manager.update_layer_decode_new_token(
        k_tensor=k_tensor, v_tensor=None,
        sequence_lengths=(write_positions if write_positions is not None
                          else q_position_ids[:, 0].to(torch.int32).contiguous()),
        layer_idx=layer_idx, batch_slice=batch_slice,
    )
    attn_out = _attend_decode(self, query, cache, table, cache_seqlens.contiguous(), max_seqlen)
    if attn_out.shape != (batch, 1, 128, 512) or attn_out.dtype != torch.bfloat16:
        raise RuntimeError("Selected kernel returned an unexpected R1 latent attention tensor")
    if entry is not None:
        entry.attn.copy_(attn_out)
        entry.post.replay()
        self._r1_sm103_projection_graph_stats["replays"] += 1
        return entry.output.clone(),k_tensor.clone()
    return _decode_output(self,attn_out,weight_scale,absorb),k_tensor
