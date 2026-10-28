# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""DeepSeek-specific wrappers for BatchGen execution.

Provides wrappers for DeepSeek-R1 and DeepSeek-V3 models with FP8 quantization:
- DeepSeekExpertWrapper: Expert wrapper with FP8 dequantization
- DeepSeekAttnWrapper: Attention wrapper with FP8 dequantization
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
from batchgen.attention.mla.decode_weight_cache import (
    clear_mla_decode_weight_cache,
    configure_mla_decode_weight_cache,
)


def _read_decode_attention_micro_batch_size():
    """Opt in to the paged decoder's existing per-layer slice/merge path.

    Zero (the default) keeps the previous one-call local attention batch. This
    is independent of the rank admission limit and MoE communication buffers.
    """
    name = "BATCHGEN_DECODE_ATTN_MICRO_BATCH_SIZE"
    raw = os.environ.get(name, "0")
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be a non-negative integer (0 disables)")
    return int(raw)


class DeepSeekExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for DeepSeek-R1/V3.

    Handles:
    - FP8 block-wise dequantization with scale factors
    - FP8 weight caching for local experts (no loading needed)
    - deepgemm kernel for forward computation

    Attributes:
        weight_dequant_scale: Dict of scale factors for dequantization
        fp8_gate: Cached FP8 gate_proj weights (when not loading)
        fp8_up: Cached FP8 up_proj weights
        fp8_down: Cached FP8 down_proj weights
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek expert wrapper.

        Args:
            module: Expert FFN module to wrap
            layer_idx: Layer index in the model
            expert_idx: Expert index (-1 for shared expert)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed.
                        False = load from buffer each forward.
            weight_dequant_scale: Dict of FP8 scale factors
        """
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}

        # FP8 weight caching for local experts
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize FP8 weights using scale factors.

        Args:
            weights_dict: Dict mapping parameter names to FP8 weights

        Returns:
            Dict mapping parameter names to BF16 weights
        """
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        """Cache FP8 weights for local experts.

        Called when persistent=True - weights stay in GPU memory.
        """
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 weights."""
        logging.debug(
            f"Clearing expert weights. Layer idx {self.layer_idx}, "
            f"expert idx {self.expert_idx}"
        )
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using deepgemm kernel.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Use deepgemm forward for both prefill and decode
        return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 handling.

        If persistent=True: Use cached FP8 weights directly (pre-loaded on GPU)
        If persistent=False: Load from core engine, dequantize, forward, cleanup

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Forward pass. Phase: {self.phase}"
        )

        if not self.persistent:
            # Load weights from core engine (non-persistent experts)
            weights = self.load_weights(self.module_key)

            # Apply weights to module (dequantization happens in deepgemm_forward)
            for name, param in self.module.named_parameters():
                param.data = weights[name]

        else:
            # Use cached FP8 weights (persistent experts)
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.up_proj.weight.data = self.fp8_up
            self.module.down_proj.weight.data = self.fp8_down

        # Micro-batch forward
        result = self.micro_batch_forward(hidden_states, "expert")

        # Cleanup for non-persistent experts
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Finish forward pass. Phase: {self.phase}"
        )

        return result


class DeepSeekAttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequantization for DeepSeek-R1/V3.

    This is a placeholder - the full implementation requires porting
    the extensive prefill/decode logic from the original Wrapper.py.
    For now, it inherits the base behavior.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek attention wrapper."""
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

        # FP8 weight caching for MLA architecture
        # DeepSeek V3 uses: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None
        # Only _register_fp8_weights may enable reuse after resident loading.
        configure_mla_decode_weight_cache(self.module, enabled=False)
        self.decode_attn_micro_batch_size = _read_decode_attention_micro_batch_size()
        if self.decode_attn_micro_batch_size and self.layer_idx == 0:
            logging.info(
                "[ATTN-MICROBATCH] paged decode attention chunk=%d; "
                "merge all local rows before the decoder layer's MoE",
                self.decode_attn_micro_batch_size,
            )

    def _register_fp8_weights(self):
        """Cache FP8 attention weights for MLA architecture."""
        from batchgen.attention.mla.r1_decode_cuda_graph import release_attention_graph
        release_attention_graph(self)
        clear_mla_decode_weight_cache(self.module)
        self.fp8_q_a_proj = self.module.q_a_proj.weight.data
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        self.fp8_kv_a_proj = self.module.kv_a_proj_with_mqa.weight.data
        self.fp8_kv_b_proj = self.module.kv_b_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data
        configure_mla_decode_weight_cache(self.module, enabled=self.persistent)

    def _unregister_fp8_weights(self):
        """Clear cached FP8 attention weights."""
        configure_mla_decode_weight_cache(self.module, enabled=False)
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def apply_weights(self, weights_dict):
        # Base loading uses .data assignments, which may bypass version bumps.
        from batchgen.attention.mla.r1_decode_cuda_graph import release_attention_graph
        release_attention_graph(self)
        clear_mla_decode_weight_cache(self.module)
        return super().apply_weights(weights_dict)

    def clear_weights(self):
        from batchgen.attention.mla.r1_decode_cuda_graph import release_attention_graph
        release_attention_graph(self)
        clear_mla_decode_weight_cache(self.module)
        return super().clear_weights()

    def _apply(self, fn, recurse=True):
        # Plain cached tensors are not registered buffers; release them before
        # nn.Module.to()/cuda()/dtype conversion can move the source weights.
        from batchgen.attention.mla.r1_decode_cuda_graph import release_attention_graph
        release_attention_graph(self)
        clear_mla_decode_weight_cache(self.module)
        return super()._apply(fn, recurse=recurse)

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged - NO dequantization.

        For DeepSeek with FP8 DeepGEMM, weights stay in FP8:
        - w8a16_gemm quantizes activations to FP8 (not weights)
        - deep_gemm.fp8_gemm_nt() handles FP8 weights directly
        - Dequantization would break w8a16_gemm which expects FP8 weights
        """
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward using FP8 DeepGEMM.

        In prepack mode (cu_seqlens-based batching), calls prepacked attention
        which doesn't need attention_mask. In regular mode, calls standard
        attention which requires attention_mask for unpadding.

        Calls prefill_attn_w8a16 which uses w8a16_gemm:
        - Weights stay in FP8 format
        - Activations are quantized to FP8 via act_quant()
        - deep_gemm.fp8_gemm_nt() outputs BF16

        Returns:
            Tuple of (attn_output, attn_weights, kv_cache) to match decoder_layer's
            expected unpacking.
        """
        if self.prepack_mode:
            # Keep at most one layer's KV offloads live. Unlike discarding the
            # native shared_future (which blocks in its final destructor), an
            # explicit retirement is visible and propagates native exceptions.
            AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
                self.layer_idx, device=hidden_states.device,
            )
            # Prepack mode: hidden_states is [1, total_tokens, hidden_dim]
            # Prepacked attention expects [total_tokens, hidden_dim]
            hidden_states_2d = hidden_states.squeeze(0)
            if (os.environ.get("BATCHGEN_R1_SM103_MLA_COMPAT") == "flashinfer-fa2"
                    and os.environ.get("BATCHGEN_R1_PREFILL_PREPARATION_REUSE") == "1"):
                from batchgen.attention.mla.common_flashinfer_backend import packed_prefill_cpu_metadata
                _, cpu_cu = packed_prefill_cpu_metadata(
                    self.prepack_seq_lengths, self.prepack_num_sequences,
                    hidden_states_2d.shape[0], self.prepack_max_seqlen,
                )
                # The compatibility adapter plans on CPU. Native attention
                # paths keep the original CUDA metadata contract unchanged.
                cu_seqlens = torch.tensor(cpu_cu, dtype=torch.int32)
            else:
                cu_seqlens = self.prepack_cu_seqlens.to(hidden_states_2d.device)

            attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                cu_seqlens,
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
                self.weight_dequant_scale
            )

            # Offload KV cache per-sequence to host
            # offload_kv is [total_tokens, kv_lora_rank + qk_rope_head_dim]
            self._offload_prepacked_kv(offload_kv)

            # Reshape back to [1, total_tokens, hidden_dim] for decoder_layer
            attn_output = attn_output.unsqueeze(0)

            # Return 3-tuple to match decoder_layer's expected unpacking
            return (attn_output, None, None)
        else:
            # Regular mode: use attention_mask-based unpadding
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            attn_output, offload_kv = self.module.prefill_attn_w8a16(
                hidden_states,
                attention_mask,
                position_ids,
                self.weight_dequant_scale  # Inverse scales for w8a16_gemm
            )

            # Return 3-tuple for consistency
            return (attn_output, None, offload_kv)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory.

        For MLA, offload_kv contains [compressed_kv, k_pe] for each token.
        Split by cu_seqlens and offload each sequence.

        Args:
            offload_kv: [total_tokens, kv_lora_rank + qk_rope_head_dim]
        """
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch
        if num_sequences:
            AttnWrapperBase.pin_prefill_offload_tensor(offload_kv, self.layer_idx)

        if (os.environ.get("BATCHGEN_R1_SM103_MLA_COMPAT") == "flashinfer-fa2"
                and os.environ.get("BATCHGEN_R1_PREFILL_PREPARATION_REUSE") == "1"):
            from batchgen.attention.mla.common_flashinfer_backend import packed_prefill_cpu_metadata
            lengths, _ = packed_prefill_cpu_metadata(
                self.prepack_seq_lengths, num_sequences,
                offload_kv.shape[0], self.prepack_max_seqlen,
            )
            if num_sequences:
                # A single native future owns the complete packed source;
                # native code resolves each sequence's actual physical pages.
                # Keep the same pinned-source and next-layer retirement rules.
                task = self.core_engine.host_paged_kv_worker_view.async_offload_packed_layer_kv_to_host(
                    layer_idx=self.layer_idx,
                    sequence_ids=list(global_sequence_ids),
                    k_tensor=offload_kv.unsqueeze(1),
                    v_tensor=None,
                    sequence_lengths=list(lengths),
                )
                AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)
            return

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # Extract KV for this sequence
            seq_kv = offload_kv[start_idx:end_idx]  # [seq_len, kv_dim]

            # Reshape to [1, seq_len, 1, kv_dim] for KV cache API
            seq_kv = seq_kv.unsqueeze(0).unsqueeze(2)

            seq_global_id = [global_sequence_ids[seq_idx]]

            # MLA has no V (K contains compressed KV + k_pe)
            task = self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward using FlashMLA backend.

        FlashMLA decode functions handle FP8 weights correctly via w8a8_deepgemm,
        unlike vanilla DeepseekV3Attention.forward() which expects BF16 weights.

        Routes to appropriate decode backend based on KV cache management:
        - If gpu_paged_kv_manager is set: Use decoding_attn_mode_3_bf16 (paged KV)
        - Otherwise: Use decoding_attn_mode_3_fp8 (FP8 KV with tensor references)

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]
            **kwargs: Unused (all state from class-level attributes)

        Returns:
            Tuple of (attn_output, past_key_states, scale) for KV cache update.
        """
        from batchgen.attention.mla.r1_decode_cuda_graph import requested, run_attention_graph
        if requested():
            return run_attention_graph(self, hidden_states)

        # Get class-level decode state
        past_key_states = AttnWrapperBase.past_key_states
        attention_mask = AttnWrapperBase.attention_mask
        position_ids = AttnWrapperBase.position_ids  # q_position_ids for decode
        scale = AttnWrapperBase.scale
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        # Check if using paged KV (GPUPagedKVCacheManager)
        if gpu_paged_kv_manager is not None:
            # BF16 paged KV path
            chunk_size = self.decode_attn_micro_batch_size
            batch_size = hidden_states.shape[0]
            if chunk_size and batch_size > chunk_size:
                if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("MLA decode micro-batching currently requires CUDA Graph OFF")
                from batchgen.attention.mla.decode_microbatch_scope import (
                    decode_attention_microbatch_scope,
                )

                if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
                    raise ValueError("MLA decode micro-batching requires [batch, 1, hidden] inputs")
                if (position_ids is None or cache_seqlens is None
                        or position_ids.shape[0] != batch_size
                        or cache_seqlens.shape[0] != batch_size):
                    raise ValueError("MLA decode micro-batch metadata must match the full local batch")
                # Same per-layer order as Wrapper_dep.Attn_Wrapper mode 3:
                # finish each attention slice, assemble local rows, then return
                # once so the decoder layer calls MoE once on the full batch.
                attn_output = torch.empty_like(hidden_states)
                all_k_tensors = []
                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    with decode_attention_microbatch_scope(
                        gpu_paged_kv_manager, position_ids, cache_seqlens,
                        start, end,
                    ) as (micro_positions, micro_lengths):
                        micro_output, micro_k = self.module.decoding_attn_mode_3_bf16(
                            hidden_states[start:end], micro_positions, micro_lengths,
                            max_seqlen, self.weight_dequant_scale,
                            gpu_paged_kv_manager, self.layer_idx, (start, end),
                        )
                    if (micro_output.shape != hidden_states[start:end].shape
                            or micro_k.shape[0] != end - start):
                        raise ValueError("MLA decode backend returned inconsistent micro-batch rows")
                    attn_output[start:end].copy_(micro_output)
                    all_k_tensors.append(micro_k)
                k_tensor = torch.cat(all_k_tensors, dim=0)
            else:
                attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                    hidden_states,
                    position_ids,
                    cache_seqlens,
                    max_seqlen,
                    self.weight_dequant_scale,
                    gpu_paged_kv_manager,
                    self.layer_idx,
                    None  # batch_slice (handled by gpu_paged_kv_manager)
                )

            # Offload k_tensor via KV append callback if available
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)

            # Return tuple (attn_output is all that's needed - KV managed externally)
            return (attn_output, None, None)
        else:
            if self.decode_attn_micro_batch_size:
                raise ValueError(
                    "BATCHGEN_DECODE_ATTN_MICRO_BATCH_SIZE currently requires the paged BF16 KV path"
                )
            # FP8 KV cache with tensor references
            # Get layer-specific KV cache slice
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None

            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_mode_3_fp8(
                hidden_states,
                layer_past_key,
                None,  # past_value_states (None for MLA - K contains compressed KV)
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale  # FP8 weight scales for w8a8_deepgemm
            )

            return (attn_output, updated_past_key, updated_scale)
