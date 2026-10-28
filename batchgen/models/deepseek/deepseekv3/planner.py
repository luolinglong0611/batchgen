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

"""DeepSeek-V3/R1 specific planner for BatchGen configuration."""

import os
import re

from batchgen.planner.base_planner import BasePlanner


class DeepSeekV3Planner(BasePlanner):
    """Planner specialized for DeepSeek-V3/R1 models.

    Optimized for H20 GPUs with MoE architecture.
    """

    # Model-specific constants
    MAGIC_NUM = 672_000  # 224000 * 3, optimized for DeepSeek MoE architecture
    DEFAULT_MEM_FRAC = 0.85  # Higher utilization for DeepSeek
    NUM_EXPERTS = 256  # Total experts in DeepSeek-V3/R1

    def __version__(self):
        """Version for H20 and DeepSeek-R1/V3."""
        return "0.1.6"

    def _adjust_config_for_model(self):
        """Opt in to the existing resident EP decode path on eight GPUs."""
        from .resident_prefill import (
            moe_backend, resident_prefill_requested, resident_prefill_token_cap,
        )
        backend = moe_backend()
        resident_prefill = resident_prefill_requested()
        name = "BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP"
        requested = os.environ.get(name, "0")
        if requested not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        ll_requested = bool(getattr(
            self.config.Module_Batching_Config, "r1_deepep_ll", False))
        if backend == "deep_gemm" and (self.world_size != 8 or requested != "1" or not ll_requested):
            raise ValueError("R1 DeepGEMM requires EP8 resident decode and --enable-deepep")
        if ll_requested and self.world_size <= 8:
            if self.world_size != 8 or requested != "1":
                raise ValueError(
                    "R1 --enable-deepep on one node requires world_size=8 and "
                    f"{name}=1; legacy decode is not a DeepEP LL fallback"
                )
        if requested == "0":
            return

        if self.world_size != 8:
            raise ValueError(f"{name} requires world_size=8")
        ep = self.config.EP_Config
        if ep.enable_offloading or ep.offloading_ratio != 0:
            raise ValueError(f"{name} requires EP offloading disabled and ratio=0")

        # The worker's admission cap defaults to 128. Its legacy spelling
        # controls the strategy manager's communication buffers separately.
        cap_name = "BATCHGEN_MAX_DECODE_RANK_BSZ"
        legacy_name = "BATCHGEN_MAX_RANK_BSZ"
        cap = os.environ.get(cap_name, "128")
        legacy_cap = os.environ.get(legacy_name, cap)
        for key, value in ((cap_name, cap), (legacy_name, legacy_cap)):
            if not re.fullmatch(r"[1-9][0-9]*", value):
                raise ValueError(f"{key} must be a positive integer")
        if cap != legacy_cap:
            raise ValueError(f"{cap_name} and {legacy_name} must match")

        # Select the same fully resident DP-attention/EP-MoE path already
        # used by multi-node configurations, retaining the real world size.
        self.config.Basic_Config.attn_mode = 3
        ep.enable = True
        ep.num_local_expert_per_layer = self.NUM_EXPERTS // self.world_size
        self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = int(cap)
        buffers = self.config.GPU_Buffer_Config
        buffers.num_decoding_module_buffer = {
            "attn": 0, "routed_expert": 0, "shared_expert": 0,
        }
        buffers.num_k_buffer = 0
        buffers.num_v_buffer = 0
        buffers.kv_buffer_num_tokens = 0
        if resident_prefill:
            # A normal-EP prefill processes the whole local prompt microbatch,
            # not many LL exchanges. Match the explicit SG8K/DP activation
            # budget by default; this is a new opt-in execution condition.
            self.config.Module_Batching_Config.prefill_micro_batch_token_cap = resident_prefill_token_cap()
            buffers.num_prefill_module_buffer = {
                "attn": 0, "routed_expert": 0, "shared_expert": 0,
            }

    def get_module_shapes(self) -> dict:
        """Return DeepSeek-V3/R1 specific tensor shapes."""
        return {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_kv_heads": 8,  # MLA uses 8 KV heads
            "num_layers": 61,
            "num_experts": 256,
            "num_experts_per_tok": 8,  # Top-8 experts selected
            "expert_hidden_size": 2048,
            "kv_lora_rank": 512,  # MLA KV compression rank
            "q_lora_rank": 1536,  # MLA Q compression rank
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }


# Alias for backward compatibility
Scheduler = DeepSeekV3Planner
