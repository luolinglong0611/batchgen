"""CPU configuration contracts for the explicit resident EP8 deployment."""
import os
from dataclasses import asdict
from unittest.mock import patch

import pytest

from batchgen.config.config import EngineConfig
from batchgen.models.deepseek.deepseekv3.planner import DeepSeekV3Planner


SWITCH = "BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP"
CAP = "BATCHGEN_MAX_DECODE_RANK_BSZ"
LEGACY_CAP = "BATCHGEN_MAX_RANK_BSZ"


def plan(env, *, world_size=8, offloading=False, ratio=0.0):
    config = EngineConfig()
    config.Basic_Config.kv_dtype = "bfloat16"
    config.Basic_Config.set_max_prompt_length(8192)
    config.Basic_Config.max_decoding_length = 2048
    config.Basic_Config.world_size = world_size
    config.EP_Config.enable_offloading = offloading
    config.EP_Config.offloading_ratio = ratio
    with patch.dict(os.environ, env, clear=True):
        return DeepSeekV3Planner().generate_config(config)


@pytest.mark.parametrize("world_size", [8, 16])
def test_disabled_preserves_all_native_configuration(world_size):
    native = plan({}, world_size=world_size)
    disabled = plan({SWITCH: "0", CAP: "bad", LEGACY_CAP: "0"}, world_size=world_size)
    assert asdict(native) == asdict(disabled)
    assert native.Basic_Config.attn_mode == (1 if world_size == 8 else 3)


@pytest.mark.parametrize("env, expected_cap", [
    ({SWITCH: "1"}, 128),
    ({SWITCH: "1", CAP: "192"}, 192),
    ({SWITCH: "1", CAP: "256", LEGACY_CAP: "256"}, 256),
    ({SWITCH: "1", LEGACY_CAP: "128"}, 128),
])
def test_explicit_resident_path_and_prefill_preservation(env, expected_cap):
    native = plan({})
    config = plan(env)
    assert config.Basic_Config.attn_mode == 3
    assert config.Basic_Config.world_size == 8
    assert config.EP_Config.enable is True
    assert config.EP_Config.num_local_expert_per_layer == 32
    assert config.EP_Config.enable_offloading is False
    assert config.EP_Config.offloading_ratio == 0
    assert config.Module_Batching_Config.MoE_decoding_micro_batch_size == expected_cap
    buffers = config.GPU_Buffer_Config
    assert buffers.num_decoding_module_buffer == {
        "attn": 0, "routed_expert": 0, "shared_expert": 0,
    }
    assert (buffers.num_k_buffer, buffers.num_v_buffer, buffers.kv_buffer_num_tokens) == (0, 0, 0)
    assert buffers.num_prefill_module_buffer == native.GPU_Buffer_Config.num_prefill_module_buffer
    assert asdict(config.KV_Storage_Config) == asdict(native.KV_Storage_Config)
    assert asdict(config.Device_Paged_KV_Config) == asdict(native.Device_Paged_KV_Config)
    unchanged_batching = asdict(config.Module_Batching_Config)
    unchanged_batching["MoE_decoding_micro_batch_size"] = native.Module_Batching_Config.MoE_decoding_micro_batch_size
    assert unchanged_batching == asdict(native.Module_Batching_Config)


@pytest.mark.parametrize("value", ["", "true", "2", "-1", "1.0", " 1", "01", "１"])
def test_invalid_switch_rejected(value):
    with pytest.raises(ValueError, match=SWITCH):
        plan({SWITCH: value})


@pytest.mark.parametrize("world_size", [1, 4, 16])
def test_wrong_topology_rejected(world_size):
    with pytest.raises(ValueError, match="world_size=8"):
        plan({SWITCH: "1"}, world_size=world_size)


@pytest.mark.parametrize("offloading, ratio", [(True, 0), (True, 0.2), (False, 0.2)])
def test_offloading_rejected_including_zero_ratio_flag(offloading, ratio):
    with pytest.raises(ValueError, match="offloading disabled"):
        plan({SWITCH: "1"}, offloading=offloading, ratio=ratio)


@pytest.mark.parametrize("key", [CAP, LEGACY_CAP])
@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", " 128", "+128", "0128", "１２８"])
def test_nonpositive_or_malformed_caps_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        plan({SWITCH: "1", key: value})


@pytest.mark.parametrize("env", [
    {CAP: "192", LEGACY_CAP: "256"},
    {LEGACY_CAP: "256"},  # The real worker admission cap would remain 128.
])
def test_admission_and_communication_caps_must_agree(env):
    with pytest.raises(ValueError, match="must match"):
        plan({SWITCH: "1", **env})
