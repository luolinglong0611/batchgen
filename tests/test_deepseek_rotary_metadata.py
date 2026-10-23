"""CPU contracts for the real RoPE constructors used during meta enumeration."""
import ast
import logging
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py"


def rotary_classes():
    names = {"DeepseekV3RotaryEmbedding", "DeepseekV3YarnRotaryEmbedding",
             "yarn_find_correction_dim", "yarn_find_correction_range",
             "yarn_get_mscale", "yarn_linear_ramp_mask"}
    tree = ast.parse(MODEL.read_text(), filename=str(MODEL))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"torch": torch, "nn": nn, "math": math,
                 "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("kind", ["DeepseekV3RotaryEmbedding", "DeepseekV3YarnRotaryEmbedding"])
def test_metadata_context_keeps_all_rotary_buffers_meta_without_cuda_query(kind):
    constructor = rotary_classes()[kind]
    with patch.object(torch.cuda, "is_available", side_effect=AssertionError("must not query CUDA")):
        with torch.device("meta"):
            assert torch.get_default_device().type == "meta"
            module = constructor(dim=8, max_position_embeddings=16)
    assert module.init_device.type == "meta"
    assert {name for name, _ in module.named_buffers()} == {"inv_freq", "cos_cached", "sin_cached"}
    assert all(buffer.is_meta and buffer.data_ptr() == 0 for buffer in module.buffers())
    assert module.cos_cached.shape == module.sin_cached.shape == (16, 8)
    assert torch.get_default_device().type == "cpu"


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_explicit_base_device_overrides_meta_context(device):
    constructor = rotary_classes()["DeepseekV3RotaryEmbedding"]
    with patch.object(torch.cuda, "is_available", side_effect=AssertionError("explicit device bypasses CUDA query")):
        with torch.device("meta"):
            module = constructor(dim=8, max_position_embeddings=16, device=device)
    assert module.init_device == device
    assert all(buffer.device.type == "cpu" and not buffer.is_meta for buffer in module.buffers())


@pytest.mark.parametrize("kind", ["DeepseekV3RotaryEmbedding", "DeepseekV3YarnRotaryEmbedding"])
def test_real_cpu_construction_values_and_buffer_devices_unchanged(kind):
    constructors = rotary_classes()
    with patch.object(torch.cuda, "is_available", return_value=False):
        actual = constructors[kind](dim=8, max_position_embeddings=16)
        expected = constructors[kind](dim=8, max_position_embeddings=16, device="cpu")
    assert actual.init_device == torch.device("cpu")
    for name, value in actual.named_buffers():
        assert value.device.type == "cpu" and not value.is_meta
        torch.testing.assert_close(value, dict(expected.named_buffers())[name], rtol=0, atol=0)


def test_normal_default_still_selects_cuda_when_available_without_allocating_it():
    # Execute just the production constructor. Intercept its arange allocation
    # after device selection; this is a CPU test, not a CUDA emulator.
    constructor = rotary_classes()["DeepseekV3RotaryEmbedding"]
    seen = []

    def stop_before_allocation(*args, **kwargs):
        seen.append(kwargs["device"])
        raise RuntimeError("allocation intercepted")

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch, "arange", stop_before_allocation):
        with pytest.raises(RuntimeError, match="allocation intercepted"):
            constructor(dim=8, max_position_embeddings=16)
    assert seen == [torch.device("cuda")]
