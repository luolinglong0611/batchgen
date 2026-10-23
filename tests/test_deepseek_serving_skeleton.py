"""CPU contracts for R1's empty, explicitly loaded serving parameters.

Extract small production methods to keep CUDA/native imports out of CPU tests.
Full distributed model execution remains a separate GPU qualification gate.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py"
PSM = MODEL.with_name("Parallel_Strategy_Manager.py")
WRAPPERS = MODEL.with_name("wrappers.py")
SPEC = importlib.util.spec_from_file_location("r1_serving_skeleton", MODEL.with_name("skeleton.py"))
SKELETON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SKELETON)


def _method(path, class_name, method_name, **extra):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == method_name)
    namespace = {
        "torch": torch, "nn": nn, "logging": logging,
        "expected_parameter_shape": SKELETON.expected_parameter_shape,
        "validate_parameter_residency": SKELETON.validate_parameter_residency,
        **extra,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("kind", ["linear", "embedding"])
def test_empty_parameters_preserve_module_contract_and_accept_real_weights(kind):
    factory = getattr(SKELETON, f"make_{kind}")
    args = (7, 5)
    kwargs = {"bias": True} if kind == "linear" else {"padding_idx": 0}
    ordinary = factory(SimpleNamespace(), *args, **kwargs)
    skeleton = factory(SimpleNamespace(batchgen_skeleton=True), *args, **kwargs)
    assert list(dict(ordinary.named_parameters())) == list(dict(skeleton.named_parameters()))
    assert all(p.numel() == 0 and not p.is_meta and p.device.type == "cpu"
               for p in skeleton.parameters())
    assert all(p.untyped_storage().nbytes() == 0 for p in skeleton.parameters())
    for name, parameter in ordinary.named_parameters():
        assert SKELETON.expected_parameter_shape(skeleton, name) == tuple(parameter.shape)
    if kind == "linear":
        assert (skeleton.in_features, skeleton.out_features) == (7, 5)
    else:
        assert (skeleton.num_embeddings, skeleton.embedding_dim, skeleton.padding_idx) == (7, 5, 0)

    # This is the actual post_init callback, including the embedding padding row.
    initialize = _method(MODEL, "DeepseekV3PreTrainedModel", "_init_weights")
    initialize(SimpleNamespace(config=SimpleNamespace(initializer_range=0.02)), skeleton)
    skeleton.to(device="cpu", dtype=torch.float64)
    assert all(p.numel() == 0 and p.untyped_storage().nbytes() == 0
               for p in skeleton.parameters())
    for name, parameter in ordinary.named_parameters():
        skeleton.get_parameter(name).data = parameter.detach().clone()
    SKELETON.validate_parameter_residency(skeleton, set())
    inputs = torch.randn(2, 7) if kind == "linear" else torch.tensor([0, 1, 6])
    torch.testing.assert_close(skeleton(inputs), ordinary(inputs))


def test_only_explicitly_streamed_empty_weights_are_allowed():
    model = SKELETON.make_linear(SimpleNamespace(batchgen_skeleton=True), 7, 5, bias=False)
    with pytest.raises(RuntimeError, match="Missing resident weight"):
        SKELETON.validate_parameter_residency(model, set())
    SKELETON.validate_parameter_residency(model, {"weight"})
    model.weight.data = torch.zeros(5, 6)
    with pytest.raises(RuntimeError, match="Weight shape mismatch"):
        SKELETON.validate_parameter_residency(model, {"weight"})


def _manager(phase):
    config = SimpleNamespace(batchgen_skeleton=True, phase=phase)
    model = nn.Module()
    model.norm = nn.LayerNorm(3)
    model.attn = SKELETON.make_linear(config, 3, 3, bias=False)
    model.local = SKELETON.make_linear(config, 3, 3, bias=False)
    model.remote = SKELETON.make_linear(config, 3, 3, bias=False)
    SKELETON.capture_parameter_shapes(model)
    return SimpleNamespace(
        model=model, loaded_model_config=config, world_size=16, global_rank=0,
        rank=1, dequant_scale={},
        state_dict_name_map={
            "attn.weight": {"module_key": "attn_0"},
            "local.weight": {"module_key": "routed_expert_0_0"},
            "remote.weight": {"module_key": "routed_expert_0_16"},
        },
        weight_copy_task={"attn": [], "routed_expert": []},
        skeleton_state_dict={"norm.weight": torch.tensor([2., 3., 4.]),
                             "norm.bias": torch.tensor([1., 2., 3.])},
    )


def test_prefill_and_decode_have_distinct_empty_parameter_whitelists():
    validate = _method(PSM, "DeepseekV3ParallelStrategyManager", "_validate_loaded_parameters")
    prefill = _manager("prefill")
    prefill.weight_copy_task = {
        "attn": ["attn_0"], "routed_expert": ["routed_expert_0_0", "routed_expert_0_16"],
    }
    validate(prefill)
    prefill.weight_copy_task["routed_expert"].remove("routed_expert_0_16")
    with pytest.raises(RuntimeError, match="Missing resident weight: remote.weight"):
        validate(prefill)
    decode = _manager("decode")
    with pytest.raises(RuntimeError, match="Missing resident weight: attn.weight"):
        validate(decode)
    decode.model.attn.weight.data = torch.ones(3, 3)
    with pytest.raises(RuntimeError, match="Missing resident weight: local.weight"):
        validate(decode)
    decode.model.local.weight.data = torch.ones(3, 3)
    validate(decode)  # Only rank 1's expert remains empty on rank 0.
    decode.model.to("cpu", dtype=torch.float64)
    assert decode.model.remote.weight.numel() == 0


def test_skeleton_load_rejects_missing_small_weights_and_wrong_shapes():
    load = _method(PSM, "DeepseekV3ParallelStrategyManager", "_load_model_skeleton")
    manager = _manager("prefill")
    load(manager)
    torch.testing.assert_close(manager.model.norm.weight, torch.tensor([2., 3., 4.]))
    del manager.skeleton_state_dict["norm.bias"]
    with pytest.raises(RuntimeError, match="Missing skeleton weight: norm.bias"):
        load(manager)
    manager.skeleton_state_dict["norm.bias"] = torch.ones(4)
    with pytest.raises(RuntimeError, match="Skeleton weight shape mismatch: norm.bias"):
        load(manager)


def test_small_parameter_shapes_remain_checked_after_data_replacement():
    manager = _manager("prefill")
    manager.model.norm.weight.data = torch.ones(4)
    with pytest.raises(RuntimeError, match="Weight shape mismatch for norm.weight"):
        SKELETON.validate_parameter_residency(
            manager.model, {"attn.weight", "local.weight", "remote.weight"})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_weight_assignment_preserves_checkpoint_dtype(dtype):
    module = SKELETON.make_linear(SimpleNamespace(batchgen_skeleton=True), 3, 2, bias=False)
    weight = torch.ones(2, 3).to(dtype)
    module.weight.data = weight
    assert module.weight.dtype == dtype
    assert module.weight.data_ptr() == weight.data_ptr()
    SKELETON.validate_parameter_residency(module, set())


def test_nonlocal_expert_cannot_be_called_as_a_resident_expert():
    forward = _method(WRAPPERS, "DeepSeekExpertWrapper", "forward")
    wrapper = SimpleNamespace(_batchgen_nonlocal_expert=True,
                              module_key="routed_expert_0_16")
    with pytest.raises(RuntimeError, match="Cannot execute non-local expert"):
        forward(wrapper, torch.ones(1, 3))


def test_all_causal_lm_large_parameter_constructors_use_skeleton_factories():
    tree = ast.parse(MODEL.read_text())
    names = {"DeepseekV3MLP", "DeepseekV3Attention", "DeepseekV3Model", "DeepseekV3ForCausalLM"}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in names:
            continue
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__")
        for node in ast.walk(init):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "nn"
                            and node.func.attr in {"Linear", "Embedding"})


def test_real_attention_constructor_keeps_norms_and_rope_buffers(monkeypatch):
    # Exercise the actual attention/RoPE constructors with small dimensions.
    # Explicitly keep this a CPU test even when executed in a GPU container.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    selected = {
        "DeepseekV3RMSNorm": {"__init__"},
        "DeepseekV3RotaryEmbedding": {"__init__", "_set_cos_sin_cache"},
        "DeepseekV3Attention": {"__init__", "_init_rope"},
    }
    classes = []
    for cls in ast.parse(MODEL.read_text()).body:
        if isinstance(cls, ast.ClassDef) and cls.name in selected:
            cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                        and node.name in selected[cls.name]]
            classes.append(cls)
    namespace = {
        "torch": torch, "nn": nn, "Optional": Optional,
        "DeepseekV3Config": SimpleNamespace, "make_linear": SKELETON.make_linear,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=classes, type_ignores=[])), str(MODEL), "exec"), namespace)
    config = SimpleNamespace(
        attention_dropout=0., hidden_size=8, num_attention_heads=2,
        max_position_embeddings=16, rope_theta=10000., q_lora_rank=4,
        qk_rope_head_dim=4, kv_lora_rank=2, v_head_dim=4, qk_nope_head_dim=4,
        attention_bias=False, rope_scaling=None,
    )
    ordinary = namespace["DeepseekV3Attention"](config, layer_idx=0)
    config.batchgen_skeleton = True
    skeleton = namespace["DeepseekV3Attention"](config, layer_idx=0)
    assert skeleton.q_a_proj.weight.numel() == 0
    assert skeleton.q_a_layernorm.weight.numel() == 4
    assert skeleton.kv_a_layernorm.weight.numel() == 2
    for name, value in ordinary.named_buffers():
        actual = skeleton.get_buffer(name)
        assert not actual.is_meta
        torch.testing.assert_close(actual, value)
    for name, parameter in ordinary.named_parameters():
        assert SKELETON.expected_parameter_shape(skeleton, name) == tuple(parameter.shape)
