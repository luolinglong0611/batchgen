"""CPU-only coverage of the R1 parameter-server enumeration path.

Extract the production method rather than importing the server's native CUDA
extension. The small model uses the production MLP constructor; attention is a
minimal named-parameter fixture because no attention execution is involved.
"""

import ast
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "batchgen/models/deepseek/deepseek_parameter_server.py"
MODEL = ROOT / "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py"
_SPEC = importlib.util.spec_from_file_location("r1_metadata_skeleton", MODEL.with_name("skeleton.py"))
_SKELETON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SKELETON)


def _method(path, class_name, method_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == method_name)


def _small_model(config):
    # Use the real MLP allocation logic with tiny dimensions.
    init = _method(MODEL, "DeepseekV3MLP", "__init__")
    cls = ast.ClassDef(name="DeepseekV3MLP", bases=[ast.Attribute(
        value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
        keywords=[], body=[init], decorator_list=[])
    namespace = {"nn": nn, "ACT2FN": {"silu": torch.nn.functional.silu},
                 "make_linear": _SKELETON.make_linear}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[cls], type_ignores=[])), str(MODEL), "exec"), namespace)
    mlp = namespace["DeepseekV3MLP"]
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for index in range(config.num_hidden_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(config.hidden_size, 4, bias=False)
        layer.self_attn.q_norm = nn.LayerNorm(4)
        if index < config.first_k_dense_replace:
            layer.mlp = mlp(config)
        else:
            layer.mlp = nn.Module()
            layer.mlp.shared_experts = mlp(config)
            layer.mlp.experts = nn.ModuleList([
                mlp(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ])
        model.model.layers.append(layer)
    return model


def _enumeration_method(factory):
    method = _method(SERVER, "DeepSeek_Parameter_Server", "_parse_state_dict")
    namespace = {
        "torch": torch, "gc": gc,
        "DeepseekV3ForCausalLM": factory,
        "trange": lambda count, **kwargs: range(count),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[])), str(SERVER), "exec"), namespace)
    return namespace["_parse_state_dict"]


@pytest.mark.parametrize("architecture", ["DeepseekV3ForCausalLM", "DeepseekV4ForCausalLM"])
def test_metadata_enumeration_matches_cpu_shapes_without_parameter_storage(architecture):
    config = SimpleNamespace(
        architectures=[architecture], hidden_size=8, intermediate_size=12,
        moe_intermediate_size=4, hidden_act="silu", num_hidden_layers=3,
        first_k_dense_replace=1, n_routed_experts=2,
    )
    reference = _small_model(config)
    expected_shapes = {name: tuple(param.shape)
                       for name, param in reference.named_parameters()}
    observed = {}

    class Factory:
        @staticmethod
        def _from_config(actual_config):
            assert actual_config is config
            model = _small_model(actual_config)
            observed.update({name: tuple(param.shape)
                             for name, param in model.named_parameters()})
            assert all(param.is_meta and param.data_ptr() == 0
                       for param in model.parameters())
            return model

    server = SimpleNamespace(
        hf_config=config, model_config=config,
        weight_copy_task={}, state_dict_name_map={},
    )
    _enumeration_method(Factory)(server)
    assert observed == expected_shapes

    expected = {}
    for name in expected_shapes:
        parts = name.split(".")
        index = int(parts[2])
        if parts[3] == "self_attn":
            module_key, tensor_key = f"attn_{index}", ".".join(parts[4:])
        elif parts[4] == "shared_experts":
            module_key, tensor_key = f"shared_expert_{index}", ".".join(parts[5:])
        elif parts[4] == "experts":
            module_key = f"routed_expert_{index}_{parts[5]}"
            tensor_key = ".".join(parts[6:])
        else:
            continue  # Dense layers remain in the skeleton state dict.
        expected[name] = {"module_key": module_key, "tensor_key": tensor_key}
    assert server.state_dict_name_map == expected
    assert server.weight_copy_task == {
        "attn": ["attn_0", "attn_1", "attn_2"],
        "shared_expert": ["shared_expert_1", "shared_expert_2"],
        "routed_expert": [f"routed_expert_{layer}_{expert}"
                          for layer in (1, 2) for expert in (0, 1)],
    }
    assert torch.empty(0).device.type == "cpu"


def test_metadata_constructor_failure_restores_default_device():
    class Factory:
        @staticmethod
        def _from_config(config):
            assert torch.empty(0).is_meta
            raise RuntimeError("constructor failure")

    server = SimpleNamespace(
        hf_config=SimpleNamespace(),
        model_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
    )
    with pytest.raises(RuntimeError, match="constructor failure"):
        _enumeration_method(Factory)(server)
    assert torch.empty(0).device.type == "cpu"
