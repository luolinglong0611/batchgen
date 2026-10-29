"""CPU dispatch/lifetime contracts; native FP8/CUDA equality is separate."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native(monkeypatch):
    name = "batchgen.gemm.r1_native_gemm"
    spec = importlib.util.spec_from_file_location(name, ROOT / "batchgen/gemm/r1_native_gemm.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    import batchgen.gemm
    monkeypatch.setattr(batchgen.gemm, "r1_native_gemm", module, raising=False)
    monkeypatch.setenv(module.ENV, "deep_gemm")
    return module


@pytest.fixture
def math(native, monkeypatch):
    """Real CPU tensors; accelerator entry points are explicit test doubles."""
    calls = {"quant": [], "gemm": []}

    def quant(x):
        calls["quant"].append(x.clone())
        return x.to(torch.float8_e4m3fn), torch.zeros(x.shape[0], x.shape[1] // 512, dtype=torch.int32)

    def gemm(lhs, rhs, out):
        calls["gemm"].append((lhs[0], lhs[1], rhs[0]))
        out.copy_(lhs[0].float() @ rhs[0].float().T)

    monkeypatch.setattr(native, "quantize", quant)
    monkeypatch.setattr(native, "_runtime", lambda: (SimpleNamespace(fp8_gemm_nt=gemm), None, None))
    banks = []

    def bank(rows, cols=512):
        values = ((torch.arange(rows * cols).view(rows, cols) % 7) - 3).float() / 32
        weight = values.to(torch.float8_e4m3fn)
        scale = torch.ones((rows + 127) // 128, cols // 128)
        entry = native.PreparedWeight(weight, scale, native._key(weight, scale),
            (weight._version, scale._version), weight.clone(), torch.zeros(rows, cols // 512, dtype=torch.int32))
        native._LOOKUP[entry.source_key] = entry
        banks.append(entry)
        return entry

    return calls, bank


def function(path, name, globals_, *, until=None):
    node = next(n for n in ast.walk(ast.parse((ROOT / path).read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    if until is not None:
        cut = next(i for i, stmt in enumerate(node.body) if until(stmt))
        node.body = node.body[:cut] + [ast.parse("return query_states, compressed_kv").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), globals_)
    return globals_[name]


def test_selector_default_native_only_and_invalid_value(native, monkeypatch):
    monkeypatch.delenv(native.SHARED_INPUT_QUANT_ENV, raising=False)
    assert not native.shared_input_quant_reuse_enabled()
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "1")
    assert native.shared_input_quant_reuse_enabled()
    monkeypatch.setenv(native.ENV, "legacy")
    assert not native.shared_input_quant_reuse_enabled()
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native.shared_input_quant_reuse_enabled()


@pytest.mark.parametrize("shape", [(0, 512), (1, 512), (180, 512), (3, 1, 512)])
def test_pair_reuses_one_activation_and_preserves_both_outputs(native, math, shape):
    calls, bank = math
    a, b = bank(16), bank(32)
    x = torch.full(shape, .125, dtype=torch.bfloat16)
    expected = (native.linear(a.source, a.source_scale, x), native.linear(b.source, b.source_scale, x))
    calls["quant"].clear(); calls["gemm"].clear()
    actual = native.linear_pair(a.source, a.source_scale, b.source, b.source_scale, x)
    assert all(torch.equal(old, new) for old, new in zip(expected, actual))
    assert len(calls["quant"]) == 1
    if x.numel():
        assert len(calls["gemm"]) == 2
        assert calls["gemm"][0][0] is calls["gemm"][1][0]
        assert calls["gemm"][0][1] is calls["gemm"][1][1]


def test_each_pair_consumes_current_input_without_cross_call_cache(native, math):
    calls, bank = math
    a, b = bank(16), bank(32)
    x = torch.full((2, 512), .125, dtype=torch.bfloat16)
    first = native.linear_pair(a.source, a.source_scale, b.source, b.source_scale, x)
    x.mul_(2)
    second = native.linear_pair(a.source, a.source_scale, b.source, b.source_scale, x)
    assert len(calls["quant"]) == 2
    assert all(torch.equal(after, before * 2) for before, after in zip(first, second))


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("mutation", ["inplace", "rebind", "scale"])
def test_pair_retains_each_weight_source_guard(native, math, which, mutation):
    _, bank = math
    a, b = bank(16), bank(32)
    target = (a, b)[which]
    if mutation == "inplace":
        target.source.copy_(torch.zeros_like(target.source))
    elif mutation == "rebind":
        target.source.data = torch.zeros_like(target.source)
    else:
        target.source_scale.add_(1)
    with pytest.raises(RuntimeError, match="absent/stale"):
        native.linear_pair(a.source, a.source_scale, b.source, b.source_scale,
                           torch.zeros(2, 512, dtype=torch.bfloat16))


@pytest.mark.parametrize("shape", [(0, 512), (3, 512), (3, 1, 512)])
def test_actual_shared_mlp_body_preserves_three_gemms_with_one_less_quant(native, math, monkeypatch, shape):
    calls, bank = math
    up, gate, down = bank(512), bank(512), bank(16)
    owner = SimpleNamespace(up_proj=SimpleNamespace(weight=up.source), gate_proj=SimpleNamespace(weight=gate.source),
        down_proj=SimpleNamespace(weight=down.source), act_fn=torch.nn.functional.silu)
    scales = {name + ".weight_scale_inv": entry.source_scale for name, entry in
              (("up_proj", up), ("gate_proj", gate), ("down_proj", down))}
    fn = function("batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py", "deepgemm_forward",
                  {"torch": torch, "w8a16_gemm": native.linear})
    x = torch.full(shape, .125, dtype=torch.bfloat16)
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "0")
    expected = fn(owner, x, scales)
    assert len(calls["quant"]) == 3
    calls["quant"].clear(); calls["gemm"].clear()
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "1")
    actual = fn(owner, x, scales)
    assert torch.equal(expected, actual)
    assert len(calls["quant"]) == 2
    assert len(calls["gemm"]) == (3 if x.numel() else 0)


@pytest.mark.parametrize("dequant,native_backend,expected_quants", [(False, True, 2), (True, True, 3), (False, False, 3)])
def test_actual_prefill_projection_body_reuses_qkv_input_and_keeps_fallbacks(
        native, math, monkeypatch, dequant, native_backend, expected_quants):
    calls, bank = math
    qa, qb, kva = bank(512), bank(384), bank(576)
    owner = SimpleNamespace(q_a_proj=SimpleNamespace(weight=qa.source), q_b_proj=SimpleNamespace(weight=qb.source),
        kv_a_proj_with_mqa=SimpleNamespace(weight=kva.source), q_a_layernorm=lambda x: x,
        num_heads=2, q_head_dim=192, qk_nope_head_dim=128, qk_rope_head_dim=64)
    scales = {name + ".weight_scale_inv": entry.source_scale for name, entry in
              (("q_a_proj", qa), ("q_b_proj", qb), ("kv_a_proj_with_mqa", kva))}
    until = lambda stmt: (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Tuple)
                         and [n.id for n in stmt.targets[0].elts] == ["compressed_kv", "k_pe"])
    fn = function("batchgen/attention/mla/fa3_backend.py", "mla_prefill_flashattention3_w8a16_deepgemm_prepacked",
        {"torch": torch, "w8a16_gemm": native.linear, "w8a16_gemm_dequant": native.linear}, until=until)
    x = torch.full((3, 512), .125, dtype=torch.bfloat16)
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "0")
    expected = fn(owner, x, torch.arange(3), torch.tensor([0, 3]), 3, 1, scales)
    calls["quant"].clear(); calls["gemm"].clear()
    monkeypatch.setenv(native.SHARED_INPUT_QUANT_ENV, "1")
    monkeypatch.setenv(native.ENV, "deep_gemm" if native_backend else "legacy")
    monkeypatch.setenv("BATCHGEN_W8A16_DEQUANT", "1" if dequant else "0")
    actual = fn(owner, x, torch.arange(3), torch.tensor([0, 3]), 3, 1, scales)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert len(calls["quant"]) == expected_quants
    assert len(calls["gemm"]) == 3
