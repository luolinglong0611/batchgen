"""CPU lifetime/dispatch contracts; actual FP8 math is qualified on CUDA."""
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "batchgen/gemm/r1_native_gemm.py"
    spec = importlib.util.spec_from_file_location("native_ordinary_cpu", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result


def bank(module):
    weight = torch.ones(16, 512, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 4)
    native = weight.clone()
    packed = torch.zeros(16, 1, dtype=torch.int32)
    entry = module.PreparedWeight(weight, scale, module._key(weight, scale),
                                  (weight._version, scale._version), native, packed)
    module._LOOKUP[entry.source_key] = entry
    return entry


def test_forward_reuses_prepared_bank_without_quantizing_weights(module, monkeypatch):
    entry = bank(module)
    source_before, scale_before = entry.source.float().clone(), entry.source_scale.clone()
    calls = []

    def native(lhs, rhs, output):
        calls.append((lhs, rhs))
        assert rhs[0] is entry.weight and rhs[1] is entry.scale
        output.copy_(lhs[0].float() @ rhs[0].float().T)

    monkeypatch.setattr(module, "_runtime", lambda: (SimpleNamespace(fp8_gemm_nt=native), None, None))
    monkeypatch.setattr(module, "prepare_weight", lambda *a: pytest.fail("Weight conversion in forward"))
    x = torch.ones(3, 512, dtype=torch.float8_e4m3fn)
    packed = torch.zeros(3, 1, dtype=torch.int32)
    for _ in range(2):
        output = module.gemm(x, packed, entry.source, entry.source_scale)
        assert output.shape == (3, 16) and output.dtype == torch.bfloat16
        assert torch.all(output == 512)
    assert len(calls) == 2
    assert torch.equal(entry.source.float(), source_before)
    assert torch.equal(entry.source_scale, scale_before)


@pytest.mark.parametrize("mutation", ["weight_inplace", "weight_storage", "scale_inplace"])
def test_source_mutation_refuses_stale_native_bank_before_dispatch(module, monkeypatch, mutation):
    entry = bank(module)
    monkeypatch.setattr(module, "_runtime", lambda: pytest.fail("Native GEMM dispatched stale weights"))
    if mutation == "weight_inplace":
        entry.source.copy_(torch.full_like(entry.source, 2))
    elif mutation == "weight_storage":
        entry.source.data = torch.full_like(entry.source, 2)
    else:
        entry.source_scale.add_(1)
    with pytest.raises(RuntimeError, match="absent/stale"):
        module.gemm(torch.ones(2, 512, dtype=torch.float8_e4m3fn),
                    torch.zeros(2, 1, dtype=torch.int32), entry.source, entry.source_scale)


def test_module_bank_is_owner_and_global_lookup_does_not_leak_it(module):
    entry = bank(module)
    owner = SimpleNamespace(_r1_native_gemm_weights=[entry])
    source, key = entry.source, entry.source_key
    reference = weakref.ref(entry)
    del entry
    gc.collect()
    assert reference() is not None and key in module._LOOKUP
    assert len(module.graph_tensors(owner)) == 2
    del owner
    gc.collect()
    assert reference() is None and key not in module._LOOKUP
    assert source.shape == (16, 512)


def test_empty_forward_returns_without_native_kernel(module, monkeypatch):
    entry = bank(module)
    monkeypatch.setattr(module, "_runtime", lambda: pytest.fail("Empty GEMM dispatched"))
    output = module.gemm(torch.empty(0, 512, dtype=torch.float8_e4m3fn),
                         torch.empty(0, 1, dtype=torch.int32), entry.source, entry.source_scale)
    assert output.shape == (0, 16) and output.dtype == torch.bfloat16


def test_selector_keeps_legacy_default_and_rejects_typo(module, monkeypatch):
    monkeypatch.delenv(module.ENV, raising=False)
    assert not module.enabled()
    monkeypatch.setenv(module.ENV, "deep_gemm")
    assert module.enabled()
    monkeypatch.setenv(module.ENV, "deepgemm")
    with pytest.raises(ValueError):
        module.enabled()
