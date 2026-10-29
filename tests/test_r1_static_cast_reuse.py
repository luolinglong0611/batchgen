"""Real CPU tensors qualify conversion parity and cache invalidation/lifetime."""
import gc
import importlib.util
from pathlib import Path
import weakref

import pytest
import torch


SPEC = importlib.util.spec_from_file_location("_r1_static_cast_contract", Path(__file__).resolve().parents[1] / "batchgen/gemm/r1_static_cast.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_router_cast_is_identical_and_reused(monkeypatch):
    monkeypatch.setenv(m.ENV, "1")
    w = torch.randn(256, 32).to(torch.bfloat16)
    a, b = m.router_weight_fp32(w), m.router_weight_fp32(w)
    assert a is b and torch.equal(a, w.float())
    x = torch.randn(7, 32).to(torch.bfloat16)
    assert torch.equal(x.float() @ a.T, x.float() @ w.float().T)


def test_source_write_invalidates_conversion():
    w = torch.ones(16, dtype=torch.bfloat16)
    old = m.cached_cast(w, torch.float32)
    w.add_(1)
    new = m.cached_cast(w, torch.float32)
    assert new is not old and torch.equal(new, w.float())


def test_source_storage_rebind_invalidates_conversion():
    w = torch.ones(16, dtype=torch.bfloat16)
    old = m.cached_cast(w, torch.float32)
    w.data = torch.full_like(w, 3)
    new = m.cached_cast(w, torch.float32)
    assert new is not old and torch.equal(new, w.float())


def test_cached_result_mutation_does_not_poison_subsequent_calls():
    w = torch.ones(16, dtype=torch.bfloat16)
    old = m.cached_cast(w, torch.float32)
    old.add_(9)
    new = m.cached_cast(w, torch.float32)
    assert new is not old and torch.equal(new, w.float())


def test_cache_does_not_retain_original_after_source_lifetime():
    w = torch.ones(16, dtype=torch.bfloat16)
    value = m.cached_cast(w, torch.float32)
    wr, vr = weakref.ref(w), weakref.ref(value)
    del w, value
    gc.collect()
    assert wr() is None and vr() is None


def test_rotary_slices_share_full_cast_and_preserve_rounding(monkeypatch):
    monkeypatch.setenv(m.ENV, "1")
    cos, sin = torch.randn(128, 64), torch.randn(128, 64)
    x = torch.empty(1, dtype=torch.bfloat16)
    a, b = m.rotary_values(cos, sin, x, 8)
    c, d = m.rotary_values(cos, sin, x, 127)
    assert a.data_ptr() == c.data_ptr() and b.data_ptr() == d.data_ptr()
    assert torch.equal(a, cos[:8].bfloat16()) and torch.equal(d, sin[:127].bfloat16())
    assert torch.equal(c[:8], a)


def test_shared_source_users_share_converted_storage(monkeypatch):
    monkeypatch.setenv(m.ENV, "1")
    cos, sin = torch.randn(128, 64), torch.randn(128, 64)
    users = [(cos, sin)] * 61
    values = [m.rotary_values(a, b, torch.empty(1, dtype=torch.bfloat16), 64) for a, b in users]
    assert len({a.data_ptr() for a, _ in values}) == len({b.data_ptr() for _, b in values}) == 1


def test_rotary_replacement_and_source_write_refresh(monkeypatch):
    monkeypatch.setenv(m.ENV, "1")
    cos, sin = torch.ones(8, 64), torch.zeros(8, 64)
    x = torch.empty(1, dtype=torch.bfloat16)
    a, _ = m.rotary_values(cos, sin, x, 8)
    cos.mul_(.5)
    b, _ = m.rotary_values(cos, sin, x, 8)
    c, _ = m.rotary_values(torch.zeros_like(cos), sin, x, 8)
    assert b.data_ptr() != a.data_ptr() and torch.equal(b, cos.bfloat16())
    assert c.data_ptr() != b.data_ptr() and not c.any()


def test_unversioned_router_rejected_but_readonly_rotary_supported(monkeypatch):
    monkeypatch.setenv(m.ENV, "1")
    with torch.inference_mode():
        w = torch.ones(8, dtype=torch.bfloat16)
        cos, sin = torch.ones(8, 64), torch.zeros(8, 64)
    with pytest.raises(RuntimeError, match="versioned"):
        m.router_weight_fp32(w)
    assert torch.equal(m.rotary_values(cos, sin, w, 8)[0], cos.bfloat16())


def test_no_gradient_cache_and_default_disabled(monkeypatch):
    monkeypatch.delenv(m.ENV, raising=False)
    w = torch.ones(16, dtype=torch.bfloat16, requires_grad=True)
    assert m.router_weight_fp32(w).requires_grad
    monkeypatch.setenv(m.ENV, "1")
    with pytest.raises(RuntimeError, match="inference-only"):
        m.router_weight_fp32(w)
    with torch.no_grad():
        assert torch.equal(m.router_weight_fp32(w), w.float())


def test_invalid_flag_or_context_rejected(monkeypatch):
    monkeypatch.setenv(m.ENV, "true")
    with pytest.raises(ValueError, match="0 or 1"):
        m.enabled()
    monkeypatch.setenv(m.ENV, "1")
    with pytest.raises(ValueError, match="context"):
        m.rotary_values(torch.ones(4, 64), torch.ones(4, 64), torch.empty(1), 5)


def test_noop_cast_returns_source():
    x = torch.ones(4)
    assert m.cached_cast(x, x.dtype) is x
