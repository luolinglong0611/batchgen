"""CPU contract tests only. Fake LL/GEMM; these never qualify GPU correctness."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


class FakeExchange:
    def __init__(self, *, max_tokens_per_rank, hidden, num_experts, num_local_experts):
        self.cap, self.h = max_tokens_per_rank, hidden
        self.e, self.stride = num_local_experts, 8 * max_tokens_per_rank
        self.dispatch_calls = self.combine_calls = 0

    def dispatch(self, x, idx):
        self.dispatch_calls += 1
        self.last_idx = idx.clone()
        # Simulate remote tokens even when this rank has no live local rows.
        return (torch.ones((self.e, self.stride, self.h), dtype=torch.bfloat16),
                torch.full((self.e,), 2, dtype=torch.int32), self.dispatch_calls)

    def combine(self, x, idx, weights, handle, *, out):
        self.combine_calls += 1
        self.last_weights = weights.clone()
        out.copy_(((idx >= 0) * weights).sum(1)[:, None] * 2.0)
        return out


@pytest.fixture
def module(monkeypatch):
    api = types.ModuleType("batchgen.moe.deepep_ll")
    api.get_low_latency_exchange = lambda group, **kw: FakeExchange(**kw)
    api.requests = []
    api.require_deepep = lambda **kw: api.requests.append(kw)
    monkeypatch.setitem(sys.modules, "batchgen.moe.deepep_ll", api)
    p = Path(__file__).resolve().parents[1] / "batchgen/models/deepseek/deepseekv3/deepep_decode.py"
    spec = importlib.util.spec_from_file_location("r1_deepep_candidate", p)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.fake_api = api
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("NVSHMEM_QP_DEPTH", raising=False)
    return result


def pool(module, **kw):
    return module.R1DeepEPDecodePool(
        types.SimpleNamespace(size=lambda: 8),
        max_tokens_per_rank=kw.get("cap", 4), hidden=128, intermediate=128,
        num_experts=16, topk=2, device="cpu")


def adapter(module, p):
    # Avoid importing production model/Triton; test routing/lifetime contract.
    result = module.R1DeepEPDecodeAdapter.__new__(module.R1DeepEPDecodeAdapter)
    result.pool = p
    result.layer_idx = 3
    result.moe = types.SimpleNamespace(
        gate=types.SimpleNamespace(moe_gate_forward_hybrid=lambda x: (
            torch.tensor([[0, 1]]).repeat(x.shape[0], 1),
            torch.tensor([[0.25, 0.75]]).repeat(x.shape[0], 1))),
        gate_list=[], up_list=[], down_list=[], gate_ptrs_ptr=None,
        up_ptrs_ptr=None, down_ptrs_ptr=None, gate_scale_list=[],
        up_scale_list=[], down_scale_list=[], gate_scale_ptrs_ptr=None,
        up_scale_ptrs_ptr=None, down_scale_ptrs_ptr=None)
    result.quantize_into = lambda x, counts, output, scales: (
        output.copy_(x), scales.fill_(1))
    result.s1 = lambda *args, out: out.fill_(1)
    result.s3 = lambda *args, out: out.fill_(2)
    return result


def test_bf16_transport_and_fixed_capacity_are_explicit(module):
    p = pool(module)
    assert module.fake_api.requests[-1]["use_fp8"] is False
    assert p.x.dtype == torch.bfloat16
    assert p.expert_out.shape == (2, 32, 128)
    assert p.x_fp8.dtype == torch.float8_e4m3fn


def test_dynamic_rows_mask_padding_and_reuse_buffers(module):
    p = pool(module)
    a = adapter(module, p)
    ptr = p.local_out.data_ptr()
    assert torch.all(a(torch.ones(4, 128, dtype=torch.bfloat16)) == 2)
    out = a(torch.ones(1, 128, dtype=torch.bfloat16))
    assert out.shape == (1, 128) and out.data_ptr() == ptr
    assert torch.all(p.exchange.last_idx[1:] == -1)
    assert torch.all(p.exchange.last_weights[1:] == 0)
    assert torch.all(p.local_out[1:] == 0)


def test_zero_local_rank_still_dispatches_and_combines(module):
    p = pool(module)
    out = adapter(module, p)(torch.empty(0, 128, dtype=torch.bfloat16))
    assert out.shape == (0, 128)
    assert p.exchange.dispatch_calls == p.exchange.combine_calls == 1
    assert torch.all(p.exchange.last_idx == -1)
    assert p.layer_counters[3].tolist() == [1, 1, 0, 4]


def test_graph_requires_external_row_metadata(module, monkeypatch):
    p = pool(module)
    a = adapter(module, p)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="external valid_rows"):
        a(torch.ones(4, 128, dtype=torch.bfloat16))
    p.external_valid_rows = True
    p.set_valid_rows(2)
    a(torch.ones(4, 128, dtype=torch.bfloat16))
    assert p.valid_rows.item() == 2
    assert torch.all(p.exchange.last_idx[2:] == -1)
    assert p.layer_counters[3].tolist() == [1, 1, 2, 4]


def test_invalid_shapes_fail_before_ll(module):
    p = pool(module)
    a = adapter(module, p)
    with pytest.raises(ValueError, match="BF16"):
        a(torch.ones(1, 128, dtype=torch.float32))
    with pytest.raises(ValueError, match="BF16"):
        a(torch.ones(5, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="fixed LL capacity"):
        p.set_valid_rows(5)
    assert p.exchange.dispatch_calls == 0


def test_qp_depth_checked_from_real_environment(module, monkeypatch):
    monkeypatch.setenv("NVSHMEM_QP_DEPTH", "8")
    with pytest.raises(RuntimeError, match="NVSHMEM_QP_DEPTH"):
        pool(module)


def test_dependency_error_is_not_a_fallback(module, monkeypatch):
    def fail(**kw):
        raise RuntimeError("missing DeepEP LL")
    monkeypatch.setattr(module, "require_deepep", fail)
    with pytest.raises(RuntimeError, match="missing DeepEP LL"):
        pool(module)


def test_all_layer_counters_and_phase_lifetime_are_observable(module):
    p = pool(module)
    p.begin_phase()
    a = adapter(module, p)
    a(torch.ones(2, 128, dtype=torch.bfloat16))
    a(torch.ones(1, 128, dtype=torch.bfloat16))
    for layer_idx in range(4, 61):
        a.layer_idx = layer_idx
        a(torch.ones(1, 128, dtype=torch.bfloat16))
    p.external_valid_rows = True
    p.begin_phase()
    assert p.external_valid_rows is False
    snapshot = p.trace_snapshot()
    assert snapshot["phase_configurations"] == 2
    assert snapshot["layers"][3] == [2, 2, 3, 8]
    assert all(v == [1, 1, 1, 4] for v in snapshot["layers"][4:])
    assert all(v == [0, 0, 0, 0] for v in snapshot["layers"][:3])
    assert len(snapshot["layers"]) == 61
