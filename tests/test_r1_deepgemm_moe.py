"""CPU contracts only; real DeepGEMM/DeepEP/Graph correctness needs CUDA tests."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "batchgen/models/deepseek/deepseekv3/deepgemm_moe.py"
    spec = importlib.util.spec_from_file_location("r1_native_deepgemm_cpu", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    monkeypatch.setattr(result, "_capturing", lambda: False)
    return result


class FakeDG:
    def __init__(self):
        self.calls = []

    def get_mn_major_tma_aligned_packed_ue8m0_tensor(self, scale):
        exp = ((scale.contiguous().view(torch.int32) >> 23) & 255).to(torch.uint8)
        packed = exp.contiguous().view(torch.int32)
        return packed.transpose(-1, -2).contiguous().transpose(-1, -2)

    def fp8_m_grouped_gemm_nt_masked(self, lhs, rhs, out, counts, expected_m):
        self.calls.append(("masked", lhs, rhs, out, counts.clone(), expected_m))
        for e, count in enumerate(counts.tolist()):
            out[e, :count].fill_(2)

    def m_grouped_fp8_gemm_nt_contiguous(self, lhs, rhs, out, indices):
        self.calls.append(("contiguous", lhs, rhs, out, indices.clone()))
        out[indices >= 0] = 2


def runtime():
    def masked(x, y, scale, group, counts, *, scale_ue8m0):
        assert group == 128 and scale_ue8m0
        for e, count in enumerate(counts.tolist()):
            y[e, :count] = 1
            scale[e, :count] = 0.5

    def silu(x, y):
        y.fill_(1)

    def quantize(x, group, **kwargs):
        assert group == 128
        assert kwargs == dict(column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
        return x.to(torch.float8_e4m3fn), torch.full((*x.shape[:-1], x.shape[-1] // 128), 0.5)

    return SimpleNamespace(dg=FakeDG(), silu_masked=masked, silu_and_mul=silu, quantize=quantize)


def math(module, e=2, h=512, n=512):
    bank = module.R1DeepGemmWeightBank(
        torch.ones((e, 2 * n, h), dtype=torch.float8_e4m3fn),
        torch.zeros((e, h // 512, 2 * n), dtype=torch.int32).transpose(1, 2),
        torch.ones((e, h, n), dtype=torch.float8_e4m3fn),
        torch.zeros((e, n // 512, h), dtype=torch.int32).transpose(1, 2))
    return module.R1DeepGemmExpertMath(bank, runtime())


def sources(e=1, n=512, h=512):
    return [dict(gate=torch.ones((n, h), dtype=torch.float8_e4m3fn),
                 up=torch.full((n, h), 3, dtype=torch.float8_e4m3fn),
                 down=torch.full((h, n), 2, dtype=torch.float8_e4m3fn),
                 gate_scale=torch.full((n // 128, h // 128), 0.5),
                 up_scale=torch.full((n // 128, h // 128), 0.5),
                 down_scale=torch.full((h // 128, n // 128), 0.5)) for _ in range(e)]


@pytest.mark.parametrize("inference", [False, True])
def test_host_bank_conversion_gate_order_scale_layout_and_sources_unchanged(module, inference):
    original = sources()
    snapshots = {k: v.float().clone() for k, v in original[0].items()}
    with torch.inference_mode(inference):
        result = module.R1DeepGemmExpertMath.from_expert_sources(
            iter(original), device="cpu", source_identity=("host", 3, 0), runtime=runtime())
    assert result.bank is result.weights
    assert result.source_identity == ("host", 3, 0)
    # 0.5 -> scale 1/512 and FP8=256; 1.5 -> scale 1/256 and FP8=384.
    assert torch.all(result.bank.w13[0, :512].float() == 256)
    assert torch.all(result.bank.w13[0, 512:].float() == 384)
    assert torch.all(result.bank.w2.float() == 256)
    assert result.bank.w13_scale.stride(-2) == result.bank.w2_scale.stride(-2) == 1
    assert result.resident_bytes == sum(t.numel() * t.element_size() for t in result.bank.tensors())
    assert result.build_stats["temporary_upload_max_experts"] == 1
    assert result.build_stats["host_sources_modified"] is False
    assert all(isinstance(t._version, int) for t in result.graph_weight_tensors())
    for key, tensor in original[0].items():
        assert torch.equal(tensor.float(), snapshots[key])


def test_original_scale_shape_and_fp8_dtype_fail_before_conversion(module):
    bad = sources()
    bad[0]["gate_scale"] = torch.ones(4, 3)
    with pytest.raises(ValueError, match="gate_scale"):
        module.R1DeepGemmExpertMath.from_expert_sources(bad, device="cpu", runtime=runtime())
    bad = sources()
    bad[0]["up"] = bad[0]["up"].to(torch.bfloat16)
    with pytest.raises(ValueError, match="up"):
        module.R1DeepGemmExpertMath.from_expert_sources(bad, device="cpu", runtime=runtime())


def test_repacking_is_rejected_during_capture(module, monkeypatch):
    monkeypatch.setattr(module, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="convert expert weights"):
        module.R1DeepGemmExpertMath.from_expert_sources([], device="cpu", runtime=runtime())


def workspace(module, e=2, m=8, h=512, n=512):
    return module.R1DeepGemmMaskedWorkspace(
        torch.full((e, m, 2 * n), -7, dtype=torch.bfloat16),
        torch.full((e, m, n), -7, dtype=torch.float8_e4m3fn),
        torch.full((e, m, n // 128), -7, dtype=torch.float32))


def test_masked_math_uses_two_native_grouped_calls_and_preserves_padding(module):
    expert = math(module)
    counts = torch.tensor([2, 0], dtype=torch.int32)
    work = workspace(module)
    output = torch.full((2, 8, 512), -7, dtype=torch.bfloat16)
    x = torch.ones((2, 8, 512), dtype=torch.float8_e4m3fn)
    scale = torch.ones((2, 8, 4), dtype=torch.float32)
    assert expert.masked(x, scale, counts, 3, workspace=work, out=output) is output
    assert [x[0] for x in expert.runtime.dg.calls] == ["masked", "masked"]
    assert all(x[1][1].dtype == torch.int32 for x in expert.runtime.dg.calls)
    assert torch.all(output[0, :2] == 2)
    assert torch.all(output[0, 2:] == -7) and torch.all(output[1] == -7)
    assert torch.all(work.down_fp8[0, 2:].float() == -7)


def test_contiguous_math_preserves_indices_and_reuses_same_weight_bank(module):
    expert = math(module)
    x = torch.ones((4, 512), dtype=torch.float8_e4m3fn)
    scale = torch.ones((4, 4), dtype=torch.float32)
    indices = torch.tensor([0, 0, 1, -1], dtype=torch.int32)
    out = torch.full((4, 512), -7, dtype=torch.bfloat16)
    assert expert.contiguous(x, scale, indices, out=out) is out
    first, second = expert.runtime.dg.calls
    assert first[0] == second[0] == "contiguous"
    assert first[2][0] is expert.bank.w13 and second[2][0] is expert.bank.w2
    assert torch.equal(first[4], indices) and torch.equal(second[4], indices)
    assert torch.all(out[:3] == 2) and torch.all(out[3] == -7)


def test_empty_contiguous_math_is_noop_but_not_empty_ep_collective(module):
    expert = math(module)
    out = expert.contiguous(torch.empty((0, 512), dtype=torch.float8_e4m3fn),
                            torch.empty((0, 4)), torch.empty(0, dtype=torch.int32))
    assert out.shape == (0, 512) and expert.runtime.dg.calls == []


def pool():
    p = SimpleNamespace(capacity=4, hidden=512, intermediate=512, world_size=8,
                        num_experts=16, num_local_experts=2, topk=2, stride=32,
                        external_valid_rows=False)
    p.x = torch.zeros((4, 512), dtype=torch.bfloat16)
    p.topk_idx = torch.full((4, 2), -1, dtype=torch.int64)
    p.topk_weights = torch.zeros((4, 2))
    p.valid_rows = torch.tensor([4], dtype=torch.int32)
    p.row_ids = torch.arange(4, dtype=torch.int32)
    p.x_fp8 = torch.empty((2, 32, 512), dtype=torch.float8_e4m3fn)
    p.x_scale = torch.empty((2, 32, 4))
    p.intermediate_fp8 = torch.empty((2, 32, 512), dtype=torch.float8_e4m3fn)
    p.intermediate_scale = torch.empty((2, 32, 4))
    p.expert_out = torch.empty((2, 32, 512), dtype=torch.bfloat16)
    p.local_out = torch.empty_like(p.x)
    p.layer_counters = torch.zeros((61, 4), dtype=torch.int64)
    p.set_valid_rows = lambda rows: p.valid_rows.fill_(rows)
    p.dispatches = p.combines = 0

    def dispatch(x, idx, cap, e, **kwargs):
        p.dispatches += 1
        p.dispatch_kwargs = kwargs
        return ((torch.ones((2, 32, 512), dtype=torch.float8_e4m3fn),
                 torch.zeros((2, 32, 1), dtype=torch.int32)),
                torch.tensor([2, 3], dtype=torch.int32), "handle", None, None)

    def combine(experts, idx, weights, handle, *, out):
        p.combines += 1
        assert handle == "handle"
        out.copy_(weights.sum(1, keepdim=True).expand_as(out))
        return out

    p.exchange = SimpleNamespace(buffer=SimpleNamespace(low_latency_dispatch=dispatch), combine=combine)
    return p


def adapter(module, p):
    gate = SimpleNamespace(moe_gate_forward_hybrid=lambda x: (
        torch.tensor([[0, 1]], dtype=torch.int64).repeat(x.shape[0], 1),
        torch.tensor([[0.25, 0.75]]).repeat(x.shape[0], 1)))
    return module.R1DeepGemmDecodeAdapter(SimpleNamespace(gate=gate), p, layer_idx=3,
                                          expert_math=math(module))


def test_zero_rank_still_native_fp8_dispatch_math_and_combine(module):
    p = pool()
    a = adapter(module, p)
    out = a(torch.empty((0, 512), dtype=torch.bfloat16))
    assert out.shape == (0, 512) and p.dispatches == p.combines == 1
    assert p.dispatch_kwargs == dict(use_fp8=True, round_scale=True, use_ue8m0=True,
                                      async_finish=False, return_recv_hook=False)
    assert torch.all(p.topk_idx == -1) and torch.all(p.topk_weights == 0)
    assert p.layer_counters[3].tolist() == [1, 1, 0, 5]
    assert len(a.expert_math.runtime.dg.calls) == 2


def test_graph_live_row_mask_and_persistent_pool_workspace(module, monkeypatch):
    p = pool()
    a = adapter(module, p)
    assert adapter(module, p).workspace is a.workspace
    assert a.graph_weight_tensors() == a.expert_math.bank.tensors()
    monkeypatch.setattr(module, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="external valid_rows"):
        a(torch.ones((4, 512), dtype=torch.bfloat16))
    p.external_valid_rows = True
    p.valid_rows.fill_(1)
    a(torch.ones((4, 512), dtype=torch.bfloat16))
    assert torch.all(p.topk_idx[1:] == -1) and torch.all(p.local_out[1:] == 0)
    assert p.layer_counters[3].tolist() == [1, 1, 1, 5]


def test_local_jit_warmup_has_no_ep_collective_and_uses_actual_row_hint(module):
    p = pool()
    a = adapter(module, p)
    a.warmup_local_only(rows=1)
    assert p.dispatches == p.combines == 0
    assert [x[-1] for x in a.expert_math.runtime.dg.calls] == [2, 2]
    assert a._expected_m(0) == 1


def test_malformed_local_rows_do_not_enter_ep_collective(module):
    p = pool()
    a = adapter(module, p)
    with pytest.raises(ValueError, match="BF16"):
        a(torch.ones((5, 512), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16"):
        a(torch.ones((1, 512), dtype=torch.float32))
    assert p.dispatches == p.combines == 0
