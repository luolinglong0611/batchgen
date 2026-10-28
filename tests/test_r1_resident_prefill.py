"""CPU contracts for normal EP routing/lifecycle; not GPU/math qualification."""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = "batchgen.models.deepseek.deepseekv3"
PSM = ROOT / "batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py"
spec = importlib.util.spec_from_file_location(PKG + ".resident_prefill", PSM.with_name("resident_prefill.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("BATCHGEN_R1_MOE_BACKEND", "BATCHGEN_R1_RESIDENT_EP_PREFILL", "BATCHGEN_R1_PREFILL_TOKENS_PER_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, PKG + ".resident_prefill", rp)


def methods(*names, **extras):
    cls = next(n for n in ast.parse(PSM.read_text()).body if isinstance(n, ast.ClassDef))
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=selected, type_ignores=[])
    ns = {"__package__": PKG, "logging": logging, "torch": torch, "os": os, "time": time, **extras}
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), ns)
    result = SimpleNamespace()
    for n in names:
        setattr(result, n, types.MethodType(ns[n], result))
    return result


def test_opt_in_keeps_legacy_and_rejects_incomplete_native_request(monkeypatch):
    assert rp.moe_backend() == "legacy" and not rp.resident_prefill_requested()
    monkeypatch.setenv("BATCHGEN_R1_RESIDENT_EP_PREFILL", "1")
    with pytest.raises(ValueError, match="deep_gemm"):
        rp.resident_prefill_requested()
    monkeypatch.setenv("BATCHGEN_R1_MOE_BACKEND", "deep_gemm")
    assert rp.resident_prefill_requested() and rp.resident_prefill_token_cap() == 8192
    for value in ("0", "127", "8193", "-1", "NaN", "262144"):
        monkeypatch.setenv("BATCHGEN_R1_PREFILL_TOKENS_PER_RANK", value)
        with pytest.raises(ValueError):
            rp.resident_prefill_token_cap()


def test_all_ep8_expert_spans_cover_once():
    ranges = [list(rp.local_expert_ids(rank, 8)) for rank in range(8)]
    assert all(len(r) == 32 for r in ranges)
    assert [v for r in ranges for v in r] == list(range(256))
    with pytest.raises(ValueError):
        rp.local_expert_ids(0, 16)


def test_planner_native_prefill_cap_and_no_weight_stream_buffers(monkeypatch):
    import re
    path = PSM.with_name("planner.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_adjust_config_for_model")
    ns = {"__package__": PKG, "os": os, "re": re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), ns)
    config = SimpleNamespace(Module_Batching_Config=SimpleNamespace(r1_deepep_ll=True),
        Basic_Config=SimpleNamespace(attn_mode=1),
        EP_Config=SimpleNamespace(enable_offloading=False, offloading_ratio=0), GPU_Buffer_Config=SimpleNamespace())
    obj = SimpleNamespace(world_size=8, config=config, NUM_EXPERTS=256)
    monkeypatch.setenv("BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP", "1")
    monkeypatch.setenv("BATCHGEN_R1_MOE_BACKEND", "deep_gemm")
    monkeypatch.setenv("BATCHGEN_R1_RESIDENT_EP_PREFILL", "1")
    ns["_adjust_config_for_model"](obj)
    assert config.Module_Batching_Config.prefill_micro_batch_token_cap == 8192
    assert config.GPU_Buffer_Config.num_prefill_module_buffer == {"attn": 0, "routed_expert": 0, "shared_expert": 0}
    assert config.Basic_Config.attn_mode == 3 and config.EP_Config.num_local_expert_per_layer == 32


def host_manager(rank=0):
    obj = methods("_r1_host_expert_sources", "get_r1_deepgemm_expert_math")
    obj.global_rank, obj.rank, obj.world_size = rank, rank, 8
    obj._r1_native_moe = lambda: True
    obj.engine_config = SimpleNamespace(Basic_Config=SimpleNamespace(device_torch="cpu"))
    obj.skeleton_state_dict = {}
    tensors = {}
    for e in rp.local_expert_ids(rank, 8):
        key = f"routed_expert_3_{e}"
        tensors[key] = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[key][projection + ".weight"] = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
            obj.skeleton_state_dict[f"model.layers.3.mlp.experts.{e}.{projection}.weight_scale_inv"] = torch.ones((1, 1))
    obj.core_engine = SimpleNamespace(get_tensor=lambda key: tensors[key])
    return obj, tensors


def test_host_mapping_and_unique_bank_reused_then_source_change_rejected(monkeypatch):
    obj, tensors = host_manager(rank=7)
    calls = []
    class Math:
        @staticmethod
        def from_expert_sources(sources, *, device, source_identity):
            rows = list(sources)
            calls.append(rows)
            assert len(rows) == 32 and all(t.device.type == "cpu" for r in rows for t in r.values())
            return SimpleNamespace(source_identity=source_identity, resident_bytes=123)
    monkeypatch.setitem(sys.modules, PKG + ".deepgemm_moe", SimpleNamespace(R1DeepGemmExpertMath=Math))
    bank = obj.get_r1_deepgemm_expert_math(3)
    assert obj.get_r1_deepgemm_expert_math(3) is bank and len(calls) == 1
    source = tensors["routed_expert_3_224"]["gate_proj.weight"]
    source.copy_(torch.zeros_like(source))
    with pytest.raises(RuntimeError, match="changed/remapped"):
        obj.get_r1_deepgemm_expert_math(3)


def test_prefill_count_preparation_does_not_grow_decode_ll_capacity():
    obj = methods("set_num_tokens_per_rank")
    obj._r1_prefill_configuring = True
    obj._r1_deepep_pool = SimpleNamespace(capacity=4)
    obj.set_num_tokens_per_rank(8192)
    assert obj._r1_prefill_tokens_per_rank == 8192
    obj._r1_prefill_configuring = False
    with pytest.raises(RuntimeError, match="fixed process capacity"):
        obj.set_num_tokens_per_rank(8192)


@pytest.mark.parametrize("resident", [False, True])
def test_expert_configuration_drops_only_native_resident_placeholders(resident):
    registrations = []
    class Wrapper:
        def __init__(self, module, layer, expert, *args):
            self.module, self.expert = module, expert
            self.weight_dequant_scale = {}
        def _register_fp8_weights(self):
            registrations.append(self.expert)
    obj = methods("_config_expert_module", DeepSeekExpertWrapper=Wrapper)
    mlp = SimpleNamespace(experts=[object() for _ in range(256)], shared_experts=object())
    obj.model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 3 + [SimpleNamespace(mlp=mlp)]))
    obj.loaded_model_config = SimpleNamespace(first_k_dense_replace=3, phase="prefill")
    obj.core_engine = object()
    obj.engine_config = SimpleNamespace(Basic_Config=SimpleNamespace(device_torch="cpu"))
    obj.model_config = object()
    obj.skeleton_state_dict = {}
    obj.weight_copy_task = {"shared_expert": ["shared_expert_3"],
        "routed_expert": [f"routed_expert_3_{i}" for i in range(256)]}
    obj._r1_native_moe = lambda: True
    obj.prefill_uses_resident_ep = lambda: resident
    requested = []
    obj.get_r1_deepgemm_expert_math = lambda layer: requested.append(layer)
    obj._config_expert_module()
    if resident:
        assert requested == [3] and mlp.experts == [None] * 256
    else:
        assert requested == [] and all(isinstance(x, Wrapper) for x in mlp.experts)
    assert registrations == []  # No routed GPU register/list in the native path.


def test_collective_only_zero_rank_visits_all58_routed_layers():
    obj = methods("run_resident_ep_collective_only_prefill")
    obj.prefill_uses_resident_ep = lambda: True
    obj.engine_config = SimpleNamespace(Basic_Config=SimpleNamespace(device_torch="cpu"))
    obj.loaded_model_config = SimpleNamespace(first_k_dense_replace=3)
    calls = []
    obj.model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 3 + [
        SimpleNamespace(mlp=SimpleNamespace(_r1_resident_prefill_adapter=lambda x, i=i: calls.append((i, tuple(x.shape)))))
        for i in range(3, 61)]))
    obj.run_resident_ep_collective_only_prefill()
    assert calls == [(i, (0, 7168)) for i in range(3, 61)]


class CPU:
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)
    def __getattr__(self, key):
        return getattr(torch, key)
    def tensor(self, *args, **kw):
        kw.pop("pin_memory", None)
        return torch.tensor(*args, **kw)


def test_normal_routes_apply_weight_once_and_zero_rank_still_computes_peers():
    events = []
    class Exchange:
        def dispatch(self, quantized, ids, weights):
            events.append(("normal_dispatch", ids.shape[0]))
            received = torch.ones((1, 7168), dtype=torch.float8_e4m3fn)
            return ((received, torch.ones((1, 14), dtype=torch.int32)),
                    torch.tensor([[0] + [-1] * 7]), torch.tensor([[.25] + [0.] * 7]),
                    [128] + [0] * 31, "normal-handle")
        def combine(self, x, handle):
            events.append(("normal_combine", handle))
            # This rank owns no output row in the second call, but still
            # computes its received expert work before joining combine.
            return x if len(events) == 3 else x[:0]
    def quant(x, *args, **kwargs):
        assert kwargs == dict(column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)
        return x.to(torch.float8_e4m3fn), torch.ones((x.shape[0], 14), dtype=torch.int32)
    def scatter(x, scale, ids, counts, starts, out, scales, m, index, **kwargs):
        assert kwargs == {"scale_ue8m0": True} and out.shape == (128, 7168)
        m.zero_(); index.zero_()
    def gather(out, ids, weights, index, gathered):
        gathered.copy_(out[:1] * weights[0, 0])
    class Math:
        def contiguous(self, x, scale, indices):
            events.append(("contiguous", tuple(x.shape)))
            return torch.full_like(x, 2, dtype=torch.bfloat16)
    moe = SimpleNamespace(gate=lambda x: (torch.tensor([[0] + [-1] * 7]), torch.tensor([[.25] + [0.] * 7])))
    adapter = rp.R1ResidentPrefillAdapter(moe, Exchange(), Math(), layer_idx=3,
        runtime=(CPU(), scatter, gather, quant))
    result = adapter(torch.ones((1, 7168), dtype=torch.bfloat16))
    assert result.shape == (1, 7168) and torch.all(result == .5)
    empty = adapter(torch.empty((0, 7168), dtype=torch.bfloat16))
    assert empty.shape == (0, 7168)
    assert [x[0] for x in events] == ["normal_dispatch", "contiguous", "normal_combine"] * 2


def test_native_decoding_branch_uses_new_adapter_without_legacy_expert_lists(monkeypatch):
    calls = []
    class Pool:
        def __init__(self, *args, **kw):
            self.capacity = kw["max_tokens_per_rank"]
            self.world_size, self.num_experts, self.phase_configurations = 8, 256, 0
        def begin_phase(self):
            self.phase_configurations += 1
    def legacy(*args, **kwargs):
        raise AssertionError("native branch must not inspect old expert pointer lists")
    def native(moe, pool, **kwargs):
        calls.append(kwargs)
        return "native-adapter"
    monkeypatch.setitem(sys.modules, PKG + ".deepep_decode", SimpleNamespace(R1DeepEPDecodePool=Pool, R1DeepEPDecodeAdapter=legacy))
    monkeypatch.setitem(sys.modules, PKG + ".deepgemm_moe", SimpleNamespace(R1DeepGemmDecodeAdapter=native))
    obj = methods("_init_r1_deepep", dist=SimpleNamespace(group=SimpleNamespace(WORLD="world")))
    obj.enable_ep_offloading = False
    obj.world_size = 8
    obj.engine_config = SimpleNamespace(EP_Config=SimpleNamespace(enable=True, num_local_expert_per_layer=32),
        Basic_Config=SimpleNamespace(attn_mode=3, device_torch="cpu"))
    obj.loaded_model_config = SimpleNamespace(first_k_dense_replace=3, hidden_size=7168, moe_intermediate_size=2048,
        n_routed_experts=256, num_experts_per_tok=8)
    obj.model_config = SimpleNamespace(num_hidden_layers=4)
    obj.model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 3 + [SimpleNamespace(mlp=SimpleNamespace(max_num_tokens_per_rank=4))]))
    obj._r1_native_moe = lambda: True
    obj.get_r1_deepgemm_expert_math = lambda layer: "unique-math"
    obj._init_r1_deepep()
    assert calls == [{"layer_idx": 3, "expert_math": "unique-math"}]
