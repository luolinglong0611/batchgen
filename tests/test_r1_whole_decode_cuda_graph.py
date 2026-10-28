"""CPU-only whole-graph control/signature contracts and 2-rank real Gloo check."""
import importlib.util
import ast
import gc
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import time
import weakref
from unittest.mock import patch

import pytest
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "batchgen/attention/mla/r1_whole_decode_cuda_graph.py"
NAME = "batchgen.attention.mla.r1_whole_decode_cuda_graph"
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def attention_signature_boundary(monkeypatch):
    # The runtime-agent's separately tested source-signature export is the
    # public boundary here. These fixtures test whole-model invalidation.
    monkeypatch.setattr(module, "_attention_signature", lambda wrapper: tuple(
        module._tensor_signature(getattr(wrapper.module, key).weight, version=True)
        for key in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")))


def fixture_signature():
    ns = types.SimpleNamespace
    def parameter():
        return torch.nn.Parameter(torch.ones((2, 2)))
    layers = []
    for _ in range(61):
        attn = ns(**{key: ns(weight=parameter()) for key in (
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")})
        attn._mla_decode_weight_cache_entry = None
        layers.append(ns(self_attn=ns(module=attn, weight_dequant_scale={}), mlp=ns()))
    model = ns(model=ns(layers=layers), lm_head=ns(weight=parameter()))
    caches = [torch.ones(1, 64, 1, 576) for _ in range(61)]
    manager = ns(_gpu_page_table_manager=ns(), _k_cache=caches,
        get_layer_kv_with_page_table=lambda i: (caches[i], None, None))
    keys = ("positions", "lengths", "page_table", "slot_indices")
    batch = ns(rows=1, micro_batch_size=1, context_cap=10240,
        plan_identity=((1, 2, 3),), stream=100, plans=(),
        **{key: torch.zeros(1, dtype=torch.int64) for key in keys})
    pool = ns(capacity=128, phase_configurations=1)
    return ns(), model, manager, batch, pool, torch.zeros(1, 1, dtype=torch.int64)


def signature(args):
    return module._signature(*args, invalidations=0)


def test_actual_token_and_metadata_bytes_do_not_require_recapture():
    args = fixture_signature()
    before = signature(args)
    args[3].positions.add_(1)
    args[3].page_table.add_(3)
    args[5].add_(17)
    assert signature(args) == before


def test_flat_weight_guard_reuses_full_signature_and_checks_normal_mutation(monkeypatch):
    model = fixture_signature()[1]
    guard = module.CapturedWeightSignature(model)
    key = guard.signature(model)
    assert guard.refreshes == 1
    original = module._weight_signature
    def unexpected_scan(model):
        raise AssertionError("Stable step must not rebuild the full model signature")
    monkeypatch.setattr(module, "_weight_signature", unexpected_scan)
    assert guard.signature(model) is key
    monkeypatch.setattr(module, "_weight_signature", original)
    with torch.no_grad():
        model.lm_head.weight.add_(1)
    assert guard.signature(model) != key
    assert guard.refreshes == 2


@pytest.mark.parametrize("mutate", [
    lambda m: setattr(m.lm_head, "weight", torch.nn.Parameter(torch.zeros(2, 2))),
    lambda m: setattr(m.lm_head.weight, "data", torch.ones(2, 2)),
    lambda m: setattr(m.lm_head.weight, "data", m.lm_head.weight.data.t()),
    lambda m: setattr(m, "_r1_weight_generation", 2),
    lambda m: setattr(m.model.layers[0].self_attn.module, "q_a_proj", types.SimpleNamespace(weight=torch.ones(2, 2))),
    lambda m: setattr(m.model.layers[0].self_attn.module, "_mla_decode_weight_cache_entry", (None, torch.ones(2, 2))),
])
def test_flat_weight_guard_detects_binding_layout_epoch_and_cache_changes(mutate):
    model = fixture_signature()[1]
    guard = module.CapturedWeightSignature(model)
    before = guard.signature(model)
    mutate(model)
    assert guard.signature(model) != before


def test_flat_weight_guard_detects_prepared_bank_and_native_api_replacement():
    model = fixture_signature()[1]
    attn = model.model.layers[0].self_attn.module
    item = types.SimpleNamespace(weight=torch.ones(2, 2), scale=torch.ones(2, 2))
    attn._r1_native_gemm_weights = [item]
    bank = [torch.ones(2, 2) for _ in range(4)]
    mlp = model.model.layers[3].mlp
    mlp.r1_deepep_adapter = types.SimpleNamespace(graph_weight_tensors=lambda: tuple(bank))
    mlp.shared_experts = types.SimpleNamespace()
    guard = module.CapturedWeightSignature(model)
    key = guard.signature(model)
    assert guard.signature(model) is key
    item.scale.add_(1)
    changed = guard.signature(model)
    assert changed != key
    item.weight = item.weight.clone()
    again = guard.signature(model)
    assert again != changed
    bank[0] = bank[0].clone()
    assert guard.signature(model) != again


def test_flat_weight_guard_detects_parameter_registration_replacement():
    model = fixture_signature()[1]
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    guard = module.CapturedWeightSignature(model)
    key = guard.signature(model)
    model.lm_head.weight = torch.nn.Parameter(torch.zeros(2, 2))
    assert guard.signature(model) != key


@pytest.mark.parametrize("name", ["weight", "e_score_correction_bias"])
@pytest.mark.parametrize("replace", [False, True])
def test_gate_weights_and_correction_bias_are_guarded_in_full_and_cached_signatures(name, replace):
    model = fixture_signature()[1]
    gate = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 2)),
                                 e_score_correction_bias=torch.nn.Parameter(torch.ones(2)))
    model.model.layers[3].mlp.gate = gate
    guard = module.CapturedWeightSignature(model)
    key = guard.signature(model)
    full_before = module._weight_signature(model)
    if replace:
        setattr(gate, name, torch.nn.Parameter(getattr(gate, name).detach().clone()))
    else:
        with torch.no_grad():
            getattr(gate, name).add_(1)
    assert module._weight_signature(model) != full_before
    assert guard.signature(model) != key


def test_prepared_ordinary_shared_weight_and_scale_updates_invalidate_graph():
    args = fixture_signature()
    entry = types.SimpleNamespace(weight=torch.ones(2, 2),
                                  scale=torch.zeros(2, 1, dtype=torch.int32))
    args[1].model.layers[0].mlp._r1_native_gemm_weights = [entry]
    before = signature(args)
    entry.weight.add_(1)
    changed = signature(args)
    assert changed != before
    entry.scale.add_(1)
    assert signature(args) != changed


@pytest.mark.parametrize("mutate", [
    lambda a: setattr(a[3], "plan_identity", ((1, 2, 4),)),
    lambda a: setattr(a[4], "phase_configurations", 2),
    lambda a: setattr(a[2], "_r1_graph_generation", 1),
    lambda a: setattr(a[3], "page_table", torch.zeros(1, dtype=torch.int64)),
    lambda a: setattr(a[1].model.layers[0].self_attn.module,
                      "_mla_decode_weight_cache_entry", (None, torch.ones(2, 2))),
])
def test_plan_lifetime_and_expanded_cache_changes_require_recapture(mutate):
    args = fixture_signature()
    before = signature(args)
    mutate(args)
    assert signature(args) != before


def test_weight_inplace_update_and_cache_reallocation_are_detected():
    args = fixture_signature()
    before = signature(args)
    with torch.no_grad():
        args[1].lm_head.weight.add_(1)
    assert signature(args) != before
    before = signature(args)
    args[2]._k_cache[0] = torch.ones_like(args[2]._k_cache[0])
    assert signature(args) != before


def test_native_expert_bank_replaces_legacy_pointer_signature():
    args = fixture_signature()
    bank = [torch.ones(2, 2) for _ in range(4)]
    mlp = args[1].model.layers[3].mlp
    mlp.shared_experts = types.SimpleNamespace()
    mlp.r1_deepep_adapter = types.SimpleNamespace(graph_weight_tensors=lambda: tuple(bank))
    # No legacy gate/up/down pointer lists exist on a native-only model.
    before = signature(args)
    bank[2].add_(1)
    assert signature(args) != before
    before = signature(args)
    bank[0] = bank[0].clone()
    assert signature(args) != before


def test_native_expert_signature_refuses_missing_bank():
    args = fixture_signature()
    mlp = args[1].model.layers[3].mlp
    mlp.r1_deepep_adapter = types.SimpleNamespace(graph_weight_tensors=lambda: ())
    with pytest.raises(RuntimeError, match="actual weight tensors"):
        signature(args)


def test_native_local_warmup_never_calls_old_mma_kernels(monkeypatch):
    calls = []
    pool = types.SimpleNamespace(x=torch.zeros(4, 8), num_local_experts=2,
        expert_out=torch.zeros(2, 4, 8))
    adapter = types.SimpleNamespace(warmup_local_only=lambda **kw: calls.append(kw))
    gate = types.SimpleNamespace(moe_gate_forward_hybrid=lambda x: None)
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(r1_deepep_adapter=adapter, gate=gate))
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[None] * 3 + [layer] * 58))
    helpers = types.ModuleType("batchgen.attention.mla.r1_decode_cuda_graph")
    helpers.compute_attention = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setattr(module, "_stream_complete", lambda stream: None)
    module._prewarm_local_kernels(model, None, None, pool, stream=None)
    assert calls == [{"rows": 0}] * 58


def test_zero_local_rows_still_execute_all_58_ll_pairs():
    calls = []
    record = module._WholeRecord.__new__(module._WholeRecord)
    record.static_ids = torch.empty((0, 1), dtype=torch.int64)
    record.empty_hidden = torch.empty((0, 7168), dtype=torch.bfloat16)
    record.empty_output = object()
    layers = [types.SimpleNamespace(mlp=types.SimpleNamespace(
        r1_deepep_adapter=lambda x, i=i: calls.append((i, tuple(x.shape))))) for i in range(61)]
    record.model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
    assert record.forward() is record.empty_output
    assert calls == [(i, (0, 7168)) for i in range(3, 61)]


def test_global_change_and_error_flags_cannot_be_ignored_locally():
    class Dist:
        ReduceOp = types.SimpleNamespace(MAX=object())
        def all_reduce(self, tensor, **kw):
            tensor.copy_(torch.tensor(self.global_flags))
        def all_gather_object(self, output, local, **kw):
            output[:] = [local, {"rank": 1, "stage": local["stage"], "error": "bad metadata"}]
    api = Dist()
    c = module.WholeDecodeCoordinator(object(), world_size=2, rank=0, dist_api=api)
    api.global_flags = [1, 0]
    assert c.agree("prepare", changed=False) is True
    api.global_flags = [0, 0]
    assert c.agree("prepare", changed=False) is False
    api.global_flags = [0, 1]
    with pytest.raises(RuntimeError, match="bad metadata"):
        c.agree("prepare", error=None)


def test_cpu_flags_are_not_inference_tensors():
    with torch.inference_mode():
        c = module.WholeDecodeCoordinator(object(), world_size=2, rank=0)
    assert c.flags.device.type == "cpu" and not c.flags.is_inference()


@pytest.mark.parametrize("worker_has_model", [True, False])
def test_real_worker_phase_teardown_releases_graph_owned_model_and_cache(worker_has_model):
    class Held:
        pass
    model, cache = Held(), Held()
    model_ref, cache_ref = weakref.ref(model), weakref.ref(cache)
    record = types.SimpleNamespace(model=model, expanded_mla_keepalive=[cache], last_done=object())
    state = module.R1WholeDecodeState.__new__(module.R1WholeDecodeState)
    state.record, state.invalidations = record, 0
    state.coordinator = object()
    control_group = state.coordinator
    events = []
    worker = types.SimpleNamespace(model=model if worker_has_model else None,
        parallel_manager=types.SimpleNamespace(model=model if worker_has_model else None),
        _r1_whole_decode_state=state, rank=0,
        _log_phase_memory_ledger=lambda label: events.append(label))
    del record, model, cache
    source = HERE.parent / "batchgen/batchgen_worker.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "deep_free_model_memory")
    namespace = {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        "sys": types.SimpleNamespace(modules={}), "time": time, "gc": gc, "logging": logging}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source), "exec"), namespace)
    with patch.object(module, "_wait_event", lambda event: events.append("wait-last-replay")):
        namespace[fn.name](worker)
    gc.collect()
    assert model_ref() is None and cache_ref() is None
    assert events[0] == "wait-last-replay"
    assert state.record is None and state.invalidations == 1
    assert worker._r1_whole_decode_state is state and state.coordinator is control_group


def test_invalidation_waits_for_replay_before_dropping_storage():
    state = module.R1WholeDecodeState.__new__(module.R1WholeDecodeState)
    event = object()
    state.record = types.SimpleNamespace(last_done=event)
    state.invalidations = 0
    seen = []
    with patch.object(module, "_wait_event", lambda e: seen.append(e)):
        state.invalidate(reason="weight-clear")
    assert seen == [event] and state.record is None and state.invalidations == 1


def gloo_worker(rank, world, rendezvous, output):
    import datetime
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=world, timeout=datetime.timedelta(seconds=30))
    with torch.inference_mode():
        c = module.WholeDecodeCoordinator(dist.group.WORLD, world_size=world, rank=rank)
        assert not c.flags.is_inference()
        results = [c.agree("initial", changed=True), c.agree("steady", changed=False),
                   c.agree("remote-changed", changed=(rank == 1))]
        try:
            c.agree("one-rank-error", error="bad page-table" if rank == 1 else None)
        except RuntimeError as exc:
            results.append("bad page-table" in str(exc))
        else:
            results.append(False)
    Path(output, str(rank) + ".json").write_text(json.dumps(results))
    dist.destroy_process_group()


if __name__ == "__main__" and "--gloo-check" in sys.argv:
    import torch.multiprocessing as mp
    assert not torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory(prefix="r1-whole-control-") as directory:
        mp.spawn(gloo_worker, args=(2, str(Path(directory, "rendezvous")), directory),
                 nprocs=2, join=True)
        results = [json.loads(Path(directory, str(i) + ".json").read_text()) for i in range(2)]
        assert results == [[True, False, True, True]] * 2, results
        print(json.dumps({"CPU_Gloo_ranks": 2, "results": results,
                          "GPU_operations": False, "status": "CONTROL_PROTOCOL_PASSED"}))
