"""Same-LL graph/eager diagnostic protocol; CPU tensors and strict scope fake."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
NAME = "batchgen.attention.mla.checked_whole_compare"
SOURCE = HERE.parent / "batchgen/attention/mla/r1_whole_decode_cuda_graph.py"
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)


@pytest.fixture
def context(monkeypatch):
    class Collector:
        def __init__(self, expected_layers, stream):
            self.expected_layers = expected_layers
            self.capture_stream = stream
            self.kv_by_layer = {}
            self.last_staged_replay = 0
    active = []
    @contextmanager
    def scope(collector):
        active.append(collector)
        try:
            yield collector
        finally:
            active.pop()
    api = types.ModuleType("batchgen.attention.mla.r1_decode_cuda_graph")
    api.WholeModelKVOutputs, api.whole_model_attention_scope = Collector, scope
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: types.SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(module, "_stream_complete", lambda stream: None)
    return active


def fixture(context, *, rows=1, global_need=False, corrupt=False):
    stages = []
    class Coordinator:
        def agree(self, stage, *, changed=False, error=None):
            stages.append((stage, changed, error))
            if error is not None:
                raise RuntimeError(stage + ": " + str(error))
            return global_need or changed
    state = module.R1WholeDecodeState.__new__(module.R1WholeDecodeState)
    state.coordinator = Coordinator()
    state.compared_nonzero_rows = False
    state.diagnostic_eager_passes = 0
    layers = tuple(range(61)) if rows else ()
    logits = torch.arange(rows * 8).float().view(rows, 1, 8)
    kv = {i: torch.full((rows, 1, 1, 576), float(i + 1)) for i in layers}
    captured = types.SimpleNamespace(expected_layers=layers, kv_by_layer=kv,
                                     last_staged_replay=1)
    calls = []
    def forward():
        calls.append("all-rank-eager-participation")
        assert context and context[-1] is not captured
        context[-1].kv_by_layer = {i: value.clone() for i, value in kv.items()}
        if corrupt:
            context[-1].kv_by_layer[3].add_(1)
        return types.SimpleNamespace(logits=logits.clone())
    record = types.SimpleNamespace(static_ids=torch.zeros(rows, 1, dtype=torch.int64),
        output=types.SimpleNamespace(logits=logits), outputs=captured,
        forward=forward, replays=1)
    return state, record, calls, stages


def test_first_nonzero_diagnostic_is_once_and_preserves_graph_collector(context):
    state, record, calls, stages = fixture(context)
    captured_kv = record.outputs.kv_by_layer
    state.compare_first_nonzero_replay(types.SimpleNamespace(rank=0), record)
    assert calls == ["all-rank-eager-participation"]
    assert state.compared_nonzero_rows is True and state.diagnostic_eager_passes == 1
    assert record.outputs.kv_by_layer is captured_kv
    assert record.outputs.last_staged_replay == 1
    state.compare_first_nonzero_replay(types.SimpleNamespace(rank=0), record)
    assert len(calls) == 1


def test_zero_rank_joins_when_another_rank_requests_comparison(context):
    state, record, calls, stages = fixture(context, rows=0, global_need=True)
    state.compare_first_nonzero_replay(types.SimpleNamespace(rank=7), record)
    assert calls == ["all-rank-eager-participation"]
    assert stages[0] == ("diagnostic-selection", False, None)
    assert state.compared_nonzero_rows is False
    assert record.outputs.kv_by_layer == {} and record.outputs.last_staged_replay == 1


def test_layerwise_kv_mismatch_is_coherently_rejected(context):
    state, record, calls, stages = fixture(context, corrupt=True)
    with pytest.raises(RuntimeError, match="diagnostic-numeric-comparison"):
        state.compare_first_nonzero_replay(types.SimpleNamespace(rank=0), record)
    assert state.compared_nonzero_rows is False
    assert torch.all(record.outputs.kv_by_layer[3] == 4)


def test_default_diagnostic_is_off_and_invalid_value_is_rejected(monkeypatch):
    monkeypatch.delenv(module.COMPARE_ENVIRONMENT, raising=False)
    assert module.comparison_requested() is False
    monkeypatch.setenv(module.COMPARE_ENVIRONMENT, "1")
    assert module.comparison_requested() is True
    monkeypatch.setenv(module.COMPARE_ENVIRONMENT, "true")
    with pytest.raises(ValueError, match="explicitly 0 or 1"):
        module.comparison_requested()


def test_logits_mismatch_and_nonfinite_values_are_rejected():
    with pytest.raises((AssertionError, ValueError)):
        module._compare_tensor(torch.ones(16), torch.ones(16) * 2)
    with pytest.raises(ValueError, match="non-finite"):
        module._compare_tensor(torch.tensor([float("nan")]), torch.zeros(1))
