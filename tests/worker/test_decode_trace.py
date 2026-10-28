import json
from types import SimpleNamespace

import pytest
from batchgen.worker.decode_trace import DecodeStepTrace


def test_disabled_trace_does_not_require_a_directory():
    assert DecodeStepTrace.from_env(rank=0, rows=180, env={}) is None
    assert DecodeStepTrace.from_env(rank=0, rows=8,
        env={"BATCHGEN_R1_DECODE_TRACE": "1", "BATCHGEN_R1_DECODE_TRACE_MIN_ROWS": "180"}) is None


def test_arm_file_is_checked_once_at_decode_round_start(tmp_path):
    arm = tmp_path / 'ARM'
    env = {"BATCHGEN_R1_DECODE_TRACE": "1", "BATCHGEN_R1_DECODE_TRACE_DIR": str(tmp_path),
           "BATCHGEN_R1_DECODE_TRACE_ARM_FILE": str(arm)}
    assert DecodeStepTrace.from_env(rank=0, rows=180, env=env) is None
    arm.touch()
    assert DecodeStepTrace.from_env(rank=0, rows=180, env=env) is not None


def test_trace_is_bounded_and_preserves_step_and_completed_event_identity(tmp_path):
    clock = [1.0]
    usage = SimpleNamespace(ru_minflt=0, ru_majflt=0, ru_nvcsw=0, ru_nivcsw=0)
    trace = DecodeStepTrace(tmp_path, rank=2, max_steps=1,
                            clock=lambda: clock[0], cpu_clock=lambda: 0.01,
                            usage=lambda: usage)
    trace.begin(64, 180)
    clock[0] = 1.03
    trace.mark("setup")
    trace.on_gpu_sample(64, 201.0)
    clock[0] = 1.25
    trace.finish(forward_marks=[1.03, 1.04, 1.06, 1.07, 1.23, 1.24, 1.25],
                 graph_timing={"signature": 0.8}, contexts=[8256, 8256], page_boundary=True)
    data = json.loads(trace.path.read_text())
    assert data['recorded_steps'] == 1
    r = data['records'][0]
    assert r['step'] == 64 and r['page_boundary']
    assert r['GPU_current_stream_forward_event_span_ms'] == 201.0
    assert r['CPU_segments_ms']['token_wait'] == pytest.approx(160.0)
    assert r['wall_ms_including_boundary'] == pytest.approx(250.0)
    assert r['graph_submit_substages_ms'] == {'signature': 0.8}
    trace.begin(65, 180)
    trace.on_gpu_sample(65, 500.0)
    trace.finish(forward_marks=[])
    assert len(trace.records) == 1 and not trace.gpu_spans
    assert all(e['cat'] != 'GPU_kernel' for e in data['traceEvents'])


def test_gc_pause_has_an_explicit_step_and_generation_without_changing_collection(tmp_path):
    import gc
    clock = [1.0]
    usage = SimpleNamespace(ru_minflt=0, ru_majflt=0, ru_nvcsw=0, ru_nivcsw=0)
    prior = len(gc.callbacks)
    trace = DecodeStepTrace(tmp_path, rank=0, max_steps=1,
                            clock=lambda: clock[0], cpu_clock=lambda: 0.0, usage=lambda: usage)
    assert len(gc.callbacks) == prior + 1
    trace.begin(128, 180)
    trace._on_gc('start', {'generation': 2})
    clock[0] = 1.2
    trace._on_gc('stop', {'generation': 2, 'collected': 4, 'uncollectable': 0})
    trace.finish(forward_marks=[1.0] * 7)
    data = json.loads(trace.path.read_text())
    assert data['records'][0]['Python_GC_ms'] == pytest.approx(200.0)
    assert data['gc_events'][-1]['step'] == 128
    assert data['gc_events'][-1]['generation'] == 2
    assert len(gc.callbacks) == prior


@pytest.mark.parametrize("env", [
    {"BATCHGEN_R1_DECODE_TRACE": "invalid"},
    {"BATCHGEN_R1_DECODE_TRACE": "1", "BATCHGEN_R1_DECODE_TRACE_DIR": "relative"},
])
def test_invalid_trace_configuration_is_explicit(env):
    with pytest.raises(ValueError):
        DecodeStepTrace.from_env(rank=0, rows=180, env=env)
