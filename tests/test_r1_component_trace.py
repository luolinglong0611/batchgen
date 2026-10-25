"""The component probe must never turn timing into a device synchronization."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('r1_components', Path(__file__).parents[1] / 'batchgen/profiling/r1_components.py')
trace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_module)

class Event:
    def __init__(self):
        self.complete = False
        self.records = []
    def record(self, stream):
        self.records.append(stream)
    def query(self):
        return self.complete
    def elapsed_time(self, other):
        assert self.complete and other.complete
        return 2.5
    def synchronize(self):
        raise AssertionError('component tracing must not synchronize')

def test_trace_reads_elapsed_only_after_both_events_complete(tmp_path):
    events = []
    def factory():
        event = Event(); events.append(event); return event
    output = tmp_path / 'trace.jsonl'
    trace = trace_module.ComponentTrace(output, 8, factory, lambda: 'stream')
    with trace.span('down_gemm', {'call': 32, 'layer': 3}):
        pass
    assert [e.records for e in events] == [['stream'], ['stream']]
    trace.harvest(); assert not output.exists()
    events[1].complete = True
    trace.harvest(); assert not output.exists()
    events[0].complete = True
    trace.harvest()
    import json
    row = json.loads(output.read_text())
    assert row['stream_ms'] == 2.5 and row['rank'] == 8 and row['layer'] == 3
    assert trace.pending == []

def test_exception_still_closes_interval(tmp_path):
    trace = trace_module.ComponentTrace(tmp_path / 'trace', 0, Event, lambda: None)
    with pytest.raises(ValueError):
        with trace.span('gate', {}):
            raise ValueError('original failure')
    assert len(trace.pending) == 1
    assert len(trace.pending[0][2].records) == 1

def test_full_queue_skips_without_allocating_or_waiting(tmp_path):
    trace = trace_module.ComponentTrace(tmp_path / 'trace', 0, lambda: pytest.fail('bounded queue allocated'), lambda: None)
    trace.pending = [None] * 4096
    with trace.span('all_gather', {}):
        pass
    assert trace.skipped == 1
