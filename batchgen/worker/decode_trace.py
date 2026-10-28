"""Opt-in bounded CPU execution trace; consumes already-completed GPU events.

CUDA-event spans are reported as measurements, not drawn as a fabricated GPU
kernel timeline. This helper never imports Torch or synchronizes a device.
"""
from __future__ import annotations

import json
import gc
import os
from pathlib import Path
import resource
import time
import weakref


def _integer(env, key, default):
    raw = env.get(key, str(default))
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(raw)


class DecodeStepTrace:
    @classmethod
    def from_env(cls, *, rank, rows, env=None):
        env = os.environ if env is None else env
        flag = env.get("BATCHGEN_R1_DECODE_TRACE", "0")
        if flag not in ("0", "1"):
            raise ValueError("BATCHGEN_R1_DECODE_TRACE must be 0 or 1")
        if flag == "0":
            return None
        arm_file = env.get("BATCHGEN_R1_DECODE_TRACE_ARM_FILE")
        if arm_file:
            if not Path(arm_file).is_absolute():
                raise ValueError("BATCHGEN_R1_DECODE_TRACE_ARM_FILE must be absolute")
            if not Path(arm_file).is_file():
                return None
        minimum = _integer(env, "BATCHGEN_R1_DECODE_TRACE_MIN_ROWS", 1)
        if rows < minimum:
            return None
        directory = env.get("BATCHGEN_R1_DECODE_TRACE_DIR")
        if not directory or not Path(directory).is_absolute():
            raise ValueError("An absolute BATCHGEN_R1_DECODE_TRACE_DIR is required")
        return cls(directory, rank=rank,
                   max_steps=_integer(env, "BATCHGEN_R1_DECODE_TRACE_STEPS", 320))

    def __init__(self, directory, *, rank, max_steps, clock=time.perf_counter,
                 cpu_clock=time.process_time, usage=None):
        self.rank, self.max_steps = rank, max_steps
        self.clock, self.cpu_clock = clock, cpu_clock
        self.usage = usage or (lambda: resource.getrusage(resource.RUSAGE_SELF))
        self.origin = clock()
        self.path = Path(directory) / f"decode-rank{rank}-pid{os.getpid()}-{time.time_ns()}.json"
        self.records, self.events, self.gpu_spans = [], [], {}
        self.current = None
        self.written = False
        self.gc_events, self._gc_start = [], None
        owner = weakref.ref(self)
        def collection_event(phase, info):
            trace = owner()
            if trace is not None:
                trace._on_gc(phase, info)
        self._gc_callback = collection_event
        gc.callbacks.append(collection_event)

    def _on_gc(self, phase, info):
        if phase == 'start':
            self._gc_start = (self.clock(), info.get('generation'),
                              self.current['step'] if self.current is not None else None)
        elif phase == 'stop' and self._gc_start is not None:
            start, generation, step = self._gc_start
            self._gc_start = None
            stop = self.clock()
            row = {'start_perf_counter_seconds': start, 'end_perf_counter_seconds': stop,
                   'duration_ms': (stop - start) * 1000, 'generation': generation, 'step': step,
                   'collected': info.get('collected'), 'uncollectable': info.get('uncollectable')}
            self.gc_events.append(row)
            self.events.append({'name': f'Python_GC_generation{generation}', 'cat': 'CPU.GC', 'ph': 'X',
                                'ts': (start - self.origin) * 1e6, 'dur': (stop - start) * 1e6,
                                'pid': self.rank, 'tid': 1, 'args': row})

    def _remove_gc_callback(self):
        callback = getattr(self, '_gc_callback', None)
        callbacks = getattr(gc, 'callbacks', None)
        if callbacks is not None and callback in callbacks:
            callbacks.remove(callback)

    def __del__(self):
        self._remove_gc_callback()

    def begin(self, step, rows):
        if len(self.records) >= self.max_steps:
            self.current = None
            return
        self.current = {"step": int(step), "rows": int(rows),
                        "start": self.clock(), "cpu_start": self.cpu_clock(),
                        "usage_start": self.usage(), "marks": []}

    def mark(self, name):
        if self.current is not None:
            self.current["marks"].append((name, self.clock()))

    def on_gpu_sample(self, step, elapsed_ms):
        if step is not None and (self.current is not None or len(self.records) < self.max_steps):
            self.gpu_spans[int(step)] = float(elapsed_ms)

    def finish(self, *, forward_marks, graph_timing=None, contexts=(), page_boundary=False):
        c = self.current
        if c is None:
            return
        end, cpu_end, u = self.clock(), self.cpu_clock(), self.usage()
        names = ("setup", "forward_submit", "KV_submit", "token_wait", "Host_KV_drain", "bookkeeping")
        segments = {name: (forward_marks[i + 1] - forward_marks[i]) * 1000
                    for i, name in enumerate(names)}
        record = {"step": c["step"], "rows": c["rows"],
                  "start_perf_counter_seconds": c["start"],
                  "wall_ms_including_boundary": (end - c["start"]) * 1000,
                  "process_CPU_ms": (cpu_end - c["cpu_start"]) * 1000,
                  "page_boundary": bool(page_boundary), "CPU_segments_ms": segments,
                  "graph_submit_substages_ms": dict(graph_timing or {}),
                  "Python_GC_ms": sum(x['duration_ms'] for x in self.gc_events if x['step'] == c['step']),
                  "GPU_current_stream_forward_event_span_ms": self.gpu_spans.pop(c["step"], None),
                  "context_min": min(contexts, default=None), "context_max": max(contexts, default=None),
                  "minor_faults": u.ru_minflt - c["usage_start"].ru_minflt,
                  "major_faults": u.ru_majflt - c["usage_start"].ru_majflt,
                  "voluntary_context_switches": u.ru_nvcsw - c["usage_start"].ru_nvcsw,
                  "involuntary_context_switches": u.ru_nivcsw - c["usage_start"].ru_nivcsw}
        self.records.append(record)
        marks = [("loop", c["start"]), *c["marks"], ("end", end)]
        for (name, start), (_, stop) in zip(marks, marks[1:]):
            self.events.append({"name": name, "cat": "CPU", "ph": "X",
                                "ts": (start - self.origin) * 1e6, "dur": (stop - start) * 1e6,
                                "pid": self.rank, "tid": 0, "args": {"step": c["step"]}})
        self.events.append({"name": "decode_step_measurements", "cat": "measurements", "ph": "i", "s": "t",
                            "ts": (end - self.origin) * 1e6, "pid": self.rank, "tid": 0, "args": record})
        self.current = None
        if len(self.records) == self.max_steps:
            self.flush()

    def flush(self):
        self._remove_gc_callback()
        if self.written or not self.records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": "r1-bounded-decode-execution-trace-v1", "rank": self.rank,
                "recorded_steps": len(self.records), "max_steps": self.max_steps,
                "limitations": ["Opt-in diagnostic instrumentation; separate from uninstrumented throughput.",
                    "CUDA-event spans include CPU and peer waits; they are not pure GPU kernel busy time.",
                    "No extra device synchronization; missing GPU span means no completed event was available."],
                "records": self.records, "gc_events": self.gc_events, "traceEvents": self.events}
        self.path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        self.written = True
