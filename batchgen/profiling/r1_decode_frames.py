"""Bounded decode-frame measurements, without any added device wait.

Thread runqueue delay is measured locally, not inferred from collective CUDA
intervals. Wall minus thread CPU also includes driver blocking. Device event
intervals include host launch gaps and peer waits, not occupied SM time.
"""
import json
import logging
import os
import time


def schedstat():
    try:
        with open('/proc/thread-self/schedstat') as source:
            values = list(map(int, source.read().split()))
        return values if len(values) == 3 else None
    except (OSError, ValueError):
        return None


class FrameTrace:
    def __init__(self, rank, torch):
        self.rank, self.torch, self.call = rank, torch, 0
        self.rows, self.pending = [], []

    def begin(self):
        self.call += 1
        if self.pending:
            self.harvest()
        if self.call % 16:
            return None
        if len(self.pending) >= 16:
            return None
        before = schedstat()
        first, last = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        first.record(self.torch.cuda.current_stream())
        return (self.call, before, first, last, time.perf_counter_ns(), time.thread_time_ns())

    def end(self, token):
        if token is None:
            return
        call, before, first, last, wall, cpu = token
        cpu = time.thread_time_ns() - cpu
        wall = time.perf_counter_ns() - wall
        last.record(self.torch.cuda.current_stream())
        after = schedstat()
        row = {'call': call, 'unix': time.time(), 'forward_host_ms': wall / 1e6,
               'forward_thread_cpu_ms': cpu / 1e6, 'forward_off_cpu_ms': max(0, wall - cpu) / 1e6,
               'runqueue_wait_ms': (after[1] - before[1]) / 1e6 if before is not None and after is not None else None,
               'timeslices': after[2] - before[2] if before is not None and after is not None else None}
        self.pending.append((row, first, last))

    def harvest(self):
        waiting = []
        for row, first, last in self.pending:
            if first.query() and last.query():
                self.rows.append({**row, 'forward_stream_ms': first.elapsed_time(last)})
            else:
                waiting.append((row, first, last))
        self.pending = waiting
        if self.rows:
            logging.info('[R1_DECODE_FRAMES] %s', json.dumps({
                'schema': 'r1-decode-frames-v1', 'rank': self.rank,
                'samples': self.rows,
                'scope': 'Every16th forward; asynchronous GPU harvest, local thread CPU/runqueue observations. Collective intervals include arrival waits.',
            }, separators=(',', ':')))
            self.rows = []


def install(model, rank, torch):
    if os.environ.get('BATCHGEN_R1_SM103_DECODE_DIAGNOSTICS') != '1':
        return
    original = model.forward
    trace = FrameTrace(rank, torch)
    model._r1_frame_trace = trace

    def forward(*args, **kwargs):
        token = trace.begin()
        try:
            return original(*args, **kwargs)
        finally:
            trace.end(token)
    model.forward = forward
