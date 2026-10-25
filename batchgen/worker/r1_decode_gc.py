"""Keep stable R1 decode objects out of cyclic scans until model teardown.

Reference counting and collection of newly allocated cycles remain enabled.
The worker must close the scope before dropping either model owner; a model
finalizer cannot do that because a frozen model cycle cannot collect itself.
"""
import gc
import json
import logging
import os
import threading
import time


class DecodeGCScope:
    def __init__(self, rank):
        self.rank = rank
        self.attempted = self.closed = self.owns_freeze = False
        self.frozen_objects = 0
        self.forward_call = 0
        self.in_forward = False
        self.pending = [None] * 3
        self.stats = [dict(collections=0, wall_ms=0.0, cpu_ms=0.0,
                           maximum_wall_ms=0.0, collected=0) for _ in range(3)]
        self.callback = self._observe
        gc.callbacks.append(self.callback)

    def _log(self, event, **fields):
        logging.info('[R1_DECODE_GC] %s', json.dumps({
            'schema': 'r1-decode-gc-scope-v1', 'rank': self.rank,
            'event': event, 'unix': time.time(),
            'thread_id': threading.get_native_id(),
            'forward_call': self.forward_call, 'in_forward': self.in_forward, **fields,
        }, separators=(',', ':')))

    def _observe(self, phase, info):
        generation = info['generation']
        if phase == 'start':
            self.pending[generation] = time.perf_counter_ns(), time.thread_time_ns()
        elif phase == 'stop' and self.pending[generation] is not None:
            wall, cpu = self.pending[generation]
            wall = (time.perf_counter_ns() - wall) / 1e6
            cpu = (time.thread_time_ns() - cpu) / 1e6
            self.pending[generation] = None
            row = self.stats[generation]
            row['collections'] += 1
            row['wall_ms'] += wall
            row['cpu_ms'] += cpu
            row['maximum_wall_ms'] = max(row['maximum_wall_ms'], wall)
            row['collected'] += info.get('collected', 0)
            if generation == 2:
                self._log('collection', generation=2, wall_ms=wall, cpu_ms=cpu,
                          collected=info.get('collected', 0), frozen=self.owns_freeze)

    def activate(self):
        if self.closed:
            raise RuntimeError('Closed R1 GC scope reused')
        if self.attempted:
            return
        self.attempted = True
        # Never take ownership of an existing application's permanent generation.
        if not gc.isenabled() or gc.get_freeze_count():
            self._log('skipped_existing_GC_policy', enabled=gc.isenabled(),
                      existing_frozen_objects=gc.get_freeze_count())
            return
        started = time.perf_counter()
        gc.collect()
        if gc.get_freeze_count():
            self._log('skipped_external_freeze_during_collection')
            return
        gc.freeze()
        self.owns_freeze = True
        self.frozen_objects = gc.get_freeze_count()
        self._log('activated', frozen_objects=self.frozen_objects,
                  prepare_seconds=time.perf_counter() - started,
                  automatic_collection_enabled=gc.isenabled())

    def close(self):
        if self.closed:
            return
        if self.owns_freeze:
            gc.unfreeze()
            self.owns_freeze = False
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)
        self.closed = True
        self._log('closed', originally_frozen_objects=self.frozen_objects,
                  remaining_frozen_objects=gc.get_freeze_count(),
                  automatic_collection_enabled=gc.isenabled(), generations=self.stats)


def install(model, rank, *, persistent_phase_instances=False):
    value = os.environ.get('BATCHGEN_R1_SM103_FREEZE_DECODE_GC', '0')
    if value not in ('0', '1'):
        raise ValueError('BATCHGEN_R1_SM103_FREEZE_DECODE_GC must be 0 or 1')
    if value == '0':
        return
    if (os.environ.get('BATCHGEN_R1_SM103_FAST_DIAGNOSTIC') != '1'
            or os.environ.get('BATCHGEN_R1_SM103_ALLOW_CAPTURE') != '0'
            or persistent_phase_instances):
        raise ValueError('R1 decode GC scope requires the nonpersistent eager diagnostic')
    if hasattr(model, '_r1_decode_gc_scope'):
        raise RuntimeError('R1 GC scope already installed on this model')
    scope = DecodeGCScope(rank)
    original = model.forward

    def forward(*args, **kwargs):
        scope.forward_call += 1
        scope.in_forward = True
        try:
            output = original(*args, **kwargs)
            # First-call projections and local-MoE graphs now belong to this model.
            # Include this one-time preparation in the native forward timing.
            scope.activate()
            return output
        finally:
            scope.in_forward = False

    model._r1_decode_gc_scope = scope
    model.forward = forward
