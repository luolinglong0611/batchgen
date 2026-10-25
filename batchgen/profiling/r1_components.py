"""Opt-in, sampled R1 stream/host attribution; never synchronizes the device.

CUDA-event intervals include host launch gaps and collective waits. Nested
intervals must not be added to their parent. They are not occupied-SM time.
The normal path installs no wrappers when BATCHGEN_R1_COMPONENT_TRACE is unset.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import time

_ACTIVE = None
_TRACE = None


class ComponentTrace:
    def __init__(self, path, rank, event_factory, stream_factory):
        self.path, self.rank = path, rank
        self.event_factory, self.stream_factory = event_factory, stream_factory
        self.pending = []
        self.skipped = 0

    def harvest(self):
        ready, pending = [], []
        for row, start, end in self.pending:
            if start.query() and end.query():
                ready.append({**row, 'stream_ms': start.elapsed_time(end)})
            else:
                pending.append((row, start, end))
        self.pending = pending
        if ready:
            with open(self.path, 'a', encoding='utf-8') as out:
                for row in ready:
                    out.write(json.dumps(row, sort_keys=True) + '\n')

    @contextlib.contextmanager
    def span(self, name, metadata):
        # Strict bound even if a device stalls. Profiling never waits for space.
        if len(self.pending) >= 4096:
            self.skipped += 1
            yield
            return
        start, end = self.event_factory(), self.event_factory()
        stream = self.stream_factory()
        start.record(stream)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cpu_ms = (time.perf_counter() - t0) * 1000
            end.record(stream)
            self.pending.append(({
                'schema': 'r1-component-stream-interval-v1', 'rank': self.rank,
                'stage': name, 'cpu_ms': cpu_ms, 'unix': time.time(), **metadata,
            }, start, end))


def span(name):
    if _ACTIVE is None:
        return contextlib.nullcontext()
    trace, metadata = _ACTIVE
    return trace.span(name, metadata)


def scalar_item(value, name):
    if _ACTIVE is None:
        return value.item()
    with span(name):
        return value.item()


def _wrap_stage(fn, name):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with span(name):
            return fn(*args, **kwargs)
    return wrapped


def install(model, phase, rank, *, persistent_phase_instances=False):
    """Install only on the exact R1 model instance, after its weights are bound."""
    global _TRACE
    directory = os.getenv('BATCHGEN_R1_COMPONENT_TRACE')
    if not directory:
        return
    import torch
    from batchgen.models.deepseek.deepseekv3 import modeling_deepseek_v3 as source

    if _TRACE is None:
        os.makedirs(directory, exist_ok=True)
        _TRACE = ComponentTrace(
            os.path.join(directory, f'rank{rank}.jsonl'), rank,
            lambda: torch.cuda.Event(enable_timing=True), torch.cuda.current_stream,
        )
        # These functions are shared, but wrappers record only inside a sampled
        # routed-MoE forward. Calls in other paths retain their original result.
        for name, label in (
            ('fused_moe_token_dispatch', 'dispatch'),
            ('compact_expert_data', 'compact'), ('act_quant', 'quantize'),
            ('fused_fp8_moe_stage_1_tma', 'gate_up_gemm'),
            ('fused_dequant_grouped_gemm_fp8_tma', 'down_gemm'),
            ('scatter_weight_reduce_optimized', 'scatter_reduce'),
        ):
            setattr(source, name, _wrap_stage(getattr(source, name), label))

    if phase == 'decode':
        from batchgen.worker.r1_decode_gc import install as install_gc
        install_gc(model, rank, persistent_phase_instances=persistent_phase_instances)
        from batchgen.profiling.r1_decode_frames import install as install_frames
        install_frames(model, rank, torch)

    def wrap_forward(module, label, layer):
        original = module.forward
        calls = 0
        @functools.wraps(original)
        def forward(*args, **kwargs):
            nonlocal calls
            global _ACTIVE
            calls += 1
            # Few complete steps, after first-call compilation. Same call set
            # on every layer/rank; full512 profiles retain their native length.
            selected = (phase == 'prefill' and os.getenv('BATCHGEN_R1_PHASE_PREPARE') == '1') or calls in (8, 16, 32, 64, 256) or calls % 1024 == 0 or (phase == 'decode' and os.getenv('BATCHGEN_R1_SM103_DECODE_DIAGNOSTICS') == '1' and calls % 256 == 128)
            if not selected:
                return original(*args, **kwargs)
            _TRACE.harvest()
            metadata = {'phase': phase, 'layer': layer, 'call': calls}
            previous = _ACTIVE
            _ACTIVE = (_TRACE, metadata)
            try:
                with span(label):
                    return original(*args, **kwargs)
            finally:
                _ACTIVE = previous
        module.forward = forward

    for index, layer in enumerate(model.model.layers):
        wrap_forward(layer.self_attn, 'attention', index)
        wrap_forward(layer.mlp, 'routed_moe' if index >= model.config.first_k_dense_replace else 'dense_mlp', index)
        if phase == 'prefill' and index in (3, 30, 60):
            # Separate weight readiness waits from expert math in the streamed
            # prefill path. These are nested in routed_moe, not additive to it.
            for expert in layer.mlp.experts:
                expert.load_weights = _wrap_stage(expert.load_weights, 'prefill_weight_wait')
                expert.micro_batch_forward = _wrap_stage(expert.micro_batch_forward, 'prefill_expert_compute')
                expert.free_weights = _wrap_stage(expert.free_weights, 'prefill_weight_release')
        if phase == 'decode' and index >= model.config.first_k_dense_replace:
            moe = layer.mlp
            moe.grouped_dequant_moe_fp8 = _wrap_stage(moe.grouped_dequant_moe_fp8, 'experts_total')
            moe.gate.moe_gate_forward_hybrid = _wrap_stage(moe.gate.moe_gate_forward_hybrid, 'gate')
            moe.shared_experts.forward = _wrap_stage(moe.shared_experts.forward, 'shared_expert')
            comm = moe.comm
            if not getattr(comm, '_r1_components_wrapped', False):
                comm.all_gather = _wrap_stage(comm.all_gather, 'all_gather')
                comm.all_reduce = _wrap_stage(comm.all_reduce, 'all_reduce')
                comm._r1_components_wrapped = True
