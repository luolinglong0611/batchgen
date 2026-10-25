#!/usr/bin/env python3
"""Attribute exact-window R1 prefill streams and native weight-copy counters.

Selected-layer event spans are nested in MoE; copy daemon time overlaps model
compute and must never be added to it. A blocking-copy interval includes host
scheduling and CUDA completion; it is not a hardware-only DMA measurement.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def analyze(records, log, start, end, ranks=16):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start or type(ranks) is not int or ranks < 1:
        raise ValueError('A finite positive window and rank count are required')
    samples = defaultdict(list)
    for row in records:
        if row.get('phase') == 'prefill':
            for field in ('unix', 'cpu_ms', 'stream_ms'):
                value = row.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or (field != 'unix' and value < 0):
                    raise ValueError('Invalid prefill event value: ' + field)
        if row.get('phase') == 'prefill' and start <= row['unix'] <= end:
            samples[row['rank'], row['call']].append(row)
    per_rank = {}
    for rank in range(ranks):
        calls = []
        for (r, call), rows in sorted(samples.items()):
            if r != rank:
                continue
            stages = defaultdict(list)
            for row in rows:
                stages[row['stage']].append(row)
            for stage, layers in [('attention', set(range(61))), ('routed_moe', set(range(3, 61))), ('dense_mlp', set(range(3)))]:
                if len(stages[stage]) != len(layers) or {v['layer'] for v in stages[stage]} != layers:
                    raise ValueError(f'Incomplete prefill sample rank={rank} call={call} stage={stage}')
            selected = {}
            for layer in (3, 30, 60):
                values = {name: [v for v in stages[name] if v['layer'] == layer] for name in ('prefill_weight_wait', 'prefill_expert_compute', 'prefill_weight_release')}
                counts = {name: len(v) for name, v in values.items()}
                if len(set(counts.values())) != 1 or not next(iter(counts.values())):
                    raise ValueError(f'Incomplete weight/compute sample: {rank}/{call}/{layer}/{counts}')
                selected[str(layer)] = {name: {'calls': len(v), 'host_seconds': sum(x['cpu_ms'] for x in v) / 1000, 'stream_seconds': sum(x['stream_ms'] for x in v) / 1000} for name, v in values.items()}
            calls.append({'call': call, 'forward_stream_seconds': {name: sum(v['stream_ms'] for v in stages[name]) / 1000 for name in ('attention', 'routed_moe', 'dense_mlp')}, 'selected_layers': selected})
        if not calls:
            raise ValueError(f'No complete prefill for rank {rank}')
        per_rank[str(rank)] = calls
    native = defaultdict(list)
    for line in log.splitlines():
        if '[R1_WEIGHT_STREAM]' not in line:
            continue
        stamp = re.search(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})', line)
        match = re.search(r'\[R1_WEIGHT_STREAM\] rank=([0-9]+) (\{.*\})', line)
        if not stamp or not match:
            raise ValueError('Malformed native weight counters')
        unix = datetime.strptime(stamp[1], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        if start <= unix <= end:
            counters = json.loads(match[2])
            if not counters['h2d']['enabled'] or not counters['consumer_wait']['enabled']:
                raise ValueError('Native weight profiling disabled')
            native[int(match[1])].append({'unix': unix, 'counters': counters})
    if set(native) != set(range(ranks)) or any(len(v) != 1 for v in native.values()):
        raise ValueError('Exactly one native phase snapshot per rank required')
    return {'schema': 'r1-prefill-weight-attribution-v1', 'window': [start, end], 'per_rank': per_rank, 'native_weight_stream': {str(k): v[0] for k, v in native.items()}, 'scope': 'All61-layer prefill samples; expert spans only layers3/30/60. Nested spans and concurrent copy-daemon time are not additive. Native bytes count actual H2D copies including prefetched modules; blocking copy elapsed includes scheduling/completion. No hardware-only DMA or SGLang kernel attribution.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--log', type=Path, action='append', required=True)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = [json.loads(line) for path in args.trace for line in path.read_text().splitlines() if line]
    result = analyze(records, '\n'.join(path.read_text() for path in args.log), args.start, args.end)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
