#!/usr/bin/env python3
"""Analyze sampled R1 event intervals inside one exact client measurement window.

Only complete all-layer decode samples enter per-step estimates. Per-rank
intervals remain separate: summing ranks or parent/child spans would double count.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

EXPECTED = {'attention': 61, 'routed_moe': 58, 'dense_mlp': 3,
            'all_gather': 58, 'all_reduce': 58, 'gate': 58, 'dispatch': 58,
            'experts_total': 58, 'scatter_reduce': 58, 'shared_expert': 58,
            'token_count_readback': 58, 'compact': 58, 'active_experts_readback': 58,
            'gate_up_gemm': 58, 'quantize': 116, 'down_gemm': 58}
DIRECT_CHILDREN = ('all_gather','gate','dispatch','experts_total',
                   'scatter_reduce','all_reduce','shared_expert')
GRAPH_EXPECTED = {name: EXPECTED[name] for name in
                  ('attention','routed_moe','dense_mlp','all_gather','all_reduce','shared_expert')}
GRAPH_EXPECTED['local_moe_graph'] = 58
GRAPH_CHILDREN = ('all_gather','local_moe_graph','all_reduce','shared_expert')


def analyze(records, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('An actual positive measurement window is required')
    groups = defaultdict(list)
    for row in records:
        if row.get('schema') != 'r1-component-stream-interval-v1':
            raise ValueError('Unknown trace schema')
        for key in ('unix','stream_ms','cpu_ms'):
            if type(row.get(key)) not in (int,float) or not math.isfinite(row[key]):
                raise ValueError('Non-finite or missing trace value')
        if row['stream_ms'] < 0 or row['cpu_ms'] < 0:
            raise ValueError('Negative interval')
        if start <= row['unix'] <= end:
            groups[(row['rank'],row['phase'],row['call'])].append(row)
    samples, rejected, prefill = [], [], []
    for (rank,phase,call), rows in sorted(groups.items()):
        stages = defaultdict(list)
        for row in rows:
            stages[row['stage']].append(row)
        values = {name:{'n':len(rs),'layers':len({r['layer']for r in rs}),
                        'stream_ms':sum(r['stream_ms']for r in rs),
                        'cpu_ms':sum(r['cpu_ms']for r in rs)}
                  for name,rs in stages.items()}
        sample = {'rank':rank,'call':call,'phase':phase,'stages':values}
        if phase == 'prefill':
            prefill.append(sample)
            continue
        # Duplicate layers also reject pooled samples from multiple model
        # instances with the same call number; no averaging partial samples up.
        graphed = bool(stages.get('local_moe_graph'))
        expected = GRAPH_EXPECTED if graphed else EXPECTED
        optional = set() if graphed else {'compact','active_experts_readback','gate_up_gemm','down_gemm','quantize'}
        complete = all(len(stages[name]) == count and
                       {r['layer']for r in stages[name]} ==
                       (set(range(61)) if name=='attention' else set(range(3)) if name=='dense_mlp' else set(range(3,61)))
                       for name,count in expected.items() if name not in optional)
        if graphed:
            # Inner Python spans disappear on replay. Mixed eager fallback or a
            # capture inside this sample cannot stand in for a complete replay.
            complete = complete and not any(stages.get(name) for name in EXPECTED if name not in GRAPH_EXPECTED)
        else:
            active_layers = {r['layer']for r in stages['down_gemm']}
            complete = complete and active_layers <= set(range(3,61)) and all(
                len(stages[name]) == len(active_layers) and {r['layer']for r in stages[name]} == active_layers
                for name in ('compact','active_experts_readback','gate_up_gemm','down_gemm'))
            complete = complete and len(stages['quantize']) == 2*len(active_layers) and all(
                sum(r['layer']==layer for r in stages['quantize'])==2 for layer in active_layers)
        for name in expected:
            values.setdefault(name, {'n':0,'layers':0,'stream_ms':0.0,'cpu_ms':0.0})
        if not complete:
            rejected.append({'rank':rank,'call':call,'counts':{k:len(v)for k,v in stages.items()}})
            continue
        sample['execution_path'] = 'local_graph_replay' if graphed else 'eager'
        sample['observed_forward_stream_ms'] = sum(values[k]['stream_ms']for k in ('attention','routed_moe','dense_mlp'))
        children = GRAPH_CHILDREN if graphed else DIRECT_CHILDREN
        sample['moe_unattributed_stream_ms'] = values['routed_moe']['stream_ms']-sum(values[k]['stream_ms']for k in children)
        samples.append(sample)
    ranks = {}
    for rank in sorted({s['rank']for s in samples}):
        rs = [s for s in samples if s['rank']==rank]
        observed_names = set.intersection(*(set(s['stages']) for s in rs))
        ranks[str(rank)] = {
            'complete_steps':len(rs), 'calls':[s['call']for s in rs],
            'execution_paths':sorted({s['execution_path'] for s in rs}),
            'median_observed_forward_stream_ms':statistics.median(s['observed_forward_stream_ms']for s in rs),
            'stages':{name:{key:statistics.median(s['stages'][name][key]for s in rs)
                            for key in ('stream_ms','cpu_ms')}for name in sorted(observed_names)},
        }
    return {'schema':'r1-component-measurement-analysis-v1','start_unix':start,'end_unix':end,
            'per_rank':ranks,'complete_decode_samples':samples,'rejected_partial_decode_samples':rejected,
            'prefill_samples':prefill,
            'scope':'Sampled stream elapsed and host call intervals. Not kernel occupancy; collective intervals include rank skew/waits. Do not sum ranks, nested spans, or CPU and stream time. Prefill samples are separate from native full-prefill/model-switch wall time.'}


def analyze_native_phases(log_text, start, end, timezone='Asia/Shanghai'):
    """Read native prefill KV-ready and busy switch timers from BOTH node logs.

    Prefill is the max participating-rank capability boundary. Switch timers
    exclude pool idle waits and overlap other wall classifications; do not add
    these fields to the request BCT or to sampled decode intervals.
    """
    import datetime as dt
    import re
    from zoneinfo import ZoneInfo
    prefills, switches, configurations = [], [], []
    stamp = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - \[BatchGenWorker-(\d+)\].*')
    for line in log_text.splitlines():
        match = stamp.search(line)
        if not match:
            continue
        unix = dt.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo(timezone)).timestamp()
        if not start <= unix <= end:
            continue
        if '[METRICS] ' in line:
            row = json.loads(line.split('[METRICS] ',1)[1])
            if row.get('component') == 'prefill_full_attention':
                prefills.append({'completed_unix':unix, **row})
        if '[PHASE_SWITCH]' in line:
            q = re.search(r'rank=(\d+) direction=(\w+) seconds=([\d.]+) persistent=(\w+)',line)
            if not q:
                raise ValueError('Malformed native phase-switch line')
            switches.append({'rank':int(q[1]),'direction':q[2],'reported_busy_seconds':float(q[3]),
                             'persistent':q[4], 'completed_unix':unix})
        q = re.search(r'\[PREFILL\] Model configured in ([\d.]+)s \(init=([\d.]+)s, skeleton=([\d.]+)s, expert=([\d.]+)s, to_device=([\d.]+)s\)',line)
        if q:
            configurations.append({'rank':int(match[2]),'completed_unix':unix,
                                   **dict(zip(('total_s','init_s','skeleton_s','expert_s','to_device_s'),map(float,q.groups())))})
    return {'schema':'r1-native-prefill-switch-attribution-v1','prefill_rows':prefills,
            'prefill_participating_ranks':sorted({r['rank']for r in prefills}),
            'prefill_total_sequences':sum(r['sequences']for r in prefills),
            'prefill_total_prompt_tokens':sum(r['prompt_tokens']for r in prefills),
            'max_rank_prefill_seconds':{k:max((r[k]for r in prefills),default=None)
                                        for k in ('prefill_forward_host_s','prefill_forward_gpu_complete_s','prefill_kv_ready_s')},
            'switch_rows':switches,'prefill_configuration_rank0':configurations,
            'scope':'Native per-rank prefill KV-ready boundary and busy switch timers completing inside the exact measurement window. Partial rank coverage stays explicit; never sum rank durations or add switch timers to BCT.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,action='append',required=True)
    parser.add_argument('--start-unix',type=float,required=True)
    parser.add_argument('--end-unix',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    for path in args.trace:
        rows.extend(json.loads(line)for line in path.read_text().splitlines()if line)
    result=analyze(rows,args.start_unix,args.end_unix)
    with args.output.open('x')as out:json.dump(result,out,indent=2);out.write('\n')

if __name__=='__main__':main()
