#!/usr/bin/env python3
"""Separate local host CPU, OS runqueue, and inclusive stream observations.

Never subtract a standalone collective floor from in-model events as an exact
wire/wait decomposition. Host wall minus thread CPU also includes driver waits.
The fixed 16-step cadence can alias page-growth/control boundaries. Its sample
distribution is not an unbiased full-step latency or stall-frequency estimate.
"""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import statistics


FIELDS=('forward_host_ms','forward_thread_cpu_ms','forward_off_cpu_ms','forward_stream_ms','runqueue_wait_ms')


def summary(values):
    values=sorted(values)
    return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p95':values[math.ceil(len(values)*.95)-1],'max':values[-1]}if values else None


def component_sample(call):
    return call in (8,16,32,64,256)or call%1024==0 or call%256==128


def analyze(log,start,end,expected_ranks=16):
    if not math.isfinite(start)or not math.isfinite(end)or end<=start:raise ValueError('Positive finite window required')
    ranks=defaultdict(list);seen=set()
    for line in log.splitlines():
        if '[R1_DECODE_FRAMES] 'not in line:continue
        data=json.loads(line.split('[R1_DECODE_FRAMES] ',1)[1]);rank=data['rank']
        if data['schema']!='r1-decode-frames-v1' or type(rank)is not int or not 0<=rank<expected_ranks:raise ValueError('Unexpected frame rank/schema')
        for row in data['samples']:
            if type(row.get('unix'))not in(int,float)or not math.isfinite(row['unix']):raise ValueError('Invalid frame timestamp')
            if not start<=row['unix']<=end:continue
            call=row['call']
            if type(call)is not int or call<=0 or call%16:raise ValueError('Unexpected frame cadence')
            if(rank,call)in seen:raise ValueError('Duplicate model frame inside measurement')
            seen.add((rank,call))
            for name in FIELDS:
                value=row.get(name)
                if value is None and name=='runqueue_wait_ms':continue
                if type(value)not in(int,float)or not math.isfinite(value)or value<0:raise ValueError('Invalid frame interval')
            ranks[rank].append({**row,'also_component_probed':component_sample(call)})
    if set(ranks)!=set(range(expected_ranks)):raise ValueError('Incomplete rank coverage')
    result={}
    call_sets=[]
    for rank,rows in sorted(ranks.items()):
        rows.sort(key=lambda r:r['call']);call_sets.append({r['call']for r in rows})
        cohorts={'all':rows,'without_component_probes':[r for r in rows if not r['also_component_probed']]}
        result[str(rank)]={'calls':[r['call']for r in rows],'observations':rows,'statistics':{scope:{name:summary([r[name]for r in rs if r.get(name)is not None])for name in FIELDS}for scope,rs in cohorts.items()}}
    common=set.intersection(*call_sets)
    if not common:raise ValueError('No common logical frames across ranks')
    aligned=[];lookups={rank:{r['call']:r for r in rows}for rank,rows in ranks.items()}
    for call in sorted(common):
        rows=[(rank,lookups[rank][call])for rank in range(expected_ranks)]
        slow_rank,slow=max(rows,key=lambda item:item[1]['forward_host_ms'])
        aligned.append({'call':call,'also_component_probed':component_sample(call),'max_host_rank':slow_rank,'max_host_ms':slow['forward_host_ms'],'slow_rank_thread_cpu_ms':slow['forward_thread_cpu_ms'],'slow_rank_runqueue_ms':slow['runqueue_wait_ms'],'median_stream_ms':statistics.median(r['forward_stream_ms']for _,r in rows),'max_runqueue_ms':max((r['runqueue_wait_ms']for _,r in rows if r['runqueue_wait_ms']is not None),default=None)})
    return {'schema':'r1-frame-attribution-analysis-v1','window':[start,end],'per_rank':result,'matched_rank_frames':aligned,'unmatched_calls':{str(rank):sorted(call_sets[rank]-common)for rank in range(expected_ranks)},'scope':'Every16th model forward; component-probed frames separated. Local OS runqueue counters are direct observations. Off-CPU wall includes driver blocking; stream intervals include host gaps and peer waits. Logical call alignment does not assume synchronized host clocks. No exact network wait or causal historical scheduling attribution.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',type=Path,action='append',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze('\n'.join(p.read_text()for p in a.log),a.start,a.end)
    with a.output.open('x')as out:json.dump(result,out,indent=2);out.write('\n')


if __name__=='__main__':main()
