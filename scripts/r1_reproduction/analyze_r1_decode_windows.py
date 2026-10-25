#!/usr/bin/env python3
"""Exact native heartbeat windows; preserve coarse means instead of inventing steps."""
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

STAMP=re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}) - \[BatchGenWorker-0\]')
TIMING=re.compile(r'step_ms=([0-9.]+).*gpu_fwd_ms=([0-9.]+).*gpu_fwd_max=([0-9.]+).*\(n=([0-9]+)\)')
SEGMENTS=('setup','fwd_launch','kv_launch','readback','kv_drain','bookkeeping')


def analyze(log,start,end,expected_steps):
    rows=[]
    for line in log.replace('\r','\n').splitlines():
        stamp=STAMP.search(line);match=TIMING.search(line)
        if not stamp or not match:continue
        unix=datetime.strptime(stamp[1],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        if not start<=unix<=end:continue
        step,gpu,maximum,n=match.groups();parts={}
        for name in SEGMENTS:
            p=re.search(r'\b'+name+r' ([0-9.]+)',line)
            if not p:raise ValueError('Missing native CPU timing segment')
            parts[name]=float(p[1])
        rows.append({'completed_unix':unix,'steps':int(n),'step_ms':float(step),'forward_stream_ms':float(gpu),'max_forward_stream_ms':float(maximum),'host_split_ms':parts})
    rows.sort(key=lambda r:r['completed_unix']);total=sum(r['steps']for r in rows)
    if total!=expected_steps:raise ValueError(f'Native window coverage {total} != {expected_steps}')
    cursor=0
    for row in rows:row['logical_steps']=[cursor+1,cursor+row['steps']];cursor+=row['steps']
    weighted=lambda fn:sum(fn(r)*r['steps']for r in rows)/total
    return {'schema':'r1-native-decode-window-analysis-v1','rows':rows,'steps':total,'weighted_step_ms':weighted(lambda r:r['step_ms']),'weighted_forward_stream_ms':weighted(lambda r:r['forward_stream_ms']),'weighted_host_split_ms':{name:weighted(lambda r:r['host_split_ms'][name])for name in SEGMENTS},'minimum_window_mean_ms':min(r['step_ms']for r in rows),'maximum_window_mean_ms':max(r['step_ms']for r in rows),'maximum_single_forward_ms':max(r['max_forward_stream_ms']for r in rows),'scope':'Rank0 complete rolling means and maxima, clipped by original measurement bounds. No reconstructed per-step values or exact stall-frequency claim.'}
