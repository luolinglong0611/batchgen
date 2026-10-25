#!/usr/bin/env python3
"""Read-only host RDMA counter deltas in an exact steady-state window.

Linux ABI: port_xmit_data/port_rcv_data count octets divided by four:
https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband
Port counters include all host traffic. They are neither per-process wire
attribution nor a way to exclude short bursts, latency or congestion.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def ports(snapshot):
    result={}
    for device in snapshot['devices']:
        for port in device['ports']:
            key=(device['node_guid'],port['port'])
            if key in result:
                raise ValueError('Duplicate physical port identity')
            rate=re.fullmatch(r'([\d.]+) Gb/sec(?: .*)?',port['rate'])
            if not rate or not port['state'].startswith('4: ACTIVE'):
                raise ValueError('An active port and explicit reported rate are required')
            result[key]={'device':device['device'],'rate_bytes_per_second':float(rate[1])*1e9/8,
                         'counters':port['counters'],'hw_counters':port.get('hw_counters',{})}
    return result


def analyze(snapshots, bindings, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end<=start:
        raise ValueError('A positive exact measurement window is required')
    by_host=defaultdict(list)
    for snapshot in snapshots:
        host=snapshot['hostname']
        if host not in bindings:
            raise ValueError('Missing actual host/container GUID binding')
        actual=ports(snapshot);expected=ports(bindings[host])
        if set(actual)!=set(expected) or {k:v['device']for k,v in actual.items()}!={k:v['device']for k,v in expected.items()}:
            raise ValueError('Counter port identities differ from allocated-container identities')
        if not math.isfinite(snapshot['observed_unix']):
            raise ValueError('Nonfinite snapshot timestamp')
        by_host[host].append(snapshot)
    intervals=[]
    for host,rows in by_host.items():
        rows.sort(key=lambda x:x['observed_unix'])
        for left,right in zip(rows,rows[1:]):
            t0,t1=left['observed_unix'],right['observed_unix']
            if t1<=t0:
                raise ValueError('Duplicate or unordered snapshot times')
            # Never interpolate an interval that crosses warmup/prefill/window
            # boundaries, or silently prorate traffic to the desired interval.
            if not start<=t0<t1<=end:
                continue
            before,after=ports(left),ports(right);values=[]
            for key in before:
                a,b=before[key],after[key]
                if a['rate_bytes_per_second']!=b['rate_bytes_per_second']:
                    raise ValueError('Port rate changed within a sampled interval')
                deltas={}
                for name in ('port_xmit_data','port_rcv_data'):
                    x,y=a['counters'][name],b['counters'][name]
                    if type(x)is not int or type(y)is not int or y<x:
                        raise ValueError('Counter reset/wrap or invalid counter; do not guess a delta')
                    deltas[name]=4*(y-x)
                seconds=t1-t0;tx=deltas['port_xmit_data']/seconds;rx=deltas['port_rcv_data']/seconds
                hw={name:b['hw_counters'][name]-v for name,v in a['hw_counters'].items()
                    if type(v)is int and type(b['hw_counters'].get(name))is int and b['hw_counters'][name]>=v}
                values.append({'device':b['device'],'node_guid':key[0],'port':key[1],
                               'tx_bytes':deltas['port_xmit_data'],'rx_bytes':deltas['port_rcv_data'],
                               'tx_GB_per_second':tx/1e9,'rx_GB_per_second':rx/1e9,
                               'reported_port_GB_per_second':b['rate_bytes_per_second']/1e9,
                               'tx_line_fraction':tx/b['rate_bytes_per_second'],
                               'rx_line_fraction':rx/b['rate_bytes_per_second'],
                               'host_hw_counter_deltas':hw})
            intervals.append({'hostname':host,'start_unix':t0,'end_unix':t1,'seconds':t1-t0,
                              'ports':values,'aggregate_tx_GB_per_second':sum(v['tx_GB_per_second']for v in values),
                              'aggregate_rx_GB_per_second':sum(v['rx_GB_per_second']for v in values)})
    hosts={}
    for host in sorted(by_host):
        rows=[r for r in intervals if r['hostname']==host];seconds=sum(r['seconds']for r in rows)
        hosts[host]={'intervals':len(rows),'covered_seconds':seconds,
                     'weighted_host_tx_GB_per_second':sum(sum(p['tx_bytes']for p in r['ports'])for r in rows)/seconds/1e9 if seconds else None,
                     'weighted_host_rx_GB_per_second':sum(sum(p['rx_bytes']for p in r['ports'])for r in rows)/seconds/1e9 if seconds else None,
                     'max_single_port_TX_line_fraction':max((p['tx_line_fraction']for r in rows for p in r['ports']),default=None),
                     'max_single_port_RX_line_fraction':max((p['rx_line_fraction']for r in rows for p in r['ports']),default=None)}
    return {'schema':'r1-host-RDMA-window-analysis-v1','start_unix':start,'end_unix':end,
            'hosts':hosts,'intervals':intervals,'data_counter_bytes_multiplier':4,
            'ABI_source':'https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband',
            'scope':'Only complete sampled intervals within the given steady window; host traffic, not exclusive model traffic. Average rates cannot exclude microbursts, latency, congestion or rank-arrival skew. TX and RX capacities are separate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,action='append',required=True)
    p.add_argument('--binding',type=Path,action='append',required=True,help='JSON with host field and actual container snapshot')
    p.add_argument('--start-unix',type=float,required=True);p.add_argument('--end-unix',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    bindings={}
    for path in args.binding:
        row=json.loads(path.read_bytes());bindings[row['host']]=row['container_snapshot']
    result=analyze([json.loads(path.read_bytes())for path in args.snapshot],bindings,args.start_unix,args.end_unix)
    result['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in args.snapshot+args.binding}
    with args.output.open('x')as out:json.dump(result,out,indent=2);out.write('\n')

if __name__=='__main__':main()
