#!/usr/bin/env python3
"""Plan or supervise ONE assigned-GPU SGLang primitive diagnostic.

Default prints a plan only. --execute is for the parent resource owner AFTER
explicit release. Never controls BatchGen or any unrelated process. Saves an
immutable child-script snapshot and signals only identity-verified members of its own process group.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import traceback

from launch_sglang_diagnostic import HOSTS, ROOT, plan

CHILD_SHA = '2bb0d71ebb6c3d15040d6c814868f30ba84331545f86a44baa0110ab0161e942'
LIMITS = {'seconds': 900, 'process_group_RSS_bytes': 8*1024**3,
          'target_GPU_bytes': 4*1024**3, 'MemAvailable_min_bytes': 64*1024**3}


def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def gpu_apps(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, timeout=5, check=True)
    apps = []
    for line in result.stdout.splitlines():
        if line.strip():
            pid, mib = line.split(',')
            apps.append({'pid': int(pid), 'bytes': int(mib)*1024**2})
    return apps


def process_identity(pid):
    """Linux start ticks survive exec; command names and numeric PGIDs are not identity."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'pgid': int(fields[2]), 'session': int(fields[3]),
                'start_ticks': int(fields[19]), 'state': fields[0],
                'RSS_bytes': max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_identity(current, saved):
    return current is not None and all(current[k] == saved[k]
        for k in ('pid', 'pgid', 'session', 'start_ticks'))


def group_processes(pgid):
    rows = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            row = process_identity(int(path.name))
            if row is not None and row['pgid'] == pgid:
                rows.append(row)
    return rows


def child_exited(child):
    # Never poll/wait before cleanup: an unreaped leader pins its PID and PGID,
    # even after exit, so compiler descendants remain safely addressable.
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def signal_owned_group(identity, sig):
    current = process_identity(identity['pid'])
    if (not same_identity(current, identity) or
            current['pid'] != current['pgid'] or current['session'] != current['pid']):
        raise RuntimeError('Refusing signal: unreaped leader identity no longer matches')
    os.killpg(identity['pgid'], sig)


def violation(sample, elapsed):
    if elapsed > LIMITS['seconds']: return 'wall_time_limit'
    if sample['MemAvailable_bytes'] < LIMITS['MemAvailable_min_bytes']: return 'host_available_memory_limit'
    if sample['process_group_RSS_bytes'] >= LIMITS['process_group_RSS_bytes']: return 'process_group_RSS_limit'
    if sample.get('target_GPU_bytes', 0) >= LIMITS['target_GPU_bytes']: return 'target_GPU_memory_limit'
    if sample.get('unexpected_GPU_pids'): return 'unexpected_process_on_assigned_GPU'
    return None


def stop_own_group(proc, identity, grace_seconds=15):
    audit = {'signals': [], 'unverified_group_members': [], 'remaining_owned': []}
    if proc is None:
        return audit
    current = process_identity(proc.pid)
    if identity is None or not same_identity(current, identity):
        audit['identity_capture_failed'] = True
        audit['unverified_group_members'] = group_processes(proc.pid)
        return audit  # No numeric PID or PGID signals without ownership proof.
    def live():
        return [p for p in group_processes(proc.pid) if p['state'] not in ('Z', 'X')]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not live():
            break
        signal_owned_group(identity, sig)
        audit['signals'].append(int(sig))
        deadline = time.monotonic() + grace_seconds
        while live() and time.monotonic() < deadline:
            time.sleep(0.1)
    # Snapshot before reaping. No signal is ever sent after this wait.
    audit['remaining_owned'] = live()
    try:
        audit['leader_exit_code'] = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        audit['leader_wait_timeout'] = True
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node-rank', type=int, choices=(0,1), required=True)
    ap.add_argument('--stage', choices=('mla','moe'), required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--gpu', type=int, choices=range(8), default=1)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',args.run_id): ap.error('Invalid run-id')
    base = ROOT / 'sglang-baseline-v8'
    out = base / args.run_id / f'node{args.node_rank}' / args.stage
    env_changes, unset, _ = plan(ROOT,args.node_rank)
    env = {k:v for k,v in os.environ.items() if k not in unset}
    env.update(env_changes,CUDA_VISIBLE_DEVICES=str(args.gpu),MAX_JOBS='4',
               TRITON_CACHE_DIR=str(out/'triton-cache'))
    command = [str(base/'baseline-env/bin/python'),str(out/'sglang_trtllm_diagnostic.py'),
               '--stage',args.stage,'--expected-hostname',HOSTS[args.node_rank],
               '--output',str(out/'result')]
    record = {'status':'plan_only_NOT_executed','hostname':socket.gethostname(),
              'command':command,'environment':{k:env[k] for k in env_changes},
              'limits':LIMITS,'physical_GPU':args.gpu,'child_sha256':CHILD_SHA,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'One isolated primitive stage; no model-service or benchmark claim'}
    if not args.execute:
        print(json.dumps(record,indent=2));return 0
    if not Path('/.dockerenv').exists() or socket.gethostname()!=HOSTS[args.node_rank]:
        raise RuntimeError('Wrong execution container')
    out.mkdir(parents=True,exist_ok=False)
    proc = None
    identity = None
    cleanup_done = False
    start = time.monotonic()
    record.update(status='failed',started_unix=time.time(),samples=[])
    def save():
        tmp=out/'guard-result.json.tmp'
        tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        tmp.replace(out/'guard-result.json')
    def cleanup():
        nonlocal cleanup_done
        if not cleanup_done:
            cleanup_done = True
            record['cleanup'] = stop_own_group(proc, identity)
    try:
        if not all(hasattr(os, name) for name in ('waitid', 'WNOWAIT', 'WNOHANG', 'WEXITED', 'P_PID')):
            raise RuntimeError('Linux WNOWAIT support required before starting child')
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise RuntimeError('SIGCHLD must retain children for WNOWAIT ownership')
        record['cleanup_policy'] = 'Unreaped leader WNOWAIT + start_ticks/pgrp/session; wait only after last signal'
        child=Path(__file__).with_name('sglang_trtllm_diagnostic.py')
        if hashlib.sha256(child.read_bytes()).hexdigest()!=CHILD_SHA:
            raise RuntimeError('Child script changed; audit and update explicit pin before execution')
        shutil.copyfile(child,out/'sglang_trtllm_diagnostic.py')
        if hashlib.sha256((out/'sglang_trtllm_diagnostic.py').read_bytes()).hexdigest()!=CHILD_SHA:
            raise RuntimeError('Copied child script hash mismatch')
        record['initial_GPU_apps']=gpu_apps(args.gpu)
        record['initial_MemAvailable_bytes']=available()
        if record['initial_GPU_apps']: raise RuntimeError('Assigned GPU is not empty')
        if record['initial_MemAvailable_bytes']<LIMITS['MemAvailable_min_bytes']:
            raise RuntimeError('Insufficient available host memory')
        save()
        with (out/'child.log').open('x') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity = process_identity(proc.pid)
            record['pid']=record['pgid']=proc.pid
            while not child_exited(proc):
                processes=group_processes(proc.pid)
                sample={'elapsed_seconds':time.monotonic()-start,'MemAvailable_bytes':available(),
                        'process_group_RSS_bytes':sum(p['RSS_bytes'] for p in processes)}
                apps=gpu_apps(args.gpu)
                pids={p['pid'] for p in processes}
                sample.update(target_GPU_bytes=sum(p['bytes'] for p in apps),
                              unexpected_GPU_pids=[p['pid'] for p in apps if p['pid'] not in pids])
                sample['process_identities'] = processes
                record['samples'].append(sample)
                reason=violation(sample,sample['elapsed_seconds'])
                save()
                if reason:
                    record['stopped_reason']=reason
                    cleanup()
                    break
                time.sleep(1)
            cleanup()
            record['exit_code']=record['cleanup'].get('leader_exit_code')
        child_report=out/'result/report.json'
        if child_report.is_file():
            data=json.loads(child_report.read_text())
            record['child_status']=data.get('status')
            record['child_report_sha256']=hashlib.sha256(child_report.read_bytes()).hexdigest()
        if record.get('stopped_reason'):
            record['status']='stopped_by_resource_guard'
        elif record['exit_code']==0:
            record['status']='diagnostics_completed_NOT_model_qualified'
        elif record['exit_code']==2 and record.get('child_status')=='strict_attention_gate_failed':
            record['status']='strict_attention_gate_failed_preserved_for_audit'
        else:
            record['status']='child_failed'
    except BaseException as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
        record['traceback']=traceback.format_exc()
    finally:
        try:
            cleanup()
        except BaseException as exc:
            record['cleanup_error']=f'{type(exc).__name__}: {exc}'
            record['status']='cleanup_failed'
        if any(record.get('cleanup', {}).get(k) for k in ('remaining_owned', 'unverified_group_members', 'identity_capture_failed', 'leader_wait_timeout')):
            record['status_before_cleanup_audit'] = record['status']
            record['status'] = 'cleanup_requires_orphan_audit'
        record['finished_unix']=time.time()
        record['elapsed_seconds']=time.monotonic()-start
        try: record['final_GPU_apps']=gpu_apps(args.gpu)
        except Exception as exc: record['final_GPU_check_error']=repr(exc)
        save()
        print(json.dumps({k:v for k,v in record.items() if k!='samples'},indent=2),flush=True)
    return 0 if record['status']=='diagnostics_completed_NOT_model_qualified' else (2 if record['status']=='strict_attention_gate_failed_preserved_for_audit' else 1)


if __name__=='__main__':
    raise SystemExit(main())
