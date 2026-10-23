#!/usr/bin/env python3
"""Bound an existing-client SGLang QA run; default is a CPU-only plan.

No service launches, GPU queries, retries, or performance/quality qualification.
An operator supplies an immutable admission with the observed service identity.
Only the new CPU client's session can be signalled. HTTP work may outlive an
interrupted client; a failure explicitly requires the resource owner to check it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

import qa_sanity as qa
import run_sglang_trtllm_guard as ownership

ROOT = Path('/home/admin/batchgen-r1-repro-20261023')
HOST = 'serverless-010066009163.wm226'
ENDPOINT = 'http://127.0.0.1:30000'
LIMITS = {'seconds': 5400, 'client_session_RSS_bytes': 2 * 1024**3,
          'MemAvailable_min_bytes': 128 * 1024**3}
OWNERSHIP_SHA = 'd3b22fe45b89fc6630d6a3407b928fdaab2ae7775db458766f8c75192601e905'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified_plan(args):
    if qa.sha(args.manifest) != args.manifest_sha:
        raise ValueError('Explicit prepared QA manifest SHA differs')
    if qa.sha(Path(qa.__file__)) != args.qa_source_sha:
        raise ValueError('Reviewed QA source SHA differs')
    if qa.sha(Path(ownership.__file__)) != OWNERSHIP_SHA:
        raise ValueError('Reviewed process ownership implementation differs')
    admission = json.loads(args.admission.read_text())
    if (admission.get('scope') != qa.SCOPE or admission.get('manifest_sha256') != args.manifest_sha
            or admission.get('server_identity') != 'sglang0.5.5.post3+R1-routing-compat'
            or admission.get('cohort') != 'sglang-r1-diagnostic-v4'
            or admission.get('endpoint') != ENDPOINT or admission.get('kv_dtype') != 'torch.bfloat16'
            or admission.get('model_id') != 'deepseek-ai/DeepSeek-R1'):
        raise ValueError('Admission is not the reviewed v4 R1 QA deployment')
    declaration = qa.client_plan(args.manifest, 'sglang', ENDPOINT, args.output / 'client')
    command = declaration['command']
    command[0] = str(ROOT / 'client-env/bin/python')
    return admission, {'status': 'plan_only_NOT_requested', **declaration, 'command': command,
        'limits': LIMITS, 'admission_sha256': qa.sha(args.admission),
        'qa_source_sha256': args.qa_source_sha, 'guard_source_sha256': qa.sha(__file__),
        'ownership_source_sha256': OWNERSHIP_SHA,
        'scope': qa.SCOPE, 'quality_pass': False, 'performance_qualified': False}


def service_current(admission):
    saved = admission['service_process_identity']
    current = ownership.process_identity(saved['pid'])
    return ownership.same_identity(current, saved) and current['state'] not in ('Z', 'X')


def cleanup_client(child, identity):
    # A second stop request must not interrupt the first request's cleanup.
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        return ownership.stop_own_group(child, identity)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute(args, admission, record):
    if socket.gethostname() != HOST or not Path('/.dockerenv').exists():
        raise RuntimeError('Only the allocated node0 container may execute')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError('Default SIGCHLD is required for WNOWAIT ownership')
    if not service_current(admission):
        raise RuntimeError('Admitted service leader has exited or changed identity')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'admission.json').write_bytes(args.admission.read_bytes())
    (args.output / 'guard-source.py').write_bytes(Path(__file__).read_bytes())
    record.update(status='started', started_unix=time.time(), samples=[],
                  server_requests_may_remain=False, service_admission=admission)
    write(args.output / 'plan.json', record)
    child = None
    identity = None
    cleaned = False
    start = time.monotonic()
    def interrupted(signum, frame):
        raise InterruptedError(f'Guard received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        environment.pop('OPENAI_API_KEY', None)  # Allocated local service needs no external API credential.
        with (args.output / 'client.log').open('x') as log:
            child = subprocess.Popen(record['command'], env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT/'repo', start_new_session=True)
            identity = ownership.process_identity(child.pid)
            if identity is None or identity['pid'] != identity['pgid'] or identity['pid'] != identity['session']:
                raise RuntimeError('Cannot establish CPU client ownership')
            record['client_process_identity'] = identity
            while not ownership.child_exited(child):
                processes = ownership.session_processes(child.pid)
                sample = {'elapsed_seconds': time.monotonic()-start,
                    'client_session_RSS_bytes': sum(p['RSS_bytes'] for p in processes),
                    'MemAvailable_bytes': ownership.available(), 'service_current': service_current(admission)}
                record['samples'].append(sample)
                reason = ('wall_time_limit' if sample['elapsed_seconds'] >= LIMITS['seconds'] else
                    'client_memory_limit' if sample['client_session_RSS_bytes'] >= LIMITS['client_session_RSS_bytes'] else
                    'host_memory_floor' if sample['MemAvailable_bytes'] < LIMITS['MemAvailable_min_bytes'] else
                    'service_leader_exited' if not sample['service_current'] else None)
                write(args.output / 'guard.json', record)
                if reason:
                    record['stopped_reason'] = reason
                    break
                time.sleep(2)
            record['cleanup'] = cleanup_client(child, identity)
            cleaned = True
        record['exit_code'] = record['cleanup'].get('leader_exit_code')
        clean = not record['cleanup'].get('remaining_owned') and not record['cleanup'].get('remaining_session_members')
        if record.get('stopped_reason') or record['exit_code'] != 0 or not clean:
            raise RuntimeError('QA client failed or was stopped; inspect service requests before further work')
        summary = json.loads((args.output/'client/summary.json').read_text())
        if summary.get('status') != 'completed' or not summary['measurement']['all_requests_valid']:
            raise RuntimeError('QA collection has missing or invalid responses')
        record['score'] = qa.score(args.manifest, args.output/'client/requests.jsonl',
            args.output/'client/config.json', 'sglang', args.output/'score.json')
        record['status'] = 'collected_and_scored_NOT_quality_or_performance_pass'
    except BaseException:
        record['status'] = 'failed_requires_service_request_check'
        record['error'] = traceback.format_exc()
        record['server_requests_may_remain'] = child is not None
    finally:
        if child is not None and not cleaned:
            try:
                record['cleanup'] = cleanup_client(child, identity)
            except BaseException:
                record['cleanup_error'] = traceback.format_exc()
        record['finished_unix'] = time.time()
        record['elapsed_seconds_NOT_benchmark'] = time.monotonic()-start
        write(args.output/'guard.json', record)
    return 0 if record['status'] == 'collected_and_scored_NOT_quality_or_performance_pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--qa-source-sha', required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    admission, record = verified_plan(args)
    if not args.execute:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    return execute(args, admission, record)


if __name__ == '__main__':
    raise SystemExit(main())
