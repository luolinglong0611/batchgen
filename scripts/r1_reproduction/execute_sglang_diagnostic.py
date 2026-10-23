#!/usr/bin/env python3
"""Two-phase, parent-coordinated SGLang diagnostic executor (not a benchmark).

Default: print a plan only. After selected GPU kernels are qualified and BatchGen
is released on BOTH nodes, run --prepare on each node with a fresh --run-dir and
shared --cohort-id. Exchange receipt.json without credentials. The parent alone
then authorizes --execute on each node using --peer-receipt. Receipts expire in
600 seconds; resource ownership between preflight and launch is the parent's
responsibility. A failure on either node requires the parent to stop the peer.

Signal this executor with SIGTERM/SIGINT for controlled shutdown. It keeps the
child group leader unreaped until cleanup, binding group signals to /proc PID
starttime to avoid PID reuse. No process-name kills, remote actions or repeated
GPU queries. READY does not end this supervisor or impose an evaluation timeout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path('/home/admin/batchgen-r1-repro-20261023')
HOSTS = ('serverless-010066009163.wm226', 'serverless-010066009166.wm226')
GIB = 1024 ** 3
RECEIPT_TTL = 600
STARTUP_SECONDS = 7200
TERM_GRACE_SECONDS = 120
POLL_SECONDS = 5
MODEL = 'deepseek-ai/DeepSeek-R1'
READINESS_PINS = {
    'sglang/srt/entrypoints/http_server.py': 'ea07502b405064aff9874d4b9d36f58adeeeaf29e8036f9704a38fa0c1d178cc',
    'sglang/srt/entrypoints/engine.py': '6c58c81a1226fa41ce522c2d8b11617d0ad1bc48794eae4961a10a7b3fb8f95b',
    'sglang/srt/managers/scheduler.py': '14476e32f930726c0838e2044811c0ced4fbf7f50582e635b77b0e4f651262b5',
    'sglang/srt/managers/data_parallel_controller.py': 'e7eea6ddc9f6825782c1e8c9c3be9af92e5cfd196a894cf39ad55aef1a23cbba',
    'sglang/srt/utils/common.py': 'ed60f73e55147782aba9c893d39a1d7283e8b97a1b6402de1ace0744b193ee41',
}
MARKERS = (
    'The server is fired up and ready to roll!',
    'Dummy health check server started in background thread at 0.0.0.0:30000',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
    else:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(path)


def load_planner():
    path = Path(__file__).with_name('launch_sglang_diagnostic.py')
    spec = importlib.util.spec_from_file_location('r1_sglang_plan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes():
    here = Path(__file__).parent
    names = ('execute_sglang_diagnostic.py', 'launch_sglang_diagnostic.py',
             'sglang_diagnostic_source_pins.json', 'r1_model_manifest.json')
    return {name: sha(here / name) for name in names}


def require_container(rank, root):
    if root != ROOT or not Path('/.dockerenv').exists() or socket.gethostname() != HOSTS[rank]:
        raise RuntimeError('Expected allocated container and fixed experiment root required')


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def require_memory(min_gib):
    available = memory_available()
    if available < min_gib * GIB:
        raise RuntimeError(f'MemAvailable {available / GIB:.3f} GiB < {min_gib} GiB')
    return available


def parse_idle_gpus(xml):
    """One nvidia-smi XML snapshot: exactly 8 devices, no processes, zero util.

    <=64 MiB/device permits driver bookkeeping only. Unknown/N/A values fail
    closed. A fresh snapshot cannot reserve hardware; parent owns allocation.
    """
    root = ET.fromstring(xml)
    gpus = root.findall('gpu')
    if len(gpus) != 8 or int(root.findtext('attached_gpus', '0')) != 8:
        raise RuntimeError('Exactly eight visible GPUs required')
    result = []
    for gpu in gpus:
        used = gpu.findtext('fb_memory_usage/used', '')
        util = gpu.findtext('utilization/gpu_util', '')
        processes = gpu.find('processes')
        if not re.fullmatch(r'\d+ MiB', used) or not re.fullmatch(r'\d+ %', util):
            raise RuntimeError('GPU snapshot has unknown memory/utilization')
        if (processes is None or processes.findall('process_info') or
                (processes.text or '').strip() not in ('', 'None')):
            raise RuntimeError('GPU process inventory unavailable or occupied')
        if int(used.split()[0]) > 64 or int(util.split()[0]) != 0:
            raise RuntimeError('GPU not idle')
        result.append({'uuid': gpu.findtext('uuid'), 'name': gpu.findtext('product_name'),
                       'memory_used_mib': int(used.split()[0]), 'utilization_percent': 0})
    if len({g['uuid'] for g in result}) != 8 or any(not g['uuid'] for g in result):
        raise RuntimeError('GPU UUID inventory invalid')
    return result


def bundle(planner, root, rank):
    env, unset, command = planner.plan(root, rank)
    return {'environment': env, 'unset_environment': unset, 'command': command}


def child_environment(configuration, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = {k: v for k, v in inherited.items() if k not in configuration['unset_environment']}
    env.update(configuration['environment'])
    return env


def check_readiness_sources(root):
    site = root / 'sglang-baseline-v8/baseline-env/lib/python3.11/site-packages'
    actual = {name: sha(site / name) for name in READINESS_PINS}
    if actual != READINESS_PINS:
        raise RuntimeError('Readiness source changed; re-audit exact log/HTTP criteria')
    return actual


def prepare(args, planner):
    require_container(args.node_rank, args.root)
    # exist_ok=False is deliberate: even a failed preparation is not reusable.
    args.run_dir.mkdir(parents=False, exist_ok=False)
    try:
        config = bundle(planner, args.root, args.node_rank)
        write_json(args.run_dir / 'plan.json', config, exclusive=True)
        sources = source_hashes()
        inspection = planner.inspect(args.root, args.node_rank, config['environment'],
                                     config['unset_environment'], config['command'])
        if inspection.get('status') != 'CPU_artifact_inspection_passed_NOT_GPU_qualified':
            raise RuntimeError('CPU artifact inspection has not passed')
        write_json(args.run_dir / 'cpu-inspection.json', inspection, exclusive=True)
        readiness_sources = check_readiness_sources(args.root)
        before = require_memory(850)
        gpu = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True,
                             text=True, timeout=5, check=True)
        (args.run_dir / 'gpu-preflight.xml').write_text(gpu.stdout)
        devices = parse_idle_gpus(gpu.stdout)
        after = require_memory(850)
        receipt = {'status': 'prepared_NOT_started', 'cohort_id': args.cohort_id,
                   'node_rank': args.node_rank, 'hostname': socket.gethostname(),
                   'root': str(args.root), 'run_dir': str(args.run_dir.resolve()),
                   'created_unix': time.time(), 'ttl_seconds': RECEIPT_TTL,
                   'source_sha256': sources, 'plan_sha256': digest(config),
                   'inspection_sha256': sha(args.run_dir / 'cpu-inspection.json'),
                   'readiness_source_sha256': readiness_sources,
                   'memory_before_bytes': before, 'memory_after_bytes': after,
                   'gpus': devices, 'gpu_query_count': 1,
                   'gpu_snapshot_sha256': sha(args.run_dir / 'gpu-preflight.xml'),
                   'selected_gpu_kernel_qualification': 'Parent must verify separate reports before prepare/execute'}
        write_json(args.run_dir / 'receipt.json', receipt, exclusive=True)
        return receipt
    except BaseException as exc:
        write_json(args.run_dir / 'prepare-failed.json', {'error': repr(exc), 'time_unix': time.time()})
        raise


def validate_receipts(local, peer, rank, cohort, now, sources, config):
    for receipt, expected_rank in ((local, rank), (peer, 1 - rank)):
        if (receipt.get('status') != 'prepared_NOT_started' or receipt.get('node_rank') != expected_rank
                or receipt.get('hostname') != HOSTS[expected_rank] or receipt.get('cohort_id') != cohort
                or receipt.get('root') != str(ROOT) or receipt.get('source_sha256') != sources
                or receipt.get('readiness_source_sha256') != READINESS_PINS):
            raise RuntimeError('Preparation identity/cohort/source mismatch')
        age = now - receipt['created_unix']
        if age < 0 or age > RECEIPT_TTL or receipt.get('ttl_seconds') != RECEIPT_TTL:
            raise RuntimeError('Preparation receipt expired or from future')
        if receipt.get('memory_after_bytes', 0) < 850 * GIB or len(receipt.get('gpus', [])) != 8:
            raise RuntimeError('Peer/local resource receipt incomplete')
    if local['plan_sha256'] != digest(config):
        raise RuntimeError('Launch plan/environment differs from preparation')


def process_identity(pid):
    # comm can contain spaces and parentheses; numeric fields start after last ).
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'pgid': int(fields[2]), 'sid': int(fields[3]),
            'starttime_ticks': int(fields[19])}


def same_group_leader(identity):
    try:
        current = process_identity(identity['pid'])
    except FileNotFoundError:
        return False
    return (current['pid'] == current['pgid'] == identity['pgid'] and
            current['sid'] == identity['sid'] == identity['pid'] and
            current['starttime_ticks'] == identity['starttime_ticks'])


def signal_owned_group(identity, sig):
    if not same_group_leader(identity):
        raise RuntimeError('Refusing signal: child PGID/starttime identity no longer matches')
    os.killpg(identity['pgid'], sig)


def live_group_members(identity):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            item = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if item['pgid'] == identity['pgid'] and item['state'] not in ('Z', 'X'):
            members.append(item)
    return members


def live_session_members(identity):
    """Audit all live members of this session, including detached PGIDs.

    Do not signal these additional groups: their recorded identities allow the
    parent to review and clean them separately. The leader must remain unreaped
    throughout this snapshot, so its SID cannot be recycled during the audit.
    """
    if not same_group_leader(identity):
        raise RuntimeError('Cannot audit session: unreaped leader identity changed')
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            item = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if item['sid'] == identity['sid'] and item['state'] not in ('Z', 'X'):
            members.append(item)
    return sorted(members, key=lambda item: item['pid'])


def child_exited(child):
    # WNOWAIT preserves leader identity for safe cleanup even after early exit.
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def stop_owned_group(child, identity, grace=TERM_GRACE_SECONDS):
    record = {'started_unix': time.time(), 'identity': identity, 'signals': []}
    signal_owned_group(identity, signal.SIGTERM)
    record['signals'].append('SIGTERM')
    deadline = time.monotonic() + grace
    while live_group_members(identity) and time.monotonic() < deadline:
        time.sleep(0.5)
    remaining = live_group_members(identity)
    if remaining:
        signal_owned_group(identity, signal.SIGKILL)
        record['signals'].append('SIGKILL')
    # Wait without reaping: the leader's PID/SID identity must remain pinned
    # through the FINAL session audit, even when its own PGID is already empty.
    exit_deadline = time.monotonic() + 10
    exited = child_exited(child)
    while not exited and time.monotonic() < exit_deadline:
        time.sleep(0.1)
        exited = child_exited(child)
    record['session_audit_scope'] = (
        'All live processes with original SID, including other PGIDs; '
        'does not claim discovery of descendants that created a new SID')
    record['session_audit_before_leader_reap'] = True
    try:
        session_members = live_session_members(identity)
        record['remaining_live_session_members'] = session_members
        record['remaining_live_members'] = [
            item for item in session_members if item['pgid'] == identity['pgid']]
        record['remaining_live_other_pgid_members'] = [
            item for item in session_members if item['pgid'] != identity['pgid']]
        record['session_audit_complete'] = True
    except Exception as exc:
        record['session_audit_complete'] = False
        record['session_audit_error'] = repr(exc)
    # Reap only after the audit; no signals or identity-based scans after reap.
    record['exit_code'] = child.wait(timeout=0) if exited else None
    record['finished_unix'] = time.time()
    record['cleanup_complete'] = (record['session_audit_complete'] and exited and
                                  not record['remaining_live_session_members'])
    record['parent_cleanup_required'] = not record['cleanup_complete']
    return record


def http_get(url):
    # Fixed private/local endpoints; explicitly bypass proxy credentials/settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=1) as response:
        data = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(data) > 1024 * 1024:
            raise RuntimeError('Health response status/size invalid')
        return data


def probe_ready(rank):
    root = 'http://127.0.0.1:30000' if rank == 0 else 'http://10.66.9.163:30000'
    http_get(root + '/health')
    models = json.loads(http_get(root + '/v1/models'))
    if MODEL not in [item.get('id') for item in models.get('data', [])]:
        raise RuntimeError('Rank0 model identity missing')
    if rank == 1:
        http_get('http://127.0.0.1:30000/health')
    return {'rank0_health_http': 200, 'rank0_models': models,
            'rank1_local_dummy_health_http': 200 if rank else None}


class LogMarker:
    def __init__(self, path, marker):
        self.path, self.marker, self.offset, self.tail, self.seen = path, marker, 0, b'', False

    def scan(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            data = stream.read(1024 * 1024)
            self.offset += len(data)
        combined = self.tail + data
        self.seen |= self.marker.encode() in combined
        self.tail = combined[-len(self.marker.encode()):]
        return self.seen


def supervise(child, identity, run_dir, rank, stop_requested):
    started = time.monotonic()
    ready = False
    marker = LogMarker(run_dir / 'service.log', MARKERS[rank])
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    state = {'status': 'starting', 'pid_identity': identity, 'started_unix': time.time(),
             'ready': False, 'peer_cleanup_required': True}
    reason = 'supervisor_error'
    try:
        while True:
            available = memory_available()
            state.update(memory_available_bytes=available, updated_unix=time.time())
            write_json(run_dir / 'status.json', state)
            if stop_requested():
                reason = 'parent_stop_signal'
                break
            if available < 128 * GIB:
                reason = 'host_memory_below_128GiB'
                break
            if child_exited(child):
                reason = 'service_exited'
                break
            if not ready and time.monotonic() - started >= STARTUP_SECONDS:
                reason = 'startup_timeout_7200s'
                break
            if not ready and marker.scan():
                if pending is None:
                    pending = pool.submit(probe_ready, rank)
                elif pending.done():
                    try:
                        evidence = pending.result()
                    except Exception as exc:
                        state['last_readiness_error'] = repr(exc)
                        pending = None
                    else:
                        evidence.update(log_marker=MARKERS[rank], readiness_source_sha256=READINESS_PINS,
                                        ready_unix=time.time(), startup_seconds=time.monotonic() - started,
                                        meaning='HTTP and local initialization ready; NOT model quality or performance qualification')
                        write_json(run_dir / 'ready.json', evidence, exclusive=True)
                        ready = True
                        state.update(status='ready', ready=True)
                        write_json(run_dir / 'status.json', state)
            # Signals interrupt this wait through the event predicate within 0.2s.
            deadline = time.monotonic() + POLL_SECONDS
            while time.monotonic() < deadline and not stop_requested():
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        state.update(status='stopping', stop_reason=reason, updated_unix=time.time())
        # Disk/log errors must never bypass resource cleanup.
        try:
            write_json(run_dir / 'status.json', state)
        except Exception as exc:
            state['status_write_error'] = repr(exc)
        try:
            cleanup = stop_owned_group(child, identity)
            write_json(run_dir / 'cleanup.json', cleanup, exclusive=True)
            state.update(status='stopped' if cleanup['cleanup_complete'] else 'cleanup_incomplete',
                         exit_code=cleanup['exit_code'], cleanup_complete=cleanup['cleanup_complete'])
        except Exception as exc:
            state.update(status='cleanup_failed', cleanup_error=repr(exc), cleanup_complete=False)
        write_json(run_dir / 'status.json', state)
    return state


def execute(args, planner):
    require_container(args.node_rank, args.root)
    config = bundle(planner, args.root, args.node_rank)
    local = json.loads((args.run_dir / 'receipt.json').read_text())
    peer = json.loads(args.peer_receipt.read_text())
    validate_receipts(local, peer, args.node_rank, args.cohort_id, time.time(), source_hashes(), config)
    if local['run_dir'] != str(args.run_dir.resolve()):
        raise RuntimeError('Preparation run directory differs')
    if sha(args.run_dir / 'cpu-inspection.json') != local['inspection_sha256']:
        raise RuntimeError('CPU inspection receipt changed')
    if sha(args.run_dir / 'gpu-preflight.xml') != local['gpu_snapshot_sha256']:
        raise RuntimeError('GPU snapshot changed')
    parse_idle_gpus((args.run_dir / 'gpu-preflight.xml').read_text())
    check_readiness_sources(args.root)
    memory = require_memory(850)
    # Exclusive lock prevents retry/reuse, even after a failed launch.
    write_json(args.run_dir / 'execution-authorized.json', {
        'time_unix': time.time(), 'cohort_id': args.cohort_id, 'explicit_execute': True,
        'peer_receipt_sha256': sha(args.peer_receipt), 'parent_coordinates_peer_cleanup': True,
        'memory_before_bytes': memory, 'executor_identity': process_identity(os.getpid()),
        'source_sha256': source_hashes(), 'plan_sha256': digest(config)}, exclusive=True)
    stopping = [False]
    def request_stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (args.run_dir / 'service.log').open('xb') as log:
        child = subprocess.Popen(config['command'], env=child_environment(config), cwd=args.root,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        identity = process_identity(child.pid)
        if not identity['pid'] == identity['pgid'] == identity['sid']:
            raise RuntimeError('Child did not acquire its own process group and session')
        try:
            write_json(args.run_dir / 'child.json', identity, exclusive=True)
            return supervise(child, identity, args.run_dir, args.node_rank, lambda: stopping[0])
        except BaseException:
            # Includes failure recording child identity before supervisor entry.
            # supervise normally cleans/reaps first, in which case no signal.
            if same_group_leader(identity):
                stop_owned_group(child, identity)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cohort-id', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--peer-receipt', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', args.cohort_id):
        parser.error('cohort-id must use 1..100 letters, digits, underscores or hyphens')
    if not args.run_dir.is_absolute() or args.root not in args.run_dir.parents:
        parser.error('run-dir must be an absolute child of the experiment root')
    if args.execute and not args.peer_receipt:
        parser.error('--execute requires --peer-receipt')
    planner = load_planner()
    if args.prepare:
        result = prepare(args, planner)
    elif args.execute:
        result = execute(args, planner)
    else:
        result = {'status': 'plan_only_NOT_executed', 'plan': bundle(planner, args.root, args.node_rank),
                  'limits': {'receipt_ttl_seconds': RECEIPT_TTL, 'preflight_memory_gib': 850,
                             'runtime_memory_floor_gib': 128, 'memory_poll_seconds': POLL_SECONDS,
                             'startup_seconds': STARTUP_SECONDS, 'term_grace_seconds': TERM_GRACE_SECONDS},
                  'readiness_log_markers': MARKERS, 'readiness_source_sha256': READINESS_PINS,
                  'cross_node_coordination': 'Parent must stop peer on any failure; no remote actions'}
    print(json.dumps(result, indent=2))
    if args.execute and (result['stop_reason'] != 'parent_stop_signal' or not result.get('cleanup_complete')):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
