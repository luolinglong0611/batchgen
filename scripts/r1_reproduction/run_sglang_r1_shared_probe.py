#!/usr/bin/env python3
"""Bound one owner-approved torchrun and aggregate two closed node receipts.

Default plan-only. No imports of Torch/SG. Only --execute launch starts the
explicit shared-layer probe; no server/model launch, installation or old-PID
control is present. Existing external GPU/RSS guards remain required.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


PROBE = Path(__file__).with_name('qualify_sglang_r1_local_shared.py').resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_module():
    spec = importlib.util.spec_from_file_location('r1_bounded_shared_probe', PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACKING_METHOD = 'linux_proc_ppid_start_ticks_subreaper_pidfd_v1'


def parse_proc_stat(pid, text):
    """Parse /proc stat after comm's last ')'; comm may itself contain ')'."""
    if text.find('(') <= 0 or text.rfind(')') < text.find('('):
        raise ValueError('Malformed /proc process command field')
    fields = text[text.rfind(')') + 2:].split()
    if len(fields) < 20 or int(text.split(' ', 1)[0]) != pid:
        raise ValueError('Malformed /proc process identity')
    return {'pid': pid, 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'sid': int(fields[3]), 'start_ticks': int(fields[19]), 'state': fields[0]}


def read_proc(pid):
    return parse_proc_stat(pid, Path(f'/proc/{pid}/stat').read_text())


def proc_snapshot(root=Path('/proc')):
    """Unreadable existing identities fail closed; exit races may disappear."""
    result = {}
    for entry in root.iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            result[pid] = parse_proc_stat(pid, (entry / 'stat').read_text())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def identity(record):
    return record['pid'], record['start_ticks']


def discover_owned(snapshot, known, owner_pid):
    """Follow live PPid identities, including new SIDs and adopted orphans.

    owner_pid is an exclusive, single-threaded subreaper with no pre-existing
    children. Therefore a new direct child after the one Popen is its adopted
    descendant; a process outside this ancestry is never admitted by SID/PGID.
    """
    owned = {pid for pid, value in snapshot.items() if identity(value) in known}
    owned.update(pid for pid, value in snapshot.items() if value['ppid'] == owner_pid)
    while True:
        expanded = owned | {pid for pid, value in snapshot.items() if value['ppid'] in owned}
        if expanded == owned:
            return {identity(snapshot[pid]): snapshot[pid] for pid in owned}
        owned = expanded


def require_linux_tracking():
    if (sys.platform != 'linux' or not Path('/proc/self/stat').is_file()
            or not callable(getattr(os, 'pidfd_open', None))
            or not callable(getattr(signal, 'pidfd_send_signal', None))):
        raise RuntimeError('Execution requires Linux /proc, pidfd_open and pidfd_send_signal; no killpg fallback')
    if threading.active_count() != 1 or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Execution requires an exclusive single-threaded launcher')


def subreaper_mode(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER')
    if value is not None and libc.prctl(36, int(value), 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    return current.value


class OwnedTree:
    def __init__(self, owner_pid):
        self.owner_pid, self.known, self.fds = owner_pid, {}, {}

    def observe(self, snapshot):
        current = discover_owned(snapshot, self.known, self.owner_pid)
        for key, value in current.items():
            self.known.setdefault(key, dict(value, initial_sid=value['sid']))
            self.known[key]['last_state'] = value['state']
            self.known[key]['sid'] = value['sid']
            if key in self.fds or value['state'] in ('Z', 'X'):
                continue
            try:
                fd = os.pidfd_open(value['pid'], 0)
            except ProcessLookupError:
                continue
            try:
                confirmed = read_proc(value['pid'])
                if identity(confirmed) != key:
                    raise RuntimeError('Process identity changed while binding its pidfd')
            except (FileNotFoundError, ProcessLookupError):
                os.close(fd)
                continue
            except BaseException:
                os.close(fd)
                raise
            self.fds[key] = fd
        return current

    def send(self, key, sig, snapshot):
        """Never signal a bare PID, changed identity, SID or process group."""
        try:
            value = read_proc(key[0])
        except (FileNotFoundError, ProcessLookupError):
            return False
        if identity(value) != key or value['state'] in ('Z', 'X'):
            return False
        if key not in self.known or key not in self.fds:
            raise RuntimeError('Cannot signal an unbound owned descendant')
        self.known[key]['sid'] = value['sid']
        try:
            signal.pidfd_send_signal(self.fds[key], sig, None, 0)
            return True
        except ProcessLookupError:
            return False

    def reap_adopted(self, snapshot, leader_pid):
        for value in snapshot.values():
            if (value['pid'] != leader_pid and identity(value) in self.known
                    and value['ppid'] == self.owner_pid and value['state'] in ('Z', 'X')):
                try:
                    os.waitpid(value['pid'], os.WNOHANG)
                except ChildProcessError:
                    pass

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def run_bounded(argv, cwd, log, wall_seconds, grace_seconds=10):
    """Bound the exact Popen tree across new SIDs; uncertainty is unclosed.

    Linux subreaper adoption retains ancestry when torchrun/rank leaders exit
    between scans. pidfds bind signal targets to their observed start identity.
    The outer owner still enforces GPU/RSS caps and the independent 900s guard.
    """
    if (not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 840
            or not math.isfinite(grace_seconds) or not 0 < grace_seconds <= 10):
        raise ValueError('Finite wall<=840s and cleanup grace<=10s required')
    require_linux_tracking()
    owner_pid = os.getpid()
    initial = proc_snapshot()
    if any(row['ppid'] == owner_pid for row in initial.values()):
        raise RuntimeError('Dedicated launcher must have no pre-existing direct children')
    previous_subreaper = subreaper_mode(1)
    started, tree, child = time.monotonic(), OwnedTree(owner_pid), None
    result = {'argv': argv, 'wall_seconds': wall_seconds, 'timed_out': False,
              'signals': [], 'signal_records': [], 'termination_requested': False,
              'tracking_method': TRACKING_METHOD, 'closed': False, 'clean_exit': False,
              'closure_errors': [], 'initial_direct_children': [], 'subreaper_enabled': True}
    requested_signals = []
    def request_stop(sig, _frame):
        requested_signals.append(signal.Signals(sig).name)
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with Path(log).open('xb') as output:
            child = subprocess.Popen(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            result.update(pid=child.pid, owned_session_id=child.pid)
            first = read_proc(child.pid)
            if first['ppid'] != owner_pid or first['sid'] != child.pid:
                raise RuntimeError('New Popen leader does not have the promised parent/session identity')
            tree.observe({child.pid: first})
            phase, next_stage, empty_scans, sent = 'running', None, 0, set()
            deadline = started + wall_seconds
            cleanup_limit = deadline + 3 * grace_seconds + 2
            leader_key = identity(first)
            while time.monotonic() < cleanup_limit:
                now = time.monotonic()
                try:
                    snapshot = proc_snapshot()
                    current = tree.observe(snapshot)
                    if leader_key is None and child.pid in snapshot:
                        leader_key = identity(snapshot[child.pid])
                        if leader_key not in tree.known:
                            raise RuntimeError('New Popen leader ancestry was not observed')
                    tree.reap_adopted(snapshot, child.pid)
                except BaseException as error:
                    result['closure_errors'].append(repr(error))
                    # Known pidfds remain safe, but missing scans can never
                    # establish complete closure. Retry bounded cleanup only.
                    snapshot, current = {}, {}
                returncode = child.poll()
                absent = not any(identity(row) in tree.known for row in snapshot.values())
                if (returncode is not None and leader_key is not None and absent
                        and not current and not result['closure_errors']):
                    empty_scans += 1
                    if empty_scans >= 2:
                        result.update(closed=True, stable_empty_scans=empty_scans)
                        break
                else:
                    empty_scans = 0
                if phase == 'running' and (now >= deadline or result['closure_errors'] or requested_signals):
                    result['timed_out'] = now >= deadline
                    result['termination_requested'] = True
                    phase, next_stage = 'leader_TERM', now + grace_seconds
                    cleanup_limit = now + 3 * grace_seconds + 2
                targets = []
                if phase == 'leader_TERM':
                    if leader_key is not None:
                        targets = [(leader_key, signal.SIGTERM)]
                    if now >= next_stage:
                        phase, next_stage = 'descendant_TERM', now + grace_seconds
                if phase == 'descendant_TERM':
                    targets = [(key, signal.SIGTERM) for key in tree.known]
                    if leader_key is not None:
                        targets.append((leader_key, signal.SIGKILL))
                    if now >= next_stage:
                        phase, next_stage = 'descendant_KILL', now + grace_seconds
                if phase == 'descendant_KILL':
                    targets = [(key, signal.SIGKILL) for key in tree.known]
                for key, sig in targets:
                    if (key, sig) in sent:
                        continue
                    try:
                        if tree.send(key, sig, snapshot):
                            sent.add((key, sig))
                            result['signals'].append(sig.name)
                            result['signal_records'].append({'pid': key[0], 'start_ticks': key[1],
                                'sid': tree.known[key]['sid'], 'signal': sig.name, 'phase': phase})
                    except BaseException as error:
                        result['closure_errors'].append(repr(error))
                time.sleep(0.05)
            result['returncode'] = child.poll()
            result['known_owned_identities'] = [value for _, value in sorted(tree.known.items())]
            result['new_session_descendants'] = [value for key, value in sorted(tree.known.items())
                if key != leader_key and value['sid'] != child.pid]
            result['all_owned_identities_absent'] = result['closed']
            result['leader_reaped'] = result['returncode'] is not None
            result['external_stop_signals'] = requested_signals
            result['clean_exit'] = (result['closed'] and result['returncode'] == 0
                                    and not result['termination_requested'])
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        tree.close()
        subreaper_mode(previous_subreaper)
    return dict(result, elapsed_seconds=time.monotonic() - started)


def launch(args):
    world = args.nnodes * args.nproc_per_node
    if world not in (2, 4, 16) or not (0 <= args.node_rank < args.nnodes):
        raise ValueError('Only 2/4-rank local prechecks or the full 2x8 layout are supported')
    if world == 16 and (args.nnodes, args.nproc_per_node) != (2, 8):
        raise ValueError('Final layout requires two nodes with eight ranks each')
    if world != 16 and args.nnodes != 1:
        raise ValueError('A partial precheck must be local to one node')
    if not 0 < args.wall_seconds <= 840:
        raise ValueError('Inner wall limit must be positive and <=840s, inside the outer900s guard')
    extra = args.probe_args[1:] if args.probe_args[:1] == ['--'] else args.probe_args
    reserved = ('--execute', '--world-size', '--output-dir', '--aggregate')
    if any(value.split('=', 1)[0] in reserved for value in extra):
        raise ValueError('Shared probe launch mode/world/output are owned by this bounded launcher')
    allowed = {'--allowed-hostname', '--layer', '--adapter-source-sha256', '--weights-bundle',
               '--weights-sha256', '--tensor-manifest', '--tensor-manifest-sha256',
               '--numerical-contract', '--numerical-contract-sha256'}
    if any(value.startswith('--') and value.split('=', 1)[0] not in allowed for value in extra):
        raise ValueError('Only explicit shared-probe source/weight/host options may be forwarded')
    argv = [str(args.python), '-m', 'torch.distributed.run', f'--nnodes={args.nnodes}',
            f'--nproc-per-node={args.nproc_per_node}', f'--node-rank={args.node_rank}',
            f'--master-addr={args.master_addr}', f'--master-port={args.master_port}',
            '--max-restarts=0', '--monitor-interval=1', str(PROBE), '--execute',
            f'--world-size={world}', f'--output-dir={args.job_dir / "ranks"}', *extra]
    receipt = {'schema': 'r1-bounded-shared-probe-launch-v1', 'status': 'plan_only_NOT_executed',
               'nnodes': args.nnodes, 'nproc_per_node': args.nproc_per_node,
               'node_rank': args.node_rank, 'world_size': world,
               'hostname': socket.gethostname(), 'source_sha256': sha(__file__),
               'probe_sha256': sha(PROBE), 'argv': argv, 'external_GPU_RSS_guard_required': True,
               'formal_test_permitted': False}
    if not args.execute:
        return receipt
    if os.environ.get('R1_SG_SHARED_GUARDED') != '1' or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Explicit parent guard and exact hostname required')
    require_linux_tracking()
    if not args.job_dir.is_absolute() or not args.repo_root.is_absolute() or not args.python.is_absolute():
        raise ValueError('Execution paths must be absolute')
    args.job_dir.mkdir(parents=True, exist_ok=False)
    (args.job_dir / 'ranks').mkdir()
    (args.job_dir / 'launch-plan.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result = run_bounded(argv, args.repo_root, args.job_dir / 'torchrun.log', args.wall_seconds)
    expected = list(range(args.node_rank * args.nproc_per_node, (args.node_rank + 1) * args.nproc_per_node))
    records, errors = {}, []
    for rank in expected:
        path = args.job_dir / f'ranks/rank-{rank}.json'
        try:
            value = json.loads(path.read_text())
            if (value.get('rank') != rank or value.get('world_size') != world
                    or value.get('status') != 'passed_real_shared_branch_NOT_full_model_qualified'
                    or value.get('source_sha256') != sha(PROBE)):
                raise ValueError('Rank did not pass with the frozen source')
            records[str(rank)] = sha(path)
        except (OSError, ValueError) as error:
            errors.append({'rank': rank, 'error': str(error)})
    passed = result['closed'] and result['clean_exit'] and not result['timed_out'] and result['returncode'] == 0 and not errors
    receipt.update(result, status='closed_passed_local_rank_set_NOT_final16' if passed else 'failed_or_unclosed_local_rank_set',
                   expected_ranks=expected, rank_result_sha256=records, rank_errors=errors,
                   final16_layout_qualified=False)
    (args.job_dir / 'launcher-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def aggregate(job_dirs):
    if len(job_dirs) != 2:
        raise ValueError('Exactly two closed two-node launcher receipts are required')
    probe, records, nodes, receipts = probe_module(), [], set(), []
    for directory in job_dirs:
        path = directory / 'launcher-result.json'
        launch_receipt = json.loads(path.read_text())
        if (launch_receipt.get('status') != 'closed_passed_local_rank_set_NOT_final16'
                or not launch_receipt.get('closed') or launch_receipt.get('timed_out')
                or launch_receipt.get('clean_exit') is not True
                or launch_receipt.get('tracking_method') != TRACKING_METHOD
                or launch_receipt.get('all_owned_identities_absent') is not True
                or launch_receipt.get('leader_reaped') is not True
                or launch_receipt.get('stable_empty_scans', 0) < 2
                or launch_receipt.get('closure_errors') != []
                or launch_receipt.get('returncode') != 0 or launch_receipt.get('probe_sha256') != sha(PROBE)
                or launch_receipt.get('source_sha256') != sha(__file__)
                or (launch_receipt.get('nnodes'), launch_receipt.get('nproc_per_node'), launch_receipt.get('world_size')) != (2, 8, 16)):
            raise ValueError('Launcher did not close successfully with the frozen two-node source')
        node = launch_receipt['node_rank']
        if node not in (0, 1) or node in nodes:
            raise ValueError('Missing/duplicate real node rank')
        nodes.add(node)
        expected = list(range(node * 8, (node + 1) * 8))
        if launch_receipt.get('expected_ranks') != expected or set(launch_receipt['rank_result_sha256']) != {str(r) for r in expected}:
            raise ValueError('Launcher lacks its complete eight-rank closeout')
        for rank in expected:
            rank_path = directory / f'ranks/rank-{rank}.json'
            if sha(rank_path) != launch_receipt['rank_result_sha256'][str(rank)]:
                raise ValueError('Closed rank receipt changed after launcher exit')
            records.append(json.loads(rank_path.read_text()))
        receipts.append({'path': str(path), 'sha256': sha(path)})
    return dict(probe.aggregate_rank_records(records, 16), all16_rank_receipts_closed=True,
                both_torchrun_launchers_closed=True, launcher_receipts=receipts,
                aggregate_source_sha256=sha(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('launch')
    run.add_argument('--execute', action='store_true')
    run.add_argument('--python', type=Path, required=True)
    run.add_argument('--repo-root', type=Path, required=True)
    run.add_argument('--job-dir', type=Path, required=True)
    run.add_argument('--expected-hostname', required=True)
    run.add_argument('--nnodes', type=int, choices=(1, 2), default=2)
    run.add_argument('--nproc-per-node', type=int, choices=(2, 4, 8), default=8)
    run.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    run.add_argument('--master-addr', required=True)
    run.add_argument('--master-port', type=int, required=True)
    run.add_argument('--wall-seconds', type=float, default=840)
    run.add_argument('probe_args', nargs=argparse.REMAINDER)
    close = sub.add_parser('aggregate')
    close.add_argument('--job-dir', type=Path, action='append', required=True)
    close.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'launch':
        result = launch(args)
    else:
        result = aggregate(args.job_dir)
        with args.output.open('x') as output:
            output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] != 'failed_or_unclosed_local_rank_set' else 1


if __name__ == '__main__':
    raise SystemExit(main())
