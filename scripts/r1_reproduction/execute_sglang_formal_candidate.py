#!/usr/bin/env python3
"""Fixed-cohort SGLang A/B admission adapter; default is local plan-only.

This consumes an exact, reconstructed formal candidate. It does not generate
requests or qualify model quality/capacity/performance. Prepare and execute
require separate parent-owned allocation of BOTH nodes. The mature diagnostic
supervisor supplies WNOWAIT, memory monitoring and owned-session cleanup.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import time
import types

HERE = Path(__file__).resolve().parent
PINS = {
    'execute_sglang_diagnostic.py': '9b7da876347ed9f6defeedc3bf5b3d3c54574dfc1bd951f2713890b537ce7d19',
    'plan_sglang_formal_candidate.py': '20ed116f4591a55712b2660f21eec7d05b5f8fc841660797d26a6d579636548e',
    'launch_sglang_diagnostic.py': '86f4af26428fe98a2aafd43dceb53d655225ad1a6d818569aaca05a525bc102f',
    'r1_launch_safety.py': '3b06a6644b22d53343f89427834553c462fba0ee44f3ebf8a3cde3c5bccfe130',
    'sglang_diagnostic_source_pins.json': '019e1524ad03c45455620fa425cb707054baa866ffeb6d77bc6936d1c34dff5f',
    'r1_model_manifest.json': 'd9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253',
    'patch_sglang_r1_routing.py': '274db85e1f0bebbad43c0456e1c96ddb7036364aa8854788c5d30626191e47ca',
}
FORMAL_SCHEMA = 'r1-sglang-formal-execution-v1'
SHA_RE = re.compile(r'[0-9a-f]{64}')


def stable_read(path, expected=None, maximum=4 * 1024**2):
    """Bounded regular-file read; reject symlinks, replacement and mutation."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f'Not a bounded regular file: {path}')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(maximum + 1)
        after, current = os.fstat(fd), path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (len(data) > maximum or identity(before) != identity(after)
                or identity(after) != identity(current) or path.is_symlink()):
            raise ValueError(f'Input changed during read: {path}')
    finally:
        os.close(fd)
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and (not SHA_RE.fullmatch(expected) or actual != expected):
        raise ValueError(f'SHA mismatch: {path}')
    return data, actual


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def pinned_module(name):
    """Execute verified source bytes, never a stale pyc or shared module."""
    path = HERE / name
    data, _ = stable_read(path, PINS[name])
    module = types.ModuleType('_formal_owned_' + path.stem)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def check_ports(ports):
    """Point-in-time bind test, not a reservation; parent owns allocation."""
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('0.0.0.0', port))
    return {'ports': ports, 'status': 'available_at_check_NOT_reserved'}


class FormalContext:
    def __init__(self, args):
        self.args = args
        self.base = pinned_module('execute_sglang_diagnostic.py')
        self.formal = pinned_module('plan_sglang_formal_candidate.py')
        self.provider = pinned_module('launch_sglang_diagnostic.py')
        self.safety = pinned_module('r1_launch_safety.py')
        self.paths = {name: (HERE / name, expected) for name, expected in PINS.items()}
        _, own_sha = stable_read(__file__)
        self.paths['execute_sglang_formal_candidate.py'] = (Path(__file__), own_sha)
        raw, candidate_sha = stable_read(args.candidate_plan, args.expected_plan_sha256)
        self.paths['candidate_plan_raw'] = (args.candidate_plan, candidate_sha)
        receipt, _ = stable_read(args.source_audit / 'COMBINED_SOURCE_RECEIPT.json', self.formal.RECEIPT_SHA)
        audit = strict_json(receipt)
        self.paths['source-audit/COMBINED_SOURCE_RECEIPT.json'] = (
            args.source_audit / 'COMBINED_SOURCE_RECEIPT.json', self.formal.RECEIPT_SHA)
        for name, expected in self.formal.SOURCE_PINS.items():
            stable_read(args.source_audit / name, expected)
            if audit['files'][name]['sha256'] != expected:
                raise ValueError(f'Source audit receipt mismatch: {name}')
            self.paths['source-audit/' + name] = (args.source_audit / name, expected)
        help_text = stable_read(args.source_audit / 'launch-server-help.stdout',
                                self.formal.SOURCE_PINS['launch-server-help.stdout'])[0].decode()
        candidate = strict_json(raw)
        expected = self.formal.build_candidate(self.provider, help_text, candidate['topology'],
            candidate['nodes'][0]['effective_per_dp_rank_running_requests_target'],
            candidate['nodes'][0]['effective_per_dp_rank_prefill_chunk_tokens'],
            candidate['kv_policy']['memory_fraction_static'],
            candidate['cuda_graph_policy']['mode'],
            candidate['cuda_graph_policy']['max_bs_per_dp_rank'])
        # Canonical bytes also distinguish JSON true/1 and 16/16.0.
        if self.base.digest(candidate) != self.base.digest(expected):
            raise ValueError('Whole candidate differs from pinned planner reconstruction')
        self.candidate = candidate
        self.sources = {name: expected for name, (_, expected) in self.paths.items()}
        self.binding = {'schema': FORMAL_SCHEMA, 'candidate_raw_sha256': candidate_sha,
            'candidate_id': candidate['candidate_id'], 'topology': candidate['topology'],
            'cohort_id': args.cohort_id, 'source_sha256': self.sources}
        self.configs = [{
            'formal_receipt_schema': FORMAL_SCHEMA,
            'formal_binding': self.binding, 'physical_node': rank,
            'physical_node_role': ('distributed_root' if rank == 0 else 'distributed_peer')
                if candidate['topology'] == 'A' else 'independent_full_server',
            'environment_policy': 'replace_with_exact_allowlisted_environment',
            'environment': node['environment'], 'unset_environment': [], 'command': node['argv'],
        } for rank, node in enumerate(candidate['nodes'])]
        self.configure_leaves()
        self.current_sources()

    def current_sources(self):
        for path, expected in self.paths.values():
            stable_read(path, expected)
        return {**self.sources, 'formal_global_binding': self.base.digest(self.binding)}

    def configure_leaves(self):
        e = self.base
        writer, validate, merge = e.write_json, e.validate_receipts, e.child_environment
        original_probe = e.probe_ready
        def write(path, value, exclusive=False):
            value.update(formal_receipt_schema=FORMAL_SCHEMA, formal_binding=self.binding,
                         physical_node=self.args.node_rank,
                         physical_node_role=self.configs[self.args.node_rank]['physical_node_role'])
            if Path(path).name == 'ready.json':
                value['meaning'] = 'HTTP/local initialization ready; NOT quality, capacity or performance qualification'
            writer(path, value, exclusive)
        def validate_formal(local, peer, rank, cohort, now, sources, config):
            for item, physical in ((local, rank), (peer, 1 - rank)):
                if (item.get('formal_receipt_schema') != FORMAL_SCHEMA
                        or item.get('formal_binding') != self.binding
                        or item.get('physical_node') != physical
                        or item.get('physical_node_role') != self.configs[physical]['physical_node_role']
                        or item.get('plan_sha256') != e.digest(self.configs[physical])):
                    raise RuntimeError('Formal candidate/schema/peer plan binding mismatch')
                if (item.get('gpu_query_count') != 1
                        or item.get('memory_before_bytes', 0) < 850 * e.GIB
                        or item.get('memory_after_bytes', 0) < 850 * e.GIB):
                    raise RuntimeError('Formal preparation resource evidence incomplete')
                devices = item.get('gpus', [])
                ids = [g.get('uuid') for g in devices]
                if (len(ids) != 8 or len(set(ids)) != 8 or any(not x for x in ids)
                        or any(type(g.get('memory_used_mib')) is not int
                               or not 0 <= g['memory_used_mib'] <= 64
                               or g.get('utilization_percent') != 0 for g in devices)):
                    raise RuntimeError('Formal idle GPU inventory invalid')
            if set(g['uuid'] for g in local['gpus']) & set(g['uuid'] for g in peer['gpus']):
                raise RuntimeError('Physical nodes must have distinct GPU UUID inventories')
            validate(local, peer, rank, cohort, now, sources, config)
        def probe(rank):
            if self.candidate['topology'] == 'A':
                return original_probe(rank)
            e.http_get('http://127.0.0.1:30000/health')
            models = strict_json(e.http_get('http://127.0.0.1:30000/v1/models'))
            if e.MODEL not in [item.get('id') for item in models.get('data', [])]:
                raise RuntimeError('Local independent R1 server model identity missing')
            return {'physical_node': rank, 'local_full_server_health_http': 200, 'local_models': models}
        e.bundle = lambda planner, root, rank: copy.deepcopy(self.configs[rank])
        e.source_hashes = self.current_sources
        e.write_json, e.validate_receipts = write, validate_formal
        e.child_environment = lambda config: merge(config, inherited={})
        e.probe_ready = probe
        if self.candidate['topology'] == 'B':
            e.MARKERS = (e.MARKERS[0], e.MARKERS[0])

    def safety_preflight(self, phase):
        env = dict(self.configs[self.args.node_rank]['environment'],
                   CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
        # Only this owned safety-module instance changes; no global subprocess patch.
        self.safety.subprocess = types.SimpleNamespace(DEVNULL=subprocess.DEVNULL,
            run=lambda *a, **kw: subprocess.run(*a, env=env, **kw))
        receipt = self.safety.prepare_launch_safety(self.args.root)
        self.base.write_json(self.args.run_dir / f'launch-safety-{phase}.json', receipt, exclusive=True)
        return receipt

    def inspect(self, root, rank, env, unset, command):
        self.current_sources()
        self.safety_preflight('prepare')
        self.base.write_json(self.args.run_dir / 'ports-prepare.json',
            check_ports(self.candidate['nodes'][rank]['reserved_known_ports_to_recheck']), exclusive=True)
        site = root / 'sglang-baseline-v8/baseline-env/lib/python3.11/site-packages'
        current = {name: stable_read(site / name, expected)[1]
                   for name, expected in self.formal.SOURCE_PINS.items() if name.endswith('.py')}
        # Provider otherwise merges ambient env. Removing ALL inherited keys makes
        # its metadata/help/ptxas children use precisely SG's own allowlist.
        result = self.provider.inspect(root, rank, env, list(os.environ), command)
        if result['launch_help_sha256'] != self.formal.SOURCE_PINS['launch-server-help.stdout']:
            raise RuntimeError('Actual hidden-device installed help differs from frozen audit')
        result['formal_installed_semantic_source_sha256'] = current
        result['inspection_environment_policy'] = 'exact_SG_allowlist_plus_CUDA_hidden'
        return result

    def capture_child_identity(self, child):
        identity = self.base.process_identity(child.pid)
        proc = Path(f'/proc/{child.pid}')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        identity.update(uid=proc.stat().st_uid, ppid=int(fields[1]))
        if (not identity['pid'] == identity['pgid'] == identity['sid']
                or identity['uid'] != os.getuid() or identity['ppid'] != os.getpid()):
            raise RuntimeError('Child UID/session/group ownership mismatch')
        return identity

    def execute(self):
        """Small launch adapter; supervision/stop/reap remain the pinned code."""
        e, args = self.base, self.args
        e.require_container(args.node_rank, args.root)
        config = e.bundle(self, args.root, args.node_rank)
        local = strict_json(stable_read(args.run_dir / 'receipt.json')[0])
        peer_raw, peer_sha = stable_read(args.peer_receipt, args.expected_peer_receipt_sha256)
        peer = strict_json(peer_raw)
        e.validate_receipts(local, peer, args.node_rank, args.cohort_id, time.time(),
                            self.current_sources(), config)
        if local['run_dir'] != str(args.run_dir.resolve()):
            raise RuntimeError('Preparation run directory differs')
        for filename, key in (('cpu-inspection.json', 'inspection_sha256'),
                              ('gpu-preflight.xml', 'gpu_snapshot_sha256')):
            stable_read(args.run_dir / filename, local[key])
        e.parse_idle_gpus(stable_read(args.run_dir / 'gpu-preflight.xml')[0].decode())
        e.check_readiness_sources(args.root)
        memory = e.require_memory(850)
        self.safety_preflight('execute')
        e.write_json(args.run_dir / 'ports-execute.json',
            check_ports(self.candidate['nodes'][args.node_rank]['reserved_known_ports_to_recheck']), exclusive=True)
        e.write_json(args.run_dir / 'execution-authorized.json', {
            'time_unix': time.time(), 'cohort_id': args.cohort_id, 'explicit_execute': True,
            'peer_receipt_sha256': peer_sha, 'parent_coordinates_peer_cleanup': True,
            'memory_before_bytes': memory, 'executor_identity': e.process_identity(os.getpid()),
            'executor_uid': os.getuid(), 'source_sha256': self.current_sources(),
            'plan_sha256': e.digest(config)}, exclusive=True)
        stopping = [False]
        def request_stop(signum, frame):
            stopping[0] = True
        previous = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        child, identity = None, None
        try:
            with (args.run_dir / 'service.log').open('xb') as log:
                self.current_sources()
                self.safety.assert_self_core_policy()
                if stopping[0]:
                    raise RuntimeError('Stop requested before Popen; no service started')
                child = subprocess.Popen(config['command'], env=e.child_environment(config), cwd=args.root,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                # Popen guarantees setsid before return. Capture fallback ownership
                # before any fallible identity receipt write, retaining WNOWAIT.
                identity = self.capture_child_identity(child)
                e.write_json(args.run_dir / 'child.json', identity, exclusive=True)
                return e.supervise(child, identity, args.run_dir, args.node_rank, lambda: stopping[0])
        except BaseException as original:
            # Popen.wait sets returncode on reap. Never scan or signal by PID
            # after the supervisor has reaped, even if its final disk write fails.
            if child is not None and child.returncode is None:
                try:
                    if identity is None:
                        # The unreaped direct child pins its PID. Retry capture
                        # after a transient first /proc failure, before signalling.
                        identity = e.process_identity(child.pid)
                    if not e.same_group_leader(identity):
                        raise RuntimeError('Unreaped child identity unavailable; parent cleanup required')
                    cleanup = e.stop_owned_group(child, identity)
                    e.write_json(args.run_dir / 'launch-error-cleanup.json', cleanup, exclusive=True)
                except BaseException as cleanup_error:
                    # Preserve the original launch error and report incomplete
                    # cleanup; never replace it with a secondary receipt failure.
                    original.add_note(f'Owned cleanup/receipt failed; parent review required: {cleanup_error!r}')
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--source-audit', type=Path,
        default=HERE.parents[1] / '.repro-private/baseline-recovery/formal-source-audit')
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--root', type=Path, default=Path('/home/admin/batchgen-r1-repro-20261023'))
    parser.add_argument('--cohort-id', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--peer-receipt', type=Path)
    parser.add_argument('--expected-peer-receipt-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', args.cohort_id):
        parser.error('Invalid fresh cohort id')
    if (args.root != Path('/home/admin/batchgen-r1-repro-20261023')
            or not args.run_dir.is_absolute() or args.root not in args.run_dir.parents):
        parser.error('Expected fixed experiment root and absolute child run directory')
    if args.execute and (not args.peer_receipt or not args.expected_peer_receipt_sha256):
        parser.error('--execute requires peer receipt AND its explicit SHA')
    # An ELF preload cannot be removed after this interpreter has started.
    if (args.prepare or args.execute) and os.environ.get('LD_PRELOAD'):
        parser.error('Start this executor in a clean environment without LD_PRELOAD')
    context = FormalContext(args)
    if args.prepare:
        result = context.base.prepare(args, context)
    elif args.execute:
        result = context.execute()
    else:
        result = {'status': 'plan_only_NOT_executed', 'formal_binding': context.binding,
            'plan': context.configs[args.node_rank],
            'expected_node_plan_sha256': [context.base.digest(x) for x in context.configs],
            'readiness_log_markers': context.base.MARKERS,
            'limits': {'receipt_ttl_seconds': 600, 'preflight_memory_gib': 850,
                       'runtime_memory_floor_gib': 128, 'startup_seconds': 7200},
            'not_admitted': ['model_quality', 'capacity', 'performance', 'workload_submission']}
    print(json.dumps(result, indent=2))
    return int(args.execute and (result.get('stop_reason') != 'parent_stop_signal'
                                 or not result.get('cleanup_complete')))


if __name__ == '__main__':
    raise SystemExit(main())
