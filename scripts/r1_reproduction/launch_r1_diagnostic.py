#!/usr/bin/env python3
"""Print a two-node R1 diagnostic command; --execute explicitly loads the model.

No torch import, CUDA context, filesystem changes, or remote operation in plan
mode. Execute mode validates all 163 converted shard receipts and file hashes
(or explicit unchanged-file evidence from a prior local launch), then uses the normal HTTP entry point (including its real preflight).
This is an eager correctness experiment, not a performance-qualified launcher.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys

MODEL = 'deepseek-ai/DeepSeek-R1'
REVISION = '20f2865710d161dc4a32c4e8c3d7c42c7b9852f8'
MANIFEST_SHA = 'd9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253'
HOSTS = ('serverless-010066009163.wm226', 'serverless-010066009166.wm226')
ROOT = Path('/home/admin/batchgen-r1-repro-20261023')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def plan(root, rank, run_id, attention_implementation='triton_precision'):
    if attention_implementation not in ('triton_precision', 'fa4_flashinfer'):
        raise ValueError('Only the two explicitly audited eager diagnostics are supported')
    toolkit = root / 'cuda130-packages/nvidia/cu13'
    env = {
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'CUDA_HOME': str(toolkit),
        'TRITON_PTXAS_PATH': str(toolkit / 'bin/ptxas'),
        'BATCHGEN_R1_SM103_EXPERIMENTAL': '1',
        'BATCHGEN_R1_SM103_ATTENTION': attention_implementation,
        'DISABLE_MMA_V5': '1', 'TORCH_CUDA_CHANNEL': 'cu130',
        'BATCHGEN_W8A16_DEQUANT': '0',
        'BATCHGEN_HOST_REGISTER_CHUNK_GIB': '16',
        'BATCHGEN_ENABLE_ALL_TO_ALL': '0', 'GLOO_SOCKET_IFNAME': 'eth0',
        'NCCL_SOCKET_IFNAME': 'eth0', 'NCCL_IB_GID_INDEX': '3',
        # NCCL 2.27.7 RAS peers-update SIGSEGV observed before the first R1
        # request. Keep collectives/RDMA enabled; disable only RAS telemetry.
        'NCCL_RAS_ENABLE': '0',
        'NCCL_IB_HCA': '=' + ','.join(f'mlx5_bond_{i}:1' for i in range(8)),
        'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    }
    if attention_implementation == 'fa4_flashinfer':
        env['BATCHGEN_R1_SM103_FAST_DIAGNOSTIC'] = '1'
    command = [str(root / 'env/bin/python'), '-m', 'batchgen.launch_http_server',
               '--model', MODEL,
               '--cache-dir', '/mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8',
               '--converted-ckpt-dir', str(root / 'converted'),
               '--instance-id', run_id, '--runtime-mode', 'exclusive',
               '--listen-ip', '0.0.0.0', '--listen-port', '10900',
               '--world-size', '16', '--nnodes', '2', '--node-rank', str(rank),
               '--dist-init-addr', '10.66.9.163:34155',
               '--restart-control-addr', '10.66.9.163:34156',
               '--pynccl-port-base', '34200', '--pynccl-port-span', '100',
               '--kv-dtype', 'bfloat16', '--host-kv-cache-size', '64',
               '--input-ids-pool-size-gb', '1', '--gpu-memory-frac', '0.85',
               '--disable-cuda-graphs', '--storage-path', str(root / 'runs' / run_id / f'node{rank}' / 'storage'),
               '--save-result']
    return env, command


def validate_registration_qualification(root, rank, path):
    """Require local, completed native evidence, not a driver-only probe.

    No imports of native/PyTorch modules. Child reports are read explicitly so
    an incomplete or stale eight-worker summary cannot authorize model loading.
    """
    def require(condition, message):
        if not condition:
            raise RuntimeError('Registration qualification: ' + message)

    def read_report(report_path):
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                require(key not in obj, f'duplicate JSON field {key}')
                obj[key] = value
            return obj
        return json.loads(report_path.read_text(), object_pairs_hook=unique)

    def integer(value):
        return type(value) is int

    def sha256(value):
        return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None

    path = path.resolve(strict=True)
    data = read_report(path)
    require(data.get('status') == 'passed', 'owner report is not passed')
    require(data.get('hostname') == HOSTS[rank], 'report belongs to another node')
    require(data.get('registration_mode') == 'native', 'driver-only probes cannot qualify native storage')
    require(data.get('native_register_chunk_gib') == 16, 'expected native tensor-aware chunk size 16 GiB')
    require(data.get('reservation_bytes') == 675 * 1024**3, 'expected full 675 GiB reservation')
    require(data.get('worker_count') == 8, 'expected eight simultaneous workers')
    require(data.get('hugetlbfs') is False and data.get('thp') is False, 'page policy differs from launch')
    require(data.get('clean_native_release') is True, 'missing successful native unregister/munmap postvalidation')
    candidates = list((root / 'repo/batchgen').glob('core_engine*.so'))
    require(len(candidates) == 1, 'expected exactly one repository core_engine native library')
    core = candidates[0].resolve(strict=True)
    require(core.is_relative_to((root / 'repo/batchgen').resolve()), 'core library escapes repository')
    recorded_path = data.get('core_engine_path')
    require(isinstance(recorded_path, str) and Path(recorded_path).resolve() == core, 'native library path mismatch')
    core_sha = digest(core)
    require(data.get('core_engine_sha256') == core_sha, 'native library SHA256 differs from qualification')

    fixture = data.get('derived_fixture', {})
    require(fixture.get('kind') == 'shifted original tensor bytes with sparse zero prefix', 'missing shifted real-weight fixture')
    prefix, boundary = fixture.get('prefix_bytes'), fixture.get('nominal_boundary')
    start, size = fixture.get('tensor_start'), fixture.get('tensor_bytes')
    require(all(integer(v) for v in (prefix, boundary, start, size)), 'invalid shifted fixture extents')
    require(prefix > 0 and prefix % 4096 == 0 and boundary == 16 * 1024**3,
            'expected page-aligned shift across nominal 16 GiB cut')
    require(size > 4 * 1024**2 and 0 <= start < boundary < start + size <= 675 * 1024**3,
            'real tensor does not straddle the nominal cut')
    require(sha256(fixture.get('derived_metadata_sha256')), 'missing derived metadata hash')
    stem = 'model-00001-of-000163'
    receipt = read_report(root / 'converted.receipts' / (stem + '.safetensors.converted.json'))
    metadata_path = root / 'converted' / (stem + '.json')
    metadata = read_report(metadata_path)
    require(digest(metadata_path) == receipt['validation']['metadata_sha256'], 'first-shard metadata differs from receipt')
    require(fixture.get('copied_source_sha256') == receipt['validation']['binary_sha256'],
            'fixture is not from the verified original first shard')
    original = metadata['state_dict']
    crossing = fixture.get('crossing_tensor')
    require(crossing in original and len(original) == 270, 'unexpected first-shard tensor set')
    require(start == original[crossing]['offset'] + prefix and size == original[crossing]['byte_size'],
            'shifted crossing tensor disagrees with original metadata')
    derived = json.loads(json.dumps(metadata))
    for tensor in derived['state_dict'].values():
        tensor['offset'] += prefix
    derived['total_byte_size'] += prefix
    require(hashlib.sha256(json.dumps(derived).encode()).hexdigest() == fixture['derived_metadata_sha256'],
            'derived metadata does not match the declared shift of the verified first shard')
    workers = data.get('workers', [])
    require(isinstance(workers, list) and len(workers) == 8, 'missing worker summaries')
    child_hashes, seen_gpu, seen_pid, tensor_hashes = {}, set(), set(), None
    for summary in workers:
        require(summary.get('status') == 'passed' and summary.get('exit_code') == 0
                and summary.get('simultaneous_registration_barrier_passed') is True
                and summary.get('tensor_count') == 270, 'failed/incomplete worker summary')
        child_path = Path(summary['report']).resolve(strict=True)
        require(child_path.parent == path.parent, 'worker report is outside qualification directory')
        child = read_report(child_path)
        gpu, pid = child.get('physical_gpu'), child.get('pid')
        require(integer(gpu) and 0 <= gpu < 8 and gpu not in seen_gpu, 'duplicate/invalid physical GPU')
        require(child_path == path.with_suffix(f'.gpu{gpu}.json'), 'worker report filename mismatch')
        require(integer(pid) and pid > 0 and pid not in seen_pid and pid == summary.get('pid'), 'worker PID mismatch')
        seen_gpu.add(gpu)
        seen_pid.add(pid)
        require(child.get('status') == 'passed' and child.get('simultaneous_registration_barrier_passed') is True
                and child.get('tensor_count') == 270, 'child did not complete the native barrier and DMA')
        require(child.get('core_engine') == recorded_path, 'child used another native library path')
        require(child.get('core_engine_sha256') == core_sha, 'child native library SHA256 mismatch')
        tensors = child.get('tensors', [])
        require(len(tensors) == 270 and {t.get('name') for t in tensors} == set(original), 'child tensor evidence incomplete')
        hashes = {}
        for tensor in tensors:
            meta = original[tensor['name']]
            require(tensor.get('host_registered') is True and tensor.get('host_exact') is True
                    and tensor.get('roundtrip_exact') is True, 'tensor registration or byte comparison failed')
            require(tensor.get('bytes') == meta['byte_size'] and tensor.get('shape') == meta['shape']
                    and tensor.get('dtype') == meta['dtype'] and sha256(tensor.get('sha256')), 'tensor evidence differs from metadata')
            hashes[tensor['name']] = tensor['sha256']
        require(tensor_hashes is None or hashes == tensor_hashes, 'workers copied different tensor bytes')
        tensor_hashes = hashes
        child_hashes[child_path.name] = digest(child_path)
    require(seen_gpu == set(range(8)), 'not all eight local GPUs were qualified')
    return {'report': str(path), 'report_sha256': digest(path), 'worker_report_sha256': child_hashes,
            'core_engine_path': str(core), 'core_engine_sha256': core_sha,
            'native_register_chunk_gib': 16, 'reservation_bytes': 675 * 1024**3,
            'worker_count': 8, 'clean_native_release': True, 'derived_fixture': fixture}


def file_identity(path):
    """Local regular-file identity; access time is deliberately not included."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f'Missing/nonregular converted artifact: {path}')
    return {'path': str(path.absolute()), 'dev': info.st_dev, 'ino': info.st_ino,
            'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def stable_digest(path):
    before = file_identity(path)
    # Hash an opened regular file, never follow a last-component symlink swapped
    # in after lstat. Verify both the fd and pathname still identify that file.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                before['dev'], before['ino'], before['size'], before['mtime_ns'], before['ctime_ns']):
            raise RuntimeError(f'File changed before SHA256 read: {path}')
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
        after = os.fstat(stream.fileno())
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                before['size'], before['mtime_ns'], before['ctime_ns']):
            raise RuntimeError(f'File changed during SHA256 read: {path}')
    if file_identity(path) != before:
        raise RuntimeError(f'File changed during SHA256 read: {path}')
    return h.hexdigest(), before


def load_weights_evidence(root, rank, path, entries, converter_sha):
    """Accept a prior local launcher record, never a standalone checksum manifest.

    This is an audit cache on a trusted local filesystem, not an authentication
    scheme against privileged edits to reports, clocks or filesystem metadata.
    """
    if path.is_symlink():
        raise RuntimeError('Prior launch report must not be a symlink')
    path = path.resolve(strict=True)
    runs = (root / 'runs').resolve()
    if (not path.is_relative_to(runs) or len(path.relative_to(runs).parts) != 3
            or path.name != 'launch.json' or path.parent.name != f'node{rank}'):
        raise RuntimeError('Prior weight evidence must be a local runs/<run-id>/nodeN/launch.json')
    before = file_identity(path)
    payload = path.read_bytes()
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise RuntimeError(f'Duplicate prior report field: {key}')
            obj[key] = value
        return obj
    data = json.loads(payload, object_pairs_hook=unique)
    if file_identity(path) != before:
        raise RuntimeError('Prior launch report changed while reading')
    if data.get('hostname') != HOSTS[rank] or data.get('status') not in ('exited', 'failed'):
        raise RuntimeError('Prior launch report must belong to this node and have terminated')
    cmd = data.get('command', [])
    prior_implementation = data.get('environment', {}).get('BATCHGEN_R1_SM103_ATTENTION')
    env, expected = plan(root, rank, path.parent.parent.name, prior_implementation)
    legacy_ras_plan = (data.get('launcher_source_sha256', {}).get('launch_r1_diagnostic.py')
                       == 'aec80861e8bbdc2dafb81acad264148f652b1e28cb4b989da87ea44acdfeb12d'
                       and 'NCCL_RAS_ENABLE' not in data.get('environment', {}))
    if legacy_ras_plan:
        # The frozen pre-workaround launcher still proves unchanged weight
        # identities. Accept precisely its old plan; never replay its settings.
        env.pop('NCCL_RAS_ENABLE')
    if cmd != expected or data.get('environment') != env:
        raise RuntimeError('Prior report does not match this diagnostic launch command/environment')
    if data.get('source_sha256', {}).get('batchgen/ckpt_converter/ckpt_converter.py') != converter_sha:
        raise RuntimeError('Prior converter source differs; omit reuse and perform full validation')
    started = datetime.fromisoformat(data['utc'])
    if started.tzinfo is None or started > datetime.now(timezone.utc):
        raise RuntimeError('Prior launch UTC is missing timezone or is in the future')
    delta = started.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    started_ns = (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000
    records = data.get('weights', [])
    if (not isinstance(records, list) or len(records) != 163
            or {r.get('shard') for r in records} != {e['Path'] for e in entries}):
        raise RuntimeError('Prior report lacks all 163 validated shards')
    evidence = {'report': str(path), 'report_sha256': hashlib.sha256(payload).hexdigest(),
                'prior_status': data['status'], 'prior_exit_code': data.get('exit_code'),
                'prior_error': data.get('error'), 'prior_validation_started_utc': data['utc'],
                'legacy_pre_ras_workaround_weight_evidence': legacy_ras_plan,
                'trust_scope': 'local launcher record and unchanged filesystem metadata; not signed evidence'}
    return {record['shard']: record for record in records}, evidence, started_ns


def validate_weights(root, validated_from=None, rank=None):
    manifest = root / 'repo/scripts/r1_reproduction/r1_model_manifest.json'
    if digest(manifest) != MANIFEST_SHA:
        raise RuntimeError('Pinned model manifest changed')
    entries = [e for e in json.loads(manifest.read_text()) if e['Path'].endswith('.safetensors')]
    if len(entries) != 163:
        raise RuntimeError('Expected exactly 163 original R1 shards')
    converter_sha = digest(root / 'repo/batchgen/ckpt_converter/ckpt_converter.py')
    previous, evidence, started_ns = {}, None, None
    if validated_from is not None:
        previous, evidence, started_ns = load_weights_evidence(root, rank, validated_from, entries, converter_sha)
    expected_files, records = set(), []
    for entry in entries:
        name = entry['Path']
        stem = Path(name).stem
        binary = root / 'converted' / (stem + '.bin')
        metadata = root / 'converted' / (stem + '.json')
        receipt_path = root / 'converted.receipts' / (name + '.converted.json')
        binary_before = file_identity(binary)
        receipt_sha, receipt_identity = stable_digest(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if file_identity(receipt_path) != receipt_identity:
            raise RuntimeError(f'Receipt changed while reading: {receipt_path}')
        for key, value in {'source_sha256': entry['Sha256'], 'source_size': entry['Size'],
                           'source_revision': REVISION, 'manifest_sha256': MANIFEST_SHA,
                           'converter_sha256': converter_sha, 'marlin': False}.items():
            if receipt.get(key) != value:
                raise RuntimeError(f'{receipt_path}: {key} mismatch')
        v = receipt['validation']
        if binary_before['size'] != v['binary_size']:
            raise RuntimeError(f'{binary}: size mismatch')
        metadata_sha, metadata_identity = stable_digest(metadata)
        if metadata_sha != v['metadata_sha256']:
            raise RuntimeError(f'{metadata}: SHA256 mismatch')
        prior = previous.get(name)
        reuse_mode, fallback_reason = None, 'no prior evidence'
        if prior is not None:
            # Even unchanged metadata/receipts are revalidated above. A changed
            # receipt, expected binary hash, metadata hash or identity forces
            # a new full read of this binary, never an optimistic cache hit.
            validation_matches = all(prior.get(key) == value for key, value in v.items())
            receipt_matches = prior.get('receipt_sha256', receipt_sha) == receipt_sha
            small_file_identities_match = (
                prior.get('metadata_identity', metadata_identity) == metadata_identity
                and prior.get('receipt_identity', receipt_identity) == receipt_identity)
            if not validation_matches or not receipt_matches or not small_file_identities_match:
                fallback_reason = 'prior validation or receipt differs'
            elif 'binary_identity' in prior:
                if prior['binary_identity'] == binary_before:
                    reuse_mode = 'reused_exact_identity'
                else:
                    fallback_reason = 'binary filesystem identity changed'
            elif (0 < binary_before['ctime_ns'] < started_ns
                  and 0 < binary_before['mtime_ns'] < started_ns):
                reuse_mode = 'reused_legacy_prevalidation_timestamps'
            else:
                fallback_reason = 'legacy timestamps do not prove unchanged since validation began'
        if reuse_mode is None:
            binary_sha, identity_after_hash = stable_digest(binary)
            if binary_sha != v['binary_sha256']:
                raise RuntimeError(f'{binary}: SHA256 mismatch')
            if identity_after_hash != binary_before:
                raise RuntimeError(f'Binary changed during validation: {binary}')
        if file_identity(binary) != binary_before:
            raise RuntimeError(f'Binary changed during validation: {binary}')
        verification = {'mode': reuse_mode or 'full_sha256', 'fallback_reason': None if reuse_mode else fallback_reason}
        if evidence:
            verification['prior_report_sha256'] = evidence['report_sha256']
        records.append({'shard': name, **v, 'binary_identity': binary_before,
                        'metadata_identity': metadata_identity, 'receipt_identity': receipt_identity,
                        'receipt_sha256': receipt_sha, 'binary_verification': verification})
        expected_files.update((binary.name, metadata.name))
        print(f'Validated {len(records)}/163: {name} [{verification["mode"]}]', flush=True)
    if {p.name for p in (root / 'converted').iterdir()} != expected_files:
        raise RuntimeError('Converted directory has unexpected files: native loader scans this directory')
    return records, evidence


def check_weight_identities(records):
    """Cheap final check after all validation and immediately before launch."""
    for record in records:
        for field in ('binary_identity', 'metadata_identity', 'receipt_identity'):
            identity = record[field]
            if file_identity(Path(identity['path'])) != identity:
                raise RuntimeError(f'Validated weight artifact changed before launch: {identity["path"]}; rerun validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id', default='r1-diagnostic-v1')
    parser.add_argument('--attention-implementation', choices=('triton_precision', 'fa4_flashinfer'),
                        default='triton_precision', help='Explicit eager diagnostic only; neither choice implies model/performance qualification')
    parser.add_argument('--registration-qualification', type=Path,
                        help='Required for --execute: local passed native 675GiB/8worker/shifted-tensor report')
    parser.add_argument('--validated-weights-from', type=Path,
                        help='Optional prior terminated local launch.json; rehash any shard lacking unchanged-file proof')
    parser.add_argument('--execute', action='store_true', help='Validate and launch the full model in this allocated container')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        parser.error('run-id must use letters, numbers, dash, underscore')
    env, command = plan(args.root, args.node_rank, args.run_id, args.attention_implementation)
    print(json.dumps({'mode': 'execute' if args.execute else 'plan', 'environment': env,
                      'cwd': str(args.root / 'repo'), 'command': command,
                      'scope': 'Full-model eager diagnostic; not a paper speedup measurement'}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    if args.registration_qualification is None:
        parser.error('--execute requires --registration-qualification; driver-only probes are insufficient')
    if not Path('/.dockerenv').exists() or socket.gethostname() != HOSTS[args.node_rank]:
        raise RuntimeError('Execution requires the allocated container and matching node rank')
    if os.environ.get('BATCHGEN_PARAMETER_SERVER_ENDPOINT'):
        raise RuntimeError('Unset BATCHGEN_PARAMETER_SERVER_ENDPOINT: this launch uses the local embedded PS')
    if not Path(command[0]).is_file() or not Path(env['TRITON_PTXAS_PATH']).is_file():
        raise RuntimeError('Missing isolated Python or private CUDA13.0 ptxas')
    report_dir = args.root / 'runs' / args.run_id / f'node{args.node_rank}'
    report_dir.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'utc': datetime.now(timezone.utc).isoformat(),
             'hostname': socket.gethostname(), 'environment': env, 'command': command,
             'attention_implementation': args.attention_implementation,
             'model_quality_qualified': False, 'performance_qualified': False}
    report = report_dir / 'launch.json'
    try:
        from r1_launch_safety import (prepare_launch_safety,
                                      assert_self_core_policy, preflight_local_disk)
        state['launch_safety'] = prepare_launch_safety(args.root)
        if args.validated_weights_from is not None:
            state['requested_weight_reuse'] = {
                'report': str(args.validated_weights_from.absolute()),
                'report_sha256': stable_digest(args.validated_weights_from)[0]}
        state['registration_qualification'] = validate_registration_qualification(
            args.root, args.node_rank, args.registration_qualification)
        state['weights'], state['weight_validation_reuse'] = validate_weights(
            args.root, args.validated_weights_from, args.node_rank)
        state['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root / 'repo', text=True).strip()
        state['source_sha256'] = {str(p.relative_to(args.root / 'repo')): digest(p)
                                  for p in (args.root / 'repo/batchgen').rglob('*.py')}
        state['launcher_source_sha256'] = {
            name: digest(args.root / 'repo/scripts/r1_reproduction' / name)
            for name in ('launch_r1_diagnostic.py', 'r1_launch_process.py', 'r1_launch_safety.py')}
        state['network_environment'] = {k: v for k, v in os.environ.items() if k.startswith(('UCX_', 'NCCL_'))}
        state['status'] = 'launching'
        report.write_text(json.dumps(state, indent=2) + '\n')
        child_env = os.environ.copy()
        child_env.update(env)
        qualified_core = state['registration_qualification']
        if digest(Path(qualified_core['core_engine_path'])) != qualified_core['core_engine_sha256']:
            raise RuntimeError('Native library changed while validating weights; rerun qualification')
        check_weight_identities(state['weights'])
        state['pre_popen_core_policy'] = assert_self_core_policy()
        state['pre_popen_local_disk'] = preflight_local_disk(args.root)
        # Normal entry point retains all version, native-origin, freshness and
        # tokenizer checks; no fabricated capability, fallback, or preflight skip.
        child = subprocess.Popen(command, cwd=args.root / 'repo', env=child_env,
                                 start_new_session=True)
        from r1_launch_process import supervise_child

        def save_supervision(audit):
            state['process_supervision'] = audit
            temporary = report.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(report)

        # Sole direct-child wait ownership retains the HTTP leader with
        # WNOWAIT through ordered native release and the final group audit.
        audit = supervise_child(child, on_update=save_supervision)
        state['process_supervision'] = audit
        if audit['parent_cleanup_required'] or 'supervision_error' in audit:
            state.update(status='failed', exit_code=2,
                         error='Process supervision failed; inspect the preserved ownership/cleanup audit')
            return 2
        code = audit['exit_code']
        state['status'], state['exit_code'] = 'exited', code
        return code
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if hasattr(exc, 'record'):
            state['failed_prelaunch_check'] = exc.record
        raise
    finally:
        report.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
