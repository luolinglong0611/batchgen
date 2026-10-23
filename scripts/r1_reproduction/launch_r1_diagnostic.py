#!/usr/bin/env python3
"""Print a two-node R1 diagnostic command; --execute explicitly loads the model.

No torch import, CUDA context, filesystem changes, or remote operation in plan
mode. Execute mode validates all 163 converted shard receipts and complete file
hashes, then uses the normal HTTP entry point (including its real preflight).
This is an eager correctness experiment, not a performance-qualified launcher.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
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


def plan(root, rank, run_id):
    toolkit = root / 'cuda130-packages/nvidia/cu13'
    env = {
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'CUDA_HOME': str(toolkit),
        'TRITON_PTXAS_PATH': str(toolkit / 'bin/ptxas'),
        'BATCHGEN_R1_SM103_EXPERIMENTAL': '1',
        'BATCHGEN_R1_SM103_ATTENTION': 'triton_precision',
        'DISABLE_MMA_V5': '1', 'TORCH_CUDA_CHANNEL': 'cu130',
        'BATCHGEN_W8A16_DEQUANT': '0',
        'BATCHGEN_ENABLE_ALL_TO_ALL': '0', 'GLOO_SOCKET_IFNAME': 'eth0',
        'NCCL_SOCKET_IFNAME': 'eth0', 'NCCL_IB_GID_INDEX': '3',
        'NCCL_IB_HCA': '=' + ','.join(f'mlx5_bond_{i}:1' for i in range(8)),
        'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    }
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


def validate_weights(root):
    manifest = root / 'repo/scripts/r1_reproduction/r1_model_manifest.json'
    if digest(manifest) != MANIFEST_SHA:
        raise RuntimeError('Pinned model manifest changed')
    entries = [e for e in json.loads(manifest.read_text()) if e['Path'].endswith('.safetensors')]
    if len(entries) != 163:
        raise RuntimeError('Expected exactly 163 original R1 shards')
    converter_sha = digest(root / 'repo/batchgen/ckpt_converter/ckpt_converter.py')
    expected_files, records = set(), []
    for entry in entries:
        name = entry['Path']
        stem = Path(name).stem
        binary = root / 'converted' / (stem + '.bin')
        metadata = root / 'converted' / (stem + '.json')
        receipt_path = root / 'converted.receipts' / (name + '.converted.json')
        for path in (binary, metadata, receipt_path):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f'Missing/nonregular converted artifact: {path}')
        receipt = json.loads(receipt_path.read_text())
        for key, value in {'source_sha256': entry['Sha256'], 'source_size': entry['Size'],
                           'source_revision': REVISION, 'manifest_sha256': MANIFEST_SHA,
                           'converter_sha256': converter_sha, 'marlin': False}.items():
            if receipt.get(key) != value:
                raise RuntimeError(f'{receipt_path}: {key} mismatch')
        v = receipt['validation']
        if binary.stat().st_size != v['binary_size']:
            raise RuntimeError(f'{binary}: size mismatch')
        for path, field in ((binary, 'binary_sha256'), (metadata, 'metadata_sha256')):
            if digest(path) != v[field]:
                raise RuntimeError(f'{path}: SHA256 mismatch')
        records.append({'shard': name, **v})
        expected_files.update((binary.name, metadata.name))
        print(f'Validated {len(records)}/163: {name}', flush=True)
    if {p.name for p in (root / 'converted').iterdir()} != expected_files:
        raise RuntimeError('Converted directory has unexpected files: native loader scans this directory')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id', default='r1-diagnostic-v1')
    parser.add_argument('--execute', action='store_true', help='Validate and launch the full model in this allocated container')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        parser.error('run-id must use letters, numbers, dash, underscore')
    env, command = plan(args.root, args.node_rank, args.run_id)
    print(json.dumps({'mode': 'execute' if args.execute else 'plan', 'environment': env,
                      'cwd': str(args.root / 'repo'), 'command': command,
                      'scope': 'Full-model eager diagnostic; not a paper speedup measurement'}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    if not Path('/.dockerenv').exists() or socket.gethostname() != HOSTS[args.node_rank]:
        raise RuntimeError('Execution requires the allocated container and matching node rank')
    if os.environ.get('BATCHGEN_PARAMETER_SERVER_ENDPOINT'):
        raise RuntimeError('Unset BATCHGEN_PARAMETER_SERVER_ENDPOINT: this launch uses the local embedded PS')
    if not Path(command[0]).is_file() or not Path(env['TRITON_PTXAS_PATH']).is_file():
        raise RuntimeError('Missing isolated Python or private CUDA13.0 ptxas')
    report_dir = args.root / 'runs' / args.run_id / f'node{args.node_rank}'
    report_dir.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'utc': datetime.now(timezone.utc).isoformat(),
             'hostname': socket.gethostname(), 'environment': env, 'command': command}
    report = report_dir / 'launch.json'
    try:
        state['weights'] = validate_weights(args.root)
        state['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root / 'repo', text=True).strip()
        state['source_sha256'] = {str(p.relative_to(args.root / 'repo')): digest(p)
                                  for p in (args.root / 'repo/batchgen').rglob('*.py')}
        state['network_environment'] = {k: v for k, v in os.environ.items() if k.startswith(('UCX_', 'NCCL_'))}
        state['status'] = 'launching'
        report.write_text(json.dumps(state, indent=2) + '\n')
        child_env = os.environ.copy()
        child_env.update(env)
        # Normal entry point retains all version, native-origin, freshness and
        # tokenizer checks; no fabricated capability, fallback, or preflight skip.
        child = subprocess.Popen(command, cwd=args.root / 'repo', env=child_env,
                                 start_new_session=True)
        def stop_owned_group(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop_owned_group)
        signal.signal(signal.SIGINT, stop_owned_group)
        code = child.wait()
        state['status'], state['exit_code'] = 'exited', code
        return code
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
