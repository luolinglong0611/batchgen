#!/usr/bin/env python3
"""Plan/inspect an isolated SGLang R1 diagnostic. NEVER starts a model service.

--inspect reads current artifacts, metadata, receipts and installed --help with
GPUs hidden. Weight receipts attest historical verification only: this script
does not rehash 641 GiB of source weights or claim full-tree source verification.
"""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

ROOT = Path('/home/admin/batchgen-r1-repro-20261023')
WEIGHTS = Path('/mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8')
HOSTS = ('serverless-010066009163.wm226', 'serverless-010066009166.wm226')
MANIFEST_SHA = 'd9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253'
VERSIONS = {'sglang': '0.5.5.post3', 'torch': '2.9.0+cu130',
            'triton': '3.5.0+git4caa0328', 'nvidia-nccl-cu13': '2.28.3',
            'sgl-kernel': '0.3.17.post1', 'flashinfer-python': '0.5.2',
            'flashinfer-cubin': '0.5.2', 'nvidia-cutlass-dsl': '4.3.0.dev0',
            'transformers': '4.57.1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(root, rank, inherited=None):
    inherited = os.environ if inherited is None else inherited
    base = root / 'sglang-baseline-v8'
    toolkit = root / 'cuda130-packages/nvidia/cu13'
    unset = sorted(set(k for k in inherited if k.startswith(('BATCHGEN_', 'SGLANG_', 'SGL_', 'NCCL_', 'FLASHINFER_'))) | {
        'PYTHONPATH', 'PYTHONHOME', 'DISABLE_MMA_V5', 'TRITON_PTXAS_PATH',
        'TRITON_PTXAS_BLACKWELL_PATH', 'NVIDIA_VISIBLE_DEVICES', 'TORCH_CUDA_CHANNEL',
    })
    env = {
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'CUDA_HOME': str(toolkit),
        'PATH': f'{base}/baseline-env/bin:{toolkit}/bin:/usr/local/bin:/usr/bin:/bin',
        'LD_LIBRARY_PATH': str(base / 'private-cxx-runtime/lib'),
        'TRITON_CACHE_DIR': str(base / 'cache/triton-runtime'),
        # TP16 shared-expert down K=128 is unsupported by FlashInfer TRTLLM
        # dense GEMM (requires K>=256). Disable UE8M0 re-quantization too.
        'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
        'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '0',
        'SGLANG_SUPPORT_CUTLASS_BLOCK_FP8': '0',
        'SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN': '0',
        'NCCL_SOCKET_IFNAME': 'eth0', 'GLOO_SOCKET_IFNAME': 'eth0',
        'NCCL_IB_GID_INDEX': '3',
        'NCCL_IB_HCA': '=' + ','.join(f'mlx5_bond_{i}:1' for i in range(8)),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTHONUNBUFFERED': '1',
    }
    command = [str(base / 'baseline-env/bin/python'), '-m', 'sglang.launch_server',
        '--model-path', str(WEIGHTS), '--tokenizer-path', str(WEIGHTS),
        '--served-model-name', 'deepseek-ai/DeepSeek-R1',
        '--load-format', 'safetensors', '--model-loader-extra-config',
        '{"enable_multithread_load":false,"num_threads":1}',
        '--dtype', 'bfloat16', '--quantization', 'fp8', '--kv-cache-dtype', 'auto',
        '--tp-size', '16', '--dp-size', '16', '--enable-dp-attention', '--ep-size', '1',
        '--nnodes', '2', '--node-rank', str(rank), '--dist-init-addr', '10.66.9.163:33155',
        '--host', '0.0.0.0', '--port', '30000',
        '--attention-backend', 'trtllm_mla', '--moe-runner-backend', 'flashinfer_trtllm',
        '--moe-a2a-backend', 'none', '--page-size', '64',
        '--disable-cuda-graph', '--disable-radix-cache', '--disable-shared-experts-fusion',
        '--chunked-prefill-size', '65536', '--context-length', '32768',
        '--mem-fraction-static', '0.85', '--max-total-tokens', '32768',
        '--max-running-requests', '16', '--random-seed', '20261023',
        '--dist-timeout', '7200', '--watchdog-timeout', '600']
    return env, unset, command


def check_help(command, help_text):
    accepted = set(re.findall(r'--[a-z][a-z0-9-]*', help_text))
    unknown = {s for s in command if s.startswith('--')} - accepted
    if unknown:
        raise RuntimeError(f'Installed --help does not support: {sorted(unknown)}')


def inspect_weights(weights, manifest_path):
    if sha(manifest_path) != MANIFEST_SHA:
        raise RuntimeError('Model manifest pin changed')
    manifest = json.loads(manifest_path.read_text())
    selected = {f['Path']: f for f in manifest if f['Type'] == 'blob' and '/' not in f['Path']}
    names = sorted(n for n in selected if n.endswith('.safetensors'))
    if len(names) != 163:
        raise RuntimeError('Expected 163 original shards')
    records = []
    for name in names:
        expected = selected[name]
        receipt_path = weights / (name + '.verified.json')
        receipt = json.loads(receipt_path.read_text())
        size = (weights / name).stat().st_size
        if receipt != {'sha256': expected['Sha256'], 'size': expected['Size'],
                       'revision': expected['Revision']} or size != expected['Size']:
            raise RuntimeError(f'Original-shard receipt/size mismatch: {name}')
        records.append({'name': name, 'current_size': size,
                        'historical_sha256': receipt['sha256'], 'receipt_sha256': sha(receipt_path)})
    small = {}
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'model.safetensors.index.json'):
        actual = sha(weights / name)
        if actual != selected[name]['Sha256']:
            raise RuntimeError(f'Current metadata SHA mismatch: {name}')
        small[name] = actual
    index = json.loads((weights / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != set(names):
        raise RuntimeError('Safetensors index does not match 163-shard manifest')
    config = json.loads((weights / 'config.json').read_text())
    if config['quantization_config'] != {
        'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8',
        'weight_block_size': [128, 128],
    } or config['torch_dtype'] != 'bfloat16':
        raise RuntimeError('Original FP8 block128/BF16 config required')
    with (weights / names[0]).open('rb') as stream:
        with mmap.mmap(stream.fileno(), 4096, access=mmap.ACCESS_READ) as mapped:
            head_hash = hashlib.sha256(mapped[:4096]).hexdigest()
    return {'path': str(weights), 'shards': records, 'current_metadata_sha256': small,
            'source_binary_sha256_verified_now': False,
            'receipt_scope': 'Historical downloader verification plus current size; not current full-byte proof',
            'mmap_probe': {'bytes': 4096, 'first_shard_head_sha256': head_hash},
            'weight_bytes': sum(f['Size'] for n, f in selected.items() if n in names)}


def inspect(root, rank, env, unset, command):
    if not Path('/.dockerenv').exists() or socket.gethostname() != HOSTS[rank]:
        raise RuntimeError('Inspect only inside the expected allocated container')
    base = root / 'sglang-baseline-v8'
    site = base / 'baseline-env/lib/python3.11/site-packages'
    pins = json.loads(Path(__file__).with_name('sglang_diagnostic_source_pins.json').read_text())
    current = {name: sha(site / name) for name in pins['files']}
    if current != pins['files']:
        raise RuntimeError('Reviewed installed source files changed; re-audit required')
    provenance = json.loads((base / 'private-cxx-runtime/provenance.json').read_text())
    for name, item in provenance['libraries'].items():
        if sha(base / 'private-cxx-runtime/lib' / name) != item['sha256']:
            raise RuntimeError('Private C++ runtime changed')
    cpu_env = {k: v for k, v in os.environ.items() if k not in unset}
    cpu_env.update(env, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    result = subprocess.run([command[0], '-c',
        'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in '+repr(list(VERSIONS))+'}))'],
        env=cpu_env, text=True, capture_output=True, timeout=60, check=True)
    versions = json.loads(result.stdout)
    if versions != VERSIONS:
        raise RuntimeError('Installed package versions differ from pinned baseline')
    help_result = subprocess.run(command[:3] + ['--help'], env=cpu_env,
                                 text=True, capture_output=True, timeout=120, check=True)
    check_help(command, help_result.stdout)
    weight_record = inspect_weights(WEIGHTS, Path(__file__).with_name('r1_model_manifest.json'))
    ptxas = site / 'triton/backends/nvidia/bin/ptxas-blackwell'
    ptxas_version = subprocess.run([str(ptxas), '--version'], env=cpu_env,
                                  text=True, capture_output=True, timeout=30, check=True).stdout
    if 'release 12.9' not in ptxas_version:
        raise RuntimeError('Baseline bundled ptxas is no longer CUDA12.9; re-audit required')
    return {'status': 'CPU_artifact_inspection_passed_NOT_GPU_qualified',
            'hostname': socket.gethostname(), 'versions': versions,
            'critical_installed_source_sha256': current, 'source_scope': pins['scope'],
            'launch_help_sha256': hashlib.sha256(help_result.stdout.encode()).hexdigest(),
            'weights': weight_record,
            'MemAvailable': next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
            'triton_bundled_ptxas_blackwell_sha256': sha(ptxas),
            'triton_bundled_ptxas_version': ptxas_version,
            'installation_audit_sha256': {name: sha(base / 'installation' / name) for name in
                ('cpu-environment-ready.json', 'metadata-audit.json', 'triton-fixed-source-wheel.json')},
            'required_before_launch': ['Release BatchGen workers AND parameter-server memory on both nodes',
                'Both nodes 8 GPUs idle; conservative initial MemAvailable >=850GiB/node',
                'GPU-qualify selected FP32-scale Triton dense + TRTLLM MoE/MLA before generation',
                'Monitor host available memory during loading; abort OWN process group below128GiB',
                'Bound startup+warmup externally to7200s; cleanup only this run process groups']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    env, unset, command = plan(args.root, args.node_rank)
    shell = shlex.join(['env', *[part for key in unset for part in ('-u', key)],
                        *[f'{k}={v}' for k, v in env.items()], *command])
    result = {'status': 'plan_only_NOT_executed', 'environment': env, 'unset_environment': unset,
              'command': command, 'shell_command': shell,
              'scope': 'Fixed six greedy probes,128new tokens,2repeats; diagnostics only, not performance',
              'dense_backend': 'existing SGLang Triton FP8 with original FP32 block128 scales',
              'speculative_decoding': False, 'cuda_graphs': False, 'prefix_cache': False,
              'kv_dtype_expected': 'torch.bfloat16; must confirm runtime log',
              'runtime_precision_qualified': False}
    if args.inspect:
        result['inspection'] = inspect(args.root, args.node_rank, env, unset, command)
    if args.output:
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
