#!/usr/bin/env python3
"""Restore the historical SG CPU environment from verified artifacts.

The default prints a plan without writes, subprocesses or network access.
--execute creates a new isolated environment and a new exclusive recovery job.
It never contacts a model server, submits inference or initializes CUDA.
"""
from __future__ import annotations

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
import sys
import tarfile
import time
import urllib.parse
import urllib.request


ROOT = Path('/home/admin/batchgen-r1-alignment-20261026')
OSS = Path('/mnt/fuse/oss/luolinglong.lll/batchgen-r1-repro-20261023/resumed-r1/sglang-baseline-v8')
OLD_WHEELS = '/home/admin/batchgen-r1-repro-20261023/baseline-wheel-recovery/'
ARTIFACTS = {
    'resolved-wheels.lock': {
        'bytes': 315295,
        'sha256': '7da8b8246cf6e3a5265e0ce36b6267b42ed5462a286be5fc7114d51d81447e4e',
    },
    'triton-3.5.0+git4caa0328-cp311-cp311-linux_x86_64.whl': {
        'bytes': 350558713,
        'sha256': 'e9f2b9066dd05c853601116b72830711fc1439de5197ca5e86b97b8140368e11',
    },
    'private-cxx-runtime.tar.gz': {
        'bytes': 14374836,
        'sha256': '9e2ff14f10adc8187e1e569f7fb9bfe8a35e0d25564205c48ae3c7425f1fd81a',
    },
}
BINARY_WHEELS = {
    'sgl_kernel-0.3.17.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl': {
        'bytes': 380986924,
        'sha256': '51a97eb73076f9e2b6bfc2dea0c152bd20ae1ce9914017de096efb5ed9a24371',
        'url': 'https://github.com/sgl-project/whl/releases/download/v0.3.17.post1/sgl_kernel-0.3.17.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl',
    },
    'triton-3.5.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': {
        'bytes': 170433488,
        'sha256': 'f34bfa21c5b3a203c0f0eab28dcc1e49bd1f67d22724e77fb6665a659200a4ec',
        'url': 'https://files.pythonhosted.org/packages/3d/78/949a04391c21956c816523678f0e5fa308eb5b1e7622d88c4e4ef5fceca0/triton-3.5.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl',
    },
}
CXX_HASHES = {
    'libstdc++.so': 'd3c4bd998c94a632c3887c8605ca95e6b102a3bee5fd9d9f6075d130dd3d809d',
    'libstdc++.so.6': 'd3c4bd998c94a632c3887c8605ca95e6b102a3bee5fd9d9f6075d130dd3d809d',
    'libgcc_s.so.1': '4f74724a5d9026964eafefc57e46f3363a24c5ad8c83d52b6b7fb270fcb64655',
}
EXPECTED_VERSIONS = {
    'sglang': '0.5.5.post3', 'torch': '2.9.0+cu130',
    'torchaudio': '2.9.0+cu130', 'torchvision': '0.24.0+cu130',
    'sgl-kernel': '0.3.17.post1', 'flashinfer-python': '0.5.2',
    'flashinfer-cubin': '0.5.2', 'triton': '3.5.0+git4caa0328',
    'nvidia-cutlass-dsl': '4.3.0.dev0', 'nvidia-nccl-cu13': '2.28.3',
    'transformers': '4.57.1',
}
SG_SOURCE_FILES = (
    'sglang/srt/server_args.py', 'sglang/srt/models/deepseek_v2.py',
    'sglang/srt/layers/communicator.py', 'sglang/srt/layers/dp_attention.py',
    'sglang/srt/layers/vocab_parallel_embedding.py',
    'sglang/srt/layers/logits_processor.py',
    'sglang/srt/layers/moe/fused_moe_triton/layer.py',
    'sglang/srt/layers/quantization/fp8.py',
    'sglang/srt/distributed/parallel_state.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def verified_copy(source, destination, record):
    """Copy once to a private partial file, verify, then publish without overwrite."""
    source, destination = Path(source), Path(destination)
    if source.stat().st_size != record['bytes']:
        raise ValueError('Source artifact size mismatch: ' + source.name)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + '.partial')
    h, count = hashlib.sha256(), 0
    with source.open('rb') as incoming, partial.open('xb') as outgoing:
        for chunk in iter(lambda: incoming.read(8 * 1024**2), b''):
            count += len(chunk)
            if count > record['bytes']:
                raise ValueError('Source artifact grew while copying: ' + source.name)
            outgoing.write(chunk)
            h.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if count != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('Source artifact SHA256 mismatch: ' + source.name)
    # The newly created exclusive job owns both paths; no other job writes here.
    partial.rename(destination)
    return {'source': str(source), 'path': str(destination), 'bytes': count,
            'sha256': h.hexdigest()}


def download_verified(destination, record, timeout):
    """Read bounded network chunks; retain partial evidence on any failure."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + '.partial')
    h, count, started = hashlib.sha256(), 0, time.monotonic()

    def check_deadline():
        if time.monotonic() - started >= timeout:
            raise TimeoutError('Artifact download deadline exceeded: ' + destination.name)

    request = urllib.request.Request(record['url'], headers={'User-Agent': 'r1-alignment-recovery/1'})
    with urllib.request.urlopen(request, timeout=min(10, timeout)) as response, partial.open('xb') as outgoing:
        if not callable(getattr(response, 'read1', None)):
            raise TypeError('Artifact download requires a response.read1 stream: ' + destination.name)
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != record['bytes']:
            raise ValueError('Download Content-Length mismatch: ' + destination.name)
        while True:
            check_deadline()
            # HTTPResponse.read(n) can wait for n bytes across many socket reads.
            # read1 performs at most one underlying read, allowing the total
            # deadline to be checked even during a continuously slow transfer.
            chunk = response.read1(64 * 1024)
            check_deadline()
            if not chunk:
                if count != record['bytes']:
                    raise EOFError('Artifact download ended before pinned size: ' + destination.name)
                break
            count += len(chunk)
            if count > record['bytes']:
                raise ValueError('Download exceeds pinned size: ' + destination.name)
            outgoing.write(chunk)
            h.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if count != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('Downloaded artifact size/SHA256 mismatch: ' + destination.name)
    check_deadline()
    partial.rename(destination)
    return {'url': record['url'], 'path': str(destination), 'bytes': count,
            'sha256': h.hexdigest()}


def remap_lock(original, wheel_directory):
    """Only remap the two known file-URI paths; keep every other character."""
    if hashlib.sha256(original).hexdigest() != ARTIFACTS['resolved-wheels.lock']['sha256']:
        raise ValueError('Original dependency lock SHA256 mismatch')
    text = original.decode('utf-8')
    counts = {name: 0 for name in BINARY_WHEELS}
    replacements = []

    def replace(match):
        old = match.group(0)
        parsed = urllib.parse.urlsplit(old)
        decoded = urllib.parse.unquote(parsed.path)
        name = Path(decoded).name
        if parsed.netloc or parsed.query or name not in counts or decoded != OLD_WHEELS + name:
            raise ValueError('Unexpected local dependency URI in pinned lock')
        counts[name] += 1
        new = (Path(wheel_directory) / name).as_uri()
        if parsed.fragment:
            new += '#' + parsed.fragment
        replacements.append({'old': old, 'new': new, 'wheel_sha256': BINARY_WHEELS[name]['sha256']})
        return new

    mapped = re.sub(r'file://[^\s\\]+', replace, text)
    if any(count != 1 for count in counts.values()):
        raise ValueError('Expected exactly one URI for each pinned local wheel')
    return mapped.encode('utf-8'), replacements


def extract_cxx(archive, job):
    root = Path(job) / 'private-cxx-runtime'
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    with tarfile.open(archive, 'r:gz') as incoming:
        members = incoming.getmembers()
        for member in members:
            target = (Path(job) / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('C++ archive member escapes private runtime')
            if member.isdev() or member.isfifo():
                raise ValueError('Unsupported C++ archive member')
            if member.issym() and not (target.parent / member.linkname).resolve().is_relative_to(root):
                raise ValueError('C++ archive symlink escapes private runtime')
            if member.islnk() and not (Path(job) / member.linkname).resolve().is_relative_to(root):
                raise ValueError('C++ archive hardlink escapes private runtime')
        incoming.extractall(job, members=members)
    for name, expected in CXX_HASHES.items():
        path = root / 'lib' / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError('Extracted C++ runtime identity mismatch: ' + name)
    return root


def child_environment(args, cxx=None):
    env = os.environ.copy()
    for key in list(env):
        if key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'LIBRARY_PATH',
                   'TRITON_WHEEL_NAME', 'TRITON_WHEEL_VERSION_SUFFIX', 'DISABLE_MMA_V5',
                   'TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH', 'TRITON_APPEND_CMAKE_ARGS',
                   'UV_DEFAULT_INDEX'):
            env.pop(key, None)
        if key.startswith(('UV_INDEX', 'UV_EXTRA_INDEX', 'PIP_INDEX', 'PIP_EXTRA_INDEX')):
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null',
               UV_CACHE_DIR=str(args.cache_dir), UV_PYTHON_DOWNLOADS='never',
               UV_CONCURRENT_DOWNLOADS='8', UV_CONCURRENT_INSTALLS='8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if cxx is not None:
        env['LD_LIBRARY_PATH'] = str(cxx / 'lib')
        env['LIBRARY_PATH'] = str(cxx / 'lib')
    return env


def run_step(argv, name, job, env, timeout):
    started = time.time()
    record = {'stage': name, 'argv': list(map(str, argv)), 'started_unix': started,
              'timeout_seconds': timeout, 'CUDA_VISIBLE_DEVICES': ''}
    with (job / (name + '.stdout')).open('xb') as out, (job / (name + '.stderr')).open('xb') as err:
        process = subprocess.Popen(record['argv'], stdout=out, stderr=err, env=env,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        record['pid'] = process.pid
        save_json(job / (name + '.run.json'), record)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            # Only this still-unreaped child process group belongs to this step.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
        record.update(exit_code=code, finished_unix=time.time())
        save_json(job / (name + '.run.json'), record)
    if code or record.get('timed_out'):
        raise RuntimeError('CPU recovery step failed: ' + name)
    return record


def audit_program():
    return '''import hashlib, importlib, importlib.metadata as md, json, pathlib, platform, sys
expected = json.loads(sys.argv[1])
sources = json.loads(sys.argv[2])
report = {'python': platform.python_version(), 'python_executable': sys.executable,
          'python_sha256': hashlib.sha256(pathlib.Path(sys.executable).resolve().read_bytes()).hexdigest(),
          'versions': {}, 'imports': {}, 'sources': {}, 'gpu_qualification': False,
          'model_requests': 0, 'routing_compat_patch_applied_by_restore': False}
assert report['python'] == '3.11.15', 'Expected shared CPython 3.11.15'
for name, version in expected.items():
    report['versions'][name] = md.version(name)
report['version_mismatches'] = {name: {'expected': version, 'observed': report['versions'][name]}
    for name, version in expected.items() if report['versions'][name] != version}
import torch
assert not torch.cuda.is_initialized(), 'Unexpected CUDA initialization before CPU imports'
for name in ('torch', 'triton', 'sgl_kernel', 'flashinfer', 'sglang'):
    try:
        module = importlib.import_module(name)
        report['imports'][name] = {'ok': True, 'file': str(getattr(module, '__file__', None))}
    except Exception as error:
        report['imports'][name] = {'ok': False, 'error': type(error).__name__ + ': ' + str(error)}
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized during CPU-only import: ' + name)
distribution = md.distribution('sglang')
for name in sources:
    path = pathlib.Path(distribution.locate_file(name))
    report['sources'][name] = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        report['sources'][name].update(bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
report['cuda_initialized'] = torch.cuda.is_initialized()
report['torch_cuda_build'] = torch.version.cuda
report['status'] = 'cpu_imports_and_versions_passed' if not report['version_mismatches'] and all(
    item['ok'] for item in report['imports'].values()) and all(
    item['exists'] for item in report['sources'].values()) and torch.version.cuda == '13.0' else 'failed'
print(json.dumps(report, indent=2), flush=True)
pathlib.Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + '\\n')
raise SystemExit(0 if report['status'] == 'cpu_imports_and_versions_passed' else 1)
'''


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--env-root', type=Path, default=ROOT / 'sglang-env')
    p.add_argument('--recovery-root', type=Path, default=ROOT / 'sglang-recovery')
    p.add_argument('--job-name', default='restore-sglang-v1')
    p.add_argument('--python', type=Path, default=ROOT / 'python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11')
    p.add_argument('--uv', type=Path, default=Path('/opt/conda310/bin/uv'))
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    p.add_argument('--oss-root', type=Path, default=OSS)
    p.add_argument('--sgl-kernel-wheel', type=Path)
    p.add_argument('--bootstrap-triton-wheel', type=Path)
    p.add_argument('--index-url', choices=('https://mirrors.aliyun.com/pypi/simple/', 'https://pypi.org/simple/'),
                   default='https://mirrors.aliyun.com/pypi/simple/')
    p.add_argument('--download-timeout', type=int, default=1200)
    p.add_argument('--install-timeout', type=int, default=2400)
    return p


def validate_args(args):
    for name in ('env_root', 'recovery_root', 'python', 'uv', 'cache_dir', 'oss_root'):
        if not getattr(args, name).is_absolute():
            raise ValueError('--' + name.replace('_', '-') + ' must be absolute')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}', args.job_name):
        raise ValueError('Invalid exclusive job name')
    if args.download_timeout <= 0 or args.install_timeout <= 0:
        raise ValueError('Timeouts must be positive')
    if args.env_root == args.recovery_root or args.env_root in args.recovery_root.parents or args.recovery_root in args.env_root.parents:
        raise ValueError('Environment and recovery roots must be disjoint')
    for candidate in (args.sgl_kernel_wheel, args.bootstrap_triton_wheel):
        if candidate is not None and not candidate.is_absolute():
            raise ValueError('Local wheel paths must be absolute')


def plan(args):
    return {'schema': 'r1-sglang-alignment-environment-restore-v1',
            'mode': 'execute' if args.execute else 'plan_only',
            'env_root': str(args.env_root), 'job_root': str(args.recovery_root / args.job_name),
            'python': str(args.python), 'uv': str(args.uv), 'cache_dir': str(args.cache_dir),
            'oss_root': str(args.oss_root), 'oss_artifacts': ARTIFACTS,
            'binary_wheels': BINARY_WHEELS, 'expected_versions': EXPECTED_VERSIONS,
            'gpu_operations': False, 'model_requests': False,
            'qualification': 'CPU environment restoration only; no alignment/model/GPU pass',
            'steps': ['exclusive_job', 'verify_shared_python', 'verified_artifacts',
                      'remap_two_file_uris_only', 'private_cxx_runtime', 'new_venv',
                      'hashed_binary_lock', 'fixed_source_triton', 'cpu_import_source_audit']}


def execute(args):
    if sys.platform != 'linux' or not Path('/.dockerenv').is_file():
        raise RuntimeError('--execute requires the allocated Linux container')
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('Run recovery without ambient LD_PRELOAD')
    if args.recovery_root.is_symlink() or args.env_root.exists() or args.env_root.is_symlink():
        raise FileExistsError('Recovery root is a symlink or environment already exists')
    if not args.python.is_file() or not args.uv.is_file():
        raise FileNotFoundError('Explicit shared Python/uv executable is missing')
    args.recovery_root.mkdir(parents=True, exist_ok=True)
    job = args.recovery_root / args.job_name
    job.mkdir(exist_ok=False)
    record = plan(args)
    record.update(status='running', stage='exclusive_job', hostname=socket.gethostname(),
                  pid=os.getpid(), started_unix=time.time(), script_sha256=digest(__file__))
    status = job / 'status.json'
    save_json(job / 'plan.json', record)
    save_json(status, record)
    try:
        # This lock is retained as provenance; an existing/partial environment
        # needs an explicitly new destination rather than an overwrite/resume.
        with (args.recovery_root / 'environment-restore.lock').open('x') as lock:
            lock.write(json.dumps({'job': str(job), 'env_root': str(args.env_root), 'pid': os.getpid()}) + '\n')
        if shutil.disk_usage(args.recovery_root).free < 25 * 1024**3:
            raise RuntimeError('Less than 25GiB local free space')
        env = child_environment(args)
        record['stage'] = 'verify_shared_python'; save_json(status, record)
        run_step([args.python, '-c', 'import sys; assert sys.version_info[:3] == (3,11,15); print(sys.version)'],
                 'python_identity', job, env, 30)
        artifacts = job / 'artifacts'; artifacts.mkdir()
        record['stage'] = 'verified_artifacts'; save_json(status, record)
        record['artifacts'] = [verified_copy(args.oss_root / name, artifacts / name, item)
                               for name, item in ARTIFACTS.items()]
        local_sources = (args.sgl_kernel_wheel, args.bootstrap_triton_wheel)
        for (name, item), local in zip(BINARY_WHEELS.items(), local_sources):
            result = (verified_copy(local, artifacts / name, item) if local is not None else
                      download_verified(artifacts / name, item, args.download_timeout))
            record['artifacts'].append(result)
            save_json(status, record)
        original = (artifacts / 'resolved-wheels.lock').read_bytes()
        mapped, replacements = remap_lock(original, artifacts)
        mapped_path = job / 'resolved-wheels.remapped.lock'
        mapped_path.write_bytes(mapped)
        record['lock_mapping'] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                                  'remapped_sha256': hashlib.sha256(mapped).hexdigest(),
                                  'replacements': replacements, 'all_other_bytes_preserved': True}
        save_json(job / 'lock-mapping.json', record['lock_mapping'])
        cxx = extract_cxx(artifacts / 'private-cxx-runtime.tar.gz', job)
        env = child_environment(args, cxx)
        save_json(job / 'runtime-environment.json', {'private_cxx_library_directory': str(cxx / 'lib'),
            'required_process_environment': {'LD_LIBRARY_PATH': str(cxx / 'lib')},
            'recovery_children_GPU_hidden': True, 'gpu_qualification': False})
        record['stage'] = 'new_venv'; save_json(status, record)
        run_step([args.uv, '--no-config', 'venv', '--python', args.python, args.env_root],
                 'create_venv', job, env, 120)
        py = args.env_root / 'bin/python'
        record['stage'] = 'hashed_binary_lock'; save_json(status, record)
        run_step([args.uv, '--no-config', 'pip', 'install', '--python', py,
                  '--no-deps', '--no-build', '--require-hashes', '--default-index', args.index_url,
                  '--torch-backend', 'cu130', '-r', mapped_path],
                 'install_hashed_lock', job, env, args.install_timeout)
        record['stage'] = 'fixed_source_triton'; save_json(status, record)
        fixed = next(name for name in ARTIFACTS if name.endswith('.whl'))
        fixed_lock = job / 'fixed-source-triton.lock'
        fixed_lock.write_text('triton @ ' + (artifacts / fixed).as_uri() +
                              ' --hash=sha256:' + ARTIFACTS[fixed]['sha256'] + '\n')
        run_step([args.uv, '--no-config', 'pip', 'install', '--python', py, '--no-deps',
                  '--no-build', '--require-hashes', '--reinstall-package', 'triton', '-r', fixed_lock],
                 'install_fixed_triton', job, env, 300)
        record['stage'] = 'cpu_import_source_audit'; save_json(status, record)
        run_step([py, '-c', audit_program(), json.dumps(EXPECTED_VERSIONS), json.dumps(SG_SOURCE_FILES),
                  job / 'cpu-audit.json'],
                 'cpu_audit', job, env, 180)
        audit = json.loads((job / 'cpu-audit.json').read_text())
        if audit['status'] != 'cpu_imports_and_versions_passed' or audit['cuda_initialized']:
            raise RuntimeError('CPU audit did not qualify imports and versions')
        run_step([args.uv, '--no-config', 'pip', 'freeze', '--python', py],
                 'environment_freeze', job, env, 60)
        record.update(status='succeeded', stage='complete', finished_unix=time.time(),
                      cpu_audit=str(job / 'cpu-audit.json'), cuda_initialized=False,
                      routing_compat_patch_applied=False, strict_alignment_qualified=False,
                      model_quality_qualified=False)
        save_json(status, record)
    except BaseException as error:
        record.update(status='failed', finished_unix=time.time(),
                      error=type(error).__name__ + ': ' + str(error))
        save_json(status, record)
        raise
    return record


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    if not args.execute:
        print(json.dumps(plan(args), indent=2))
        return 0
    try:
        record = execute(args)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': type(error).__name__ + ': ' + str(error),
                          'status_path': str(args.recovery_root / args.job_name / 'status.json')}), flush=True)
        return 1
    print(json.dumps({'status': record['status'], 'status_path': str(args.recovery_root / args.job_name / 'status.json'),
                      'env_root': str(args.env_root), 'gpu_qualification': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
