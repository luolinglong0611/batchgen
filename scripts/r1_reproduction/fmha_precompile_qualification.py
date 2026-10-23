#!/usr/bin/env python3
"""One fresh, CPU-only FlashInfer 0.6.13 fmha_gen build/load qualification.

Private reproduction artifact. No model, weights, HTTP, GPU query, package
installation, source patch, or runtime/kernel qualification. The supervisor
reuses the pinned, WNOWAIT-qualified production process guardian unchanged.
Run only after parent review. A pre-existing output directory is an error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/admin/batchgen-r1-repro-20261023')
SITE = ROOT / 'env/lib/python3.11/site-packages'
CUDA = ROOT / 'cuda130-packages/nvidia/cu13'
GUARD = ROOT / 'repo/scripts/r1_reproduction/r1_launch_process.py'
GUARD_SHA = '3d11b8dfed470d885b5eed9e0cad234634f7d942efb0ba7989c74ec9f951d2aa'
SAFETY = ROOT / 'repo/scripts/r1_reproduction/r1_launch_safety.py'
SAFETY_SHA = '3b06a6644b22d53343f89427834553c462fba0ee44f3ebf8a3cde3c5bccfe130'
STAGE = ROOT / 'cuda-fmha-precompile-v2'
HOSTS = {'serverless-010066009163.wm226', 'serverless-010066009166.wm226'}
GIB = 1024 ** 3
DEADLINE = 900
RSS_LIMIT = 8 * GIB
PINS = {
    'cuda130-packages/nvidia/cu13/bin/nvcc': 'b8e2347d9d7fcadf8812a27324d515de99d033d3e48611811ece7e66d7895f8a',
    'cuda130-packages/nvidia/cu13/bin/ptxas': 'daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2',
    'cuda130-packages/nvidia/cu13/include/crt/host_runtime.h': '4a2fb43bf85dff114387d78fbde4c16f02ab77c3936ec0606f380f2933f54604',
    'cuda130-packages/nvidia/cu13/include/crt/host_config.h': '78036716b2d45a71dbb90e175fbfa12eb2ed54d129c6697156d6e4f9905aaa23',
    'cuda130-packages/nvidia/cu13/include/cuda_runtime.h': 'e1ef0a26e49f1776004fde1217e2fa52605adb139381ba79fe919145d993bb62',
    'cuda130-packages/nvidia/cu13/lib/libcudart.so.13': '468a02b712c80445c88abbc9538d660926e68a8ce92835f87c857000d5dfb50e',
    'env/lib/python3.11/site-packages/nvidia/cu13/include/cublas_v2.h': 'ab131d0798dbf7b96e11fc3ad4b10007e5a5afc03d0cb06f3b8ad1ca90c234ac',
    'env/lib/python3.11/site-packages/nvidia/cu13/include/cublasLt.h': 'e8c933456dda3a6c3834a3e7fbaea69e9e518be94cf329208cd9c0aca43c6373',
    'env/lib/python3.11/site-packages/nvidia/cu13/include/cublas_api.h': '10b07c516b549846a5ebe0d323232420410e28b49da1831e99e365b14b789837',
    'env/lib/python3.11/site-packages/nvidia/cu13/include/nvrtc.h': '9e4415b5ff5a2c58ec2b1a02eefcc743187c94ce48b00743c68092efe8c0f86e',
    'env/lib/python3.11/site-packages/flashinfer/jit/core.py': '836f04a87bca0a36e86c04219aac2aabbe28d87c91a577584e81f8c59e4c7a7c',
    'env/lib/python3.11/site-packages/flashinfer/jit/cpp_ext.py': '9d20f28baed969411220456b140a2786a081a2b7195d2433a70daec3c8ae7403',
    'env/lib/python3.11/site-packages/flashinfer/jit/env.py': '09d2e6d6d03770d765acb8d89db2b58cd9600babacf6fbaad9dde0002a7d2cb9',
    'env/lib/python3.11/site-packages/flashinfer/compilation_context.py': 'e55c7f58a83810590e18686954527cc62735a6e91c3fa67286a5e7f00afc9be5',
    'env/lib/python3.11/site-packages/flashinfer/jit/attention/modules.py': '5c49219256c20f72f58a2e8ca563cc598d54c2c092c125dd3f987955e63ccaea',
    'env/lib/python3.11/site-packages/flashinfer/mla/_core.py': 'cdd0743ff84520c5268e7f8eba39931fd39b6bb77763d22fe3a3e1867b68bb0f',
}
CUBIN_REL = '158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen'
CHECKSUM_SHA = 'c2d9399b2537be785882354a4f9902ed6c03136c0ea341e201eac40c3923e1dc'
META_SHA = 'f6781dfd70669167b7356fa77a805e7306443348a4b5d6a93732a27ebbb3b418'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def record(path):
    p = Path(path)
    return {'path': str(p), 'resolved': str(p.resolve(strict=True)),
            'bytes': p.stat().st_size, 'sha256': digest(p)}


def write_json(path, data):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(p)


def checked_files(pins, root=ROOT):
    rows = []
    for rel, expected in pins.items():
        row = record(root / rel)
        if row['sha256'] != expected:
            raise RuntimeError(f'Pinned source/input SHA mismatch: {rel}: {row["sha256"]}')
        rows.append(row)
    return rows


def environment(stage, root=ROOT):
    """Whitelist; never inherit mixed CUDA include/compiler/cache overrides."""
    cuda = root / 'cuda130-packages/nvidia/cu13'
    site = root / 'env/lib/python3.11/site-packages'
    return {
        'HOME': '/home/admin', 'USER': 'admin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'PATH': f'{root}/env/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'none',
        'CUDA_HOME': str(cuda), 'CUDA_PATH': str(cuda), 'CUDA_LIB_PATH': str(cuda / 'lib'),
        'FLASHINFER_NVCC': str(cuda / 'bin/nvcc'),
        'TRITON_PTXAS_PATH': str(cuda / 'bin/ptxas'),
        'CPATH': f'{cuda}/include:{stage}/include-aux',
        'LIBRARY_PATH': f'{stage}/lib:{cuda}/lib:{site}/nvidia/cu13/lib',
        'LD_LIBRARY_PATH': f'{cuda}/lib:{site}/nvidia/cu13/lib',
        'LD_PRELOAD': str(cuda / 'lib/libcudart.so.13'),
        'FLASHINFER_CUDA_ARCH_LIST': '10.3a',
        'FLASHINFER_WORKSPACE_BASE': str(stage / 'workspace'),
        'TRITON_CACHE_DIR': str(stage / 'triton-cache'),
        'XDG_CACHE_HOME': str(stage / 'xdg-cache'),
        'TMPDIR': str(stage / 'tmp'),
        'MAX_JOBS': '1', 'FLASHINFER_NVCC_THREADS': '1', 'FLASHINFER_JIT_VERBOSE': '1',
        'FLASHINFER_JIT_DEBUG': '0',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NCCL_RAS_ENABLE': '0', 'DISABLE_MMA_V5': '1', 'BATCHGEN_W8A16_DEQUANT': '0',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
    }


def make_overlay(stage, root=ROOT):
    aux = stage / 'include-aux'
    aux.mkdir()
    rows = []
    for name in ('cublasLt.h', 'cublas_api.h', 'cublas_v2.h', 'nvrtc.h'):
        src = root / 'env/lib/python3.11/site-packages/nvidia/cu13/include' / name
        dst = aux / name
        shutil.copyfile(src, dst, follow_symlinks=True)
        if dst.is_symlink() or digest(src) != digest(dst):
            raise RuntimeError('Aux header must be an exact regular-file copy')
        rows.append({'source': record(src), 'copy': record(dst)})
    (stage / 'lib').mkdir()
    link = stage / 'lib/libcudart.so'
    link.symlink_to(root / 'cuda130-packages/nvidia/cu13/lib/libcudart.so.13')
    return {'headers': rows, 'libcudart_link': str(link),
            'symlink_target': os.readlink(link), 'target': record(link)}


def deny_network(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo',
                 'socket.bind', 'socket.sendto'):
        raise RuntimeError(f'Network operation forbidden in CPU build: {event}')


def run_command(stage, name, command):
    write_json(stage / f'{name}.command.json', command)
    # Sole parent guardian owns process-group lifetime; no timeout() reaper here.
    with (stage / f'{name}.log').open('wb') as out:
        code = subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f'{name} failed, exit {code}; preserve {name}.log')


def check_cuda_hidden():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'none':
        raise RuntimeError('Explicit CUDA-hidden environment required')


def local_cubin_inputs():
    # Locate without importing FlashInfer or executing a package initializer.
    matches = list((SITE / 'flashinfer_cubin').glob(f'**/{CUBIN_REL}/checksums.txt'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one installed local fmha checksum file; got {len(matches)}')
    checksums = matches[0]
    meta = checksums.parent / 'include/flashInferMetaInfo.h'
    for p, sha in ((checksums, CHECKSUM_SHA), (meta, META_SHA)):
        if digest(p) != sha:
            raise RuntimeError(f'Local cubin metadata missing or SHA mismatch: {p}')
    return [record(checksums), record(meta)]


def header_trace_paths(text):
    return sorted({str(Path(m.group(1)).resolve()) for line in text.splitlines()
                   if (m := re.match(r'^\.+\s+(/\S*?/crt/host_runtime\.h)(?:\s|$)', line))})


def worker(stage):
    check_cuda_hidden()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = {'schema': 'fmha-cpu-build-v1', 'status': 'started',
              'pid': os.getpid(), 'hostname': socket.gethostname(),
              'started_unix': time.time(), 'runtime_qualified': False}
    write_json(stage / 'worker-report.json', report)
    pinned = []
    tracked = []
    torch = None
    absent_historical_headers = []
    try:
        pinned = checked_files(PINS)
        # Read-only historical extras: node0 has these older exact copies;
        # node1 has none. Explicit four-header overlay supplies both nodes.
        report['historical_private_aux_headers'] = []
        historical = {
            'cublas_v2.h': 'ab131d0798dbf7b96e11fc3ad4b10007e5a5afc03d0cb06f3b8ad1ca90c234ac',
            'cublas.h': '6b494bab2fa4e3b36ec320e3bae782dd6d0ca5cf74f0c4d4ee8e6c309a84fb5c',
            'cublas_api.h': PINS['env/lib/python3.11/site-packages/nvidia/cu13/include/cublas_api.h'],
            'cublasLt.h': PINS['env/lib/python3.11/site-packages/nvidia/cu13/include/cublasLt.h'],
            'nvrtc.h': PINS['env/lib/python3.11/site-packages/nvidia/cu13/include/nvrtc.h'],
        }
        should_exist = socket.gethostname() == 'serverless-010066009163.wm226'
        for name, expected in historical.items():
            path = CUDA / 'include' / name
            if path.exists() != should_exist:
                raise RuntimeError(f'Historical private auxiliary header presence drift: {path}')
            if should_exist:
                row = record(path)
                if row['sha256'] != expected:
                    raise RuntimeError(f'Historical private auxiliary header SHA drift: {path}')
                report['historical_private_aux_headers'].append(row)
                pinned.append(row)
            else:
                absent_historical_headers.append(str(path))
                report['historical_private_aux_headers'].append({'path': str(path), 'exists': False})
        tracked = [record(__file__), record(GUARD), record(SAFETY)] + pinned + local_cubin_inputs()
        for name in ('trtllm_fmha_kernel_launcher.cu', 'fmhaReduction.cu'):
            tracked.append(record(SITE / 'flashinfer/data/csrc' / name))
        report['inputs'] = tracked
        report['package_versions'] = {name: importlib.metadata.version(name) for name in (
            'torch', 'flashinfer-python', 'flashinfer-cubin', 'nvidia-cuda-runtime',
            'nvidia-cuda-crt', 'nvidia-cuda-nvcc', 'nvidia-nvvm')}
        report['package_metadata'] = [record(p) for base in (SITE, ROOT / 'cuda130-packages')
            for pattern in ('nvidia*.dist-info/METADATA', 'flashinfer*.dist-info/METADATA')
            for p in sorted(base.glob(pattern))]
        tracked += report['package_metadata']
        if report['package_versions']['flashinfer-python'] != '0.6.13' or report['package_versions']['flashinfer-cubin'] != '0.6.13':
            raise RuntimeError('FlashInfer/cubin exact 0.6.13 required')
        report['overlay'] = json.loads((stage / 'overlay.json').read_text())
        tracked += [row['copy'] for row in report['overlay']['headers']]
        tracked += [report['overlay']['target']]
        write_json(stage / 'inputs.json', tracked)
        sys.addaudithook(deny_network)
        run_command(stage, 'nvcc-version', [str(CUDA / 'bin/nvcc'), '--version'])
        run_command(stage, 'ptxas-version', [str(CUDA / 'bin/ptxas'), '--version'])
        probe = stage / 'header-probe.cu'
        probe.write_text('#include <cuda_runtime.h>\n#include <cublasLt.h>\n#include <cublas_v2.h>\n#include <nvrtc.h>\n__global__ void bg_header_probe(int* p) { if (threadIdx.x == 0) *p = 1; }\n')
        keep = stage / 'probe-keep'
        keep.mkdir()
        run_command(stage, 'header-probe', [str(CUDA / 'bin/nvcc'), '-std=c++17',
            '-arch=sm_103a', '--keep', '--keep-dir', str(keep), '-Xcompiler=-H',
            '-c', str(probe), '-o', str(stage / 'header-probe.o')])
        text = (stage / 'header-probe.log').read_text(errors='replace')
        private_header = str((CUDA / 'include/crt/host_runtime.h').resolve())
        selected_crt = header_trace_paths(text)
        if selected_crt != [private_header]:
            raise RuntimeError('Header include trace did not prove private CRT selection')
        stubs = list(keep.glob('*.stub.c'))
        if not stubs or not any('__cudaLaunch' in p.read_text(errors='replace') for p in stubs):
            raise RuntimeError('Preserved nvcc stub with CUDA launch unavailable')
        report['preprocessor_evidence'] = {'private_crt_in_include_trace': True,
            'BGenv_crt_in_include_trace': False,
            'resolved_CRT_trace_paths': selected_crt,
            'stub_files': [record(p) for p in stubs], 'include_log': record(stage / 'header-probe.log')}
        import torch as imported_torch
        torch = imported_torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA unexpectedly initialized before build')
        report['torch_cuda_initialized_before'] = False
        from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
        from flashinfer.jit import env as fi_env
        expected_cache = stage / 'workspace/.cache/flashinfer/0.6.13/103a/cached_ops'
        if fi_env.FLASHINFER_JIT_DIR != expected_cache:
            raise RuntimeError(f'FlashInfer workspace differs: {fi_env.FLASHINFER_JIT_DIR}')
        spec = gen_trtllm_gen_fmha_module()
        so = Path(spec.jit_library_path)
        if so.exists() or not so.is_relative_to(expected_cache):
            raise RuntimeError('Cold isolated fmha_gen library required')
        report['spec_name'] = spec.name
        report['cache_was_cold'] = True
        write_json(stage / 'worker-report.json', report)
        spec.build(verbose=True)
        loaded = spec.load(so)  # Deliberately no AOT/cache shortcut.
        if not callable(loaded.trtllm_paged_attention_decode):
            raise RuntimeError('Loaded module lacks actual MLA decode entrypoint')
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA unexpectedly initialized by CPU build/load')
        report['torch_cuda_initialized_after'] = False
        report['compiled_library'] = record(so)
        report['build_ninja'] = record(spec.ninja_path)
        ninja = Path(spec.ninja_path).read_text()
        if '--device-debug' in ninja or '-O3' not in ninja or str(SITE / 'nvidia/cu13/include') in ninja:
            raise RuntimeError('Actual ninja flags do not match the coherent nondebug toolchain')
        run_command(stage, 'compiled-elf-dynamic', ['readelf', '-d', str(so)])
        maps = Path('/proc/self/maps').read_text()
        (stage / 'worker-maps.txt').write_text(maps)
        report['loaded_cuda_runtime_maps'] = [s for s in maps.splitlines() if 'libcudart.so' in s]
        if not report['loaded_cuda_runtime_maps'] or any(str(CUDA / 'lib/libcudart.so.13') not in s for s in report['loaded_cuda_runtime_maps']):
            raise RuntimeError('Loaded libcudart not exclusively private 13.0.88')
        report['status'] = 'compiled_and_loaded_CPU_only_NOT_runtime_qualified'
    except BaseException:
        report['status'] = 'failed_CPU_build_NOT_runtime_qualified'
        report['error'] = traceback.format_exc()
    finally:
        mutation = []
        for old in tracked:
            try:
                if digest(old['path']) != old['sha256']:
                    mutation.append(old['path'])
            except BaseException as error:
                mutation.append(f'{old["path"]}: {error}')
        mutation += [p + ': unexpectedly appeared' for p in absent_historical_headers if Path(p).exists()]
        report['input_mutations'] = mutation
        if mutation:
            report['status'] = 'fatal_input_mutation'
        report['torch_cuda_initialized_final'] = None if torch is None else torch.cuda.is_initialized()
        if report['torch_cuda_initialized_final']:
            report['status'] = 'fatal_CUDA_initialization'
        report['finished_unix'] = time.time()
        write_json(stage / 'worker-report.json', report)
    return 0 if report['status'] == 'compiled_and_loaded_CPU_only_NOT_runtime_qualified' else 1


def supervisor(stage):
    if sys.platform != 'linux' or socket.gethostname() not in HOSTS or not Path('/.dockerenv').exists():
        raise RuntimeError('Only the two authorized Linux containers are admitted')
    if stage != STAGE or Path(sys.prefix) != ROOT / 'env':
        raise RuntimeError('Exact private output path and BG interpreter required')
    checked_files(PINS)
    if digest(GUARD) != GUARD_SHA:
        raise RuntimeError('Guardian SHA mismatch')
    if digest(SAFETY) != SAFETY_SHA:
        raise RuntimeError('Launch safety SHA mismatch')
    stage.mkdir()  # No resume, reuse, overwrite, or cache warming.
    for name in ('workspace', 'triton-cache', 'xdg-cache', 'tmp'):
        (stage / name).mkdir()
    write_json(stage / 'overlay.json', make_overlay(stage))
    env = environment(stage)
    write_json(stage / 'environment.json', env)
    spec = importlib.util.spec_from_file_location('fmha_existing_guard', GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    safety_spec = importlib.util.spec_from_file_location('fmha_existing_safety', SAFETY)
    safety = importlib.util.module_from_spec(safety_spec)
    safety_spec.loader.exec_module(safety)
    launch_safety = safety.prepare_launch_safety(ROOT)
    command = [str(ROOT / 'env/bin/python'), str(Path(__file__).resolve()), '--worker', '--out', str(stage)]
    receipt = {'schema': 'fmha-cpu-supervisor-v1', 'status': 'preparing',
               'source': record(__file__), 'guardian': record(GUARD), 'command': command,
               'launch_safety_source': record(SAFETY), 'launch_safety': launch_safety,
               'environment': env, 'deadline_seconds': DEADLINE,
               'RSS_limit_bytes': RSS_LIMIT, 'CUDA_queries': False,
               'runtime_qualified': False, 'started_unix': time.time()}
    write_json(stage / 'guard.json', receipt)
    audit_state = {}
    timer_armed = False
    child = None
    started = time.monotonic()
    pending_stop = []

    def before_guard_stop(signum, _frame):
        if not pending_stop:
            pending_stop.append(signum)

    def persist(audit):
        nonlocal audit_state, timer_armed
        audit_state = audit
        if audit.get('status') != 'waiting':
            signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            if not timer_armed:
                signal.setitimer(signal.ITIMER_REAL, .5, .5)
                timer_armed = True
            if pending_stop:
                os.kill(os.getpid(), pending_stop.pop())
        receipt['supervision'] = audit
        write_json(stage / 'guard.json', receipt)

    def limits(_sig, _frame):
        if audit_state.get('status') != 'waiting':
            signal.setitimer(signal.ITIMER_REAL, 0)
            return
        rows = guard.session_members(child.pid)
        rss = 0
        for row in rows:
            try:
                vals = Path(f'/proc/{row["pid"]}/statm').read_text().split()
                rss += int(vals[1]) * os.sysconf('SC_PAGE_SIZE')
            except FileNotFoundError:
                pass
        receipt['max_session_RSS_sum_bytes'] = max(rss, receipt.get('max_session_RSS_sum_bytes', 0))
        receipt['last_sample_unix'] = time.time()
        if time.monotonic() - started > DEADLINE or rss > RSS_LIMIT:
            signal.setitimer(signal.ITIMER_REAL, 0)
            receipt['limit_violation'] = 'deadline' if time.monotonic() - started > DEADLINE else 'session_RSS_sum'
            # Guardian owns TERM/INT: a requested flag cannot be swallowed by
            # its persistence callback's exception shielding. No disk write in
            # this handler, avoiding interruption of an atomic guard.json write.
            os.kill(os.getpid(), signal.SIGTERM)

    old_alarm = signal.signal(signal.SIGALRM, limits)
    old_stops = {sig: signal.signal(sig, before_guard_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (stage / 'worker.log').open('wb') as log:
            receipt['core_policy_immediately_before_Popen'] = safety.assert_self_core_policy()
            child = subprocess.Popen(command, env=env, cwd=stage, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            # No poll()/wait() or competing reaper before the guardian.
            audit = guard.supervise_child(child, release_grace_seconds=180,
                group_term_grace_seconds=10, kill_release_grace_seconds=30, on_update=persist)
        worker_report = json.loads((stage / 'worker-report.json').read_text())
        okay = (audit.get('status') == 'exited' and audit.get('exit_code') == 0
                and audit.get('child_reaped') and not audit.get('parent_cleanup_required')
                and not receipt.get('limit_violation')
                and worker_report.get('status') == 'compiled_and_loaded_CPU_only_NOT_runtime_qualified')
        receipt['status'] = ('passed_CPU_build_load_NOT_runtime_qualified' if okay
                             else 'failed_preserved_parent_review_required')
        return 0 if okay else 1
    except BaseException:
        receipt['status'] = 'failed_preserved_parent_review_required'
        receipt['supervisor_error'] = traceback.format_exc()
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for sig, handler in old_stops.items():
            signal.signal(sig, handler)
        receipt['finished_unix'] = time.time()
        write_json(stage / 'guard.json', receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--out', type=Path, default=STAGE)
    args = p.parse_args()
    if args.out != STAGE:
        p.error('Only the fixed private new stage is admitted')
    return worker(args.out) if args.worker else supervisor(args.out)


if __name__ == '__main__':
    raise SystemExit(main())
