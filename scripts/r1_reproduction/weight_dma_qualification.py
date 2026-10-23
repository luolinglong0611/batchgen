#!/usr/bin/env python3
"""Single-shard, single-worker native memfd attachment and weight DMA gate.

The supervisor bounds the owner and spawned worker lifetime. The owner keeps a
Parameter_Server(False, False) alive until its worker exits. The worker attaches
Weights_Storage through the owner's /proc/PID/fd entries, registers host memory,
and round-trips each tensor on one GPU. No model service or full-model allocation.
"""
import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
BLOCK = 8 * 1024**2
MODULE = 'r1_dma_single_shard'
SCOPE = 'Single converted R1 shard, one spawned worker, native host registration and tensor DMA; not full runtime or a performance benchmark'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def save(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def interrupt_supervisor(signum, frame):
    raise KeyboardInterrupt(f'Supervisor received signal {signum}')


def kill_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def stop_worker(worker):
    if worker is None:
        return
    if worker.is_alive():
        worker.terminate()
        worker.join(10)
    if worker.is_alive():
        worker.kill()
        worker.join(10)
    if worker.is_alive():
        raise RuntimeError('Worker could not be reaped; owner must remain alive until supervisor kills group')


def worker_main(settings):
    report = {'status': 'failed', 'pid': os.getpid(), 'owner_pid': settings['owner_pid'],
              'scope': settings.get('scope', SCOPE), 'tensors': []}
    storage = None
    tensors = tensor = gpu = returned = raw = host_raw = None
    stream = None
    start = time.monotonic()
    try:
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        import libucx
        libucx.load_library()
        from batchgen import core_engine
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(settings['physical_gpu']):
            raise RuntimeError('Unexpected CUDA_VISIBLE_DEVICES')
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError('Require exactly one visible SM103 device')
        torch.cuda.set_device(0)
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(0), logical_device=0,
                      physical_gpu=settings['physical_gpu'], core_engine=core_engine.__file__,
                      core_engine_sha256=digest(Path(core_engine.__file__)))
        for fd_name in ('weights_fd', 'metadata_fd'):
            fd_path = Path(f"/proc/{settings['owner_pid']}/fd/{settings[fd_name]}")
            if not fd_path.exists():
                raise RuntimeError(f'Missing owner descriptor: {fd_name}')
        storage = core_engine.Weights_Storage(0)
        init_start = time.monotonic()
        storage.Init(settings['weight_name'], settings['size'], settings['metadata_name'],
                     False, False, settings['owner_pid'], settings['weights_fd'], settings['metadata_fd'])
        report['attach_and_register_seconds'] = time.monotonic() - init_start
        if settings.get('registration_barrier') is not None:
            print(json.dumps({'event': 'host_registered', 'physical_gpu': settings['physical_gpu'],
                              'bytes': settings['size']}), flush=True)
            # Keep all native registrations alive together before testing DMA.
            settings['registration_barrier'].wait(timeout=settings['barrier_timeout'])
            report['simultaneous_registration_barrier_passed'] = True
        tensors = storage.get_tensor(MODULE)
        metadata = settings['state_dict']
        if set(tensors) != set(metadata):
            raise AssertionError('Worker module tensor keys differ from the complete shard')
        stream = torch.cuda.Stream(device=0)
        dtypes = {}
        with Path(settings['binary']).open('rb') as source:
            for index, (name, tensor) in enumerate(sorted(tensors.items()), 1):
                expected = metadata[name]
                dtype = str(tensor.dtype).removeprefix('torch.')
                if list(tensor.shape) != expected['shape'] or dtype != expected['dtype']:
                    raise AssertionError(f'Native shape/dtype mismatch: {name}')
                if tensor.device.type != 'cpu' or not tensor.is_contiguous() or not tensor.is_pinned():
                    raise AssertionError(f'Native attached tensor is not contiguous CUDA-registered host memory: {name}')
                if tensor.numel() * tensor.element_size() != expected['byte_size']:
                    raise AssertionError(f'Native byte count mismatch: {name}')
                # Copies preserve dtype; no FP8/BF16 arithmetic or conversion.
                gpu = torch.empty_like(tensor, device='cuda:0')
                returned = torch.empty_like(tensor, device='cpu', pin_memory=True)
                with torch.cuda.stream(stream):
                    gpu.copy_(tensor, non_blocking=True)
                    returned.copy_(gpu, non_blocking=True)
                stream.synchronize()
                raw = returned.view(torch.uint8).numpy().reshape(-1)
                host_raw = tensor.view(torch.uint8).numpy().reshape(-1)
                source.seek(expected['offset'])
                expected_hash, host_hash, roundtrip_hash = (hashlib.sha256() for _ in range(3))
                for offset in range(0, raw.size, BLOCK):
                    block = source.read(min(BLOCK, raw.size - offset))
                    if len(block) != min(BLOCK, raw.size - offset):
                        raise AssertionError(f'Truncated original binary: {name}')
                    expected_bytes = np.frombuffer(block, dtype=np.uint8)
                    host_chunk = host_raw[offset:offset + len(block)]
                    returned_chunk = raw[offset:offset + len(block)]
                    if not np.array_equal(host_chunk, expected_bytes):
                        raise AssertionError(f'Parent memfd/worker attached bytes differ: {name}, offset={offset}')
                    if not np.array_equal(returned_chunk, expected_bytes):
                        raise AssertionError(f'H2D/D2H bytes differ: {name}, offset={offset}')
                    expected_hash.update(block)
                    host_hash.update(memoryview(host_chunk))
                    roundtrip_hash.update(memoryview(returned_chunk))
                hashes = [h.hexdigest() for h in (expected_hash, host_hash, roundtrip_hash)]
                if len(set(hashes)) != 1:
                    raise AssertionError(f'SHA256 mismatch: {name}')
                report['tensors'].append({'name': name, 'dtype': dtype, 'shape': list(tensor.shape),
                                          'bytes': int(raw.size), 'sha256': hashes[0],
                                          'host_registered': True, 'host_exact': True, 'roundtrip_exact': True})
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                raw = host_raw = gpu = returned = None
                if index % 25 == 0 or index == len(tensors):
                    print(json.dumps({'dma_verified_tensors': index, 'total': len(tensors)}), flush=True)
        if not {'float8_e4m3fn', 'float32', 'bfloat16'} <= set(dtypes):
            raise AssertionError(f'Required FP8/scales/BF16 coverage missing: {dtypes}')
        report.update(status='passed', dtype_tensor_counts=dtypes,
                      tensor_count=len(tensors), bytes_verified=sum(t['bytes'] for t in report['tensors']))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        # All from_blob aliases must die before native storage unregisters and
        # unmaps; the parent holds its independent ownership until we exit.
        if stream is not None:
            try:
                stream.synchronize()
            except Exception:
                report['status'] = 'failed'
                report['cleanup_error'] = traceback.format_exc()
        tensors = tensor = raw = host_raw = gpu = returned = None
        # NumPy chunk views can also retain the last from_blob mapping.
        if 'host_chunk' in locals():
            del host_chunk, returned_chunk
        gc.collect()
        storage = None
        gc.collect()
        report['storage_released_before_exit'] = True
        report['seconds'] = time.monotonic() - start
        save(settings['worker_report'], report)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


def owner_main(args):
    report = {'status': 'failed', 'scope': SCOPE, 'owner_pid': os.getpid(),
              'hostname': socket.gethostname(), 'physical_gpu': args.physical_gpu,
              'hugetlbfs': False, 'thp': False, 'worker_start_method': 'spawn'}
    owner = worker = None
    start = time.monotonic()
    try:
        binary = args.converted / (args.shard + '.bin')
        metadata_path = args.converted / (args.shard + '.json')
        receipt_path = args.receipts / (args.shard + '.safetensors.converted.json')
        receipt = json.loads(receipt_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        size = binary.stat().st_size
        before = (binary.stat().st_ino, size, binary.stat().st_mtime_ns)
        validation = receipt['validation']
        if not 0 < size < 8 * 1024**3 or metadata['total_byte_size'] != size:
            raise ValueError('Only one converted shard smaller than 8 GiB is allowed')
        if metadata['file_name'] != binary.name or validation['binary_size'] != size:
            raise ValueError('Binary metadata/receipt mismatch')
        if digest(metadata_path) != validation['metadata_sha256'] or digest(binary) != validation['binary_sha256']:
            raise ValueError('Converted shard no longer matches its verified receipt')
        records = metadata['state_dict']
        if len(records) != args.expected_tensors:
            raise ValueError(f'Expected all {args.expected_tensors} tensors, found {len(records)}')
        for name, record in records.items():
            if record['offset'] < 0 or record['byte_size'] < 1 or record['offset'] + record['byte_size'] > size:
                raise ValueError(f'Invalid tensor extent: {name}')
        sys.path.insert(0, str(ROOT))
        import torch
        import libucx
        libucx.load_library()
        from batchgen import core_engine
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError('Require one visible SM103 device')
        report.update(core_engine=core_engine.__file__, torch=torch.__version__,
                      shard=args.shard, binary_size=size, receipt=receipt,
                      script_sha256=digest(Path(__file__)))
        # A directory containing exactly this shard prevents the loader from
        # scanning the rest of the concurrently converted model.
        with tempfile.TemporaryDirectory(prefix='r1-weight-dma-', dir=args.converted.parent) as directory:
            view = Path(directory)
            os.link(binary, view / binary.name)
            os.link(metadata_path, view / metadata_path.name)
            name = f'/r1_dma_{os.getpid()}'
            allocated_size = math.ceil(size / 4096) * 4096
            owner = core_engine.Parameter_Server(False, False)
            owner.Init(name, name + '_metadata', allocated_size, str(view),
                       {key: {'module_key': MODULE, 'tensor_key': key} for key in records})
            if owner.get_skeleton_state_dict():
                raise AssertionError('All shard tensors must be mapped into the single worker module')
            settings = {'owner_pid': os.getpid(), 'weights_fd': owner.weights_memfd_fd(),
                        'metadata_fd': owner.tensor_meta_memfd_fd(), 'size': allocated_size,
                        'weight_name': name, 'metadata_name': name + '_metadata',
                        'state_dict': records, 'binary': str(binary.resolve()),
                        'worker_report': str(args.report.with_suffix('.worker.json')),
                        'physical_gpu': args.physical_gpu}
            for key in ('weights_fd', 'metadata_fd'):
                if settings[key] < 0:
                    raise AssertionError(f'Missing native memfd {key}')
            if os.fstat(settings['weights_fd']).st_size < size:
                raise AssertionError('Weight memfd is too small')
            worker = multiprocessing.get_context('spawn').Process(target=worker_main, args=(settings,))
            worker.start()
            report['worker_pid'] = worker.pid
            worker.join(args.worker_timeout)
            if worker.is_alive():
                raise TimeoutError('DMA worker exceeded its timeout')
            report['worker_exit_code'] = worker.exitcode
            worker_report = Path(settings['worker_report'])
            if worker_report.exists():
                report['worker'] = json.loads(worker_report.read_text())
            if worker.exitcode != 0 or report.get('worker', {}).get('status') != 'passed':
                raise RuntimeError('DMA worker failed; see its report or native log')
            after = (binary.stat().st_ino, binary.stat().st_size, binary.stat().st_mtime_ns)
            if after != before:
                raise RuntimeError('Source binary changed during qualification')
            report['status'] = 'passed'
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        stop_worker(worker)
        report['worker_reaped_before_owner_release'] = True
        owner = None
        gc.collect()
        report['owner_released'] = True
        report['seconds'] = time.monotonic() - start
        save(args.report, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('worker', 'receipt')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--shard', default='model-00001-of-000163')
    parser.add_argument('--expected-tensors', type=int, default=270)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--physical-gpu', type=int, default=1)
    parser.add_argument('--worker-timeout', type=int, default=600)
    parser.add_argument('--total-timeout', type=int, default=900)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--owner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
        parser.error('Run only inside the explicitly allocated container')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        parser.error('Set CUDA_VISIBLE_DEVICES to exactly the allocated physical GPU before launching')
    if not 0 < args.worker_timeout < args.total_timeout or args.expected_tensors < 1:
        parser.error('Invalid timeout/tensor-count settings')
    args.report = args.report.resolve()
    if args.owner:
        return owner_main(args)
    if args.report.exists() or args.report.with_suffix('.worker.json').exists():
        parser.error('Use a new report name; previous evidence must be preserved')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupt_supervisor)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--owner'],
                               start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        code = process.wait(timeout=args.total_timeout)
        if code != 0:
            # Also reap orphaned descendants if native owner code aborted.
            kill_group(process.pid, signal.SIGKILL)
        if code != 0 and not args.report.exists():
            save(args.report, {'status': 'failed', 'scope': SCOPE, 'owner_exit_code': code,
                               'error': 'Owner process failed before publishing its report'})
        return code
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        kill_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            kill_group(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        save(args.report, {'status': 'failed', 'scope': SCOPE, 'owner_exit_code': process.returncode,
                           'error': 'Supervisor timeout/interruption; owner and worker process group terminated'})
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
