#!/usr/bin/env python3
"""Probe the R1 host reservation and simultaneous GPU worker registrations.

Only the first verified shard contains model weights. The remaining reservation
is zero-filled by the native allocator. This tests capacity, shared attachment,
registration and real tensor DMA, not full-model loading or inference. Defaults
are 675 GiB and eight workers; smaller reservations and one worker isolate failures.
"""
import argparse
import gc
import json
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

import weight_dma_qualification as dma


GIB = 1024**3
SCOPE = __doc__


def memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if fields[0] in ('MemTotal:', 'MemAvailable:', 'MemFree:', 'Shmem:', 'Mlocked:'):
            values[fields[0][:-1]] = int(fields[1]) * 1024
    return values


def worker(settings):
    # Set visibility before importing torch in this fresh spawned process.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['physical_gpu'])
    dma.worker_main(settings)


def owner(args):
    report = {'status': 'failed', 'scope': SCOPE, 'hostname': socket.gethostname(),
              'reservation_bytes': args.reservation_gib*GIB, 'worker_count': args.workers,
              'workers': [], 'memory_before': memory(),
              'hugetlbfs': False, 'thp': False, 'source_sha256': dma.digest(Path(__file__)),
              'dma_source_sha256': dma.digest(Path(dma.__file__))}
    native = None
    children = []
    started = time.monotonic()
    try:
        if report['memory_before']['MemAvailable'] < (args.reservation_gib+175)*GIB:
            raise RuntimeError('Require reservation plus at least 175 GiB available headroom')
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                              check=True, text=True, capture_output=True).stdout.strip()
        if busy:
            raise RuntimeError('GPU processes are present; do not overlap this qualification')
        shard = 'model-00001-of-000163'
        binary, metadata_path = args.converted/(shard+'.bin'), args.converted/(shard+'.json')
        receipt = json.loads((args.receipts/(shard+'.safetensors.converted.json')).read_text())
        metadata = json.loads(metadata_path.read_text())
        validation = receipt['validation']
        if (dma.digest(binary) != validation['binary_sha256'] or
                dma.digest(metadata_path) != validation['metadata_sha256']):
            raise RuntimeError('Converted shard receipt mismatch')
        if binary.stat().st_size != validation['binary_size'] or len(metadata['state_dict']) != 270:
            raise RuntimeError('Unexpected first shard')
        sys.path.insert(0, str(dma.ROOT))
        import libucx
        libucx.load_library()
        from batchgen import core_engine
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(args.workers)
        with tempfile.TemporaryDirectory(prefix='r1-reservation-', dir=args.converted.parent) as directory:
            view = Path(directory)
            os.link(binary, view/binary.name)
            os.link(metadata_path, view/metadata_path.name)
            name = f'/r1_registration_{os.getpid()}'
            native = core_engine.Parameter_Server(False, False)
            print(json.dumps({'event': 'allocate_native_reservation', 'bytes': args.reservation_gib*GIB}), flush=True)
            native.Init(name, name+'_metadata', args.reservation_gib*GIB, str(view),
                        {key: {'module_key': dma.MODULE, 'tensor_key': key} for key in metadata['state_dict']})
            report['memory_after_reservation'] = memory()
            report['reservation_seconds'] = time.monotonic()-started
            settings = {'owner_pid': os.getpid(), 'weights_fd': native.weights_memfd_fd(),
                        'metadata_fd': native.tensor_meta_memfd_fd(), 'size': args.reservation_gib*GIB,
                        'weight_name': name, 'metadata_name': name+'_metadata',
                        'state_dict': metadata['state_dict'], 'binary': str(binary.resolve()),
                        'scope': SCOPE, 'registration_barrier': barrier, 'barrier_timeout': 900}
            if settings['weights_fd'] < 0 or settings['metadata_fd'] < 0:
                raise RuntimeError('Missing owner memfd')
            if os.fstat(settings['weights_fd']).st_size != args.reservation_gib*GIB:
                raise RuntimeError('Native reservation size differs')
            print(json.dumps({'event': 'reservation_ready', 'memory': memory()}), flush=True)
            for gpu in range(args.workers):
                path = args.report.with_suffix(f'.gpu{gpu}.json')
                child = context.Process(target=worker, args=({**settings, 'physical_gpu': gpu,
                                                             'worker_report': str(path)},))
                child.start()
                children.append((child, path))
            deadline = time.monotonic()+1200
            while any(child.is_alive() for child, _ in children):
                if time.monotonic() > deadline:
                    raise TimeoutError('Workers exceeded 1200 seconds')
                if any(child.exitcode not in (None, 0) for child, _ in children):
                    raise RuntimeError('A worker failed; stop remaining workers and retain reports')
                for child, _ in children:
                    child.join(timeout=0.25)
            for child, path in children:
                result = json.loads(path.read_text()) if path.exists() else {}
                report['workers'].append({'pid': child.pid, 'exit_code': child.exitcode,
                                          'report': str(path), 'status': result.get('status'),
                                          'simultaneous_registration_barrier_passed': result.get('simultaneous_registration_barrier_passed'),
                                          'attach_and_register_seconds': result.get('attach_and_register_seconds'),
                                          'tensor_count': result.get('tensor_count')})
                if child.exitcode != 0 or result.get('status') != 'passed' or not result.get('simultaneous_registration_barrier_passed'):
                    raise RuntimeError('Worker qualification failed')
            report['status'] = 'passed'
    except Exception:
        report['traceback'] = traceback.format_exc()
    finally:
        for child, _ in children:
            dma.stop_worker(child)
        # Preserve exit codes and partial worker evidence on failure as well.
        report['workers'] = []
        for child, path in children:
            result = json.loads(path.read_text()) if path.exists() else {}
            report['workers'].append({'pid': child.pid, 'exit_code': child.exitcode,
                                      'report': str(path), 'status': result.get('status'),
                                      'error': result.get('error'),
                                      'simultaneous_registration_barrier_passed': result.get('simultaneous_registration_barrier_passed'),
                                      'attach_and_register_seconds': result.get('attach_and_register_seconds'),
                                      'tensor_count': result.get('tensor_count')})
        native = None
        gc.collect()
        report.update(memory_after_cleanup=memory(), seconds_not_benchmark=time.monotonic()-started)
        dma.save(args.report, report)
        print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 8), default=8)
    parser.add_argument('--reservation-gib', type=int, choices=(8, 64, 128, 256, 675), default=675)
    parser.add_argument('--owner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
        parser.error('Run only in the allocated container')
    if args.owner:
        return owner(args)
    if args.report.exists() or any(args.report.parent.glob(args.report.stem+'.gpu*.json')):
        parser.error('Choose a new report name to preserve prior evidence')
    signal.signal(signal.SIGTERM, dma.interrupt_supervisor)
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--owner'],
                               start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        code = process.wait(timeout=1800)
        if code != 0:
            dma.kill_group(process.pid, signal.SIGKILL)
            if not args.report.exists():
                dma.save(args.report, {'status': 'failed', 'scope': SCOPE, 'owner_exit_code': code})
        return code
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        dma.kill_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dma.kill_group(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        dma.save(args.report, {'status': 'failed', 'scope': SCOPE, 'error': 'Timeout/interruption'})
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
