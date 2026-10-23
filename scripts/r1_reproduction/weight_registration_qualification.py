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
import hashlib
import shutil
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
    os.environ['BATCHGEN_HOST_REGISTER_CHUNK_GIB'] = str(settings['native_register_chunk_gib'])
    if settings.get('registration_chunk_gib'):
        chunked_worker(settings)
    else:
        dma.worker_main(settings)


def chunked_worker(settings):
    """Driver isolation: same memfd, chunked registration, sampled byte DMA.

    This branch does not call native Weights_Storage and cannot qualify it.
    """
    import ctypes
    import mmap
    import torch
    report = {'status': 'failed', 'scope': 'CUDA driver chunked registration probe; not native Weights_Storage',
              'pid': os.getpid(), 'physical_gpu': settings['physical_gpu'], 'chunks': [],
              'registered_bytes': 0, 'sampled_bytes': 0, 'cleanup_errors': []}
    mapping = source = destination = actual = None
    registered = []
    start = time.monotonic()
    try:
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError('Require one visible SM103 GPU')
        fd = os.open(f"/proc/{settings['owner_pid']}/fd/{settings['weights_fd']}", os.O_RDWR)
        try:
            mapping = mmap.mmap(fd, settings['size'], flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        runtime = torch.cuda.cudart()
        chunk_size = settings['registration_chunk_gib']*GIB
        for offset in range(0, settings['size'], chunk_size):
            size = min(chunk_size, settings['size']-offset)
            before = time.monotonic()
            error = runtime.cudaHostRegister(base+offset, size, 0)
            if int(error) != 0:
                raise RuntimeError(f'cudaHostRegister offset={offset} size={size} error={error}')
            registered.append(base+offset)
            report['registered_bytes'] += size
            report['chunks'].append({'offset': offset, 'bytes': size, 'seconds': time.monotonic()-before,
                                     'memory': memory()})
            print(json.dumps({'event': 'chunk_registered', 'physical_gpu': settings['physical_gpu'],
                              'registered_bytes': report['registered_bytes']}), flush=True)
        report['attach_and_register_seconds'] = time.monotonic()-start
        settings['registration_barrier'].wait(timeout=settings['barrier_timeout'])
        report['simultaneous_registration_barrier_passed'] = True
        for chunk in report['chunks']:
            offset, size = chunk['offset'], chunk['bytes']
            for relative in (0, (size//2//4096)*4096, size-4096):
                source = torch.frombuffer(mapping, dtype=torch.uint8, count=4096, offset=offset+relative)
                if not source.is_pinned():
                    raise RuntimeError('Registered sample is not recognized as pinned')
                destination = torch.empty_like(source, device='cuda:0')
                actual = torch.empty_like(source, pin_memory=True)
                destination.copy_(source, non_blocking=True)
                actual.copy_(destination, non_blocking=True)
                torch.cuda.synchronize()
                if not torch.equal(source, actual):
                    raise RuntimeError('Sampled H2D/D2H bytes differ')
                report['sampled_bytes'] += source.numel()
        report['cross_boundary_copies'] = 0
        for boundary in range(chunk_size, settings['size'], chunk_size):
            source = torch.frombuffer(mapping, dtype=torch.uint8, count=65536, offset=boundary-32768)
            destination = torch.empty_like(source, device='cuda:0')
            actual = torch.empty_like(source, pin_memory=True)
            destination.copy_(source, non_blocking=True)
            actual.copy_(destination, non_blocking=True)
            torch.cuda.synchronize()
            if not torch.equal(source, actual):
                raise RuntimeError('Cross-registration-boundary DMA bytes differ')
            report['cross_boundary_copies'] += 1
            report['sampled_bytes'] += source.numel()
        report['status'] = 'passed'
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        source = destination = actual = None
        gc.collect()
        for pointer in reversed(registered):
            error = runtime.cudaHostUnregister(pointer)
            if int(error) != 0:
                report['status'] = 'failed'
                report['cleanup_errors'].append(str(error))
        if mapping is not None:
            mapping.close()
        report['seconds'] = time.monotonic()-start
        dma.save(settings['worker_report'], report)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


def shifted_fixture(binary, metadata, view, boundary):
    """Keep original tensor bytes, shift one intact shard across a nominal cut.

    A single binary avoids depending on directory iteration order. Its prefix is
    sparse zero padding; this is explicitly a derived fixture, not a model shard.
    """
    selected, record = max(metadata['state_dict'].items(), key=lambda item: item[1]['byte_size'])
    if record['byte_size'] <= 4*1024**2:
        raise ValueError('Expected a large real tensor for a boundary-crossing copy')
    prefix = ((boundary - record['offset'] - 2*1024**2)//4096)*4096
    if prefix <= 0:
        raise ValueError('Cannot shift selected tensor across the nominal boundary')
    derived = json.loads(json.dumps(metadata))
    for tensor in derived['state_dict'].values():
        tensor['offset'] += prefix
    derived['total_byte_size'] += prefix
    target = view / binary.name
    source_hash = hashlib.sha256()
    with binary.open('rb') as source, target.open('wb') as output:
        output.seek(prefix)
        for block in iter(lambda: source.read(dma.BLOCK), b''):
            source_hash.update(block)
            output.write(block)
    (view / binary.with_suffix('.json').name).write_text(json.dumps(derived))
    shifted = derived['state_dict'][selected]
    if not shifted['offset'] < boundary < shifted['offset']+shifted['byte_size']:
        raise AssertionError('Selected real tensor did not cross the nominal boundary')
    return target, derived, {'kind': 'shifted original tensor bytes with sparse zero prefix',
                             'prefix_bytes': prefix, 'nominal_boundary': boundary,
                             'crossing_tensor': selected, 'tensor_start': shifted['offset'],
                             'tensor_bytes': shifted['byte_size'],
                             'copied_source_sha256': source_hash.hexdigest(),
                             'derived_metadata_sha256': dma.digest(view / binary.with_suffix('.json').name)}


def owner(args):
    report = {'status': 'failed', 'scope': SCOPE, 'hostname': socket.gethostname(),
              'reservation_bytes': args.reservation_gib*GIB, 'worker_count': args.workers,
              'registration_mode': 'chunked_driver_probe' if args.registration_chunk_gib else 'native',
              'native_register_chunk_gib': args.native_register_chunk_gib,
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
        report['core_engine_sha256'] = dma.digest(Path(core_engine.__file__))
        report['core_engine_path'] = core_engine.__file__
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(args.workers)
        with tempfile.TemporaryDirectory(prefix='r1-reservation-', dir=args.converted.parent) as directory:
            view = Path(directory)
            if args.shift_shard_across_boundary:
                binary, metadata, fixture = shifted_fixture(binary, metadata, view, args.native_register_chunk_gib*GIB)
                if fixture['copied_source_sha256'] != validation['binary_sha256']:
                    raise RuntimeError('Original shard changed while building shifted fixture')
                report['derived_fixture'] = fixture
            else:
                os.link(binary, view/binary.name)
                os.link(metadata_path, view/metadata_path.name)
            if metadata['total_byte_size'] > args.reservation_gib*GIB:
                raise RuntimeError('Fixture exceeds reservation')
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
                        'scope': SCOPE, 'registration_barrier': barrier, 'barrier_timeout': 900,
                        'registration_chunk_gib': args.registration_chunk_gib,
                        'native_register_chunk_gib': args.native_register_chunk_gib}
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
                                      'tensor_count': result.get('tensor_count'),
                                      'registered_bytes': result.get('registered_bytes'),
                                      'cross_boundary_copies': result.get('cross_boundary_copies')})
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
    parser.add_argument('--registration-chunk-gib', type=int, choices=(0, 16), default=0,
                        help='16 selects a non-native driver diagnosis with sampled DMA; 0 uses native storage')
    parser.add_argument('--native-register-chunk-gib', type=int, choices=(0, 16), default=0,
                        help='Native tensor-aware registration; default 0 preserves whole-region behavior')
    parser.add_argument('--shift-shard-across-boundary', action='store_true',
                        help='Derived real-weight fixture with a tensor crossing the nominal native chunk cut')
    parser.add_argument('--owner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.registration_chunk_gib and args.native_register_chunk_gib:
        parser.error('Choose native registration or driver isolation, not both')
    if args.shift_shard_across_boundary and not args.native_register_chunk_gib:
        parser.error('Shifted fixture requires explicit native tensor-aware registration')
    # Do not silently inherit a previous experiment setting.
    os.environ['BATCHGEN_HOST_REGISTER_CHUNK_GIB'] = str(args.native_register_chunk_gib)
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
