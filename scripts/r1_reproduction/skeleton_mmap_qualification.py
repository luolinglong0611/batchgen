#!/usr/bin/env python3
"""CPU-only tiny memfd qualification; does not load R1 or alter production code.

Run in the BatchGen environment with CUDA_VISIBLE_DEVICES=''. Default Linux
MAP_PRIVATE shares clean backing pages and isolates writes via COW; this is
not an enforced read-only mapping. This fixture cannot establish full-model
memory savings, GPU transfer correctness, or increased host-KV capacity.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import multiprocessing as mp
import os
import resource
from pathlib import Path
import time
import traceback


def create_memfd(name):
    # Match production's libc fallback for Python builds without os.memfd_create.
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, os.MFD_CLOEXEC)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'memfd_create failed')
    return fd


def tensor_manifest(tensors):
    """Compare complete logical bytes and complete serialized storage bytes."""
    import torch
    result = {}
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
        logical = tensor.contiguous().view(torch.uint8)
        result[name] = {
            'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'dtype': str(tensor.dtype), 'storage_offset': tensor.storage_offset(),
            'storage_bytes': storage.nbytes(),
            'logical_sha256': hashlib.sha256(logical.numpy().tobytes()).hexdigest(),
            'storage_sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    return result


def mapping_stats(inode):
    """Only this memfd's mappings; RSS must not be summed to claim savings."""
    result = {'mappings': 0, 'Rss_kib': 0, 'Pss_kib': 0,
              'Private_Dirty_kib': 0, 'Shared_Dirty_kib': 0}
    selected = False
    for line in Path('/proc/self/smaps').read_text().splitlines():
        columns = line.split()
        if columns and '-' in columns[0] and len(columns) >= 5:
            selected = columns[4] == str(inode)
            if selected:
                result['mappings'] += 1
        elif selected and columns and columns[0].rstrip(':') + '_kib' in result:
            result[columns[0].rstrip(':') + '_kib'] += int(columns[1])
    return result


def fixture():
    import torch
    # 4 MiB main storage: large enough for meaningful COW/page-sharing evidence.
    base = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1024, 1024)
    bf16 = torch.arange(256, dtype=torch.float32).to(torch.bfloat16).reshape(16, 16)
    fp8 = (torch.arange(256, dtype=torch.float32) / 16).to(torch.float8_e4m3fn).reshape(16, 16)
    return {'fp32': base, 'alias': base.view(-1), 'noncontiguous': base.t(),
            'offset_view': base[1:5, 2:19:2], 'bf16': bf16,
            'bf16_view': bf16.t(), 'fp8': fp8, 'fp8_view': fp8.t()}


def alias_contract(tensors):
    groups = [('fp32', 'alias', 'noncontiguous', 'offset_view'),
              ('bf16', 'bf16_view'), ('fp8', 'fp8_view')]
    return all(len({tensors[k].untyped_storage().data_ptr() for k in group}) == 1
               for group in groups)


def _child(conn, path, inode):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import mmap
        import torch
        torch.set_num_threads(1)
        assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
        assert not torch.cuda.is_initialized()
        ordinary = torch.load(path, map_location='cpu', weights_only=True)
        mapped = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        before = tensor_manifest(mapped)
        assert before == tensor_manifest(ordinary), 'ordinary/mmap bytes differ'
        assert alias_contract(mapped) and alias_contract(ordinary)
        assert not mapped['noncontiguous'].is_contiguous()
        del ordinary
        gc.collect()
        conn.send({'event': 'loaded', 'pid': os.getpid(), 'manifest': before,
                   'alias_preserved': True, 'mmap_flags': 'MAP_PRIVATE',
                   'cuda_initialized': torch.cuda.is_initialized(),
                   'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        while True:
            command = conn.recv()
            if command == 'mutate':
                mapped['fp32'][0, 0] = 31415.0
                assert mapped['alias'][0].item() == 31415.0
                assert mapped['noncontiguous'][0, 0].item() == 31415.0
                conn.send({'event': 'mutated', 'manifest': tensor_manifest(mapped),
                           'mapping': mapping_stats(inode)})
            elif command == 'snapshot':
                conn.send({'event': 'snapshot', 'manifest': tensor_manifest(mapped),
                           'mapping': mapping_stats(inode)})
            elif command == 'owner_closed':
                still_readable = tensor_manifest(mapped)
                try:
                    torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                except FileNotFoundError as exc:
                    error = type(exc).__name__
                else:
                    raise AssertionError('New load unexpectedly succeeded after owner close')
                conn.send({'event': 'owner_closed', 'manifest': still_readable,
                           'new_load_error': error, 'mapping': mapping_stats(inode)})
            elif command == 'release':
                del mapped
                gc.collect()
                stats = mapping_stats(inode)
                assert stats['mappings'] == 0, stats
                conn.send({'event': 'released', 'mapping': stats,
                           'cuda_initialized': torch.cuda.is_initialized(),
                   'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
                break
            else:
                raise ValueError(command)
    except BaseException:
        conn.send({'event': 'error', 'traceback': traceback.format_exc()})
    finally:
        conn.close()


def _receive(conn, event, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not conn.poll(remaining):
        raise TimeoutError(f'Waiting for child {event}')
    result = conn.recv()
    if result.get('event') != event:
        raise RuntimeError(f'Expected {event}, received {result}')
    return result


def qualify(timeout=90):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' explicitly for this CPU-only test")
    if not Path('/proc/self/smaps').exists():
        raise RuntimeError('Linux memfd and /proc/smaps are required')
    import mmap
    import torch
    torch.set_num_threads(1)
    assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    deadline = started + timeout
    fd = create_memfd('batchgen_skeleton_mmap_qualification')
    path = f'/proc/{os.getpid()}/fd/{fd}'
    children, connections = [], []
    report = {'status': 'failed', 'scope': 'tiny CPU memfd only; not full-model/GPU/memory-capacity qualification',
              'torch_version': torch.__version__, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        tensors = fixture()
        expected = tensor_manifest(tensors)
        with os.fdopen(os.dup(fd), 'wb') as handle:
            torch.save(tensors, handle)
        del tensors
        inode = os.fstat(fd).st_ino
        report.update(serialized_bytes=os.fstat(fd).st_size, memfd_inode=inode,
                      expected_manifest=expected)
        ctx = mp.get_context('spawn')
        for _ in range(2):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_child, args=(child, path, inode))
            process.start()
            child.close()
            children.append(process)
            connections.append(parent)
        loaded = [_receive(c, 'loaded', deadline) for c in connections]
        assert all(r['manifest'] == expected for r in loaded)
        report['loaded'] = loaded
        for c in connections:
            c.send('snapshot')
        report['both_mapped'] = [_receive(c, 'snapshot', deadline) for c in connections]
        assert all(r['mapping']['mappings'] > 0 for r in report['both_mapped'])
        assert all(r['mapping']['Pss_kib'] < r['mapping']['Rss_kib']
                   for r in report['both_mapped']), 'No measured page sharing'
        connections[0].send('mutate')
        mutated = _receive(connections[0], 'mutated', deadline)
        connections[1].send('snapshot')
        unaffected = _receive(connections[1], 'snapshot', deadline)
        assert mutated['manifest']['fp32']['logical_sha256'] != expected['fp32']['logical_sha256']
        assert unaffected['manifest'] == expected, 'COW contaminated sibling mapping'
        assert mutated['mapping']['Private_Dirty_kib'] > report['both_mapped'][0]['mapping']['Private_Dirty_kib'], 'COW page not observed'
        # A fresh ordinary read establishes that the serialized backing bytes stayed unchanged.
        reread = torch.load(path, map_location='cpu', weights_only=True)
        assert tensor_manifest(reread) == expected, 'COW modified memfd backing'
        del reread
        report['cow'] = {'writer': mutated, 'reader': unaffected, 'backing_unchanged': True}
        os.close(fd)
        fd = -1
        assert not Path(path).exists()
        for c in connections:
            c.send('owner_closed')
        closed = [_receive(c, 'owner_closed', deadline) for c in connections]
        assert closed[0]['manifest'] == mutated['manifest']
        assert closed[1]['manifest'] == expected
        report['after_owner_close'] = closed
        for c in connections:
            c.send('release')
        report['released'] = [_receive(c, 'released', deadline) for c in connections]
        for process in children:
            process.join(max(0, deadline - time.monotonic()))
            assert process.exitcode == 0, (process.pid, process.exitcode)
            assert not Path(f'/proc/{process.pid}').exists()
        report.update(status='passed', child_exit_codes=[p.exitcode for p in children],
                      child_processes_absent=True, cuda_initialized=torch.cuda.is_initialized(),
                      parent_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_seconds=time.monotonic() - started)
    except BaseException:
        report['error'] = traceback.format_exc()
    finally:
        if fd >= 0:
            os.close(fd)
        for process in children:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for conn in connections:
            conn.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 110:
        parser.error('timeout must be between 1 and 110 seconds')
    report = qualify(args.timeout)
    # Never overwrite previous evidence.
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'elapsed_seconds': report.get('elapsed_seconds'), 'error': report.get('error')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
