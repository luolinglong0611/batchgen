#!/usr/bin/env python3
"""Isolated sixteen-rank BG PyNccl timings at the actual R1 message sizes.

Run only after retiring the model. Barrier-separated repetitions intentionally
remove model arrival skew; these are not in-model communication contributions.
Each NVLS setting requires a fresh supervised torchrun and communicator.
"""
from pathlib import Path
import argparse
import ast
import logging
import hashlib
import json
import os
import socket
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-port', type=int, required=True)
    p.add_argument('--expected-pynccl-sha256', required=True)
    p.add_argument('--expected-wrapper-sha256', required=True)
    p.add_argument('--expected-nccl-setup-sha256', required=True)
    a = p.parse_args()
    assert Path('/.dockerenv').is_file() and os.getuid() == 500
    setup_file = a.repo / 'batchgen/server_worker_main_loop.py'
    setup_bytes = setup_file.read_bytes()
    assert hashlib.sha256(setup_bytes).hexdigest() == a.expected_nccl_setup_sha256
    setup_ast = ast.parse(setup_bytes)
    setup_function = next(n for n in setup_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_setup_nccl_env')
    setup_globals = {'os': os, 'logging': logging}
    exec(compile(ast.Module(body=[setup_function], type_ignores=[]), str(setup_file), 'exec'), setup_globals)
    setup_globals['_setup_nccl_env']()
    assert os.environ['NCCL_IB_GID_INDEX'] == '3'
    assert os.environ['NCCL_IB_HCA'] == '=' + ','.join(f'mlx5_bond_{i}:1' for i in range(8))
    assert os.environ['NCCL_BUFFSIZE'] == '16777216'
    assert os.environ['NCCL_RAS_ENABLE'] == '0'
    assert os.environ['NCCL_NVLS_ENABLE'] in ('0', '1')
    assert not any(k in os.environ for k in ('NCCL_ALGO', 'NCCL_PROTO'))
    for name, pin in [('pynccl.py', a.expected_pynccl_sha256),
                      ('pynccl_wrapper.py', a.expected_wrapper_sha256)]:
        f = a.repo / 'batchgen/distributed/device_communicators' / name
        assert hashlib.sha256(f.read_bytes()).hexdigest() == pin
    sys.path.insert(0, str(a.repo))
    import torch
    import torch.distributed as dist
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    assert world == 16
    torch.cuda.set_device(local)
    assert torch.cuda.get_device_capability() == (10, 3)
    from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(local).total_memory)
    from batchgen.distributed.utils import StatelessProcessGroup
    # Match BG's CPU-only metadata bootstrap. A second GPU NCCL control group
    # would introduce cross-communicator ordering into this isolated test.
    assert 1024 <= a.control_port <= 65535 and a.control_port != int(os.environ['MASTER_PORT'])
    metadata = StatelessProcessGroup.create(os.environ['MASTER_ADDR'], a.control_port, rank, world)

    communicator = None
    try:
        communicator = PyNcclCommunicator(metadata, torch.device('cuda', local))
        assert communicator.available and communicator.nccl.ncclGetVersion() == '2.27.7'
        stream = torch.cuda.default_stream(local)
        cases = []
        with communicator.change_state(enable=True):
            for local_batch in (1, 16, 32):
                x = torch.full((local_batch, 7168), rank, dtype=torch.bfloat16, device='cuda')
                y = torch.empty((world * local_batch, 7168), dtype=x.dtype, device='cuda')
                reduction = torch.full_like(y, rank + 1)
                communicator.all_gather(y, x, stream=stream)
                communicator.all_reduce(reduction, stream=stream)
                stream.synchronize()
                expected = torch.arange(world, device='cuda', dtype=x.dtype).repeat_interleave(local_batch)
                assert torch.equal(y, expected[:, None].expand_as(y))
                assert torch.equal(reduction, torch.full_like(reduction, world * (world + 1) // 2))
                reduction.zero_()
                operations = {'all_gather': lambda: communicator.all_gather(y, x, stream=stream),
                              'all_reduce': lambda: communicator.all_reduce(reduction, stream=stream)}
                for _ in range(20):
                    operations['all_gather']()
                    operations['all_reduce']()
                stream.synchronize()
                timings = {name: [] for name in operations}
                for name in ('all_gather', 'all_reduce', 'all_reduce', 'all_gather') * 3:
                    metadata.barrier()
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    started = time.perf_counter()
                    for _ in range(200):
                        operations[name]()
                    end.record(stream)
                    end.synchronize()
                    timings[name].append({'stream_ms_per_collective': begin.elapsed_time(end) / 200,
                                          'wall_ms_per_collective': (time.perf_counter() - started) * 1000 / 200})
                single_calls = {name: [] for name in operations}
                for name in ('all_gather', 'all_reduce', 'all_reduce', 'all_gather') * 3:
                    metadata.barrier()
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    started = time.perf_counter()
                    operations[name]()
                    end.record(stream)
                    end.synchronize()
                    single_calls[name].append({'stream_ms_per_collective': begin.elapsed_time(end),
                                               'wall_ms_per_collective': (time.perf_counter() - started) * 1000})
                assert torch.equal(reduction, torch.zeros_like(reduction))
                assert torch.equal(y, expected[:, None].expand_as(y))
                cases.append({'local_batch': local_batch, 'global_padded_batch': local_batch * world,
                              'hidden_size': 7168, 'dtype': 'bfloat16',
                              'all_gather_input_bytes': x.numel() * x.element_size(),
                              'all_gather_output_bytes': y.numel() * y.element_size(),
                              'all_reduce_bytes': reduction.numel() * reduction.element_size(),
                              'exact_collective_validation': True, 'ABBA_repetitions': timings,
                              'queued_calls_per_repetition': 200,
                              'single_call_ABBA_repetitions': single_calls,
                              'single_call_medians': {name: {key: statistics.median(r[key] for r in rows)
                                                           for key in rows[0]} for name, rows in single_calls.items()},
                              'medians': {name: {key: statistics.median(r[key] for r in rows)
                                                for key in rows[0]} for name, rows in timings.items()}})
                del x, y, reduction, expected
        libraries = []
        for line in Path('/proc/self/maps').read_text().splitlines():
            if 'libnccl.so' in line and line.split()[-1] not in libraries:
                libraries.append(line.split()[-1])
        result = {'schema': 'r1-isolated-BG-PyNccl-actual-message-benchmark-v1',
                  'status': 'exact_collectives_and_ABBA_timings_complete', 'rank': rank,
                  'hostname': socket.gethostname(), 'world_size': world,
                  'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                  'nccl_version': communicator.nccl.ncclGetVersion(), 'device_capability': [10, 3],
                  'control_bootstrap': 'BG StatelessProcessGroup CPU TCPStore',
                  'control_port': a.control_port, 'native_nccl_setup_sha256': a.expected_nccl_setup_sha256, 'cuda_stream': stream.cuda_stream, 'NVLS_setting': os.environ['NCCL_NVLS_ENABLE'],
                  'device_name': torch.cuda.get_device_name(local),
                  'device_uuid': str(getattr(torch.cuda.get_device_properties(local), 'uuid', 'unavailable')),
                  'loaded_NCCL_paths': libraries,
                  'local_leader_NCCL_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                               for name in libraries} if local == 0 else None,
                  'cases': cases,
                  'peak_Torch_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'environment': {k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'GLOO_', 'TORCH_NCCL_'))},
                  'scope': 'Barrier-aligned single-call and 200-call queued stream/launch-wall averages with reused buffers. Removes intentional model arrival skew; do not subtract this from in-model spans as an exact wait estimate. No CUDA graph, model weights or inference; no bandwidth saturation or model speedup claim.'}
        a.output.mkdir(exist_ok=True)
        with (a.output / f'rank{rank}.json').open('x') as out:
            json.dump(result, out, indent=2)
            out.write('\n')
        metadata.barrier()
        # Keep rank0's TCPStore alive until every client completed its reads.
        if rank == 0:
            for peer in range(1, world):
                assert metadata.broadcast_obj(None, src=peer) == 'reports_closed'
        else:
            metadata.broadcast_obj('reports_closed', src=rank)
        print(json.dumps({'status': result['status'], 'rank': rank, 'NVLS_setting': result['NVLS_setting']}), flush=True)
    finally:
        if communicator is not None:
            communicator.destroy()
        del metadata


if __name__ == '__main__':
    main()
