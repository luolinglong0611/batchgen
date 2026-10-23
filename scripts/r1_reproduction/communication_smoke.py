#!/usr/bin/env python3
"""Run under torchrun in the allocated containers before loading model weights.

This validates basic CUDA/NCCL operations; it is not a model benchmark or a
substitute for BatchGen's UCX and KV-transfer correctness tests.
"""
import datetime
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist


def main():
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Run only inside the allocated container')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 3):
        raise RuntimeError(f'Expected SM103, got {capability}')
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
    try:
        expected = world * (world + 1) // 2
        timings = []
        for elements in (1024, 2**21, 2**24):
            tensor = torch.full((elements,), rank + 1, device=device, dtype=torch.float32)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            torch.testing.assert_close(tensor, torch.full_like(tensor, expected), rtol=0, atol=0)
            tensor.zero_()
            for _ in range(3):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(10):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            timings.append({'bytes': elements * 4, 'mean_seconds': (time.perf_counter()-start)/10})

        local = torch.full((1024,), rank, device=device, dtype=torch.int64)
        gathered = torch.empty(world * local.numel(), device=device, dtype=local.dtype)
        dist.all_gather_into_tensor(gathered, local)
        expected_gather = torch.arange(world, device=device).repeat_interleave(local.numel())
        torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

        received = torch.empty_like(local)
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, local, (rank + 1) % world),
            dist.P2POp(dist.irecv, received, (rank - 1) % world),
        ])
        for request in requests:
            request.wait()
        torch.testing.assert_close(received, torch.full_like(local, (rank-1) % world), rtol=0, atol=0)
        dist.barrier()
        print(json.dumps({'status': 'passed', 'rank': rank, 'world_size': world,
                          'hostname': socket.gethostname(), 'capability': capability,
                          'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'nccl': torch.cuda.nccl.version(), 'all_reduce': timings,
                          'all_gather': True, 'ring_send_recv': True}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
