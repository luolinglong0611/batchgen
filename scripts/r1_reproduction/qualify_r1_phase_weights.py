#!/usr/bin/env python3
"""Real R1 staged scale/dense GPU parity and copy-count timing qualification.

Run under the owned GPU supervisor, after serving clients and GPUs are idle.
Input is an immutable CPU-staged subset from the live original R1 skeleton.
This does not qualify full-model quality or physical prefill H2D bandwidth.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    for name in ('candidate', 'inputs', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.candidate.read_bytes()).hexdigest() == args.candidate_sha256
    os.environ['BATCHGEN_R1_PHASE_PREPARE'] = '1'
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.get_device_capability() == (10, 3)
    spec = importlib.util.spec_from_file_location('phase_candidate', args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = torch.load(args.inputs, weights_only=True, map_location='cpu', mmap=True)
    scales = {k: v for k, v in source.items() if k.startswith('model.layers.') and any(part in k for part in ('.self_attn.', '.mlp.shared_experts.', '.mlp.experts.'))}
    rows = []
    for phase, rank in [('prefill', 0), ('decode', 0), ('decode', 8), ('decode', 15)]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        packed = candidate.pack_phase_scales(scales, phase, rank, 16, 'cuda:0')
        torch.cuda.synchronize()
        bulk_seconds = time.perf_counter() - start
        expected_keys = {k for k in scales if int(k.split('.')[2]) < 61 and (phase == 'prefill' or '.mlp.experts.' not in k or rank * 16 <= int(k.split('.')[5]) < (rank + 1) * 16)}
        assert set(packed) == expected_keys
        expected = torch.cat([scales[k].reshape(-1) for k in packed])
        actual = torch.cat([v.reshape(-1) for v in packed.values()]).cpu()
        assert torch.equal(actual, expected)
        assert len({v.untyped_storage().data_ptr() for v in packed.values()}) == 1
        assert all(v._version >= 0 and tuple(v.shape) == tuple(scales[k].shape) for k, v in packed.items())
        torch.cuda.synchronize()
        start = time.perf_counter()
        ordinary = {k: scales[k].to('cuda:0') for k in packed}
        torch.cuda.synchronize()
        ordinary_seconds = time.perf_counter() - start
        rows.append({'phase': phase, 'rank': rank, 'tensors': len(packed), 'bytes': expected.numel() * 4, 'bitwise_equal': True, 'versioned_shared_storage': True, 'bulk_transfers': 1, 'ordinary_transfers': len(ordinary), 'bulk_seconds': bulk_seconds, 'ordinary_seconds': ordinary_seconds})
        del packed, ordinary, expected, actual
    from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
    old = deepseek_v3_dequantization(source['qualification_dense_original'], source['qualification_dense_scale'])
    new = source['qualification_dense_BF16']
    assert torch.equal(old, new)
    old, new = old.cuda(), new.cuda()
    torch.manual_seed(120)
    x = torch.randn(32, old.shape[1], dtype=torch.bfloat16, device='cuda')
    assert torch.equal(torch.nn.functional.linear(x, old), torch.nn.functional.linear(x, new))
    torch.cuda.synchronize()
    report = {'status': 'actual_R1_GPU_scale_and_dense_bitwise_parity_passed', 'candidate_sha256': args.candidate_sha256, 'inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(), 'cases': rows, 'dense_GPU_linear_bitwise_equal': True, 'scope': 'Real staged R1 values; full-model/strict historical quality failures remain unresolved'}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
