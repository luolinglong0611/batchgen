#!/usr/bin/env python3
"""Native SG page64 allocator/token-address qualification, never a model run.

Default is plan-only. --execute requires a parent-owned bounded single-SM103
guard and R1_SG_ALLOCATOR_GUARDED=1. No service control or installation exists.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import traceback


LENGTHS = (63, 64, 65, 2047, 2048, 2049, 8191, 8192, 8193)
CAPACITY = 65536
PAGE = 64


def bf16_ulp_reference(value):
    """One BF16 ULP at a BF16-rounded oracle value, with subnormal floor."""
    if not math.isfinite(value):
        raise ValueError('BF16 oracle must be finite')
    exponent = math.frexp(abs(value))[1]
    return max(2.0**-133, math.ldexp(1.0, exponent - 8)) if value else 2.0**-133


def address_contract(rows, lengths):
    if (any(type(length) is not int or length < 0 for length in lengths)
            or len(rows) != len(lengths) or any(len(row) != length for row, length in zip(rows, lengths))):
        raise ValueError('Request/token address count mismatch')
    all_addresses = [value for row in rows for value in row]
    if any(type(value) is not int for value in all_addresses):
        raise ValueError('Token addresses must be integers')
    if len(set(all_addresses)) != len(all_addresses):
        raise ValueError('Live requests have overlapping token addresses')
    for row in rows:
        for start in range(0, len(row), PAGE):
            chunk = row[start:start + PAGE]
            if chunk[0] < PAGE or chunk[0] % PAGE or chunk[-1] > CAPACITY + PAGE - 1:
                raise ValueError('Invalid physical page start/range')
            if chunk != list(range(chunk[0], chunk[0] + len(chunk))):
                raise ValueError('Token offsets within a physical page are not contiguous')
    return {'unique_valid_token_addresses': len(all_addresses),
            'physical_pages': len({value // PAGE for value in all_addresses}),
            'allocator_page_size': PAGE, 'kernel_page_size': 1}


def plan(args):
    return {'schema': 'r1-sg-native-page64-qualification-v1', 'status': 'plan_only_NOT_executed',
            'lengths': list(LENGTHS), 'decode_append_lengths': [n + 1 for n in LENGTHS],
            'capacity_tokens': CAPACITY, 'allocator_page_size': PAGE, 'kernel_page_size': 1,
            'expected_hostname': args.expected_hostname,
            'limits': {'external_wall_seconds': 900, 'external_RSS_bytes': 8 * 1024**3,
                       'external_GPU_bytes': 2 * 1024**3},
            'requires_parent_guard': True, 'requires_exact_sglang': '0.5.5.post3',
            'acceptance': {'FP64_uniform_oracle': 'every_output_within_1_BF16_ULP_of_FP64_reference',
                           'tail_comparison': 'NaN_and_zero_unused_storage_outputs_bitwise_equal',
                           'cache_immutability': 'full_cache_byte_hash_unchanged_during_each_run'},
            'fixture': 'query_zero; latent_value=request_index+1+(token_index_mod17-8)/16; rope_zero',
            'requires_exact_flashinfer': '0.5.2', 'gpu_operations': False,
            'formal_test_permitted': False, 'output': str(args.output)}


def tensor_sha(tensor):
    torch = importlib.import_module('torch')
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def identity(value):
    path = Path(inspect.getsourcefile(getattr(value, 'fn', value)))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def execute(args):
    if os.environ.get('R1_SG_ALLOCATOR_GUARDED') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Native SG allocator execution requires the parent single-GPU guard')
    if not args.expected_hostname or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Native SG allocator hostname binding mismatch')
    for name, version in (('sglang', '0.5.5.post3'), ('flashinfer-python', '0.5.2')):
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Native SG allocator package mismatch: ' + name)
    torch = importlib.import_module('torch')
    if torch.__version__ != '2.9.0+cu130' or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError('Native SG allocator qualification requires common Torch and one SM103 GPU')
    allocator_module = importlib.import_module('sglang.srt.mem_cache.allocator')
    pool_module = importlib.import_module('sglang.srt.mem_cache.memory_pool')
    backend = importlib.import_module('sglang.srt.layers.attention.flashinfer_mla_backend')
    flashinfer = importlib.import_module('flashinfer')
    pool = pool_module.MLATokenToKVPool(CAPACITY, PAGE, torch.bfloat16, 512, 64, 1,
                                      'cuda', False, start_layer=0, end_layer=1)
    allocator = allocator_module.PagedTokenToKVPoolAllocator(
        CAPACITY, PAGE, torch.bfloat16, 'cuda', pool, need_sort=True)
    allocator.debug_mode = True
    cache = pool.get_key_buffer(0)
    if tuple(cache.shape) != (CAPACITY + PAGE, 1, 576) or cache.dtype != torch.bfloat16:
        raise RuntimeError('SG page64 MLA pool did not expose its native token-major page1 view')
    lengths_cpu = torch.tensor(LENGTHS, dtype=torch.int64)
    prefixes_cpu = torch.zeros_like(lengths_cpu)
    lengths = lengths_cpu.cuda()
    locations = allocator.alloc_extend(prefixes_cpu.cuda(), prefixes_cpu, lengths,
                                       lengths_cpu, torch.full_like(lengths, -1), sum(LENGTHS))
    if locations is None:
        raise RuntimeError('Native SG allocator unexpectedly exhausted the fixture pool')
    split_rows = [part.cpu().tolist() for part in locations.split(list(LENGTHS))]
    checks = [address_contract(split_rows, LENGTHS)]
    request_map = torch.full((len(LENGTHS), max(LENGTHS) + 1), -1, dtype=torch.int32, device='cuda')
    cache.fill_(float('nan'))
    for rank, row in enumerate(split_rows):
        request_map[rank, :len(row)] = torch.tensor(row, dtype=torch.int32, device='cuda')
        values = rank + 1 + (torch.arange(len(row), device='cuda') % 17 - 8).to(torch.bfloat16) / 16
        cache[torch.tensor(row, device='cuda'), 0, :512] = values[:, None]
        cache[torch.tensor(row, device='cuda'), 0, 512:] = 0
    workspace = torch.empty(128 * 1024**2, dtype=torch.uint8, device='cuda')
    wrapper = flashinfer.BatchMLAPagedAttentionWrapper(workspace, backend='fa2', use_cuda_graph=False)
    numerical = []

    def check_indices_and_attention(rows, actual_lengths):
        cpu_lengths = torch.tensor(actual_lengths, dtype=torch.int32)
        gpu_lengths = cpu_lengths.cuda()
        ptr = torch.cat((torch.zeros(1, dtype=torch.int32, device='cuda'), gpu_lengths.cumsum(0).int()))
        indices = torch.empty(sum(actual_lengths), dtype=torch.int32, device='cuda')
        backend.create_flashinfer_kv_indices_triton[(len(rows),)](
            request_map, torch.arange(len(rows), dtype=torch.int64, device='cuda'), gpu_lengths,
            ptr, None, indices, request_map.shape[1])
        flat = [value for row in rows for value in row]
        if indices.cpu().tolist() != flat:
            raise RuntimeError('SG native req_to_token index kernel changed token addresses')
        valid = torch.tensor(flat, dtype=torch.int64, device='cuda')
        before = tensor_sha(cache[valid])
        unused = torch.ones(cache.shape[0], dtype=torch.bool, device='cuda')
        unused[valid] = False
        q = torch.zeros((len(rows), 128, 512), dtype=torch.bfloat16, device='cuda')
        q_pe = torch.zeros((len(rows), 128, 64), dtype=torch.bfloat16, device='cuda')
        wrapper.plan(torch.arange(len(rows) + 1, dtype=torch.int32, device='cuda'), ptr,
                     indices, gpu_lengths, 128, 512, 64, 1, False, 0.125,
                     torch.bfloat16, torch.bfloat16)
        if wrapper._backend != 'fa2':
            raise RuntimeError('Native FI wrapper did not retain explicit fa2')
        cache[unused] = float('nan')
        poison_cache_before = tensor_sha(cache)
        poisoned = wrapper.run(q, q_pe, cache[:, :, :512], cache[:, :, 512:]).clone()
        poison_cache_after = tensor_sha(cache)
        if poison_cache_before != poison_cache_after:
            raise RuntimeError('Native FI run wrote valid or unused NaN-cache bytes')
        cache[unused] = 0
        clean_cache_before = tensor_sha(cache)
        clean = wrapper.run(q, q_pe, cache[:, :, :512], cache[:, :, 512:]).clone()
        clean_cache_after = tensor_sha(cache)
        if clean_cache_before != clean_cache_after:
            raise RuntimeError('Native FI run wrote valid or unused zero-cache bytes')
        if not torch.isfinite(poisoned).all() or not torch.equal(poisoned, clean):
            raise RuntimeError('SG allocated-token FI result reads poisoned unused storage')
        if tensor_sha(cache[valid]) != before:
            raise RuntimeError('A native read changed valid KV bytes')
        # Independent uniform-attention oracle with a predeclared primitive-only
        # one-BF16-ULP gate. This is not a full-model numerical acceptance gate.
        oracle = torch.stack([cache[torch.tensor(row, device='cuda'), 0, :512].double().mean(0) for row in rows])
        error = poisoned.double() - oracle[:, None, :]
        rounded_rows = oracle.to(torch.bfloat16).float().cpu().tolist()
        ulps = torch.tensor([[bf16_ulp_reference(value) for value in row] for row in rounded_rows],
                            dtype=torch.float64, device='cuda')[:, None, :]
        normalized = error.abs() / ulps
        if not torch.all(normalized <= 1):
            raise RuntimeError('Native FI uniform attention exceeded the frozen 1-BF16-ULP oracle bound')
        numerical.append({'lengths': list(actual_lengths), 'max_abs_to_FP64': float(error.abs().max()),
                          'maximum_BF16_ULPs_to_FP64': float(normalized.max()), 'uniform_oracle_passed': True,
                          'nan_tail_equals_zero_tail_bitwise': True, 'valid_KV_sha256': before,
                          'poison_full_cache_before_sha256': poison_cache_before,
                          'poison_full_cache_after_sha256': poison_cache_after,
                          'zero_full_cache_before_sha256': clean_cache_before,
                          'zero_full_cache_after_sha256': clean_cache_after,
                          'output_sha256': tensor_sha(poisoned), 'indices_sha256': tensor_sha(indices)})

    check_indices_and_attention(split_rows, LENGTHS)
    new_cpu = lengths_cpu + 1
    append = allocator.alloc_decode(new_cpu.cuda(), new_cpu,
                                    torch.tensor([row[-1] for row in split_rows], dtype=torch.int64, device='cuda'))
    if append is None:
        raise RuntimeError('Native SG decode append unexpectedly exhausted the fixture pool')
    append_rows = append.cpu().tolist()
    old_pages = {value // PAGE for row in split_rows for value in row}
    for rank, (row, address) in enumerate(zip(split_rows, append_rows)):
        old_length = len(row)
        if (old_length % PAGE and address != row[-1] + 1) or (
                old_length % PAGE == 0 and (address % PAGE or address // PAGE in old_pages)):
            raise RuntimeError('Native decode append violated old-tail/new-page ownership')
        row.append(address)
        request_map[rank, old_length] = address
        cache[address, 0, :512] = float(rank + 1 + (old_length % 17 - 8) / 16)
        cache[address, 0, 512:] = 0
    checks.append(address_contract(split_rows, new_cpu.tolist()))
    check_indices_and_attention(split_rows, new_cpu.tolist())
    allocator.free(torch.cat((locations, append)))
    if allocator.available_size() != CAPACITY:
        raise RuntimeError('Native SG free did not restore the page capacity')
    allocator.merge_and_sort_free()
    reused = allocator.alloc(PAGE * 3)
    if reused.cpu().tolist() != list(range(PAGE, PAGE * 4)):
        raise RuntimeError('Native SG free/reuse did not restore the first three physical pages')
    allocator.free(reused)
    torch.cuda.synchronize()
    return dict(plan(args), status='passed_native_page64_token_address_contract_NOT_model_qualified',
                gpu_operations=True, address_checks=checks, numerical_diagnostics=numerical,
                free_and_reuse_passed=True,
                source={'allocator': identity(allocator_module.PagedTokenToKVPoolAllocator),
                        'MLA_pool': identity(pool_module.MLATokenToKVPool),
                        'FI_index_updater': identity(backend.create_flashinfer_kv_indices_triton),
                        'FI_wrapper': identity(flashinfer.BatchMLAPagedAttentionWrapper)},
                peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected-hostname')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    result = plan(args)
    if args.execute:
        if not args.output.is_absolute() or args.output.exists():
            raise ValueError('Execute requires a fresh absolute output path')
        result.update(execution_requested=True, gpu_operations=None)
        try:
            result = execute(args)
        except BaseException as error:
            result.update(status='failed_native_page64_qualification', error=repr(error), traceback=traceback.format_exc())
        with args.output.open('x') as output:
            output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'].startswith(('plan_only', 'passed_')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
