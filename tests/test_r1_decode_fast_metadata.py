"""CPU scheduler-boundary contracts. GPU index arithmetic is separately qualified."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'batchgen/attention/mla/r1_decode_cuda_graph.py'
NAME = 'batchgen.attention.mla.r1_fast_metadata_contract'
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)


def test_fast_scheduler_stages_only_small_metadata_and_preserves_full_plan_key():
    class CPUKernel:
        calls = []
        def __getitem__(self, grid):
            def run(table, indptr, lengths, output, **kw):
                self.calls.append((grid, kw))
                for row, length in enumerate(lengths.tolist()):
                    offsets = torch.arange(length)
                    pages = table[row + kw['ROW_START'], offsets // 64]
                    values = pages * 64 + offsets % 64
                    begin = int(indptr[row])
                    output[begin:begin + length] = values
            return run
    item = module._AttentionPlan.__new__(module._AttentionPlan)
    item.rows, item.context_cap, item.heads, item.scale = 1, 256, 128, 0.07
    item.kv_indices = torch.full((256,), -987, dtype=torch.int32)
    item.kv_indptr = torch.empty((2,), dtype=torch.int32)
    item.lengths = torch.empty((1,), dtype=torch.int32)
    item.fast_initialized = False
    item.signature = None
    full_calls, native_calls = [], []
    def native_plan(float_workspace, int_workspace, pinned, q, indptr, lengths, heads, dim, causal):
        native_calls.append((q.clone(), indptr.clone(), lengths.clone()))
        return [len(q), int(indptr[-1]), 13, 99]
    def full_plan(q, indptr, indices, lengths, *rest):
        full_calls.append((q, indptr, indices, lengths))
        assert indices is item.kv_indices
        item.kv_indptr.copy_(indptr)
        item.lengths.copy_(lengths)
        item.wrapper._plan_info = native_plan(None, None, None, q, indptr, lengths, 128, 512, False)
    item.wrapper = SimpleNamespace(plan=full_plan, _plan_info=None,
        _cached_module=SimpleNamespace(plan=native_plan), _float_workspace_buffer=object(),
        _int_workspace_buffer=object(), _pin_memory_int_workspace_buffer=object())
    first = module.plan_metadata([91, 7], [65, 127], {91: [3, 9], 7: [8, 2]},
                                  context_cap=256, gpu_page_count=16)
    small = first.small_arrays()
    gpu_table = torch.from_numpy(small['page_table'])
    kernel = CPUKernel()
    with patch.object(module, '_index_kernel', return_value=kernel), patch('importlib.metadata.version', return_value='0.5.2'):
        item.prepare_fast(small, 1, 2, gpu_table)
        signature = item.signature
        np.testing.assert_array_equal(item.kv_indices[:127].numpy(), first.arrays()['kv_indices'][65:])
        assert bool((item.kv_indices[127:] == -987).all())
        # Cross a64-token boundary and remap all pages. No full token-index
        # upload is passed to the second native scheduler invocation.
        second = module.plan_metadata([91, 7], [66, 129], {91: [1, 5], 7: [6, 0, 4]},
                                       context_cap=256, gpu_page_count=16)
        item.kv_indices.fill_(-987)
        item.prepare_fast(second.small_arrays(), 1, 2, torch.from_numpy(second.small_arrays()['page_table']))
        np.testing.assert_array_equal(item.kv_indices[:129].numpy(), second.arrays()['kv_indices'][66:])
        assert bool((item.kv_indices[129:] == -987).all())
    assert len(full_calls) == 1 and len(native_calls) == 2
    assert signature != item.signature
    assert item.signature[-1] == 99  # No truncated/hash-only plan_info key.
    assert len(kernel.calls) == 2 and all(kw['ROW_START'] == 1 for _, kw in kernel.calls)


def test_fast_metadata_small_arrays_do_not_allocate_token_page_ids():
    plan = module.plan_metadata([3, 8], [63, 65], {3: [2], 8: [0, 4]},
                                context_cap=256, gpu_page_count=8)
    small = plan.small_arrays()
    assert set(small) == {'page_table', 'kv_indptr', 'lengths', 'positions'}
    assert small['kv_indptr'].tolist() == [0, 63, 128]
    assert small['positions'].tolist() == [[62], [64]]
    assert sum(x.nbytes for x in small.values()) < 1024
