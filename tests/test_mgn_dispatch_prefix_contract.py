"""Dispatch prefix-scan bounds regression and opt-in native CUDA qualification.

Default tests inspect the actual C++ allocation/scan contract without importing
Torch. They do not claim to execute CUB. Enable the native tests only in an
isolated CUDA process with BATCHGEN_RUN_CUDA_TESTS=1; run that same command under
compute-sanitizer with PYTORCH_NO_CUDA_MEMORY_CACHING=1 for allocation bounds.
"""
import os
import hashlib
from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    'batchgen_kernels/src/moe/mgn/fused_moe_token_dispatch.cu')


def scan_contract(source):
    """Extract the two actual CUB calls and their owning tensor allocation."""
    calls = re.findall(r'cub::DeviceScan::ExclusiveSum\(\s*'
                       r'd_temp_storage,\s*temp_storage_bytes,\s*'
                       r'(\w+)\.data_ptr<int32_t>\(\),\s*'
                       r'expert_offsets\.data_ptr<int32_t>\(\),\s*'
                       r'num_local_experts\s*\+\s*1,\s*stream\s*\)', source)
    if len(calls) != 2 or len(set(calls)) != 1:
        raise AssertionError('Expected matching scratch-query and execution scans')
    owner = calls[0]
    allocation = re.search(r'auto\s+' + re.escape(owner) + r'\s*=\s*torch::zeros\(\s*'
                           r'\{num_local_experts\s*(?:\+\s*(\d+))?\}', source)
    if not allocation:
        raise AssertionError('Scan must read an explicitly zero-initialized allocation')
    extra = int(allocation.group(1) or 0)
    if extra < 1:
        raise AssertionError('CUB E+1 input exceeds its E-element allocation')
    return owner


class ScanAllocationContract(unittest.TestCase):
    def test_scan_query_and_execution_have_real_zero_sentinel(self):
        self.assertEqual(scan_contract(SOURCE.read_text()), 'scan_counts')

    def test_regression_detector_rejects_original_undersized_allocation(self):
        source = SOURCE.read_text().replace(
            'auto scan_counts = torch::zeros(\n        {num_local_experts + 1}',
            'auto scan_counts = torch::zeros(\n        {num_local_experts}')
        self.assertNotEqual(source, SOURCE.read_text())
        with self.assertRaisesRegex(AssertionError, 'exceeds'):
            scan_contract(source)

    def test_public_counts_excludes_sentinel_but_offsets_includes_total(self):
        source = SOURCE.read_text()
        owner = scan_contract(source)
        self.assertRegex(source, r'auto expert_counts\s*=\s*' + owner +
                         r'\.narrow\(0,\s*0,\s*num_local_experts\)')
        self.assertRegex(source, r'auto expert_offsets\s*=\s*torch::empty\(\s*'
                         r'\{num_local_experts\s*\+\s*1\}')
        self.assertIn('expert_counts, expert_offsets};', source)
        self.assertIn('expert_counts.zero_();', source)


@unittest.skipUnless(os.environ.get('BATCHGEN_RUN_CUDA_TESTS') == '1',
                     'Native CUDA qualification is explicit opt-in')
class NativeDispatchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from batchgen_kernels.common.mgn import fused_moe_token_dispatch, _get_mgn
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        extension = Path(_get_mgn().__file__).resolve()
        print('MGN qualification extension:', extension,
              'sha256:', hashlib.sha256(extension.read_bytes()).hexdigest(),
              'source_sha256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest(), flush=True)
        cls.torch = torch
        cls.dispatch = staticmethod(fused_moe_token_dispatch)

    def check_case(self, ids, start, end, hidden_size):
        torch = self.torch
        # Exactly representable BF16 input; equality here detects routing/copy
        # mistakes independently of GEMM, quantization or floating reductions.
        n, k = len(ids), len(ids[0])
        x_cpu = (torch.arange(n * hidden_size).reshape(n, hidden_size) % 127).to(torch.bfloat16)
        topk = torch.tensor(ids, dtype=torch.int32, device='cuda')
        x = x_cpu.cuda()
        token_idx = torch.arange(n, dtype=torch.int32, device='cuda').repeat_interleave(k)
        topk_pos = torch.arange(k, dtype=torch.int32, device='cuda').repeat(n)
        expected = {(t, p): e for t, row in enumerate(ids) for p, e in enumerate(row)
                    if start <= e < end}
        counts_expected = [sum(e == expert for e in expected.values()) for expert in range(start, end)]
        offsets_expected = [0]
        for count in counts_expected:
            offsets_expected.append(offsets_expected[-1] + count)
        # Repeat to exercise different atomic dispatch orders and allocator reuse.
        for _ in range(5):
            out, eids, tids, positions, counts, offsets = self.dispatch(
                x, topk, token_idx, topk_pos, start, end)
            torch.cuda.synchronize()
            self.assertEqual(list(counts.shape), [end - start])
            self.assertEqual(list(offsets.shape), [end - start + 1])
            self.assertEqual(counts.cpu().tolist(), counts_expected)
            self.assertEqual(offsets.cpu().tolist(), offsets_expected)
            nnz = offsets_expected[-1]
            got_e, got_t, got_p = (v[:nnz].cpu().tolist() for v in (eids, tids, positions))
            pairs = list(zip(got_t, got_p))
            self.assertEqual(len(set(pairs)), nnz)
            self.assertEqual(dict(zip(pairs, got_e)), expected)
            out_cpu = out[:nnz].cpu()
            for i, (token, _) in enumerate(pairs):
                self.assertTrue(torch.equal(out_cpu[i], x_cpu[token]))
            for expert, (lo, hi) in enumerate(zip(offsets_expected, offsets_expected[1:]), start):
                self.assertEqual(got_e[lo:hi], [expert] * (hi - lo))

    def test_native_offsets_counts_and_mapping(self):
        cases = [
            # E=1 sentinel, last expert populated; optimized non-vector path.
            ([[3], [3], [2]], 3, 4, 17),
            # No routed rows still requires E+1 all-zero offsets.
            ([[0, 1], [6, 7]], 2, 6, 512),
            # Sparse first/last experts plus nonlocal routes; vector path.
            ([[4, 7, 8], [5, 7, 3], [4, 6, 7]], 4, 8, 512),
            # R1 EP16 partition and top8, with every local expert active.
            ([list(range(32 + (i % 2) * 8, 40 + (i % 2) * 8)) for i in range(33)],
             32, 48, 7168),
        ]
        for case in cases:
            with self.subTest(start=case[1], end=case[2], hidden=case[3]):
                self.check_case(*case)


if __name__ == '__main__':
    unittest.main()
