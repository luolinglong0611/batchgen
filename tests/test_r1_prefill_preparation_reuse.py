"""CPU contracts for the exact prefill planner; no accelerator runtime import."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'r1_prefill_reuse_cpu_contract'
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'batchgen/attention/mla')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.common_flashinfer_backend',
    ROOT / 'batchgen/attention/mla/common_flashinfer_backend.py')
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


class Tensor:
    def __init__(self, shape=(8, 128, 192), device='cuda:0', strides=None):
        self.shape = shape; self.device = device; self.dtype = 'bf16'; self.layout = 'strided'
        self._strides = strides or (shape[1] * shape[2], shape[2], 1)
    def stride(self): return self._strides


class PrefillReuseContracts(unittest.TestCase):
    def setUp(self):
        backend.clear_prefill_plan_cache()
        self.calls = []
        self.stream = 123
        def wrapper(workspace, *_args, **kwargs):
            result = SimpleNamespace(_backend=kwargs['backend'])
            def plan(**args): self.calls.append(args)
            result.plan = plan
            return result
        self.fake_fi = SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=wrapper)
        self.fake_torch = SimpleNamespace(uint8='u8', int32='i32', bfloat16='bf16',
            empty=lambda *a, **k: object(), tensor=lambda values, **kwargs: tuple(values),
            cuda=SimpleNamespace(current_stream=lambda device: SimpleNamespace(cuda_stream=self.stream)))
        self.env = patch.dict(os.environ, {backend.PREFILL_REUSE_ENV: '1'}); self.env.start()
        self.fi = patch.object(backend, '_flashinfer', return_value=self.fake_fi); self.fi.start()
        self.addCleanup(self.env.stop); self.addCleanup(self.fi.stop)
        self.addCleanup(backend.clear_prefill_plan_cache)

    def prepare(self, *, q=None, cu=(0, 3, 8), scale=0.1, device='cuda:0'):
        q = q or Tensor(device=device)
        return backend._prepare_prefill_wrapper(q, Tensor(device=device),
            Tensor((8, 128, 128), device=device), cu, scale, self.fake_torch, device)

    def test_new_layer_data_addresses_reuse_exact_plan_and_workspace(self):
        first = self.prepare(); second = self.prepare()
        self.assertIs(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]['qo_indptr'], (0, 3, 8))
        self.assertEqual(first._batchgen_prefill_resources[1], (0, 3, 8))

    def test_values_scale_stride_device_and_stream_invalidate(self):
        old = self.prepare()
        for change in ({'cu': (0, 4, 8)}, {'scale': 0.2},
                       {'q': Tensor(strides=(128*256, 256, 1))}, {'device': 'cuda:1'}):
            new = self.prepare(**change); self.assertIsNot(old, new); old = new
        old = self.prepare(); before = len(self.calls)
        self.stream += 1
        self.assertIsNot(self.prepare(), old)
        self.assertEqual(len(self.calls), before + 1)
        # One retained entry; revisiting an evicted key must replan.
        before = len(self.calls); self.stream -= 1; self.prepare()
        self.assertEqual(len(self.calls), before + 1)

    def test_default_off_stays_fresh(self):
        with patch.dict(os.environ, {backend.PREFILL_REUSE_ENV: '0'}):
            self.assertIsNot(self.prepare(), self.prepare())
        self.assertEqual(len(self.calls), 2)

    def test_cpu_metadata_is_validated_without_tensor_or_device_reads(self):
        self.assertEqual(backend.packed_prefill_cpu_metadata([3, 5], 2, 8, 5),
                         ((3, 5), (0, 3, 8)))
        for args in (([3, 5], 1, 8, 5), ([3, 5], 2, 9, 5),
                     ([3, 5], 2, 8, 4), ([0, 8], 2, 8, 8), ([True], 1, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                backend.packed_prefill_cpu_metadata(*args)


if __name__ == '__main__': unittest.main(verbosity=2)
