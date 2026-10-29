"""CPU contracts for the opt-in fresh CUTLASS path; no CUDA imports required."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "r1_blackwell_prefill_cpu_contract"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "batchgen/attention/mla")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".r1_blackwell_prefill", ROOT / "batchgen/attention/mla/r1_blackwell_prefill.py")
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


class Tensor:
    def __init__(self, shape=(8192, 128, 192), device="cuda:0", strides=None):
        self.shape = shape
        self.device = device
        self.dtype = "bf16"
        self._strides = strides or (shape[1] * shape[2], shape[2], 1)

    def stride(self):
        return self._strides


class CutlassContracts(unittest.TestCase):
    def setUp(self):
        backend.clear_plan_cache()
        self.addCleanup(backend.clear_plan_cache)
        self.plans = []
        self.allocations = []
        self.stream = 123
        self.should_fail = False

        def plan(*args):
            self.plans.append(args)
            if self.should_fail:
                raise RuntimeError("native plan failed")

        def allocate(*args, **kwargs):
            allocation = SimpleNamespace(args=args, kwargs=kwargs)
            self.allocations.append(allocation)
            return allocation

        self.native = SimpleNamespace(plan=plan)
        self.torch = SimpleNamespace(
            uint8="u8", int32="i32", int64="i64", bfloat16="bf16",
            empty=allocate, tensor=lambda values, **kwargs: tuple(values),
            cuda=SimpleNamespace(
                current_stream=lambda device: SimpleNamespace(cuda_stream=self.stream),
                get_device_capability=lambda device: (10, 3),
                get_device_properties=lambda device: SimpleNamespace(multi_processor_count=148),
            ),
        )
        native = patch.object(backend, "_native_module", return_value=self.native)
        native.start()
        self.addCleanup(native.stop)

    def prepare(self, cu=(0, 8192), scale=0.1, q=None, device="cuda:0", max_seqlen=8192):
        return backend._prepare(q or Tensor(device=device), Tensor(device=device),
                                Tensor((8192, 128, 128), device=device), cu,
                                max_seqlen, scale, self.torch)

    def test_opt_in_is_explicit_and_unknown_selectors_fail(self):
        with patch.dict(os.environ, {backend.ENV: "legacy"}):
            self.assertFalse(backend.enabled())
        with patch.dict(os.environ, {backend.ENV: "cutlass-sm100"}):
            self.assertTrue(backend.enabled())
        with patch.dict(os.environ, {backend.ENV: "auto"}):
            with self.assertRaises(ValueError):
                backend.enabled()

    def test_real_8k_fresh_mha_geometry_and_capacity(self):
        q = (8192, 128, 192)
        v = (8192, 128, 128)
        self.assertEqual(backend.packed_contract(q, q, v, (0, 8192), (0, 8192),
                                                  8192, 8192, 0.1),
                         ((0, 8192), 0.1, 4096))
        q = (262144, 128, 192)
        self.assertEqual(backend.packed_contract(q, q, (262144, 128, 128),
                         (0, 262144), (0, 262144), 262144, 262144, 0.1)[2],
                         backend.PLAN_CAPACITY)
        q = (262145, 128, 192)
        with self.assertRaisesRegex(ValueError, "capacity"):
            backend.packed_contract(q, q, (262145, 128, 128),
                                   (0, 262145), (0, 262145), 262145, 262145, 0.1)

    def test_prefix_gqa_wrong_dims_empty_and_malformed_boundaries_rejected(self):
        q = (8192, 128, 192)
        v = (8192, 128, 128)
        cases = [
            (q, q, v, (0, 4096, 8192), (0, 8192), 8192, 8192, 0.1),
            (q, q, v, (0, 8192), (0, 8192), 8192, 8193, 0.1),
            (q, (8192, 1, 192), v, (0, 8192), (0, 8192), 8192, 8192, 0.1),
            (q, q, (8192, 128, 512), (0, 8192), (0, 8192), 8192, 8192, 0.1),
            (q, q, v, (0, 8191), (0, 8191), 8192, 8192, 0.1),
            ((0, 128, 192), (0, 128, 192), (0, 128, 128), (0,), (0,), 1, 1, 0.1),
            ((1, 128, 192), (1, 128, 192), (1, 128, 128), (0, 1), (False, True), 1, 1, 0.1),
            (q, q, v, (0, 8192), (0, 8192), 8192, 8192, float("nan")),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                backend.packed_contract(*args)

    def test_61_layers_share_one_plan_and_workspace_but_not_input_addresses(self):
        first = self.prepare()
        for _ in range(60):
            self.assertIs(first, self.prepare())
        self.assertEqual(len(self.allocations), 5)
        self.assertEqual(len(self.plans), 1)
        stats = backend.cache_stats()
        self.assertEqual(stats["hits"], 60)
        self.assertEqual(stats["workspace_bytes"], 32 * 1024**2)
        self.assertEqual(stats["plan_storage_bytes"], 4 * (3 * 131072 + 149 + 2))
        self.assertEqual(stats["retained_entries"], 1)

    def test_real_metadata_values_replan_without_new_workspace(self):
        first = self.prepare()
        workspace = first.workspace
        for change in ({"cu": (0, 4096, 8192)}, {"scale": 0.2},
                       {"max_seqlen": 9000},
                       {"q": Tensor(strides=(128*256, 256, 1))}):
            entry = self.prepare(**change)
            self.assertIs(entry.workspace, workspace)
        self.assertEqual(len(self.plans), 5)
        self.assertEqual(len(self.allocations), 5)
        self.assertEqual(backend.cache_stats()["retained_entries"], 1)

    def test_stream_and_device_changes_replace_one_retained_entry(self):
        old_workspace = self.prepare().workspace
        self.stream += 1
        self.assertIsNot(self.prepare().workspace, old_workspace)
        old_workspace = self.prepare().workspace
        self.assertIsNot(self.prepare(device="cuda:1").workspace, old_workspace)
        self.assertEqual(backend.cache_stats()["retained_entries"], 1)
        self.assertEqual(len(self.plans), 3)

    def test_failed_replan_cannot_reuse_corrupted_old_plan(self):
        self.prepare()
        self.should_fail = True
        with self.assertRaisesRegex(RuntimeError, "native plan failed"):
            self.prepare(cu=(0, 4096, 8192))
        self.assertEqual(backend.cache_stats()["retained_entries"], 0)
        self.should_fail = False
        self.prepare()
        self.assertEqual(len(self.plans), 3)
        self.assertEqual(len(self.allocations), 10)

    def test_cpu_metadata_does_not_read_cuda_values(self):
        class Metadata:
            ndim = 1
            dtype = "i32"
            device = SimpleNamespace(type="cpu")

            def tolist(self):
                return [0, 8192]

        torch = SimpleNamespace(Tensor=Metadata, int32="i32", int64="i64")
        self.assertEqual(backend._host_cu(Metadata(), "cu", torch), (0, 8192))
        metadata = Metadata()
        metadata.device = SimpleNamespace(type="cuda")
        with self.assertRaisesRegex(ValueError, "CPU"):
            backend._host_cu(metadata, "cu", torch)

    def test_newer_flashinfer_abi_is_rejected_before_import(self):
        backend._flashinfer_prefill.cache_clear()
        with patch.object(backend.importlib.metadata, "version", return_value="0.7.1"):
            with self.assertRaisesRegex(RuntimeError, "requires flashinfer-python=0.5.2"):
                backend._flashinfer_prefill()
        self.addCleanup(backend._flashinfer_prefill.cache_clear)


if __name__ == "__main__":
    unittest.main(verbosity=2)
