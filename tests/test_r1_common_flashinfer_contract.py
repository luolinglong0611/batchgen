"""CPU contracts for the real explicit shims; no FlashInfer/GPU execution."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "r1_common_fi_cpu_contract_namespace"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "batchgen/attention/mla")]
    sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(
    PACKAGE + ".common_flashinfer_backend",
    ROOT / "batchgen/attention/mla/common_flashinfer_backend.py",
)
backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backend
SPEC.loader.exec_module(backend)


class FlashInferShimCPUContracts(unittest.TestCase):
    def test_fixed_version_and_no_native_packages_imported_at_module_load(self):
        self.assertEqual(backend.EXPECTED_FLASHINFER_VERSION, "0.5.2")
        self.assertEqual(backend.BACKEND_ID, "r1-common-flashinfer-0.5.2-fa2-bf16-v1")

    def test_explicit_fa2_graph_off_is_required_for_both_real_wrapper_factories(self):
        calls = []
        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(_backend=kwargs["backend"])
        package = SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=wrapper,
                                  BatchMLAPagedAttentionWrapper=wrapper)
        workspace = object()
        backend._ragged_wrapper(workspace, package)
        backend._mla_wrapper(workspace, package)
        self.assertEqual(calls[0][0], (workspace, "NHD"))
        self.assertEqual(calls[1][0], (workspace,))
        self.assertTrue(all(kwargs == {"backend": "fa2", "use_cuda_graph": False}
                            for _, kwargs in calls))

    def test_auto_or_cutlass_fallback_is_rejected(self):
        package = SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=lambda *a, **k: SimpleNamespace(_backend="auto"),
                                  BatchMLAPagedAttentionWrapper=lambda *a, **k: SimpleNamespace(_backend="cutlass"))
        for factory in (backend._ragged_wrapper, backend._mla_wrapper):
            with self.subTest(factory=factory.__name__), self.assertRaises(RuntimeError):
                factory(object(), package)

    def test_decode_token_plan_has_no_unused_nan_tail_addresses(self):
        plan = backend.plan_paged_decode((1, 1, 128, 576), (4, 64, 1, 576), [[3, 1, -1]], [65], 65)
        metadata = backend.token_page_indices(plan)
        self.assertEqual(metadata["kv_indices"], list(range(192, 256)) + [64])
        self.assertEqual(metadata["kv_indptr"], [0, 65])
        self.assertEqual(metadata["query_indptr"], [0, 1])
        self.assertEqual(metadata["kv_lengths"], [65])
        self.assertEqual(metadata["kernel_page_size"], 1)

    def test_prefill_unimplemented_options_rejected_before_torch_or_kernel_import(self):
        args = (None,) * 7
        options = ({"causal": False}, {"causal": True, "dropout_p": 0.1},
                   {"causal": True, "softcap": 1}, {"causal": True, "return_attn_probs": True},
                   {"causal": True, "window_size": (64, 0)}, {"causal": True, "unexpected": 1})
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend.flash_attn_varlen_func(*args, **kwargs)

    def test_native_split_metadata_and_fp8_modes_cannot_enter_bf16_shim(self):
        args = (None, None, None, None, 512)
        for kwargs in ({"num_splits": 1}, {"descale_q": object()}, {"descale_k": object()},
                       {"is_fp8_kvcache": True}, {"causal": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend.flash_mla_with_kvcache(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
