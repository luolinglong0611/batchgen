"""Plan/source guards always run; numerical oracle checks require CPU Torch."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/r1_reproduction/qualify_r1_common_mla.py"
SPEC = importlib.util.spec_from_file_location("r1_common_mla_worker_tests", SOURCE)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
HAS_TORCH = importlib.util.find_spec("torch") is not None


def arguments(folder):
    backend = ROOT / "batchgen/attention/mla/common_flashinfer_backend.py"
    metadata = backend.with_name("common_sdpa_backend.py")
    return SimpleNamespace(backend_source=backend, backend_source_sha256=worker.sha(backend),
                           metadata_source=metadata, metadata_source_sha256=worker.sha(metadata),
                           expected_hostname="expected-single-gpu-container", output=Path(folder) / "result")


class QualificationPlanContracts(unittest.TestCase):
    def test_plan_only_is_not_a_model_or_performance_receipt_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            report = worker.plan(args)
            self.assertEqual(report["status"], "plan_only_NOT_executed")
            self.assertEqual(report["scope"], "operator")
            self.assertFalse(report["full_model_qualified"])
            self.assertFalse(report["formal_test_permitted"])
            self.assertFalse(args.output.exists())
            self.assertEqual(report["reference"]["max_BF16_ULPs"], 1)
            self.assertEqual(report["torch_allocator_cap_bytes"], 6 * 1024**3)
            self.assertEqual(report["lengths"], [1, 63, 64, 65, 2047, 2048, 2049, 8191, 8192, 8193])
            json.dumps(report, allow_nan=False)

    def test_either_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for field in ("backend_source_sha256", "metadata_source_sha256"):
                with self.subTest(field=field):
                    args = arguments(folder)
                    setattr(args, field, "0" * 64)
                    with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                        worker.plan(args)

    def test_metadata_helper_must_be_the_explicit_sibling(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            args.metadata_source = Path(folder) / "common_sdpa_backend.py"
            with self.assertRaisesRegex(ValueError, "must be siblings"):
                worker.source_contract(args)

    def test_default_cli_prints_plan_and_does_not_import_or_execute_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            argv = ["--expected-hostname", args.expected_hostname, "--output", str(args.output),
                    "--backend-source", str(args.backend_source), "--backend-source-sha256", args.backend_source_sha256,
                    "--metadata-source", str(args.metadata_source), "--metadata-source-sha256", args.metadata_source_sha256]
            output = io.StringIO()
            with mock.patch.object(worker, "execute", side_effect=AssertionError("GPU must not execute")):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(worker.main(argv), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "plan_only_NOT_executed")
            self.assertFalse(args.output.exists())

    def test_execute_without_coordinator_guard_fails_before_import_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "parent-owned"):
                    worker.execute(args)
            self.assertFalse(args.output.exists())

    def test_explicit_source_load_binds_all_actual_shim_entrypoints_and_helper(self):
        namespace = "r1_exact_common_mla_qualification_sources"
        try:
            with tempfile.TemporaryDirectory() as folder:
                args = arguments(folder)
                module = worker.load_backend(args)
                self.assertEqual(Path(module.__file__).resolve(), args.backend_source.resolve())
                self.assertEqual(module.EXPECTED_FLASHINFER_VERSION, "0.5.2")
                helper = sys.modules[namespace + ".common_sdpa_backend"]
                self.assertEqual(Path(helper.__file__).resolve(), args.metadata_source.resolve())
        finally:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]


@unittest.skipUnless(HAS_TORCH, "Torch is unavailable; CPU FP64 oracle checks not executed")
class CPUReferenceContracts(unittest.TestCase):
    def test_packed_prefix_formula_matches_independent_dense_fp64_attention(self):
        import torch
        q, k, v, _ = worker.prefill_fixture([7, 9])
        expected = torch.empty(v.shape, dtype=torch.float64)
        for start, stop in ((0, 7), (7, 16)):
            for head in range(worker.HEADS):
                scores = q[start:stop, head].double() @ k[start:stop, head].double().T * 0.173
                scores.masked_fill_(torch.ones(stop-start, stop-start, dtype=torch.bool).triu(1), -float("inf"))
                expected[start:stop, head] = scores.softmax(-1) @ v[start:stop, head].double()
        actual = torch.cat([chunk for _, _, chunk in worker.prefill_reference(q, k, v, [7, 9], 0.173)])
        torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)

    def test_decode_formula_matches_independent_dense_fp64_attention_and_scattered_pages(self):
        import torch
        q, cache, table, lengths, history, rope = worker.decode_fixture(65)
        dense_k = torch.cat((history, rope), -1).double()
        expected = torch.empty(1, 1, worker.HEADS, 512, dtype=torch.float64)
        for head in range(worker.HEADS):
            scores = q[0, 0, head].double() @ dense_k.T * 0.091
            expected[0, 0, head] = scores.softmax(-1) @ history.double()
        actual = worker.decode_reference(q, history, rope, 0.091)[0][2]
        torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)
        self.assertEqual(int(lengths[0]), 65)
        self.assertTrue(torch.isnan(cache[table[0, 1], 1:]).all())

    def test_one_bf16_step_passes_two_steps_fail_and_nonfinite_failure_serializes(self):
        import torch
        reference = torch.tensor([[2.0]], dtype=torch.float64)
        base = reference.bfloat16()
        one = (base.view(torch.int16) + 1).view(torch.bfloat16)
        two = (base.view(torch.int16) + 2).view(torch.bfloat16)
        self.assertTrue(worker.compare_output(one, [(0, 1, reference)])["within_fixed_envelope"])
        self.assertFalse(worker.compare_output(two, [(0, 1, reference)])["within_fixed_envelope"])
        for bad in (float("nan"), float("inf")):
            result = worker.compare_output(torch.tensor([[bad]], dtype=torch.bfloat16), [(0, 1, reference)])
            self.assertFalse(result["within_fixed_envelope"])
            self.assertEqual(result["nonfinite_elements"], 1)
            json.dumps(result, allow_nan=False)

    def test_reference_cannot_skip_or_recount_rows_and_fp32_output_is_rejected(self):
        import torch
        output = torch.ones(2, 1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            worker.compare_output(output, [(1, 2, torch.ones(1, 1, dtype=torch.float64))])
        with self.assertRaisesRegex(ValueError, "must be BF16"):
            worker.compare_output(output.float(), [])
        self.assertFalse(worker.compare_output(output, [(0, 1, torch.ones(1, 1, dtype=torch.float64))])["within_fixed_envelope"])


if __name__ == "__main__":
    unittest.main()
