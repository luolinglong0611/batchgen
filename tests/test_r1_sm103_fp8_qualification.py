"""CPU-only admission, range-reader and fixed numeric-contract checks."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r1_reproduction/qualify_r1_sm103_fp8_compat.py"
spec = importlib.util.spec_from_file_location("r1_compat_qualification_contract", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class QualificationContract(unittest.TestCase):
    def command(self, output):
        paths = (ROOT / "batchgen/gemm/w8a8.py", ROOT / "batchgen/gemm/r1_sm103_compat.py",
                 ROOT / "batchgen_kernels/triton/fp8_quantize.py")
        command = [sys.executable, str(SCRIPT), "--expected-hostname", "future-container", "--output", str(output),
                   "--checkpoint-root", "/never-read-in-plan"]
        for flag, path in zip(("source", "compat-source", "quant-source"), paths):
            command += ["--" + flag, str(path), "--" + flag + "-sha256", tool.sha_file(path)]
        return command

    def test_plan_does_not_read_checkpoint_import_accelerators_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "sitecustomize.py").write_text(
                "import sys\nclass Reject:\n"
                " def find_spec(self,fullname,path=None,target=None):\n"
                "  if fullname.split('.')[0] in ('torch','triton','batchgen','batchgen_kernels'):\n"
                "   raise RuntimeError('unexpected accelerator import')\n"
                "sys.meta_path.insert(0,Reject())\n")
            output = directory / "not-created"
            result = subprocess.run(self.command(output), capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(directory)}, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "plan_only_NOT_executed")
            self.assertEqual(report["scope"], "operator")
            self.assertEqual(len(report["real_cases"]), 8)
            self.assertEqual(report["checkpoint_index_sha256"], tool.INDEX_SHA)
            self.assertEqual(report["quant_contract"]["scale_rel_error_bound"], 2 * 2**-24)
            self.assertFalse(output.exists())

    def test_execute_without_outer_guard_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            env = {k: v for k, v in os.environ.items() if k != "R1_MAIN_FP8_COMPAT_GUARDED"}
            result = subprocess.run(self.command(output) + ["--execute"], capture_output=True,
                                    text=True, env=env, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parent-owned FP8 compatibility resource guard", result.stderr)
            self.assertFalse(output.exists())

    def make_shard(self, root, metadata=None, data=None):
        data = struct.pack("<4f", 0.25, 0.375, 0.625, 0.875) if data is None else data
        metadata = {"fixture.scale": {"dtype": "F32", "shape": [2, 2], "data_offsets": [4, 20]}} if metadata is None else metadata
        header = json.dumps(metadata).encode()
        path = root / "model-00001-of-000163.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"GAP!" + data + b"AFTER")
        return path, data

    def test_reader_returns_only_frozen_tensor_bytes_and_records_identity_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, data = self.make_shard(root)
            actual, receipt = tool.read_tensor_range(root, {"fixture.scale": "model-00001-of-000163.safetensors"},
                                                    "fixture.scale", {"F32"}, (2, 2), tool.sha_bytes(data))
            self.assertEqual(actual, data)
            self.assertEqual(receipt["tensor_bytes"], 16)
            self.assertEqual(receipt["tensor_data_offsets"], [4, 20])
            self.assertFalse(receipt["full_shard_hash_verified"])
            self.assertIn("recorded now", receipt["identity_limit"])

    def test_reader_rejects_wrong_sha_dtype_shape_bounds_and_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_shard(root)
            index = {"fixture.scale": "model-00001-of-000163.safetensors"}
            for dtype, shape, digest in (({"F32"}, (2, 2), "0" * 64), ({"F8_E4M3"}, (2, 2), None),
                                         ({"F32"}, (4,), None)):
                with self.assertRaises(ValueError):
                    tool.read_tensor_range(root, index, "fixture.scale", dtype, shape, digest)
            for filename in ("../outside.safetensors", "/tmp/other.safetensors", "model.safetensors", "https://x/model-1-of-1.safetensors"):
                with self.assertRaises(ValueError):
                    tool.read_tensor_range(root, {"fixture.scale": filename}, "fixture.scale", {"F32"}, (2, 2))
            for lo, hi in ((-1, 15), (4, 21), (4, 4), (4, 1024**3)):
                self.make_shard(root, {"fixture.scale": {"dtype": "F32", "shape": [2, 2], "data_offsets": [lo, hi]}})
                with self.assertRaises(ValueError):
                    tool.read_tensor_range(root, index, "fixture.scale", {"F32"}, (2, 2))

    def test_original_index_is_required_before_any_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
            with self.assertRaisesRegex(ValueError, "Frozen original checkpoint index"):
                tool.read_index(root)

    def test_error_bound_is_frozen_fp32_plus_bf16_not_a_posthoc_relative_threshold(self):
        self.assertEqual(tool.U32, 2**-24)
        self.assertEqual(tool.UBF16, 2**-8)
        for k in (128, 256, 512, 1536, 7168, 16384, 18432):
            operations = 128 + 2 + (k + 127) // 128
            exact = Fraction(operations, 2**24 - operations)
            self.assertEqual(tool.gamma32(operations), float(exact))
            contract = tool.rounding_contract(k)
            self.assertEqual(contract["FP32_gamma_operations"], operations)
            self.assertEqual(contract["allowed_violations"], 0)
            self.assertIn("lower-precision", contract["assumption"])
        for invalid in (-1, 0.5, 2**24):
            with self.assertRaises(ValueError):
                tool.gamma32(invalid)

    def good_case(self, name="fixture", m=3, n=576, k=256):
        return {"name": name, "M": m, "N": n, "K": k, "compared_elements": m * n,
                "error_bound_violations": 0, "nonfinite_output_elements": 0,
                "quantization_passed": True, "inputs_unchanged": True,
                "composite_entry_bitwise_equal": True, "output_contract_passed": True,
                "reference_finite": True, "weight_and_scale_range_unchanged": True}

    def test_every_hard_contract_is_required_even_for_numerically_close_output(self):
        good = self.good_case()
        self.assertTrue(tool.case_passed(good))
        for flag in ("quantization_passed", "inputs_unchanged", "composite_entry_bitwise_equal",
                     "output_contract_passed", "reference_finite", "weight_and_scale_range_unchanged"):
            for value in (False, None, "true"):
                bad = copy.deepcopy(good); bad[flag] = value
                self.assertFalse(tool.case_passed(bad))
        for counter in ("error_bound_violations", "nonfinite_output_elements"):
            bad = copy.deepcopy(good); bad[counter] = 1
            self.assertFalse(tool.case_passed(bad))
        bad = copy.deepcopy(good); bad["compared_elements"] -= 1
        self.assertFalse(tool.case_passed(bad))

    def test_report_requires_every_unique_case_sources_index_and_actual_backend(self):
        common = types.SimpleNamespace(COMPAT_BACKEND="common", QUANT_BACKEND="quant", DEFAULT_BACKEND="dg")
        cases = [self.good_case(name, m, n, k)
                 for name, ((n, k), _) in tool.SCALE_AUDIT.items() for m in (1, 3)]
        cases.append(self.good_case("synthetic_two_K_blocks_medium_partial_N", 129, 576, 256))
        good = {"cases": cases, "source_unchanged": True, "checkpoint_index_unchanged": True,
                "backend_receipt": {"actual_backends": ["common", "quant"]}}
        self.assertTrue(tool.report_passed(good, common))
        for flag in ("source_unchanged", "checkpoint_index_unchanged"):
            bad = copy.deepcopy(good); bad[flag] = False
            self.assertFalse(tool.report_passed(bad, common))
        for backends in ([], ["common"], ["quant"], ["common", "quant", "dg"]):
            bad = copy.deepcopy(good); bad["backend_receipt"]["actual_backends"] = backends
            self.assertFalse(tool.report_passed(bad, common))
        bad = copy.deepcopy(good); bad["cases"][0] = bad["cases"][1]
        self.assertFalse(tool.report_passed(bad, common))
        bad = copy.deepcopy(good); bad["cases"].pop()
        self.assertFalse(tool.report_passed(bad, common))


if __name__ == "__main__":
    unittest.main()
