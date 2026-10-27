"""CPU-only admission and independent-reference contracts for the GPU probe."""
import copy
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r1_reproduction/qualify_r1_w8a8_index.py"
SOURCE = ROOT / "batchgen/gemm/w8a8.py"
spec = importlib.util.spec_from_file_location("r1_w8a8_index_contract", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class IndexAdmission(unittest.TestCase):
    def command(self, output, source=SOURCE, source_sha=None):
        return [sys.executable, str(SCRIPT), "--expected-hostname", "future-container",
                "--output", str(output), "--source", str(source), "--source-sha256",
                source_sha or hashlib.sha256(source.read_bytes()).hexdigest()]

    def test_plan_reads_source_only_and_never_imports_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "sitecustomize.py").write_text(
                "import sys\n"
                "class RejectAccelerators:\n"
                " def find_spec(self,fullname,path=None,target=None):\n"
                "  if fullname.split('.')[0] in ('torch','triton','flashinfer'):\n"
                "   raise RuntimeError('Unexpected accelerator import in plan')\n"
                "sys.meta_path.insert(0,RejectAccelerators())\n")
            output = directory / "not-created"
            result = subprocess.run(self.command(output), capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(directory)}, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "plan_only_NOT_executed")
            self.assertEqual(report["scope"], "operator")
            self.assertEqual([(r["M"], r["N"], r["K"]) for r in report["cases"]], list(tool.CASES))
            self.assertFalse(report["unsafe_original_kernel_executed"])
            self.assertEqual(report["comparison_chunk_rows"], 256)
            self.assertTrue(all(report["source_contract"]["index_widenings"].values()))
            self.assertFalse(output.exists())

    def test_execute_without_guard_is_rejected_before_output_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            env = {k: v for k, v in os.environ.items() if k != "R1_MAIN_GEMM_GUARDED"}
            result = subprocess.run(self.command(output) + ["--execute"], capture_output=True,
                                    text=True, env=env, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parent-owned GEMM resource guard", result.stderr)
            self.assertFalse(output.exists())

    def test_wrong_source_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
            tool.source_contract(SOURCE, "0" * 64)

    def test_each_missing_cast_rejects_the_large_probe(self):
        original = SOURCE.read_text()
        for name in ("offs_am", "offs_bn", "offs_k", "offs_cm", "offs_cn"):
            # Edit only the actual active assignment in a temporary fixture;
            # never run or alter the original source.
            lines = original.splitlines()
            matches = [i for i, line in enumerate(lines)
                       if line.strip().startswith(name + " =") and ".to(tl.int64)" in line]
            self.assertEqual(len(matches), 1)
            lines[matches[0]] = lines[matches[0]].replace(".to(tl.int64)", "")
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "unrepaired-fixture.py"
                source.write_text("\n".join(lines) + "\n")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "five int64"):
                    tool.source_contract(source, tool.sha(source))

    def test_mathematical_row_boundaries_and_spots_cover_overflow_edges(self):
        self.assertEqual(tool.first_unsafe_row(24576), 87382)
        self.assertEqual(tool.first_unsafe_row(32768), 65536)
        spots = set(tool.spot_coordinates(98304, 24576))
        for point in ((87381, 8191), (87381, 8192), (87381, 8193),
                      (87382, 0), (87383, 24575), (98303, 24575)):
            self.assertIn(point, spots)
        spots = set(tool.spot_coordinates(65537, 32768))
        self.assertIn((65535, 32767), spots)
        self.assertIn((65536, 0), spots)
        self.assertIn((65536, 32767), spots)

    def test_scalar_fp64_formula_agrees_with_independent_exact_rationals(self):
        for row in (0, 1, 2, 5, 6, 7, 65535, 65536, 87381, 87382, 98303):
            for column in (0, 1, 2, 127, 128, 255, 256, 24575, 32767):
                exact = (128 * (1 + row % 7) * (1 + column % 3)
                         * Fraction(1 + row % 5, 8) * Fraction(1 + (column // 128) % 5, 16))
                self.assertEqual(tool.scalar_oracle(row, column), float(exact))

    def test_incomplete_comparison_or_any_failed_contract_prevents_pass(self):
        good = {"M": 2048, "N": 24576, "compared_elements": 2048 * 24576,
                "compared_bytes": 2048 * 24576 * 2, "different_bytes": 0,
                "mismatched_elements": 0, "nonfinite_elements": 0,
                "inputs_unchanged": True, "FP64_spotchecks_passed": True,
                "output_contract_passed": True,
                "actual_output_sha256": "a" * 64, "analytic_BF16_output_sha256": "a" * 64}
        self.assertTrue(tool.case_passed(good))
        for field in ("compared_elements", "compared_bytes"):
            bad = copy.deepcopy(good); bad[field] -= 1
            self.assertFalse(tool.case_passed(bad))
        for field in ("different_bytes", "mismatched_elements", "nonfinite_elements"):
            bad = copy.deepcopy(good); bad[field] = 1
            self.assertFalse(tool.case_passed(bad))
        for field in ("inputs_unchanged", "FP64_spotchecks_passed", "output_contract_passed"):
            for invalid in (False, None, "true"):
                bad = copy.deepcopy(good); bad[field] = invalid
                self.assertFalse(tool.case_passed(bad))
        bad = copy.deepcopy(good); bad["analytic_BF16_output_sha256"] = "b" * 64
        self.assertFalse(tool.case_passed(bad))
        bad = copy.deepcopy(good); bad.pop("actual_output_sha256")
        self.assertFalse(tool.case_passed(bad))

    def test_worker_invokes_only_the_public_wrapper_with_unmodified_defaults(self):
        tree = ast.parse(SCRIPT.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "w8a8_gemm"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 4)
        self.assertEqual(calls[0].keywords, [])


if __name__ == "__main__":
    unittest.main()
