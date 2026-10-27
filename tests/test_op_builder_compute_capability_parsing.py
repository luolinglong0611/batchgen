"""Exercise the real parser and flag method without importing Torch or CUDA."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "op_builder/builder.py"
tree = ast.parse(SOURCE.read_text())
parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_parse_compute_capability")
builder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CUDAOpBuilder")
methods = [n for n in builder.body if isinstance(n, ast.FunctionDef)
           and n.name in ("compute_capability_args", "filter_ccs")]
namespace = {"os": os, "re": re, "List": list, "WARNING": "WARNING",
             "get_default_compute_capabilities": lambda: "8.0;9.0"}
exec(compile(ast.Module(body=[parser, *methods], type_ignores=[]), str(SOURCE), "exec"), namespace)


class BuilderFixture:
    name = "CPU-only-test"
    jit_mode = False
    compute_capability_args = namespace["compute_capability_args"]
    filter_ccs = namespace["filter_ccs"]


class ComputeCapabilityParsing(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k != "TORCH_CUDA_ARCH_LIST"}
        self.environment = patch.dict(os.environ, env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def flags(self, archs):
        instance = BuilderFixture()
        result = instance.compute_capability_args(archs)
        return result, instance.enable_bf16

    def test_existing_targets_and_complete_blackwell_major(self):
        for arch, target in (("8.0", "80"), ("8.9", "89"), ("9.0", "90"),
                             ("9.0a", "90a"), ("10.0", "100"), ("10.3", "103"),
                             ("10.0a", "100a"), ("10.0f", "100f")):
            with self.subTest(arch=arch):
                flags, bf16 = self.flags(arch)
                self.assertEqual(flags, [f"-gencode=arch=compute_{target},code=sm_{target}"])
                self.assertTrue(bf16)

    def test_ptx_preserves_full_digits_and_architecture_suffix(self):
        for arch, target in (("8.9+PTX", "89"), ("9.0a+PTX", "90a"),
                             ("10.0+PTX", "100"), ("10.3+PTX", "103")):
            with self.subTest(arch=arch):
                flags, _ = self.flags(arch)
                self.assertEqual(flags, [f"-gencode=arch=compute_{target},code=sm_{target}",
                                         f"-gencode=arch=compute_{target},code=compute_{target}"])

    def test_multiple_targets_and_whitespace_keep_requested_order(self):
        flags, bf16 = self.flags("8.0; 8.9\t9.0a\n10.3+PTX")
        self.assertEqual(flags, ["-gencode=arch=compute_80,code=sm_80",
                                 "-gencode=arch=compute_89,code=sm_89",
                                 "-gencode=arch=compute_90a,code=sm_90a",
                                 "-gencode=arch=compute_103,code=sm_103",
                                 "-gencode=arch=compute_103,code=compute_103"])
        self.assertTrue(bf16)

    def test_environment_takes_precedence_and_does_not_query_gpu(self):
        with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.3+PTX"}), redirect_stdout(io.StringIO()):
            flags = BuilderFixture().compute_capability_args("8.0")
        self.assertEqual(flags, ["-gencode=arch=compute_103,code=sm_103",
                                 "-gencode=arch=compute_103,code=compute_103"])

    def test_defaults_remain_available_without_an_override(self):
        flags = BuilderFixture().compute_capability_args()
        self.assertEqual(flags, ["-gencode=arch=compute_80,code=sm_80",
                                 "-gencode=arch=compute_90,code=sm_90"])

    def test_bf16_uses_full_major_and_resets_between_calls(self):
        instance = BuilderFixture()
        instance.compute_capability_args("7.5;10.3")
        self.assertFalse(instance.enable_bf16)
        instance.compute_capability_args("10.0;10.3")
        self.assertTrue(instance.enable_bf16)

    def test_jit_deduplicates_and_places_ptx_on_numeric_maximum(self):
        devices = [(10, 3), (9, 0), (8, 9), (10, 3)]
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: len(devices),
                                                    get_device_capability=lambda i: devices[i]))
        instance = BuilderFixture(); instance.jit_mode = True
        with patch.dict(namespace, {"torch": fake}):
            flags = instance.compute_capability_args()
        self.assertEqual(flags, ["-gencode=arch=compute_89,code=sm_89",
                                 "-gencode=arch=compute_90,code=sm_90",
                                 "-gencode=arch=compute_103,code=sm_103",
                                 "-gencode=arch=compute_103,code=compute_103"])
        self.assertTrue(instance.enable_bf16)

    def test_jit_without_devices_reports_a_clear_error(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
        instance = BuilderFixture(); instance.jit_mode = True
        with patch.dict(namespace, {"torch": fake}), self.assertRaisesRegex(RuntimeError, "No visible CUDA"):
            instance.compute_capability_args()

    def test_filter_receives_complete_targets_and_empty_filter_still_fails(self):
        seen = []
        class Filtered(BuilderFixture):
            def filter_ccs(self, values):
                seen.extend(values)
                return []
        with self.assertRaisesRegex(RuntimeError, "no compute capabilities remaining"):
            Filtered().compute_capability_args("9.0a;10.3+PTX")
        self.assertEqual(seen, ["9.0a", "10.3+PTX"])

    def test_malformed_targets_fail_before_any_compiler_call(self):
        invalid = ("", " ", ";", "10", "10.", ".3", "0.0", "00.3", "10.03",
                   "-1.0", "10.3.1", "sm_103", "compute_103", "10.3z", "9.0aa",
                   "10.3+ptx", "10.3++PTX", "10.3+PTX;", ";10.3", "10.3,9.0",
                   "10.3+PTXunexpected", "10.3$(id)", 103, ["10.3"])
        for arch in invalid:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                self.flags(arch)


if __name__ == "__main__":
    unittest.main()
