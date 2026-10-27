"""CPU dispatch contracts; these tests do not qualify GPU arithmetic."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "batchgen/gemm/r1_sm103_compat.py"
spec = importlib.util.spec_from_file_location("r1_sm103_contract", SOURCE)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class Tensor:
    def __init__(self, shape, dtype="fp8", device=None, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or DEVICE
        self.contiguous = contiguous
        self.copied_from = None

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def copy_(self, value):
        self.copied_from = value
        return self

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            known = math.prod(x for x in shape if x != -1)
            shape = tuple(math.prod(self.shape) // known if x == -1 else x for x in shape)
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError("fake view element count mismatch")
        return Tensor(shape, self.dtype, self.device, self.contiguous)


DEVICE = types.SimpleNamespace(type="cuda", index=0)


class DispatchContract(unittest.TestCase):
    def setUp(self):
        compat._ACTUAL_BACKENDS.clear()
        self.env = patch.dict(os.environ, {compat.COMPAT_ENV: "1", "DISABLE_MMA_V5": "1"})
        self.env.start()
        self.calls = []
        self.quant_calls = []
        self.imports = []
        self.capability = (10, 3)
        self.torch = types.SimpleNamespace(
            float8_e4m3fn="fp8", float32="fp32", bfloat16="bf16",
            cuda=types.SimpleNamespace(get_device_capability=lambda device: self.capability),
            empty=lambda shape, device, dtype: Tensor(shape, dtype, device))

    def tearDown(self):
        self.env.stop()

    def inputs(self, m=3, n=576, k=256):
        # Non-contiguous scales model lossless TMA column-major views.  No
        # contiguous(), value conversion, or scale cast is allowed in dispatch.
        return (Tensor((m, k)), Tensor((m, (k + 127) // 128), "fp32", contiguous=False),
                Tensor((n, k)), Tensor(((n + 127) // 128, (k + 127) // 128), "fp32", contiguous=False))

    def kernel(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return Tensor((args[0].shape[0], args[2].shape[0]), "bf16")

    def quant(self, *args, **kwargs):
        self.quant_calls.append((args, kwargs))
        x = args[0]
        return Tensor(x.shape), Tensor((x.shape[0], (x.shape[1] + 127) // 128), "fp32")

    def load(self, name):
        self.imports.append(name)
        if name == "torch":
            return self.torch
        if name == "batchgen.gemm.w8a8":
            return types.SimpleNamespace(w8a8_gemm=self.kernel)
        if name == "batchgen_kernels.triton.fp8_quantize":
            return types.SimpleNamespace(per_token_blocked_quantize_bf16_to_fp8_1d=self.quant)
        raise AssertionError(f"unexpected dependency {name}")

    def test_module_import_and_receipt_do_not_import_accelerators(self):
        with tempfile.TemporaryDirectory() as directory:
            script = ("import sys,importlib.util\n"
                      "class Reject:\n"
                      " def find_spec(self,fullname,path=None,target=None):\n"
                      "  if fullname.split('.')[0] in ('torch','triton','deep_gemm','batchgen','batchgen_kernels'):\n"
                      "   raise RuntimeError('accelerator import: '+fullname)\n"
                      "sys.meta_path.insert(0,Reject())\n"
                      f"s=importlib.util.spec_from_file_location('isolated',{str(SOURCE)!r})\n"
                      "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                      "assert m.backend_receipt()['actual_backends']==[]\n")
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_preserves_deepgemm_signature_and_never_queries_cuda(self):
        os.environ[compat.COMPAT_ENV] = "0"
        calls = []
        native = types.SimpleNamespace(fp8_gemm_nt=lambda *args, **kwargs: calls.append((args, kwargs)))
        a, a_scale, w, w_scale = self.inputs()
        out = Tensor((3, 576), "bf16")
        with patch.object(compat.importlib, "import_module", return_value=native) as load:
            compat.fp8_gemm_nt((a, a_scale), (w, w_scale), out, recipe=(1, 2), disable_ue8m0_cast=False)
            load.assert_called_once_with("deep_gemm")
        self.assertEqual(calls, [(((a, a_scale), (w, w_scale), out), {"recipe": (1, 2), "disable_ue8m0_cast": False})])
        self.assertEqual(compat.backend_receipt()["actual_backends"], [compat.DEFAULT_BACKEND])

    def test_four_tensor_dispatch_preserves_identity_and_native_launch(self):
        inputs = self.inputs()
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            result = compat.w8a8_gemm(*inputs)
        self.assertEqual(result.shape, (3, 576))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], {})
        self.assertTrue(all(actual is expected for actual, expected in zip(self.calls[0][0], inputs)))
        self.assertNotIn("deep_gemm", self.imports)
        self.assertEqual(compat.backend_receipt()["actual_backends"], [compat.COMPAT_BACKEND])

    def test_deepgemm_output_convention_copies_only_result(self):
        a, a_scale, w, w_scale = self.inputs()
        out = Tensor((3, 576), "bf16")
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            result = compat.fp8_gemm_nt((a, a_scale), (w, w_scale), out, recipe=None, disable_ue8m0_cast=True)
        self.assertIsNone(result)
        self.assertEqual(out.copied_from.shape, (3, 576))
        self.assertEqual(len(self.calls), 1)

    def test_sm103_opt_in_mma_and_unsupported_options_fail_closed(self):
        a, a_scale, w, w_scale = self.inputs()
        out = Tensor((3, 576), "bf16")
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            for invalid in ("true", "auto", "2", ""):
                os.environ[compat.COMPAT_ENV] = invalid
                with self.assertRaises(ValueError):
                    compat.w8a8_gemm(a, a_scale, w, w_scale)
            os.environ[compat.COMPAT_ENV] = "1"
            os.environ["DISABLE_MMA_V5"] = "0"
            with self.assertRaisesRegex(RuntimeError, "DISABLE_MMA_V5"):
                compat.w8a8_gemm(a, a_scale, w, w_scale)
            os.environ["DISABLE_MMA_V5"] = "1"
            for device in ((9, 0), (10, 0), (10, 1)):
                self.capability = device
                with self.assertRaisesRegex(RuntimeError, "SM103"):
                    compat.w8a8_gemm(a, a_scale, w, w_scale)
            self.capability = (10, 3)
            for kwargs in ({"recipe": (1,)}, {"disable_ue8m0_cast": False}, {"c": out}):
                with self.assertRaises(ValueError):
                    compat.fp8_gemm_nt((a, a_scale), (w, w_scale), out, **kwargs)
        self.assertFalse(self.calls)

    def test_invalid_dtype_shape_layout_and_device_cannot_launch(self):
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            for index, field, value in ((0, "dtype", "bf16"), (2, "dtype", "bf16"),
                                        (1, "dtype", "int32"), (3, "dtype", "int32"),
                                        (1, "shape", (3, 1)), (3, "shape", (4, 2)),
                                        (2, "shape", (576, 128)),
                                        (0, "device", types.SimpleNamespace(type="cpu"))):
                inputs = list(self.inputs()); setattr(inputs[index], field, value)
                with self.subTest(index=index, field=field), self.assertRaises((TypeError, ValueError)):
                    compat.w8a8_gemm(*inputs)
            for output in (Tensor((2, 576), "bf16"), Tensor((3, 576), "fp32"),
                           Tensor((3, 576), "bf16", contiguous=False)):
                with self.assertRaises(ValueError):
                    compat.w8a8_gemm(*self.inputs(), out=output)
        self.assertFalse(self.calls)

    def test_empty_batches_skip_quantizer_and_gemm_and_restore_rank3(self):
        _, _, w, w_scale = self.inputs(m=0)
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            result = compat.w8a16_gemm(w, w_scale, Tensor((0, 7, 256), "bf16"))
        self.assertEqual(result.shape, (0, 7, 576))
        self.assertFalse(self.calls)
        self.assertFalse(self.quant_calls)
        self.assertNotIn("batchgen_kernels.triton.fp8_quantize", self.imports)
        self.assertEqual(compat.backend_receipt()["actual_backends"], [])

    def test_common_bf16_entry_uses_exact_main_quantizer_defaults_and_fp8_weights(self):
        _, _, w, w_scale = self.inputs(m=6)
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            result = compat.w8a16_gemm(w, w_scale, Tensor((2, 3, 256), "bf16"))
        self.assertEqual(result.shape, (2, 3, 576))
        self.assertEqual(len(self.quant_calls), 1)
        self.assertEqual(self.quant_calls[0][1], {})
        self.assertEqual(self.quant_calls[0][0][0].shape, (6, 256))
        self.assertIs(self.calls[0][0][2], w)
        self.assertIs(self.calls[0][0][3], w_scale)
        self.assertIn(compat.QUANT_BACKEND, compat.backend_receipt()["actual_backends"])

    def test_bf16_checkpoint_weights_or_invalid_inputs_fail_before_quantization(self):
        _, _, w, w_scale = self.inputs()
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            bad_weight = Tensor((576, 256), "bf16")
            for weight, scale, activation in ((bad_weight, w_scale, Tensor((3, 256), "bf16")),
                                              (w, w_scale, Tensor((3, 128), "bf16")),
                                              (w, w_scale, Tensor((3, 256), "bf16", contiguous=False)),
                                              (w, Tensor((5, 2), "int32"), Tensor((3, 256), "bf16"))):
                with self.assertRaises((TypeError, ValueError)):
                    compat.w8a16_gemm(weight, scale, activation)
        self.assertFalse(self.calls)
        self.assertFalse(self.quant_calls)

    def test_invalid_kernel_result_never_produces_a_backend_receipt(self):
        inputs = self.inputs()
        for result in (Tensor((3, 575), "bf16"), Tensor((3, 576), "fp32"),
                       Tensor((3, 576), "bf16", types.SimpleNamespace(type="cpu"))):
            self.kernel = lambda *args, **kwargs: result
            with patch.object(compat.importlib, "import_module", side_effect=self.load):
                with self.assertRaisesRegex(RuntimeError, "invalid output contract"):
                    compat.w8a8_gemm(*inputs)
        self.assertEqual(compat.backend_receipt()["actual_backends"], [])

    def actual_function(self, path, name, namespace):
        # Execute the actual changed hook, avoiding unrelated optional GPU
        # modules elsewhere in its file.  This does not substitute its body.
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        fn.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), "exec"), namespace)
        return namespace[name]

    def test_actual_ordinary_wrapper_rejects_unsupported_options_and_keeps_out_semantics(self):
        inputs = self.inputs()
        function = self.actual_function(ROOT / "batchgen/gemm/w8a8_deepgemm.py", "w8a8_deepgemm", {
            "torch": self.torch, "sm103_fp8_compat_enabled": compat.sm103_fp8_compat_enabled,
            "w8a8_gemm": compat.w8a8_gemm})
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            for option in ({"c": object()}, {"num_valid_tokens": object()}, {"expected_m": 3},
                           {"recipe": (1,)}, {"disable_ue8m0_cast": False}):
                with self.assertRaises(ValueError):
                    function(*inputs, **option)
            self.assertFalse(self.calls)
            result = function(*inputs)
            self.assertEqual(result.shape, (3, 576))
            output = Tensor((3, 576), "bf16")
            self.assertIsNone(function(*inputs, out=output))
            self.assertIsNotNone(output.copied_from)

    def test_actual_fa3_hook_uses_same_common_bf16_function(self):
        function = self.actual_function(ROOT / "batchgen/attention/mla/fa3_backend.py", "w8a16_gemm", {
            "sm103_fp8_compat_enabled": compat.sm103_fp8_compat_enabled,
            "sm103_w8a16_gemm": compat.w8a16_gemm})
        _, _, w, w_scale = self.inputs()
        with patch.object(compat.importlib, "import_module", side_effect=self.load):
            result = function(w, w_scale, Tensor((3, 256), "bf16"))
        self.assertEqual(result.shape, (3, 576))
        self.assertEqual(len(self.quant_calls), 1)
        self.assertEqual(len(self.calls), 1)

    def test_hooks_are_explicit_and_do_not_import_deepgemm_eagerly(self):
        for path in (ROOT / "batchgen/attention/mla/fa3_backend.py", ROOT / "batchgen/gemm/w8a8_deepgemm.py"):
            tree = ast.parse(path.read_text())
            top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            for node in top_imports:
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                self.assertNotIn("deep_gemm", names)
        fa3 = ast.parse((ROOT / "batchgen/attention/mla/fa3_backend.py").read_text())
        fn = next(n for n in fa3.body if isinstance(n, ast.FunctionDef) and n.name == "w8a16_gemm")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "sm103_w8a16_gemm" for n in ast.walk(fn)))
        body = ast.get_source_segment((ROOT / "batchgen/gemm/w8a8_deepgemm.py").read_text(),
                                      next(n for n in ast.parse((ROOT / "batchgen/gemm/w8a8_deepgemm.py").read_text()).body
                                           if isinstance(n, ast.FunctionDef) and n.name == "w8a8_deepgemm"))
        self.assertIn("num_valid_tokens is not None or expected_m is not None", body)
        self.assertIn("c is not None", body)


if __name__ == "__main__":
    unittest.main()
