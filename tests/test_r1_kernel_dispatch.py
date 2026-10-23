"""CPU-only import isolation checks; these do not qualify GPU numerics."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "batchgen/models/deepseek/deepseekv3/kernel_dispatch.py"
LEGACY = "batchgen.attention.mla.fa3_backend"
NEW = "batchgen.attention.mla.sm103_backend"
ENVIRONMENT = {
    "BATCHGEN_R1_SM103_EXPERIMENTAL": "1",
    "BATCHGEN_R1_SM103_ATTENTION": "triton_precision",
    "DISABLE_MMA_V5": "1", "TORCH_CUDA_CHANNEL": "cu130",
}


class KernelDispatchTests(unittest.TestCase):
    def setUp(self):
        self.capability = (10, 3)
        torch = ModuleType("torch")
        torch.device = lambda device: SimpleNamespace(type=str(device).split(":")[0])
        torch.cuda = SimpleNamespace(get_device_capability=lambda _: self.capability)
        torch.version = SimpleNamespace(cuda="13.0")
        self.fake_torch = torch
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Loading dispatch must not need any optional attention/GEMM package.
        with patch.dict(sys.modules, {"torch": torch, LEGACY: None, NEW: None,
                                      "flash_mla": None, "deep_gemm": None}):
            spec = importlib.util.spec_from_file_location("r1_dispatch_test", DISPATCH)
            self.dispatch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.dispatch)

    def test_exact_sm103_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "BATCHGEN_R1_SM103_EXPERIMENTAL"):
            self.dispatch.select_sm103_experiment("cuda:0")
        os.environ.update(ENVIRONMENT)
        self.assertTrue(self.dispatch.select_sm103_experiment("cuda:0"))

    def test_hopper_other_blackwell_and_cpu_keep_existing_selection(self):
        for capability in [(9, 0), (10, 0), (12, 0)]:
            self.capability = capability
            self.assertFalse(self.dispatch.select_sm103_experiment("cuda:0"))
        self.capability = (10, 3)
        self.assertFalse(self.dispatch.select_sm103_experiment("cpu"))

    def test_old_projection_is_only_imported_when_called(self):
        self.capability = (9, 0)
        legacy = ModuleType(LEGACY)
        calls = []
        legacy.w8a16_gemm = lambda *args: calls.append(args) or "legacy output"
        activation = SimpleNamespace(device="cuda:0")
        with patch.dict(sys.modules, {LEGACY: legacy, NEW: None}):
            self.assertEqual(self.dispatch.w8a16_gemm("weight", "scale", activation), "legacy output")
        self.assertEqual(calls, [("weight", "scale", activation)])

    def test_sm103_requires_compiler_gate_before_importing_backend(self):
        os.environ.update(ENVIRONMENT)
        os.environ.pop("DISABLE_MMA_V5")
        with patch.dict(sys.modules, {LEGACY: None, NEW: None}):
            with self.assertRaisesRegex(RuntimeError, "DISABLE_MMA_V5"):
                self.dispatch.w8a16_gemm("weight", "scale", SimpleNamespace(device="cuda:0"))

    def test_sm103_projection_does_not_import_legacy(self):
        os.environ.update(ENVIRONMENT)
        backend = ModuleType(NEW)
        calls = []
        backend.w8a16_gemm = lambda *args: calls.append(args) or "experimental output"
        activation = SimpleNamespace(device="cuda:0")
        with patch.dict(sys.modules, {LEGACY: None, NEW: backend, "deep_gemm": None}):
            self.assertEqual(self.dispatch.w8a16_gemm("weight", "scale", activation), "experimental output")
        self.assertEqual(calls, [("weight", "scale", activation)])

    def test_legacy_quantization_keeps_keyword_arguments(self):
        self.capability = (9, 0)
        legacy = ModuleType(LEGACY)
        calls = []
        legacy.act_quant = lambda *args, **kwargs: calls.append((args, kwargs)) or "quantized"
        x = SimpleNamespace(device="cuda:0")
        with patch.dict(sys.modules, {LEGACY: legacy}):
            self.assertEqual(self.dispatch.act_quant(x, 64, 1e-6, "valid", True), "quantized")
        self.assertEqual(calls, [((x,), {"block_size": 64, "eps": 1e-6,
                                       "num_valid_tokens": "valid", "scale_tma_aligned": True})])

    def test_attention_binding_requires_explicit_precision_selection(self):
        backend = ModuleType(NEW)
        for name in ("mla_prefill_sm103_w8a16", "mla_prefill_sm103_w8a16_prepacked",
                     "mla_decoding_sm103_bf16_with_pagekv"):
            setattr(backend, name, lambda self, *args: self)
        layer = SimpleNamespace()
        with patch.dict(sys.modules, {NEW: backend, LEGACY: None, "flash_mla": None}):
            with self.assertRaisesRegex(RuntimeError, "BATCHGEN_R1_SM103_EXPERIMENTAL"):
                self.dispatch.bind_sm103_attention(layer)
            os.environ.update(ENVIRONMENT)
            self.dispatch.bind_sm103_attention(layer)
        self.assertTrue(layer._batchgen_r1_sm103_experimental)
        self.assertEqual(layer._r1_attention_implementation, "triton_precision")
        self.assertIs(layer.prefill_attn_w8a16(), layer)
        self.assertIs(layer.decoding_attn_mode_3_bf16(), layer)
        with self.assertRaisesRegex(RuntimeError, "BF16 paged"):
            layer.decoding_attn_mode_3_fp8()

    def test_precision_selection_and_cuda_quantization_contracts_fail_closed(self):
        for key, value in [("BATCHGEN_R1_SM103_ATTENTION", "fa4_flashinfer"),
                           ("BATCHGEN_R1_SM103_ATTENTION", ""),
                           ("TORCH_CUDA_CHANNEL", "cu128"),
                           ("BATCHGEN_W8A16_DEQUANT", "1")]:
            with self.subTest(key=key, value=value), patch.dict(os.environ, ENVIRONMENT, clear=True):
                os.environ[key] = value
                with self.assertRaisesRegex(RuntimeError, key):
                    self.dispatch.select_sm103_experiment("cuda:0")
        with patch.dict(os.environ, ENVIRONMENT, clear=True):
            self.fake_torch.version.cuda = "12.8"
            with self.assertRaisesRegex(RuntimeError, "Torch CUDA 13.0"):
                self.dispatch.select_sm103_experiment("cuda:0")

    def test_fast_attention_is_a_separate_explicit_unqualified_diagnostic(self):
        os.environ.update(ENVIRONMENT, BATCHGEN_R1_SM103_ATTENTION="fa4_flashinfer")
        with self.assertRaisesRegex(RuntimeError, "BATCHGEN_R1_SM103_FAST_DIAGNOSTIC"):
            self.dispatch.validate_sm103_environment()
        os.environ["BATCHGEN_R1_SM103_FAST_DIAGNOSTIC"] = "1"
        backend = ModuleType(NEW)
        for name in ("mla_prefill_sm103_w8a16", "mla_prefill_sm103_w8a16_prepacked",
                     "mla_decoding_sm103_bf16_with_pagekv"):
            setattr(backend, name, lambda self, *args: self)
        layer = SimpleNamespace()
        with patch.dict(sys.modules, {NEW: backend, LEGACY: None, "flash_mla": None, "deep_gemm": None}):
            self.dispatch.bind_sm103_attention(layer)
        self.assertEqual(layer._r1_attention_implementation, "fa4_flashinfer")
        self.assertIs(layer.prefill_attn_w8a16(), layer)
        self.assertIs(layer.decoding_attn_mode_3_bf16(), layer)
        os.environ["BATCHGEN_R1_SM103_ATTENTION"] = "torch_fp64_reference"
        with self.assertRaisesRegex(RuntimeError, "BATCHGEN_R1_SM103_ATTENTION"):
            self.dispatch.validate_sm103_environment()


if __name__ == "__main__":
    unittest.main()
