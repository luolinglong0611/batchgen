"""CPU-only R1 preflight regression tests, isolated from native dependencies."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    'BATCHGEN_R1_SM103_EXPERIMENTAL': '1',
    'BATCHGEN_R1_SM103_ATTENTION': 'triton_precision',
    'DISABLE_MMA_V5': '1', 'TORCH_CUDA_CHANNEL': 'cu130',
}


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class R1Preflight(unittest.TestCase):
    def setUp(self):
        self.capability = (10, 3)
        self.model_type = 'deepseek_v3'
        self.calls = []
        self.torch = ModuleType('torch')
        self.torch.__version__ = '2.9.0+cu130'
        self.torch.version = SimpleNamespace(cuda='13.0')
        self.torch.cuda = SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda: self.capability,
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(multi_processor_count=132),
        )
        modules = patch.dict(sys.modules, {'torch': self.torch})
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, ENVIRONMENT, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.dispatch = load('r1_preflight_dispatch_test', 'batchgen/models/deepseek/deepseekv3/kernel_dispatch.py')
        self.preflight = load('r1_preflight_test', 'batchgen/runtime_preflight.py')
        self.preflight._resolve_model_type = lambda _: self.model_type
        self.preflight._check_tokenizer = lambda _: self.calls.append('tokenizer')
        self.preflight._check_core_engine = lambda: self.calls.append('core-AOT')
        self.preflight._batchgen_root = lambda: ROOT
        self.preflight._import_required = self.import_required
        self.args = SimpleNamespace(model='deepseek-ai/DeepSeek-R1', enable_cuda_graph=False,
                                    enable_prefix_cache=False)

    def import_required(self, name, *, origin=None):
        self.calls.append((name, origin))
        if name.endswith('.kernel_dispatch'):
            return self.dispatch
        if name == 'libucx':
            return SimpleNamespace(load_library=lambda: self.calls.append('UCX-loaded'))
        entries = ('mla_prefill_sm103_w8a16', 'mla_prefill_sm103_w8a16_prepacked',
                   'mla_decoding_sm103_bf16_with_pagekv', 'varlen_prefill', 'paged_mla_decode',
                   'flash_attn_varlen_func', 'trtllm_batch_decode_with_kv_cache_mla')
        return SimpleNamespace(**{entry: lambda: None for entry in entries})

    def test_precision_runtime_checks_owned_entries_native_kernels_and_ucx_before_core(self):
        # Released HEAD has a CUDA-12 SM90 GQA binary in the common contract.
        # User worktrees may already narrow it; exercise both without depending
        # on unrelated user changes to _COMMON_EXTENSIONS.
        self.preflight._COMMON_EXTENSIONS = (
            'batchgen_kernels.attention._C_fused_ops',
            'batchgen_kernels.attention._C_gqa_mha_decode_bf16',
        )
        self.assertEqual(self.preflight.run_runtime_preflight(self.args), 'deepseek_v3')
        self.assertIn(('triton', 'site'), self.calls)
        for name in ('batchgen.attention.mla.sm103_backend', 'batchgen.attention.mla.triton_precision',
                     'batchgen_kernels.attention._C_fused_ops', 'batchgen_kernels.common._C_mgn_ops'):
            self.assertIn((name, 'batchgen'), self.calls)
        self.assertLess(self.calls.index('UCX-loaded'), self.calls.index('core-AOT'))
        names = [call[0] for call in self.calls if isinstance(call, tuple)]
        self.assertFalse(set(names) & {'flash_attn', 'flash_attn_interface', 'flash_mla', 'deep_gemm', 'flashinfer'})
        self.assertNotIn('batchgen_kernels.attention._C_gqa_mha_decode_bf16', names)

    def test_every_required_opt_in_is_checked_before_core(self):
        for key in ENVIRONMENT:
            with self.subTest(key=key), patch.dict(os.environ, ENVIRONMENT, clear=True):
                self.calls.clear()
                os.environ.pop(key)
                with self.assertRaises(self.preflight.RuntimePreflightError):
                    self.preflight.run_runtime_preflight(self.args)
                self.assertNotIn('core-AOT', self.calls)

    def test_graphs_and_prefix_cache_are_rejected_before_backend_import(self):
        for feature in ('enable_cuda_graph', 'enable_prefix_cache'):
            with self.subTest(feature=feature):
                self.calls.clear()
                setattr(self.args, feature, True)
                with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'eager-only'):
                    self.preflight.run_runtime_preflight(self.args)
                self.assertFalse(self.calls)
                setattr(self.args, feature, False)

    def test_wrong_cuda_abi_and_dequantization_are_rejected(self):
        self.torch.version.cuda = '12.8'
        with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'does not match'):
            self.preflight.run_runtime_preflight(self.args)
        self.torch.version.cuda = '13.0'
        os.environ['BATCHGEN_W8A16_DEQUANT'] = '1'
        with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'BATCHGEN_W8A16_DEQUANT'):
            self.preflight.run_runtime_preflight(self.args)

    def test_missing_precision_entrypoint_fails(self):
        original = self.preflight._import_required
        self.preflight._import_required = lambda name, **kwargs: (
            SimpleNamespace(varlen_prefill=lambda: None) if name.endswith('.triton_precision')
            else original(name, **kwargs))
        with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'paged_mla_decode'):
            self.preflight.run_runtime_preflight(self.args)

    def test_each_actual_native_dependency_still_blocks_startup_when_missing(self):
        for missing in ('batchgen_kernels.attention._C_fused_ops', 'batchgen_kernels.common._C_mgn_ops'):
            with self.subTest(missing=missing):
                self.calls.clear()

                def checked_import(name, **kwargs):
                    if name == missing:
                        raise self.preflight.RuntimePreflightError(f'Missing native module {name}')
                    return self.import_required(name, **kwargs)

                self.preflight._import_required = checked_import
                with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'Missing native module'):
                    self.preflight.run_runtime_preflight(self.args)
                self.assertNotIn('core-AOT', self.calls)

    def test_other_devices_and_models_keep_original_contract(self):
        self.preflight._COMMON_EXTENSIONS = (
            'batchgen_kernels.attention._C_fused_ops',
            'batchgen_kernels.attention._C_gqa_mha_decode_bf16',
        )
        # RuntimeContract instances are frozen snapshots, as in committed HEAD.
        for model_type in ('deepseek_v3', 'deepseek_v2'):
            self.preflight._CONTRACTS[model_type] = self.preflight.RuntimeContract(
                modules=('flash_attn', 'libucx'), extensions=self.preflight._COMMON_EXTENSIONS)
        for model_type, capability in [('deepseek_v3', (9, 0)), ('deepseek_v3', (10, 0)),
                                       ('deepseek_v2', (10, 3))]:
            with self.subTest(model=model_type, capability=capability), patch.dict(os.environ, {}, clear=True):
                os.environ['TORCH_CUDA_CHANNEL'] = 'cu130'
                self.model_type, self.capability = model_type, capability
                self.calls.clear()
                self.preflight.run_runtime_preflight(self.args)
                self.assertIn(('flash_attn', 'site'), self.calls)
                self.assertIn(('batchgen_kernels.attention._C_gqa_mha_decode_bf16', 'batchgen'), self.calls)
                self.assertNotIn(('batchgen_kernels.common._C_mgn_ops', 'batchgen'), self.calls)
                self.assertNotIn(('batchgen.attention.mla.triton_precision', 'batchgen'), self.calls)

    def enable_fast(self):
        os.environ.update(BATCHGEN_R1_SM103_ATTENTION='fa4_flashinfer',
                          BATCHGEN_R1_SM103_FAST_DIAGNOSTIC='1')
        return {'flash-attn-4': '4.0.0b36', 'flashinfer-python': '0.6.13',
                'flashinfer-cubin': '0.6.13', 'nvidia-cutlass-dsl': '4.8.1'}

    def test_fast_diagnostic_checks_pinned_site_functions_and_actual_native_dependencies(self):
        versions = self.enable_fast()
        with patch.object(self.preflight.importlib.metadata, 'version', side_effect=versions.__getitem__):
            self.preflight.run_runtime_preflight(self.args)
        for name in ('flash_attn.cute', 'flashinfer.mla'):
            self.assertIn((name, 'site'), self.calls)
        self.assertNotIn(('batchgen.attention.mla.triton_precision', 'batchgen'), self.calls)
        self.assertIn(('batchgen_kernels.common._C_mgn_ops', 'batchgen'), self.calls)
        self.assertIn('core-AOT', self.calls)

    def test_fast_versions_and_entrypoints_cannot_be_silently_substituted(self):
        versions = self.enable_fast()
        for distribution in versions:
            with self.subTest(distribution=distribution):
                invalid = dict(versions, **{distribution: 'wrong'})
                self.calls.clear()
                with patch.object(self.preflight.importlib.metadata, 'version', side_effect=invalid.__getitem__):
                    with self.assertRaisesRegex(self.preflight.RuntimePreflightError, distribution):
                        self.preflight.run_runtime_preflight(self.args)
                self.assertNotIn('core-AOT', self.calls)
        original = self.preflight._import_required
        self.preflight._import_required = lambda name, **kwargs: (
            SimpleNamespace() if name == 'flashinfer.mla' else original(name, **kwargs))
        with patch.object(self.preflight.importlib.metadata, 'version', side_effect=versions.__getitem__):
            with self.assertRaisesRegex(self.preflight.RuntimePreflightError, 'trtllm_batch_decode'):
                self.preflight.run_runtime_preflight(self.args)
        self.assertNotIn('core-AOT', self.calls)


if __name__ == '__main__':
    unittest.main()
