"""CPU-only explicit backend, geometry and observation checks."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/sglang_r1_attention_compat.py'
SPEC = importlib.util.spec_from_file_location('sg_attention_compat_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FakeRagged:
    def __init__(self):
        self._backend = 'fa2'

    def plan(self, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads,
             head_dim_qk, head_dim_vo, causal, q_data_type,
             kv_data_type=None, pos_encoding_mode='NONE', use_fp16_qk_reduction=False,
             window_left=-1, sm_scale=None, logits_soft_cap=None):
        self._sm_scale, self._causal = sm_scale, causal

    begin_forward = plan

    def run(self, q, k, v):
        return v

    def forward(self, q, k, v, causal=True, sm_scale=None):
        self._causal, self._sm_scale = causal, sm_scale
        return self.run(q, k, v)


def ragged_values(**kwargs):
    indptr = object()
    values = dict(qo_indptr=indptr, kv_indptr=indptr, num_qo_heads=128,
                  num_kv_heads=128, head_dim_qk=192, head_dim_vo=128,
                  causal=True, q_data_type='torch.bfloat16')
    return dict(values, **kwargs)


class AttentionCompatTests(unittest.TestCase):
    def test_install_forces_native_fa2_without_replacing_token_addresses_or_allocator(self):
        class NativeRagged(FakeRagged):
            def __init__(self, *args, backend='auto', **kwargs):
                super().__init__()
                self._backend = backend

        class NativeMla(NativeRagged):
            pass

        class Backend:
            def __init__(self, runner):
                self.enable_chunk_kv = False
                self.fmha_backend = 'cutlass'
                self.ragged = module.BatchPrefillWithRaggedKVCacheWrapper('workspace', 'NHD', backend='cutlass')
                self.decode = module.BatchMLAPagedAttentionWrapper('workspace', backend='auto')

            def init_forward_metadata(self, batch):
                self.forward_metadata = SimpleNamespace(use_ragged=True)

        module = SimpleNamespace(FlashInferMLAAttnBackend=Backend,
                                 BatchMLAPagedAttentionWrapper=NativeMla,
                                 BatchPrefillWithRaggedKVCacheWrapper=NativeRagged)
        server = SimpleNamespace(attention_backend='flashinfer', page_size=64,
                                 disable_cuda_graph=True, disable_radix_cache=True,
                                 disable_chunked_prefix_cache=True, flashinfer_mla_disable_ragged=False,
                                 enable_piecewise_cuda_graph=False, chunked_prefill_size=-1,
                                 disaggregation_mode='null')
        modules = {'sglang.srt.layers.attention.flashinfer_mla_backend': module,
                   'sglang.srt.server_args': SimpleNamespace(get_global_server_args=lambda: server)}
        versions = {'sglang': '0.5.5.post3', 'flashinfer-python': '0.5.2'}
        identity = lambda _: {'path': 'source.py', 'sha256': M.SG_BACKEND_SHA}
        with patch.dict(os.environ, {M.ENV: '1'}), \
             patch.object(M.importlib.metadata, 'version', side_effect=versions.__getitem__), \
             patch.object(M.importlib, 'import_module', side_effect=modules.__getitem__), \
             patch.object(M, '_identity', side_effect=identity), \
             patch.object(M, '_INSTALLATION', None), patch.object(M, '_ROLLBACK', None), \
             patch.object(M, '_WRAPPERS', []), patch.object(M, '_loaded_kernel_objects', return_value=[]):
            receipt = M.install()
            self.assertEqual(receipt['installation']['sg_allocator_page_size'], 64)
            self.assertFalse(receipt['installation']['native_page64_allocator_qualified'])
            self.assertEqual(M.install(), receipt)
            runner = SimpleNamespace(page_size=64, dtype='torch.bfloat16', kv_cache_dtype='torch.bfloat16')
            backend = Backend(runner)
            self.assertEqual(backend.decode._backend, 'fa2')
            self.assertEqual(backend.ragged._backend, 'fa2')
            self.assertEqual(runner.page_size, 64)
            mode = SimpleNamespace(is_decode_or_idle=lambda: False, is_extend=lambda: True)
            backend.init_forward_metadata(SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=[0]))
            with self.assertRaisesRegex(RuntimeError, 'cached prefix'):
                backend.init_forward_metadata(SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=[1]))

    def test_setup_failure_does_not_replace_any_registration(self):
        class MissingMetadata:
            pass

        module = SimpleNamespace(FlashInferMLAAttnBackend=MissingMetadata,
                                 BatchMLAPagedAttentionWrapper=FakeRagged,
                                 BatchPrefillWithRaggedKVCacheWrapper=FakeRagged)
        server = SimpleNamespace(attention_backend='flashinfer', page_size=64,
                                 disable_cuda_graph=True, disable_radix_cache=True,
                                 disable_chunked_prefix_cache=True, flashinfer_mla_disable_ragged=False,
                                 enable_piecewise_cuda_graph=False, chunked_prefill_size=-1,
                                 disaggregation_mode='null')
        modules = {'sglang.srt.layers.attention.flashinfer_mla_backend': module,
                   'sglang.srt.server_args': SimpleNamespace(get_global_server_args=lambda: server)}
        versions = {'sglang': '0.5.5.post3', 'flashinfer-python': '0.5.2'}
        with patch.dict(os.environ, {M.ENV: '1'}), patch.object(M, '_INSTALLATION', None), \
             patch.object(M.importlib.metadata, 'version', side_effect=versions.__getitem__), \
             patch.object(M.importlib, 'import_module', side_effect=modules.__getitem__), \
             patch.object(M, '_identity', return_value={'path': 'source.py', 'sha256': M.SG_BACKEND_SHA}):
            with self.assertRaises(AttributeError):
                M.install()
            self.assertIs(module.BatchMLAPagedAttentionWrapper, FakeRagged)
            self.assertIs(module.BatchPrefillWithRaggedKVCacheWrapper, FakeRagged)

    def test_plan_and_run_record_actual_geometry_scale_and_backend(self):
        with patch.object(M, '_WRAPPERS', []):
            wrapper = M._wrap_native_wrapper(FakeRagged(), 'ragged')
            wrapper.begin_forward(**ragged_values())
            q = SimpleNamespace(shape=(4, 128, 192), dtype='torch.bfloat16')
            v = SimpleNamespace(shape=(4, 128, 128), dtype='torch.bfloat16')
            self.assertIs(wrapper.forward(q, q, v, causal=True, sm_scale=0.125), v)
            record = M.receipt()['wrappers'][0]
            self.assertEqual(record['completed_plans'], 1)
            self.assertEqual(record['completed_run_api_calls'], 1)
            self.assertEqual(record['last_run']['sm_scale'], 0.125)
            self.assertEqual(record['last_run']['shapes'], [[4, 128, 192], [4, 128, 192], [4, 128, 128]])
            self.assertEqual(record['current_backend'], 'fa2')
            self.assertFalse(M.receipt()['formal_test_permitted'])

    def test_auto_or_changed_backend_is_rejected(self):
        wrapper = FakeRagged()
        wrapper._backend = 'auto'
        with self.assertRaisesRegex(RuntimeError, 'retain explicit fa2'):
            M._wrap_native_wrapper(wrapper, 'ragged')
        wrapper._backend = 'fa2'
        M._wrap_native_wrapper(wrapper, 'ragged')
        wrapper._backend = 'cutlass'
        with self.assertRaisesRegex(RuntimeError, 'changed the explicit fa2'):
            wrapper.plan(**ragged_values())

    def test_wrong_head_geometry_and_implicit_fp16_are_rejected(self):
        for kwargs in ({'head_dim_qk': 576}, {'head_dim_vo': 512},
                       {'num_qo_heads': 8}, {'q_data_type': 'float16'}, {'causal': False}):
            values = dict(ragged_values(), **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                M._checked_plan('ragged', values)

    def test_native_page1_mla_contract_and_explicit_scale(self):
        values = dict(head_dim_ckv=512, head_dim_kpe=64, page_size=1,
                      num_heads=128, causal=False, sm_scale=0.125,
                      q_data_type='torch.bfloat16', kv_data_type='torch.bfloat16')
        self.assertEqual(M._checked_plan('mla', values)['page_size'], 1)
        for kwargs in ({'page_size': 64}, {'head_dim_ckv': 128}, {'sm_scale': None}, {'sm_scale': float('nan')}, {'causal': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                M._checked_plan('mla', dict(values, **kwargs))

    def test_run_requires_plan_and_positive_effective_scale(self):
        wrapper = M._wrap_native_wrapper(FakeRagged(), 'ragged')
        q = SimpleNamespace(shape=(1, 128, 192), dtype='torch.bfloat16')
        v = SimpleNamespace(shape=(1, 128, 128), dtype='torch.bfloat16')
        with self.assertRaisesRegex(RuntimeError, 'observed fa2 plan'):
            wrapper.run(q, q, v)
        wrapper.plan(**ragged_values())
        with self.assertRaisesRegex(ValueError, 'explicit finite positive'):
            wrapper.run(q, q, v)

    def test_install_is_opt_in_before_any_gpu_library_import(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(M.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(RuntimeError, 'requires explicit'):
                M.install()
            imported.assert_not_called()


if __name__ == '__main__':
    unittest.main()
