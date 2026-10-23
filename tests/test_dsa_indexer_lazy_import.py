"""Import/first-call regression tests with a stub CUDA extension compiler."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
INDEXER = ROOT / 'batchgen_kernels/attention/dsa/indexer/__init__.py'


class IndexerLazyImport(unittest.TestCase):
    def setUp(self):
        self.loads, self.calls = [], []
        torch = ModuleType('torch')
        torch.Tensor = object
        compiler = ModuleType('torch.utils.cpp_extension')
        compiler.load = self.load_extension
        writer = ModuleType('batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8')
        writer.fused_indexer_k_write_fp8 = object()
        writer.page_split_write_plan = object()
        with patch.dict(sys.modules, {
            'torch': torch, 'torch.utils': ModuleType('torch.utils'),
            'torch.utils.cpp_extension': compiler,
            'batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8': writer,
        }):
            spec = importlib.util.spec_from_file_location('indexer_lazy_test', INDEXER)
            self.indexer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.indexer)
        self.writer = writer

    def load_extension(self, **kwargs):
        self.loads.append(kwargs)

        def called(name, *args):
            self.calls.append((name, args))
            return name + '_result'

        return SimpleNamespace(
            fast_hadamard_transform=lambda *args: called('hadamard', *args),
            fused_rope_hadamard=lambda *args: called('fused', *args),
            fused_rope_hadamard_out=lambda *args: called('fused_out', *args),
        )

    def test_package_import_compiles_nothing_and_keeps_aot_exports(self):
        self.assertEqual(self.loads, [])
        self.assertIs(self.indexer.fused_indexer_k_write_fp8, self.writer.fused_indexer_k_write_fp8)
        self.assertIs(self.indexer.page_split_write_plan, self.writer.page_split_write_plan)

    def test_hadamard_first_call_loads_once_and_preserves_build_flags(self):
        self.assertEqual(self.indexer.hadamard_transform('x'), 'hadamard_result')
        self.indexer.hadamard_transform('y', 0.25)
        self.assertEqual(self.calls, [('hadamard', ('x', 1.0)), ('hadamard', ('y', 0.25))])
        self.assertEqual(len(self.loads), 1)
        build = self.loads[0]
        self.assertEqual(build['name'], 'fast_hadamard_transform_cuda')
        self.assertEqual([Path(p).name for p in build['sources']],
                         ['hadamard_binding.cpp', 'fast_hadamard_transform_cuda.cu'])
        self.assertEqual(build['extra_cflags'], ['-O3', '-std=c++17'])
        self.assertEqual(build['extra_cuda_cflags'], [
            '-O3', '-std=c++17', '--use_fast_math', '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '--expt-relaxed-constexpr', '--expt-extended-lambda',
        ])
        self.assertEqual(build['extra_include_paths'], [str(INDEXER.parent / 'csrc')])
        self.assertFalse(build['verbose'])

    def test_both_fused_entrypoints_share_cache_even_when_out_is_called_first(self):
        out = object()
        self.assertIs(self.indexer.fused_rope_hadamard_out('x', 'cos', 'sin', 'pos', out), out)
        self.assertEqual(self.indexer.fused_rope_hadamard('y', 'cos', 'sin', 'pos', 0.5), 'fused_result')
        self.indexer.fused_rope_hadamard_out('z', 'cos', 'sin', 'pos', out, 0.25)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.loads[0]['name'], 'fused_rope_hadamard_cuda')
        self.assertEqual([Path(p).name for p in self.loads[0]['sources']],
                         ['fused_rope_hadamard_binding.cpp', 'fused_rope_hadamard.cu'])
        self.assertEqual(self.calls, [
            ('fused_out', ('x', 'cos', 'sin', 'pos', out, 128 ** -0.5)),
            ('fused', ('y', 'cos', 'sin', 'pos', 0.5)),
            ('fused_out', ('z', 'cos', 'sin', 'pos', out, 0.25)),
        ])
        self.indexer.hadamard_transform('h')
        self.assertEqual([load['name'] for load in self.loads],
                         ['fused_rope_hadamard_cuda', 'fast_hadamard_transform_cuda'])

    def test_compile_failure_propagates_and_is_not_cached_as_success(self):
        def broken(**kwargs):
            raise RuntimeError('compiler failed')

        self.indexer.load = broken
        with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
            self.indexer.fused_rope_hadamard('x', 'cos', 'sin', 'pos')
        self.indexer.load = self.load_extension
        self.assertEqual(self.indexer.fused_rope_hadamard('x', 'cos', 'sin', 'pos'), 'fused_result')
        self.assertEqual(len(self.loads), 1)


if __name__ == '__main__':
    unittest.main()
