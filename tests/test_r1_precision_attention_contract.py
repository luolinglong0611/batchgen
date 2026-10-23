"""CPU-only validation and dispatch checks; GPU numerical gates remain separate."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PRECISION = ROOT / 'batchgen/attention/mla/triton_precision.py'
BACKEND = ROOT / 'batchgen/attention/mla/sm103_backend.py'


def functions(path, names, **namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class MetadataContract(unittest.TestCase):
    def setUp(self):
        self.f = functions(PRECISION, {'_validate_boundaries', '_validate_page_contents'})

    def test_ragged_and_empty_prefill_sequences(self):
        self.f['_validate_boundaries']([0, 1, 1, 64, 129], 129, 65)
        self.f['_validate_boundaries']([0, 0], 0, 0)

    def test_bad_cumulative_boundaries_rejected(self):
        for boundaries, total, maximum in [([1, 4], 4, 4), ([0, 3], 4, 4),
                                           ([0, 3, 2, 4], 4, 4), ([0, 65], 65, 64)]:
            with self.subTest(boundaries=boundaries, maximum=maximum), self.assertRaises(ValueError):
                self.f['_validate_boundaries'](boundaries, total, maximum)

    def test_unused_sentinel_and_page_boundary_allowed(self):
        self.f['_validate_page_contents']([[3, -1], [0, 2]], [64, 65], 4, 2)

    def test_active_sentinel_or_out_of_bounds_rejected(self):
        for row in [[-1, 2], [0, 4]]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.f['_validate_page_contents']([row], [65], 4, 2)

    def test_empty_or_uncovered_decode_rejected(self):
        for length in [0, -1, 129]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.f['_validate_page_contents']([[0, 1]], [length], 4, 2)


class DispatchContract(unittest.TestCase):
    def setUp(self):
        self.namespace = functions(BACKEND, {'_attention_implementation', '_check_compiler_environment',
                                             '_attend_prefill', '_attend_decode'},
                                   os=os, __package__='r1_test_backend')
        self.layer = SimpleNamespace(softmax_scale=0.125, _r1_attention_implementation='triton_precision')

    def test_default_and_invalid_selection(self):
        select = self.namespace['_attention_implementation']
        self.assertEqual(select(SimpleNamespace()), 'fa4_flashinfer')
        with self.assertRaises(ValueError):
            select(SimpleNamespace(_r1_attention_implementation='typo'))

    def test_requires_process_compiler_opt_in(self):
        guard = self.namespace['_check_compiler_environment']
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            guard()
        with patch.dict(os.environ, {'DISABLE_MMA_V5': '1'}, clear=True):
            guard()

    @contextmanager
    def precision_module(self):
        calls = []
        module = ModuleType('r1_test_backend.triton_precision')
        module.varlen_prefill = lambda *args, **kwargs: calls.append((args, kwargs)) or 'prefill_output'
        module.paged_mla_decode = lambda *args, **kwargs: calls.append((args, kwargs)) or TensorLayout('latent')
        # Selecting precision must not import either native candidate, including
        # when neither dependency is installed in the runtime environment.
        with patch.dict(sys.modules, {'r1_test_backend.triton_precision': module,
                                      'flash_attn': None, 'flash_attn.cute': None,
                                      'flashinfer': None, 'flashinfer.mla': None}):
            yield calls

    def test_precision_prefill_does_not_import_fa4(self):
        with self.precision_module() as calls:
            result = self.namespace['_attend_prefill'](self.layer, 'q', 'k', 'v', 'cu', 65)
        self.assertEqual(result, 'prefill_output')
        self.assertEqual(calls, [(('q', 'k', 'v', 'cu', 65), {'softmax_scale': 0.125})])

    def test_precision_decode_adapts_only_singleton_query_dimension(self):
        with self.precision_module() as calls:
            result = self.namespace['_attend_decode'](self.layer, TensorLayout('query'), 'cache', 'table', 'lengths', 65)
        self.assertEqual(calls, [(('query.squeeze(1)', 'cache', 'table', 'lengths'), {'softmax_scale': 0.125})])
        self.assertEqual(result, 'latent.unsqueeze(1)')


class TensorLayout:
    def __init__(self, name):
        self.name = name

    def squeeze(self, dim):
        return f'{self.name}.squeeze({dim})'

    def unsqueeze(self, dim):
        return f'{self.name}.unsqueeze({dim})'


if __name__ == '__main__':
    unittest.main()
