"""CPU-only routing/layout contracts; GPU numerical checks are a separate tool."""
import ast
import hashlib
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'batchgen_kernels/triton/fused_rmsnorm_rope.py'
BACKEND = ROOT / 'batchgen/attention/mla/flashmla_backend.py'


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Tensor:
    """Small indexed-view stand-in to exercise the real Python wrapper."""
    dtype = 'bf16-test-bits'
    def __init__(self, data, shape, indices=None):
        self.data, self.shape = data, tuple(shape)
        self.indices = list(range(len(data))) if indices is None else indices
    def values(self): return [self.data[i] for i in self.indices]
    def is_contiguous(self): return True
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        if len(self.indices) != __import__('math').prod(shape): raise ValueError('wrong view')
        return Tensor(self.data, shape, self.indices)
    reshape = view
    def view_as(self, other): return self.view(*other.shape)
    def contiguous(self): return Tensor(self.values(), self.shape)
    clone = contiguous
    def transpose(self, a, b):
        a, b = a % len(self.shape), b % len(self.shape)
        shape = list(self.shape); shape[a], shape[b] = shape[b], shape[a]
        indices = []
        for coord in itertools.product(*(range(n) for n in shape)):
            old = list(coord); old[a], old[b] = old[b], old[a]
            flat = 0
            for c, n in zip(old, self.shape): flat = flat * n + c
            indices.append(self.indices[flat])
        return Tensor(self.data, shape, indices)
    def copy_(self, other):
        values = other.values()
        for index, value in zip(self.indices, values): self.data[index] = value
        return self
    def __getitem__(self, item):
        assert item[0] is Ellipsis
        begin = item[1].start
        width = self.shape[-1]
        indices = [index for i, index in enumerate(self.indices) if i % width >= begin]
        return Tensor(self.data, (*self.shape[:-1], width - begin), indices)


class KernelStub:
    def __getitem__(self, grid):
        def call(raw, processed, q, *args, **kwargs):
            # Already-rounded split-half Q/K, with a distinct latent block.
            processed.data[:] = [90, 91, 92, 93, 10, 20, 11, 21]
            q.data[:] = [1, 3, 2, 4, 5, 7, 6, 8]
        return call


class NativeFP32KernelStub:
    def __getitem__(self, grid):
        def call(raw, original_q, processed, q, *args, **kwargs):
            assert kwargs['enable_fp_fusion'] is False
            assert original_q.values() == list(range(1, 9)), 'must preserve unrotated Q before legacy kernel'
            assert q.values() == [1, 3, 2, 4, 5, 7, 6, 8], 'legacy kernel should already have returned'
            assert processed.values()[:4] == [90, 91, 92, 93]
            processed.data[4:] = [10, 11, 20, 21]
            q.data[:] = original_q.values()
        return call


class RopeHandoffContract(unittest.TestCase):
    def wrapper(self, receipts=None):
        namespace = {'torch': SimpleNamespace(Tensor=Tensor, empty_like=lambda x: Tensor([0] * len(x.indices), x.shape)),
                     'fused_rmsnorm_rope_with_q_kernel': KernelStub(), '_LAYOUT_AUDIT_ENABLED': receipts is not None,
                     'r1_native_rope_fp32_kernel': NativeFP32KernelStub(),
                     'r1_fused_native_rope_enabled': lambda: False,
                     'triton': SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
                     '_record_layout_call': lambda layout, kv, q: receipts.append((layout, kv.values(), q.values()))}
        return function(KERNEL, 'fused_rmsnorm_rope_with_q', namespace)

    def inputs(self):
        return (Tensor([0] * 8, (1, 1, 8)), Tensor(list(range(1, 9)), (1, 2, 1, 4)), None, None,
                Tensor([2], (1, 1)), None, 4, 4)

    def test_default_preserves_legacy_q_k_and_latent_bits(self):
        inputs = self.inputs()
        output = self.wrapper()(*inputs)
        self.assertEqual(output.values(), [90, 91, 92, 93, 10, 20, 11, 21])
        self.assertEqual(inputs[1].values(), [1, 3, 2, 4, 5, 7, 6, 8])

    def test_explicit_native_recomputes_original_q_k_and_retains_latent(self):
        inputs = self.inputs()
        output = self.wrapper()(*inputs, output_layout='native_interleaved')
        self.assertEqual(output.values(), [90, 91, 92, 93, 10, 11, 20, 21])
        self.assertEqual(inputs[1].values(), list(range(1, 9)))

    def test_invalid_layout_rejected_before_kernel(self):
        with self.assertRaises(ValueError): self.wrapper()(*self.inputs(), output_layout='auto')

    def test_receipts_record_returned_outputs_not_selected_intent(self):
        receipts = []
        call = self.wrapper(receipts)
        self.assertEqual(receipts, [])
        call(*self.inputs(), output_layout='native_interleaved')
        call(*self.inputs())
        self.assertEqual([r[0] for r in receipts], ['native_interleaved', 'split_half'])
        self.assertEqual(receipts[0][1][-4:], [10, 11, 20, 21])

    def test_only_explicit_common_r1_geometry_selects_native(self):
        enabled = [False]
        selector = function(BACKEND, '_r1_paged_rope_output_layout', {'common_attention_enabled': lambda: enabled[0]})
        geometry = SimpleNamespace(num_heads=128, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)
        self.assertEqual(selector(geometry), 'split_half')
        enabled[0] = True
        self.assertEqual(selector(geometry), 'native_interleaved')
        geometry.num_heads = 64
        with self.assertRaises(ValueError): selector(geometry)

    def test_only_paged_fp8_projection_bf16kv_entry_passes_layout_keyword(self):
        tree = ast.parse(BACKEND.read_text())
        callers = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'fused_rmsnorm_rope_with_q':
                        if any(k.arg == 'output_layout' for k in call.keywords): callers.append(node.name)
        self.assertEqual(callers, ['mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv'])

    def test_mixed_layout_corrupts_dot_product_but_consistent_permutation_does_not(self):
        q, k = [1, 2, 3, 4], [5, 6, 7, 8]
        split = lambda x: x[::2] + x[1::2]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        self.assertEqual(dot(q, k), dot(split(q), split(k)))
        self.assertNotEqual(dot(q, k), dot(split(q), k))

    def test_gpu_fixture_required_positions_and_no_execution_on_import(self):
        path = ROOT / 'scripts/r1_reproduction/qualify_r1_rope_handoff.py'
        spec = importlib.util.spec_from_file_location('r1_rope_fixture_cpu', path)
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        self.assertEqual(tool.PROMPTS, (63, 64, 65, 2048, 8192, 16381))
        self.assertEqual(tool.STEPS, 3)
        self.assertEqual(tool.HEADS, 128)
        self.assertEqual(tool.DIAGNOSTIC_VERSION, 'native-fp32-repair-v3')
        self.assertEqual(tool.V1_FAILED_REPORT_SHA256, '4bd9a0d66b19e73984825eb6a8612c2ac3ae08b9a6f6f5b3ce3a90050be3cd72')
        self.assertEqual(tool.V2_FAILED_REPORT_SHA256, '700c5ea5841a0c3b346cc881951da3ea8d57496b9fcc1fde117c8aa903d19568')
        self.assertIn('legacy', tool.SOURCES)
        self.assertEqual(tool.LEGACY_SOURCE_SHA256, 'fc0810fca79e10c4d4faaaee30733d0e78b889bf9f40f046ce7e56e1e8e30f13')

    def test_native_kernel_explicit_fp32_operands_no_latent_store(self):
        tree = ast.parse(KERNEL.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'r1_native_rope_fp32_kernel')
        casts = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'to']
        self.assertEqual(len(casts), 6)
        self.assertTrue(all(ast.unparse(n.args[0]) == 'tl.float32' for n in casts))
        source = ast.unparse(node)
        self.assertIn('base = row * (kv_lora_rank + qk_rope_head_dim) + kv_lora_rank', source)
        self.assertNotIn('norm_weight', source)
        self.assertNotIn('nan_to_num', source)

    def test_diagnostic_keeps_original_component_acceptance_condition(self):
        tree = ast.parse((ROOT / 'scripts/r1_reproduction/qualify_r1_rope_handoff.py').read_text())
        expected = ast.parse('max_ulps(q_new.view(HEADS, ROPE), oracle_q) > 1 or max_ulps(fixed.view(LATENT + ROPE), fresh_history[position]) > 1', mode='eval').body
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(expected)]
        self.assertEqual(len(candidates), 1)
        call = candidates[0].body[0].value
        self.assertEqual(call.args[0].value, 'native Q/K exceeds one BF16 ULP from fresh independent reference')

    def test_legacy_fused_kernel_math_and_launch_body_is_unchanged(self):
        tree = ast.parse(KERNEL.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_rmsnorm_rope_with_q_kernel')
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, 'e4bad2248e4f6a78470d4a243a6360d38b5639969edbd1c1ed7304970668f0af')

    def test_layout_receipt_is_read_only_cpu_metadata(self):
        namespace = {'_LAYOUT_SOURCE': None, '_LAYOUT_CALLS': {}, '_LAYOUT_AUDIT_ENABLED': True,
                     '__file__': str(KERNEL), 'Path': Path, 'hashlib': hashlib}
        record = function(KERNEL, '_record_layout_call', namespace)
        receipt = function(KERNEL, 'layout_receipt', namespace)
        self.assertEqual(receipt()['status'], 'no_wrapper_calls_observed')
        kv, q = self.inputs()[:2]
        record('native_interleaved', kv, q)
        record('native_interleaved', kv, q)
        first = receipt()
        self.assertEqual(first['calls'][0]['count'], 2)
        self.assertEqual(first['calls'][0]['rope_arithmetic'], 'FP32_separate_products_and_sum_v3')
        self.assertEqual(first['source']['sha256'], hashlib.sha256(KERNEL.read_bytes()).hexdigest())
        first['calls'][0]['count'] = 99
        first['source']['sha256'] = 'modified caller copy'
        self.assertEqual(receipt()['calls'][0]['count'], 2)
        self.assertNotEqual(receipt()['source']['sha256'], first['source']['sha256'])


if __name__ == '__main__':
    unittest.main()
