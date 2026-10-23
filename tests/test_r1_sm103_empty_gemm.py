"""CPU FakeTensor contracts: empty local GEMM must not skip routed collectives."""
import ast
import builtins
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'batchgen/attention/mla/sm103_backend.py'
MODEL = ROOT / 'batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py'


def extract(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class EmptyGemmContract(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'DISABLE_MMA_V5': '1', 'BATCHGEN_ENABLE_ALL_TO_ALL': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mode = FakeTensorMode()
        self.quantize = Mock(side_effect=AssertionError('empty input must not quantize'))
        self.ns = extract(BACKEND, {'w8a16_gemm', '_check_compiler_environment'},
                          {'torch': torch, 'os': os, '_quantize': self.quantize})
        self.gemm = self.ns['w8a16_gemm']

    def tensor(self, shape, dtype, device='cuda:0'):
        # Backing storage is meta: no CUDA context, data allocation or kernels.
        return FakeTensor(self.mode, torch.empty(shape, device='meta', dtype=dtype), torch.device(device))

    def inputs(self, shape=(0, 7168), n=2048):
        k = shape[-1]
        return (self.tensor((n, k), torch.float8_e4m3fn),
                self.tensor(((n + 127)//128, (k + 127)//128), torch.float32),
                self.tensor(shape, torch.bfloat16))

    def test_empty_2d_and_3d_shape_dtype_device_without_quantizer_or_kernel_import(self):
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(('batchgen.gemm', 'batchgen_kernels')):
                raise AssertionError('empty input must not import a kernel')
            return real_import(name, *args, **kwargs)
        for shape in [(0, 7168), (0, 1, 7168), (2, 0, 7168)]:
            with self.subTest(shape=shape), patch('builtins.__import__', guarded_import):
                result = self.gemm(*self.inputs(shape))
                self.assertEqual(result.shape, (*shape[:-1], 2048))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertEqual(result.device, torch.device('cuda:0'))
                self.assertEqual(result.numel(), 0)
        self.quantize.assert_not_called()

    def test_invalid_empty_metadata_still_rejected(self):
        w, s, x = self.inputs()
        invalid = [
            (w, s, self.tensor((0,), torch.bfloat16)),
            (w, s, self.tensor((0, 1, 1, 7168), torch.bfloat16)),
            (w, s, self.tensor((0, 7168), torch.float32)),
            (self.tensor((2048, 7168), torch.bfloat16), s, x),
            (self.tensor((1, 2048, 7168), torch.float8_e4m3fn), s, x),
            (w, s, self.tensor((0, 2048), torch.bfloat16)),
            (self.tensor((0, 7168), torch.float8_e4m3fn), s, x),
            (self.tensor((2048, 0), torch.float8_e4m3fn), s, self.tensor((0, 0), torch.bfloat16)),
            (w, self.tensor((16, 56), torch.bfloat16), x),
            (w, self.tensor((16, 55), torch.float32), x),
            (w, self.tensor((896,), torch.float32), x),
            (w, s, self.tensor((0, 7168), torch.bfloat16, 'cpu')),
            (self.tensor((2048, 7168), torch.float8_e4m3fn, 'cuda:1'), s, x),
            (w, self.tensor((16, 56), torch.float32, 'cuda:1'), x),
        ]
        for index, args in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.gemm(*args)
        self.quantize.assert_not_called()

    def test_compiler_requirement_not_bypassed_for_empty_input(self):
        with patch.dict(os.environ, {'DISABLE_MMA_V5': '0'}), self.assertRaises(RuntimeError):
            self.gemm(*self.inputs())

    def test_nonempty_retains_quantize_gemm_and_leading_dimensions(self):
        w, s, x = self.inputs((2, 3, 7168))
        quantized = (object(), object())
        self.quantize.side_effect = None
        self.quantize.return_value = quantized
        kernel = ModuleType('batchgen.gemm.w8a8')
        kernel.w8a8_gemm = Mock(return_value=self.tensor((6, 2048), torch.bfloat16))
        with patch.dict(sys.modules, {'batchgen.gemm.w8a8': kernel}):
            out = self.gemm(w, s, x)
        self.assertEqual(self.quantize.call_args.args[0].shape, (6, 7168))
        kernel.w8a8_gemm.assert_called_once_with(*quantized, w, s)
        self.assertEqual(out.shape, (2, 3, 2048))

    def test_real_shared_mlp_empty_forward_preserves_routed_collective_call(self):
        mlp_forward = extract(MODEL, {'deepgemm_forward'}, {'w8a16_gemm': self.gemm},
                              'DeepseekV3MLP')['deepgemm_forward']
        scales = {}
        mlp = SimpleNamespace(act_fn=torch.nn.functional.silu)
        for name, shape in [('up_proj', (2048, 7168)), ('gate_proj', (2048, 7168)),
                            ('down_proj', (7168, 2048))]:
            setattr(mlp, name, SimpleNamespace(weight=self.tensor(shape, torch.float8_e4m3fn)))
            scales[name + '.weight_scale_inv'] = self.tensor(
                tuple((size + 127)//128 for size in shape), torch.float32)
        calls = []
        def collective(x):
            calls.append('routed_collective')
            return x
        def shared(x):
            calls.append('shared_expert')
            return mlp_forward(mlp, x, scales)
        moe = SimpleNamespace(moe_infer_allgather_allreduce_bf16_acc=collective, shared_experts=shared)
        forward = extract(MODEL, {'forward'}, {'os': os}, 'DeepseekV3MoE_Decoding_FP8')['forward']
        out = forward(moe, self.tensor((0, 1, 7168), torch.bfloat16))
        self.assertEqual(out.shape, (0, 1, 7168))
        self.assertEqual(calls, ['routed_collective', 'shared_expert'])
        self.quantize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
