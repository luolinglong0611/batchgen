"""CPU-only contracts for one-kernel R1 native FP32 RoPE dispatch and aliasing."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'batchgen_kernels/triton/fused_rmsnorm_rope.py'
CONTRACT = ROOT / 'batchgen_kernels/triton/r1_native_rope_contract.py'
spec = importlib.util.spec_from_file_location('r1_fused_rope_contract_cpu', CONTRACT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
DEVICE = SimpleNamespace(type='cuda', index=0)


class Tensor:
    def __init__(self, shape, dtype='bf16', ptr=1000, contiguous=True, stride=None):
        self.shape, self.dtype, self.ptr = tuple(shape), dtype, ptr
        self.device, self.contiguous = DEVICE, contiguous
        self.ndim = len(shape)
        strides = []
        for i in range(len(shape)):
            strides.append(math.prod(shape[i + 1:]))
        self.strides = tuple(strides) if stride is None else tuple(stride)
    def data_ptr(self): return self.ptr
    def element_size(self): return {'bf16': 2, 'fp32': 4, 'int32': 4, 'int64': 8}[self.dtype]
    def stride(self, dim=None): return self.strides if dim is None else self.strides[dim]
    def is_contiguous(self): return self.contiguous


TORCH = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', float32='fp32', int32='int32', int64='int64')


def inputs(rows=160):
    return [Tensor((rows, 1, 576), ptr=1000), Tensor((rows, 128, 1, 64), ptr=2_000_000),
            Tensor((16384, 64), ptr=100_000_000), Tensor((16384, 64), ptr=200_000_000),
            Tensor((rows, 1), dtype='int64', ptr=300_000_000), Tensor((512,), ptr=400_000_000),
            512, 64, 1e-6]


@pytest.mark.parametrize('rows', [1, 7, 160, 192])
def test_valid_r1_geometry_has_no_tensor_data_read_or_allocation(rows):
    values = inputs(rows)
    assert contract.validate_native_fp32_inputs(*values, torch=TORCH) == (rows, 128)
    # Leading-row gaps are allowed; neither row overlaps the next one.
    values[0] = Tensor((rows, 1, 576), ptr=1000, contiguous=False, stride=(640, 576, 1))
    assert contract.validate_native_fp32_inputs(*values, torch=TORCH) == (rows, 128)


@pytest.mark.parametrize('source', [0, 2, 3, 4, 5])
def test_inplace_query_must_not_alias_any_readonly_input(source):
    values = inputs()
    values[source].ptr = values[1].ptr
    with pytest.raises(ValueError, match='aliases read-only'):
        contract.validate_native_fp32_inputs(*values, torch=TORCH)


@pytest.mark.parametrize('mutate', [
    lambda x: setattr(x[1], 'contiguous', False),
    lambda x: setattr(x[1], 'shape', (160, 64, 1, 64)),
    lambda x: setattr(x[0], 'strides', (0, 576, 1)),
    lambda x: setattr(x[2], 'contiguous', False),
    lambda x: setattr(x[3], 'shape', (16383, 64)),
    lambda x: setattr(x[4], 'dtype', 'bf16'),
    lambda x: setattr(x[5], 'shape', (511,)),
    lambda x: x.__setitem__(6, 256),
    lambda x: x.__setitem__(8, float('nan')),
])
def test_invalid_metadata_rejected_before_launch(mutate):
    values = inputs()
    mutate(values)
    with pytest.raises(ValueError):
        contract.validate_native_fp32_inputs(*values, torch=TORCH)


def get_function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_native_route_can_return_before_clone_or_legacy_launch():
    node = get_function(ast.parse(SOURCE.read_text()), 'fused_rmsnorm_rope_with_q')
    node.decorator_list = []
    calls = []
    def fused(*args):
        calls.append(args)
        return 'new independent processed KV'
    namespace = {'torch': TORCH, 'r1_fused_native_rope_enabled': lambda: True,
                 'fused_rmsnorm_rope_with_q_r1_fp32': fused, '_LAYOUT_AUDIT_ENABLED': False}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    values = inputs()
    # Tensor deliberately has no clone(): any entry into the legacy path fails.
    result = namespace['fused_rmsnorm_rope_with_q'](*values, output_layout='native_interleaved')
    assert result == 'new independent processed KV' and len(calls) == 1
    assert calls[0][0] is values[0] and calls[0][1] is values[1]


def test_legacy_oracles_and_glm_native_public_wrapper_are_unchanged():
    tree = ast.parse(SOURCE.read_text())
    expected = {
        'fused_rmsnorm_rope_with_q_kernel': 'e4bad2248e4f6a78470d4a243a6360d38b5639969edbd1c1ed7304970668f0af',
        'r1_native_rope_fp32_kernel': 'a1fe7c48ab15a9b575d10510ce31c651aa7a0da9faa3e855c59c4962ea699563',
        'fused_rmsnorm_rope_with_q_native': '928604c71015eeb9d518e27b6edb3c9fd06cfdb27366ebc66979de7f711e1127',
    }
    for name, digest in expected.items():
        assert hashlib.sha256(ast.dump(get_function(tree, name), include_attributes=False).encode()).hexdigest() == digest


def test_latent_reduction_and_fp32_rounding_contract_are_explicit():
    tree = ast.parse(SOURCE.read_text())
    old = get_function(tree, 'fused_rmsnorm_rope_with_q_kernel')
    native = get_function(tree, 'fused_rmsnorm_rope_with_q_native_kernel')
    kv_branch = next(node for node in native.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'lane_idx == 0')
    def rms_slice(body):
        begin = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'sum_sq_fp32')
        end = next(i for i, node in enumerate(body[begin:], begin) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'input_rope_offset')
        return [ast.dump(node, include_attributes=False) for node in body[begin:end]]
    assert rms_slice(old.body) == rms_slice(kv_branch.body)
    assert native.args.args[-1].arg == 'STRICT_FP32_ROPE'
    assert isinstance(native.args.defaults[-1], ast.Constant) and native.args.defaults[-1].value is False
    rotate = ast.unparse(get_function(tree, '_rotate_native_fp32_separate_rn'))
    assert rotate.count('mul.rn.f32') == 4 and 'sub.rn.f32' in rotate and 'add.rn.f32' in rotate
    assert rotate.count('.to(tl.float32)') == 4
    wrapper = ast.unparse(get_function(tree, 'fused_rmsnorm_rope_with_q_r1_fp32'))
    assert 'enable_fp_fusion=True' in wrapper and 'num_warps=4' in wrapper and 'BLOCK_SIZE=64' in wrapper
    assert '.clone(' not in wrapper
