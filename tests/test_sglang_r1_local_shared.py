"""No Torch/network/GPU imports: source-bound layout and operator-hook tests."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load('r1_sg_runtime_test', 'scripts/r1_reproduction/sglang_r1_alignment_runtime.py')
P = load('r1_sg_prepare_test', 'scripts/r1_reproduction/prepare_sglang_r1_local_shared.py')


FIXTURE = '''class DeepseekV2MoE(nn.Module):
    def __init__(self, config, quant_config):
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.shared_experts = DeepseekV2MLP(
            **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
            )
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def forward(self, hidden_states, forward_batch=None, should_allreduce_fusion=False,
                use_reduce_scatter=False, gemm_output_zero_allocator=None):
        if not self._enable_a2a_moe:
            raise AssertionError("old dual-stream selection")

    def _forward_shared_experts(self, hidden_states, gemm_output_zero_allocator=None):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(hidden_states)
        return None

class DeepseekV2DecoderLayer:
    def __init__(self):
        self.layer_communicator = LayerCommunicator(
            allow_reduce_scatter=True,
        )

    def forward(self, hidden_states, residual, forward_batch):
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        should_allreduce_fusion = self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(forward_batch)
        if isinstance(self.mlp, DeepseekV2MLP):
            pass
        hidden_states = self.mlp(hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter, None)
        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)
        return hidden_states, residual

class DeepseekV2ForCausalLM:
    def __init__(self, config, quant_config):
        self.determine_num_fused_shared_experts()

class DeepseekV2Model:
    def forward(self, forward_batch):
        total_num_layers = self.end_layer - self.start_layer
'''


def candidate_fixture():
    data = FIXTURE.encode()
    with patch.object(P, 'STOCK_SHA', hashlib.sha256(data).hexdigest()):
        return P.patched_source(data).decode()


def extract_method(text, class_name, method_name, namespace):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<prepared-method>', 'exec'), namespace)
    return namespace[method_name]


class Tensor:
    """Small FP64 CPU matrix stand-in; no accelerator dependency."""
    dtype = 'float64'
    device = 'cpu-test'

    def __init__(self, rows, width=3):
        self.rows = [list(row) for row in rows]
        self.shape = (len(rows), len(rows[0]) if rows else width)

    def __add__(self, other):
        return Tensor([[a + b for a, b in zip(x, y)] for x, y in zip(self.rows, other.rows)], self.shape[1])


def shared_mlp(x):
    # Independent nonlinear SwiGLU reference, with nontrivial full weights.
    gate = ((0.5, -0.25, 0.75), (-0.25, 0.5, 0.25))
    up = ((-0.5, 0.5, 0.25), (0.75, 0.25, -0.5))
    down = ((0.5, -0.5), (0.25, 0.75), (-0.75, 0.25))
    output = []
    for row in x.rows:
        g = [math.fsum(a * b for a, b in zip(row, w)) for w in gate]
        u = [math.fsum(a * b for a, b in zip(row, w)) for w in up]
        activated = [a / (1 + math.exp(-a)) * b for a, b in zip(g, u)]
        output.append([math.fsum(a * b for a, b in zip(activated, w)) for w in down])
    return Tensor(output, 3)


def r1_config():
    return SimpleNamespace(model_type='deepseek_v3', architectures=['DeepseekV3ForCausalLM'],
                           hidden_size=7168, n_routed_experts=256, n_shared_experts=1,
                           num_experts_per_tok=8, moe_intermediate_size=2048,
                           first_k_dense_replace=3, hidden_act='silu')


def quant_config():
    return SimpleNamespace(get_name=lambda: 'fp8', weight_block_size=[128, 128],
                           is_checkpoint_fp8_serialized=True)


class LocalSharedTests(unittest.TestCase):
    def test_prepare_writes_only_an_exclusive_candidate_and_preserves_source(self):
        data = FIXTURE.encode()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'stock.py'
            source.write_bytes(data)
            output = Path(directory) / 'candidate'
            with patch.object(P, 'STOCK_SHA', hashlib.sha256(data).hexdigest()), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(P.main(['--source', str(source)]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(P.main(['--source', str(source), '--execute', '--output-dir', str(output)]), 0)
                with self.assertRaises(FileExistsError):
                    P.main(['--source', str(source), '--execute', '--output-dir', str(output)])
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual((output / 'deepseek_v2.py.stock').read_bytes(), data)
            self.assertEqual((output / 'deepseek_v2.py').read_text(), candidate_fixture())
            self.assertIn('prepared_NOT_installed_NOT_GPU_qualified', (output / 'receipt.json').read_text())

    def test_unknown_source_and_duplicate_anchor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source SHA mismatch'):
            P.patched_source(FIXTURE.encode())
        changed = FIXTURE.replace('        self.determine_num_fused_shared_experts()\n',
                                  '        self.determine_num_fused_shared_experts()\n' * 2).encode()
        with patch.object(P, 'STOCK_SHA', hashlib.sha256(changed).hexdigest()), \
             self.assertRaisesRegex(ValueError, 'anchor mismatch'):
            P.patched_source(changed)

    def test_prepared_model_suppresses_old_shared_and_dual_stream_paths(self):
        text = candidate_fixture()
        self.assertIn('if self._r1_local_shared_tp1 or get_moe_a2a_backend()', text)
        self.assertIn('allow_reduce_scatter=not getattr(self.mlp', text)
        calls = []
        moe = SimpleNamespace(_r1_local_shared_tp1=True,
                              shared_experts=lambda x: calls.append('forbidden shared'))
        shared = extract_method(text, 'DeepseekV2MoE', '_forward_shared_experts', {})
        self.assertIsNone(shared(moe, Tensor([[1, 2, 3]])))
        self.assertEqual(calls, [])
        moe.forward_normal = lambda *args: ('routed-only', args)
        forward = extract_method(text, 'DeepseekV2MoE', 'forward', {})
        self.assertEqual(forward(moe, 'input')[0], 'routed-only')

    def test_local_shared_after_reduction_matches_oracle_with_empty_unequal_and_padded_ranks(self):
        # Exercise the actual generated decoder method and actual runtime helpers.
        for counts in ([2] * 16, [0, 3, 1, 0] + [0] * 12, [0] * 16):
            for padding in ('max', 'sum'):
                with self.subTest(counts=counts, padding=padding):
                    per_rank = [Tensor([[0.125 * (r + 1), -0.25 * (t + 1), 0.5]
                                        for t in range(n)]) for r, n in enumerate(counts)]
                    block = max(counts)
                    packed_rows = []
                    starts = []
                    for x in per_rank:
                        starts.append(len(packed_rows))
                        packed_rows.extend(x.rows)
                        if padding == 'max':
                            # Poison padding; SG scatter must never copy these rows as valid.
                            packed_rows.extend([[math.nan] * 3 for _ in range(block - x.shape[0])])
                    global_x = Tensor(packed_rows)
                    # Independent routed partials with skew: only rank 15 contributes.
                    routed = Tensor([[v * 1.5 for v in row] for row in global_x.rows])
                    for rank, count in enumerate(counts):
                        events = []
                        batch = SimpleNamespace(start=starts[rank], count=count)
                        padded_size = block if padding == 'max' else count
                        residual = Tensor([[7.0, -2.0, 1.0] for _ in range(padded_size)])
                        original_residual = [row[:] for row in residual.rows]

                        def scatter(dst, src, info):
                            events.append('scatter-input')
                            dst.rows = [[0.0] * 3 for _ in range(dst.shape[0])]
                            dst.rows[:info.count] = [row[:] for row in src.rows[info.start:info.start + info.count]]

                        def run_shared(x):
                            events.append('shared')
                            self.assertEqual(x.shape[0], padded_size)
                            return shared_mlp(x)

                        class Moe:
                            _r1_local_shared_tp1 = True
                            shared_experts = staticmethod(run_shared)

                            def __call__(self, *args):
                                events.append('routed-SUM')
                                return routed

                        moe = Moe()
                        class Shared:
                            tp_size = 1

                            def __call__(self, x):
                                return run_shared(x)

                        moe.shared_experts = Shared()

                        def postprocess(value, res, info):
                            events.append('scatter-output')
                            result = Tensor([[0.0] * 3 for _ in range(padded_size)])
                            result.rows[:count] = [row[:] for row in value.rows[info.start:info.start + count]]
                            return result, res

                        comm = SimpleNamespace(
                            prepare_mlp=lambda *_: (global_x, residual),
                            should_fuse_mlp_allreduce_with_next_layer=lambda _: False,
                            should_use_reduce_scatter=lambda _: False,
                            postprocess_layer=postprocess)
                        layer = SimpleNamespace(mlp=moe, layer_communicator=comm)
                        ns = {'_r1_capture_local_shared_input': R.capture_local_shared_input,
                              '_r1_finish_local_shared': R.finish_local_shared,
                              'DeepseekV2MLP': type('Dense', (), {})}
                        forward = extract_method(candidate_fixture(), 'DeepseekV2DecoderLayer', 'forward', ns)
                        fake_torch = SimpleNamespace(contiguous_format='contiguous',
                            empty_like=lambda x, **_: Tensor([[999] * 3 for _ in range(x.shape[0])]))

                        def import_module(name):
                            return fake_torch if name == 'torch' else SimpleNamespace(dp_scatter=scatter)

                        with patch.object(R.importlib, 'import_module', side_effect=import_module):
                            output, actual_residual = forward(layer, per_rank[rank], residual, batch)
                        expected_shared = shared_mlp(per_rank[rank])
                        for row_idx in range(count):
                            for col in range(3):
                                expected = 1.5 * per_rank[rank].rows[row_idx][col] + expected_shared.rows[row_idx][col]
                                self.assertAlmostEqual(output.rows[row_idx][col], expected, places=13)
                        self.assertEqual(output.rows[count:], [[0.0] * 3 for _ in range(padded_size - count)])
                        self.assertIs(actual_residual, residual)
                        self.assertEqual(residual.rows, original_residual)
                        self.assertEqual(events, ['scatter-input', 'routed-SUM', 'scatter-output', 'shared'])

    def test_reduction_fusion_is_rejected_in_prepared_decoder(self):
        ns = {'_r1_capture_local_shared_input': lambda *_: 'local',
              '_r1_finish_local_shared': R.finish_local_shared,
              'DeepseekV2MLP': type('Dense', (), {})}
        forward = extract_method(candidate_fixture(), 'DeepseekV2DecoderLayer', 'forward', ns)
        for fusion, rs in ((True, False), (False, True)):
            comm = SimpleNamespace(prepare_mlp=lambda *_: ('global', 'residual'),
                                   should_fuse_mlp_allreduce_with_next_layer=lambda _: fusion,
                                   should_use_reduce_scatter=lambda _: rs)
            with self.assertRaisesRegex(RuntimeError, 'routed all-reduce'):
                forward(SimpleNamespace(mlp=object(), layer_communicator=comm), None, None, None)

    def test_duplicate_full_shared_before_sum_is_not_the_reference(self):
        x = Tensor([[0.25, -0.5, 0.75]])
        shared = shared_mlp(x).rows[0]
        correct = [0.5 + v for v in shared]
        incorrect = [0.5 + 16 * v for v in shared]
        self.assertTrue(any(abs(a - b) > 0.01 for a, b in zip(correct, incorrect)))

    def test_capture_rejects_unsharded_claim_without_actual_tp1(self):
        moe = SimpleNamespace(_r1_local_shared_tp1=True, shared_experts=SimpleNamespace(tp_size=16))
        with self.assertRaisesRegex(RuntimeError, 'not constructed at TP1'):
            R.capture_local_shared_input(moe, Tensor([[1, 2, 3]]), Tensor([[0, 0, 0]]), None)


class CommonLinearHookTests(unittest.TestCase):
    def test_common_linear_preserves_original_weight_and_scale_identity(self):
        class ViewTensor:
            def __init__(self, shape):
                self.shape = shape

            def reshape(self, *shape):
                return self

        activation = ViewTensor((2, 3, 128))
        weight = ViewTensor((256, 128))
        original_scale = object()
        output = ViewTensor((6, 256))
        calls = []

        def common(w, scale, x):
            calls.append((w, scale, x))
            return output

        module = SimpleNamespace(w8a16_gemm=common)
        with patch.object(R.importlib, 'import_module', return_value=module):
            self.assertIs(R.common_block_fp8_linear(activation, weight, [128, 128], original_scale), output)
        self.assertEqual(calls, [(weight, original_scale, activation)])
        with self.assertRaisesRegex(ValueError, 'without bias'):
            R.common_block_fp8_linear(activation, weight, [128, 128], original_scale, bias=object())

    def test_shared_contract_refuses_overlap_or_ue8m0(self):
        args = SimpleNamespace(disable_shared_experts_fusion=True, disable_cuda_graph=True,
                               disable_overlap_schedule=True,
                               enable_dp_lm_head=True, moe_dense_tp_size=1, pp_size=1,
                               enable_two_batch_overlap=False, enable_single_batch_overlap=False,
                               enable_piecewise_cuda_graph=False,
                               enable_flashinfer_allreduce_fusion=False, speculative_algorithm=None)
        kwargs = dict(tp_size=16, ep_size=16, attention_tp_size=1, dp_attention=True,
                      a2a_none=True, runner_triton=True, deepgemm_enabled=False)
        R.require_shared_contract(r1_config(), quant_config(), args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'DeepGEMM off'):
            R.require_shared_contract(r1_config(), quant_config(), args, **dict(kwargs, deepgemm_enabled=True))
        args.enable_single_batch_overlap = True
        with self.assertRaisesRegex(RuntimeError, 'enable_single_batch_overlap'):
            R.require_shared_contract(r1_config(), quant_config(), args, **kwargs)

    def test_fp8_hook_binds_same_entry_without_changing_weights_or_quantizing_bf16_weights(self):
        class Method:
            def __init__(self, quant_config):
                self.block_quant = quant_config.weight_block_size is not None
                self.use_marlin = False
                self.weight = object()
                self.w8a8_block_fp8_linear = 'stock'

        common = SimpleNamespace(w8a16_gemm=lambda *_: None, __file__=R.__file__)
        modules = {'sglang.srt.layers.deep_gemm_wrapper': SimpleNamespace(ENABLE_JIT_DEEPGEMM=False, DEEPGEMM_SCALE_UE8M0=False),
                   'batchgen.gemm.r1_sm103_compat': common,
                   'sglang.srt.layers.quantization.fp8': SimpleNamespace(Fp8LinearMethod=Method)}
        env = {'BATCHGEN_R1_SM103_FP8_COMPAT': '1', 'DISABLE_MMA_V5': '1',
               'SGLANG_ENABLE_JIT_DEEPGEMM': '0', 'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '0',
               'SGLANG_R1_COMMON_FI_FA2': '0'}
        with patch.dict(os.environ, env), patch.object(R.importlib.metadata, 'version', return_value='0.5.5.post3'), \
             patch.object(R.importlib, 'import_module', side_effect=modules.__getitem__), \
             patch.object(R, '_LINEAR_INSTALLATION', None):
            receipt = R.install_fp8_linear_backend(r1_config(), quant_config())
            instance = Method(quant_config=quant_config())
            self.assertIs(instance.w8a8_block_fp8_linear, R.common_block_fp8_linear)
            self.assertFalse(receipt['formal_test_permitted'])
            self.assertEqual(R.install_fp8_linear_backend(r1_config(), quant_config()), receipt)
            unquantized = SimpleNamespace(weight_block_size=None)
            self.assertEqual(Method(unquantized).w8a8_block_fp8_linear, 'stock')
            bad = quant_config()
            bad.is_checkpoint_fp8_serialized = False
            with self.assertRaisesRegex(ValueError, 'serialized FP8'):
                Method(bad)

    def test_missing_explicit_scale_environment_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'requires explicit'):
            R.install_fp8_linear_backend(r1_config(), quant_config())


if __name__ == '__main__':
    unittest.main()
