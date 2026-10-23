"""Pure CPU policy/semantic tests; no remote service, model, or native import."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('formal_candidate', ROOT / 'scripts/r1_reproduction/plan_sglang_formal_candidate.py')
diagnostic = load('formal_test_diagnostic', ROOT / 'scripts/r1_reproduction/launch_sglang_diagnostic.py')
HELP = ' '.join(arg for arg in diagnostic.plan(diagnostic.ROOT, 0, {})[2] if arg.startswith('--')) + ' --cuda-graph-max-bs'


def candidate(topology='A', running=16, chunk=2048, fraction='0.85'):
    return plan.build_candidate(diagnostic, HELP, topology, running, chunk, fraction)


def flag(node, name):
    return node['argv'][node['argv'].index(name) + 1]


class CandidatePolicyTest(unittest.TestCase):
    def test_all_48_candidates_use_exactly_16_gpus_with_equal_per_rank_targets(self):
        ids = set()
        for topology in ('A', 'B'):
            for running in plan.RUNNING_CHOICES:
                for chunk in plan.CHUNK_CHOICES:
                    for fraction in plan.FRACTION_CHOICES:
                        with self.subTest(topology=topology, running=running, chunk=chunk, fraction=fraction):
                            record = candidate(topology, running, chunk, fraction)
                            ids.add(record['candidate_id'])
                            self.assertEqual(sum(len(node['local_gpu_ordinals']) for node in record['nodes']), 16)
                            self.assertEqual(record['cluster_running_requests_target'], 16 * running)
                            self.assertEqual(record['cluster_prefill_chunk_tokens_target'], 16 * chunk)
                            for node in record['nodes']:
                                dp = int(flag(node, '--dp-size'))
                                self.assertEqual(int(flag(node, '--max-running-requests')) // dp, running)
                                self.assertEqual(int(flag(node, '--chunked-prefill-size')) // dp, chunk)
                                self.assertEqual(flag(node, '--mem-fraction-static'), fraction)
                                self.assertEqual(flag(node, '--ep-size'), '1')
                                self.assertNotIn('--max-total-tokens', node['argv'])
        self.assertEqual(len(ids), 48)

    def test_a_has_one_service_and_one_generation_endpoint(self):
        record = candidate('A')
        self.assertEqual(record['independent_model_services'], 1)
        self.assertEqual(record['client_contract']['endpoint_urls'], ['http://10.66.9.163:30000/v1'])
        for i, node in enumerate(record['nodes']):
            self.assertEqual(flag(node, '--tp-size'), '16')
            self.assertEqual(flag(node, '--nnodes'), '2')
            self.assertEqual(flag(node, '--node-rank'), str(i))
            self.assertEqual(flag(node, '--dist-init-addr'), '10.66.9.163:33155')
            self.assertEqual(node['service_index'], 0)

    def test_b_has_local_rendezvous_and_physical_node_routing_receipt(self):
        record = candidate('B')
        self.assertEqual(record['independent_model_services'], 2)
        self.assertEqual(record['client_contract']['formal_requests_per_service'], 3000)
        for i, node in enumerate(record['nodes']):
            self.assertEqual(flag(node, '--tp-size'), '8')
            self.assertEqual(flag(node, '--nnodes'), '1')
            self.assertEqual(flag(node, '--node-rank'), '0')
            self.assertEqual(flag(node, '--dist-init-addr'), f'{plan.CONTAINER_IPS[i]}:33155')
            self.assertEqual(node['service_index'], i)
            self.assertEqual(node['routing_compat']['receipt_path'],
                             diagnostic.routing_compat_plan(diagnostic.ROOT, i)['receipt_path'])
            self.assertIn(33168, node['reserved_known_ports_to_recheck'])
            self.assertTrue(set(range(33156, 33161)).issubset(node['reserved_known_ports_to_recheck']))

    def test_preserves_original_weights_backends_and_bf16_runtime_requirement(self):
        for topology in ('A', 'B'):
            record = candidate(topology)
            self.assertEqual(record['kv_policy']['required_effective_dtype_every_rank'], 'torch.bfloat16')
            for node in record['nodes']:
                self.assertEqual(flag(node, '--model-path'), str(diagnostic.WEIGHTS))
                self.assertEqual(flag(node, '--dtype'), 'bfloat16')
                self.assertEqual(flag(node, '--quantization'), 'fp8')
                self.assertEqual(flag(node, '--kv-cache-dtype'), 'auto')
                self.assertEqual(flag(node, '--attention-backend'), 'trtllm_mla')
                self.assertEqual(flag(node, '--moe-runner-backend'), 'flashinfer_trtllm')
                self.assertEqual(flag(node, '--page-size'), '64')
                self.assertIn('--disable-cuda-graph', node['argv'])
                self.assertIn('--disable-radix-cache', node['argv'])
                self.assertFalse(any('speculative' in arg for arg in node['argv']))
                self.assertNotIn('--cpu-offload-gb', node['argv'])
                for key in ('SGLANG_ENABLE_JIT_DEEPGEMM', 'SGLANG_ENABLE_FLASHINFER_FP8_GEMM',
                            'SGLANG_SUPPORT_CUTLASS_BLOCK_FP8', 'SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN'):
                    self.assertEqual(node['environment'][key], '0')

    def test_explicit_graph128_only_changes_graph_flags_against_equal_concurrency_eager(self):
        eager = candidate('A', 128, 2048, '0.85')
        graph = plan.build_candidate(diagnostic, HELP, 'A', 128, 2048, '0.85', 'graph', 128)
        self.assertEqual(graph['candidate_id'], 'A-c128-m085-k2048-graph-bs128')
        self.assertEqual(graph['cuda_graph_policy']['capture_batch_sizes_expected'], [1, 2, 4, 8, 12] + list(range(16, 129, 8)))
        self.assertEqual(graph['cuda_graph_policy']['max_gathered_moe_tokens_per_service_expected'], 2048)
        self.assertFalse(graph['cuda_graph_policy']['actual_capture_replay_qualified'])
        self.assertFalse(graph['performance_qualified'])
        for old, new in zip(eager['nodes'], graph['nodes']):
            expected = old['argv'].copy(); expected.remove('--disable-cuda-graph')
            self.assertEqual(new['argv'], expected + ['--cuda-graph-max-bs', '128'])
            self.assertEqual(new['environment'], old['environment'])
        self.assertEqual(graph['client_contract'], eager['client_contract'])

    def test_graph_limit_mode_and_default_bucket_constraints_fail_closed(self):
        for mode, maximum in [('wrong', 16), ('graph', None), ('graph', True), ('graph', 16.0),
                              ('graph', 0), ('graph', 3), ('graph', 32), ('eager', 16)]:
            with self.subTest(mode=mode, maximum=maximum), self.assertRaises(ValueError):
                plan.build_candidate(diagnostic, HELP, 'A', 16, 2048, '0.85', mode, maximum)

    def test_clean_environment_cannot_capture_credentials_or_backend_override(self):
        with mock.patch.dict('os.environ', {'TEST_API_KEY': 'do-not-copy',
                                         'SGLANG_CUTLASS_MOE': '1', 'NCCL_RAS_ENABLE': '1'}):
            record = candidate()
        self.assertNotIn('do-not-copy', json.dumps(record))
        for node in record['nodes']:
            self.assertNotIn('SGLANG_CUTLASS_MOE', node['environment'])
            self.assertEqual(node['environment']['NCCL_RAS_ENABLE'], '0')
            rendered = shlex.split(node['review_only_shell_command'])
            self.assertEqual(rendered[:2], ['env', '-i'])
            self.assertEqual(rendered[-len(node['argv']):], node['argv'])

    def test_plan_never_claims_admission_or_actual_capacity(self):
        record = candidate('B', 128, 8192, '0.90')
        for key in ('launch_authorized', 'model_quality_qualified', 'resource_capacity_qualified', 'performance_qualified'):
            self.assertIs(record[key], False)
        self.assertIsNone(record['fixed_runtime']['nccl_binary_version_same_as_batchgen'])
        self.assertFalse(record['client_contract']['combine_replica_bct_by_adding_or_averaging'])
        for node in record['nodes']:
            self.assertIsNone(node['actual_running_request_limit'])
            self.assertIsNone(node['actual_kv_tokens_per_rank'])

    def test_invalid_or_unreviewed_choices_fail_closed(self):
        for args in [('C', 16, 2048, '0.85'), ('A', True, 2048, '0.85'),
                     ('A', 1024, 2048, '0.85'), ('B', 16, 1024, '0.85'),
                     ('B', 16, 2048, 0.85), ('B', 16, 2048, '0.95')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                candidate(*args)

    def test_removed_cli_flag_is_rejected_by_frozen_help(self):
        with self.assertRaisesRegex(RuntimeError, 'Installed --help'):
            plan.build_candidate(diagnostic, HELP.replace('--dp-size', ''), 'A', 16, 2048, '0.85')

    def test_plan_does_not_execute_or_access_network(self):
        with contextlib.ExitStack() as stack:
            for owner, name in ((subprocess, 'run'), (subprocess, 'Popen'),
                                (socket, 'create_connection'), (urllib.request, 'urlopen')):
                stack.enter_context(mock.patch.object(owner, name, side_effect=AssertionError(name)))
            candidate('A')
            candidate('B')


@unittest.skipUnless(plan.DEFAULT_AUDIT.is_dir(), 'Fixed private installed-source export unavailable')
class FixedSourceSemanticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic, cls.help_text = plan.load_inputs()
        cls.server_tree = ast.parse((plan.DEFAULT_AUDIT / 'sglang/srt/server_args.py').read_text())

    def test_cpu_only_cli_is_deterministic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'plan.json'
            args = ['--topology', 'B', '--per-dp-running', '64', '--per-dp-chunk', '8192',
                    '--memory-fraction', '0.90', '--execution-mode', 'eager']
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                plan.main(args)
            plan.main(args + ['--output', str(output)])
            self.assertEqual(output.read_text(), stdout.getvalue())
            with self.assertRaises(FileExistsError):
                plan.main(args + ['--output', str(output)])

    def test_real_pinned_cli_help_accepts_every_candidate(self):
        for topology in ('A', 'B'):
            plan.build_candidate(self.diagnostic, self.help_text, topology, 128, 8192, '0.90')

    def test_actual_dp_method_transforms_chunk_and_schedule(self):
        method = next(n for n in ast.walk(self.server_tree)
                      if isinstance(n, ast.FunctionDef) and n.name == '_handle_data_parallelism')
        namespace = {'logger': SimpleNamespace(warning=lambda *args: None)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned dp method>', 'exec'), namespace)
        for topology in ('A', 'B'):
            for node in candidate(topology, 64, 8192, '0.90')['nodes']:
                args = SimpleNamespace(dp_size=int(flag(node, '--dp-size')), tp_size=int(flag(node, '--tp-size')),
                                       enable_dp_attention=True, enable_dp_lm_head=False,
                                       chunked_prefill_size=int(flag(node, '--chunked-prefill-size')),
                                       schedule_conservativeness=1.0)
                namespace['_handle_data_parallelism'](args)
                self.assertEqual(args.chunked_prefill_size, 8192)
                self.assertEqual(args.schedule_conservativeness, 0.3)

    def test_actual_non_spec_graph_bucket_method_matches_graph128_policy(self):
        method = next(n for n in ast.walk(self.server_tree)
                      if isinstance(n, ast.FunctionDef) and n.name == '_generate_cuda_graph_batch_sizes')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned graph bucket method>', 'exec'), namespace)
        args = SimpleNamespace(disable_cuda_graph_padding=False, speculative_algorithm=None, cuda_graph_max_bs=128)
        graph = plan.build_candidate(self.diagnostic, self.help_text, 'A', 128, 2048, '0.85', 'graph', 128)
        self.assertEqual(namespace[method.name](args), graph['cuda_graph_policy']['capture_batch_sizes_expected'])

    def test_actual_worker_request_limit_is_divided_and_pool_capped(self):
        tree = ast.parse((plan.DEFAULT_AUDIT / 'sglang/srt/managers/tp_worker.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and ast.unparse(n.targets[0]) == 'self.max_running_requests')
        expression = compile(ast.Expression(assignment.value), '<pinned worker cap>', 'eval')
        for dp in (8, 16):
            args = SimpleNamespace(max_running_requests=64 * dp, dp_size=dp, enable_dp_attention=True)
            for pool, expected in ((4096, 64), (32, 32)):
                worker = SimpleNamespace(model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(size=pool)))
                self.assertEqual(eval(expression, {'server_args': args, 'self': worker}), expected)

    def test_trtllm_mla_rejects_explicit_bf16_but_accepts_auto(self):
        branch = next(n for n in ast.walk(self.server_tree) if isinstance(n, ast.If)
                      and 'self.attention_backend' in ast.unparse(n.test)
                      and ast.unparse(n.test).count('trtllm_mla') == 2)
        code = compile(ast.Module(body=[branch], type_ignores=[]), '<pinned backend check>', 'exec')
        for dtype in ('bf16', 'bfloat16', 'auto'):
            args = SimpleNamespace(attention_backend='trtllm_mla', decode_attention_backend=None,
                                   page_size=64, kv_cache_dtype=dtype)
            env = {'self': args, 'is_blackwell_supported': lambda: True}
            if dtype == 'auto':
                exec(code, env)
            else:
                with self.assertRaisesRegex(ValueError, 'only supports kv-cache-dtype'):
                    exec(code, env)

    def test_mutated_source_fails_before_plan_import(self):
        with mock.patch.object(plan, 'sha', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'Pinned local source changed'):
                plan.load_inputs()


if __name__ == '__main__':
    unittest.main()
