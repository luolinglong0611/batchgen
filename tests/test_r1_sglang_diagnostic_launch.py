"""CPU launch-policy tests; no model, native libraries, or remote operations."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sg_plan', ROOT / 'scripts/r1_reproduction/launch_sglang_diagnostic.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class SGLangDiagnosticPlanTest(unittest.TestCase):
    def test_dualnode_same_original_weights_and_precision(self):
        for rank in (0, 1):
            env, unset, cmd = launch.plan(launch.ROOT, rank, {})
            get = lambda key: cmd[cmd.index(key) + 1]
            self.assertEqual((get('--tp-size'), get('--dp-size'), get('--ep-size')), ('16', '16', '1'))
            self.assertEqual((get('--nnodes'), get('--node-rank')), ('2', str(rank)))
            self.assertEqual(get('--model-path'), str(launch.WEIGHTS))
            self.assertEqual(get('--tokenizer-path'), str(launch.WEIGHTS))
            self.assertEqual((get('--dtype'), get('--quantization'), get('--kv-cache-dtype')), ('bfloat16', 'fp8', 'auto'))
            self.assertIn('--disable-cuda-graph', cmd)
            self.assertIn('--disable-radix-cache', cmd)
            self.assertFalse(any('speculative' in arg for arg in cmd))
            self.assertEqual(get('--attention-backend'), 'trtllm_mla')
            self.assertEqual(get('--moe-runner-backend'), 'flashinfer_trtllm')
            self.assertEqual(get('--moe-a2a-backend'), 'none')
            self.assertEqual(env['SGLANG_ENABLE_JIT_DEEPGEMM'], '0')
            self.assertEqual(env['SGLANG_ENABLE_FLASHINFER_FP8_GEMM'], '0')
            self.assertEqual(env['SGLANG_SUPPORT_CUTLASS_BLOCK_FP8'], '0')

    def test_inherited_batchgen_compiler_and_backend_overrides_removed(self):
        inherited = {'CPATH': '/bad/include', 'CPLUS_INCLUDE_PATH': '/bad/cxx', 'LIBRARY_PATH': '/bad/lib', 'DISABLE_MMA_V5': '1', 'TRITON_PTXAS_PATH': '/cuda13/ptxas',
                     'TRITON_PTXAS_BLACKWELL_PATH': '/bad/ptxas',
                     'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '1', 'NCCL_IB_DISABLE': '1',
                     'BATCHGEN_ENABLE_ALL_TO_ALL': '1', 'SGL_USE_DEEPGEMM_BMM': '1'}
        env, unset, _ = launch.plan(launch.ROOT, 0, inherited)
        effective = {k: v for k, v in inherited.items() if k not in unset}
        effective.update(env)
        for key in ('DISABLE_MMA_V5', 'TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH',
                    'BATCHGEN_ENABLE_ALL_TO_ALL', 'NCCL_IB_DISABLE', 'SGL_USE_DEEPGEMM_BMM',
                    'CPLUS_INCLUDE_PATH'):
            self.assertNotIn(key, effective)
        self.assertEqual(effective['NCCL_IB_GID_INDEX'], '3')
        self.assertEqual(effective['GLOO_SOCKET_IFNAME'], 'eth0')
        self.assertIn('private-cxx-runtime/lib', effective['LD_LIBRARY_PATH'])
        self.assertEqual(effective['LIBRARY_PATH'], str(launch.ROOT / 'sglang-baseline-v8/private-cuda-link/lib'))
        self.assertEqual(effective['CPATH'], str(launch.ROOT / 'sglang-baseline-v8/baseline-env/lib/python3.11/site-packages/nvidia/cu13/include'))

    def test_loading_stays_mmap_and_shard_iterator_is_serial(self):
        _, _, cmd = launch.plan(launch.ROOT, 0, {})
        self.assertNotIn('--weight-loader-disable-mmap', cmd)
        self.assertIn('{"enable_multithread_load":false,"num_threads":1}', cmd)
        self.assertNotIn('--cpu-offload-gb', cmd)
        self.assertNotIn('--trust-remote-code', cmd)
        self.assertEqual(cmd[0], str(launch.ROOT / 'sglang-baseline-v8/baseline-env/bin/python'))

    def test_installed_help_must_accept_every_flag(self):
        _, _, cmd = launch.plan(launch.ROOT, 0, {})
        help_text = ' '.join(arg for arg in cmd if arg.startswith('--'))
        launch.check_help(cmd, help_text)
        with self.assertRaisesRegex(RuntimeError, 'Installed --help'):
            launch.check_help(cmd, help_text.replace('--moe-runner-backend', ''))

    def test_plan_does_not_spawn_or_read_remote_files(self):
        with mock.patch.object(launch.subprocess, 'run', side_effect=AssertionError('must not spawn')):
            with mock.patch.object(launch.Path, 'read_bytes', side_effect=AssertionError('must not read')):
                launch.plan(launch.ROOT, 0, {})


if __name__ == '__main__':
    unittest.main()
