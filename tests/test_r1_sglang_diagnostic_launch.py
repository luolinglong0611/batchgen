"""CPU launch-policy tests; no model, native libraries, or remote operations."""
import importlib.util
import difflib
import json
import tempfile
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
            self.assertEqual(env['BATCHGEN_SGLANG_R1_ROUTING_COMPAT'], '1')
            self.assertEqual(env['SGLANG_ENABLE_JIT_DEEPGEMM'], '0')
            self.assertEqual(env['SGLANG_ENABLE_FLASHINFER_FP8_GEMM'], '0')
            self.assertEqual(env['SGLANG_SUPPORT_CUTLASS_BLOCK_FP8'], '0')
            self.assertEqual(env['NCCL_RAS_ENABLE'], '0')

    def test_inherited_batchgen_compiler_and_backend_overrides_removed(self):
        inherited = {'CPATH': '/bad/include', 'CPLUS_INCLUDE_PATH': '/bad/cxx', 'LIBRARY_PATH': '/bad/lib', 'DISABLE_MMA_V5': '1', 'TRITON_PTXAS_PATH': '/cuda13/ptxas',
                     'TRITON_PTXAS_BLACKWELL_PATH': '/bad/ptxas',
                     'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '1', 'NCCL_IB_DISABLE': '1',
                     'BATCHGEN_SGLANG_R1_ROUTING_COMPAT': '0', 'BATCHGEN_ENABLE_ALL_TO_ALL': '1', 'SGL_USE_DEEPGEMM_BMM': '1'}
        env, unset, _ = launch.plan(launch.ROOT, 0, inherited)
        effective = {k: v for k, v in inherited.items() if k not in unset}
        effective.update(env)
        for key in ('DISABLE_MMA_V5', 'TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH',
                    'BATCHGEN_ENABLE_ALL_TO_ALL', 'NCCL_IB_DISABLE', 'SGL_USE_DEEPGEMM_BMM',
                    'CPLUS_INCLUDE_PATH'):
            self.assertNotIn(key, effective)
        self.assertEqual(effective['BATCHGEN_SGLANG_R1_ROUTING_COMPAT'], '1')
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
                launch.routing_compat_plan(launch.ROOT, 0)


class RoutingCompatibilityEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.site = self.root / 'site'
        self.source = self.site / launch.ROUTING_SOURCE
        self.source.parent.mkdir(parents=True)
        self.source.write_text('patched = True\n')
        self.scripts = self.root / 'scripts'
        self.scripts.mkdir()
        self.tool = self.scripts / 'patch_sglang_r1_routing.py'
        self.tool.write_text('# isolated test patch tool\n')
        self.manifest = self.scripts / 'r1_model_manifest.json'
        self.manifest.write_text('[]\n')
        self.weights = self.root / 'weights'
        self.weights.mkdir()
        (self.weights / 'config.json').write_text('{}\n')
        self.patchdir = self.root / launch.ROUTING_RECEIPT_DIRS[0]
        self.patchdir.mkdir(parents=True)
        self.backup = self.patchdir / 'deepseek_v2.py.stock'
        self.backup.write_text('patched = False\n')
        # Use small deterministic files to exercise real hashing/receipt/diff IO.
        # Fixed production digests are checked separately below, never relaxed by the launcher.
        values = {'__file__': str(self.scripts / 'launch_sglang_diagnostic.py'),
                  'WEIGHTS': self.weights, 'ROUTING_STOCK_SHA': launch.sha(self.backup),
                  'ROUTING_PATCHED_SHA': launch.sha(self.source),
                  'ROUTING_TOOL_SHA': launch.sha(self.tool),
                  'CONFIG_SHA': launch.sha(self.weights / 'config.json'),
                  'MANIFEST_SHA': launch.sha(self.manifest)}
        for key, value in values.items():
            patch = mock.patch.object(launch, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.pins = {'baseline_identity': launch.BASELINE_IDENTITY,
                     'files': {launch.ROUTING_SOURCE: launch.ROUTING_PATCHED_SHA},
                     'routing_compat': {'stock_sha256': launch.ROUTING_STOCK_SHA,
                         'patched_sha256': launch.ROUTING_PATCHED_SHA,
                         'patch_tool_sha256': launch.ROUTING_TOOL_SHA,
                         'receipt_directories_by_rank': list(launch.ROUTING_RECEIPT_DIRS)}}
        self.receipt = {'status': 'applied_NOT_GPU_qualified',
            'identity': 'sglang0.5.5.post3+r1-routing-None-compat',
            'source_path': str(self.source), 'backup_path': str(self.backup),
            'stock_sha256': launch.ROUTING_STOCK_SHA, 'patched_sha256': launch.ROUTING_PATCHED_SHA,
            'config_sha256': launch.CONFIG_SHA, 'manifest_sha256': launch.MANIFEST_SHA,
            'patch_tool_sha256': launch.ROUTING_TOOL_SHA, 'activation_env': launch.ROUTING_ENV}
        self.write_receipt()
        (self.patchdir / 'patch.diff').write_text(''.join(difflib.unified_diff(
            self.backup.read_text().splitlines(True), self.source.read_text().splitlines(True),
            fromfile=launch.ROUTING_SOURCE + '.stock', tofile=launch.ROUTING_SOURCE + '.r1-compat')))

    def write_receipt(self):
        (self.patchdir / 'receipt.json').write_text(json.dumps(self.receipt))

    def inspect(self):
        return launch.inspect_routing_compat(self.root, 0, self.site, self.pins, launch.ROUTING_ENV)

    def test_valid_receipt_records_actual_provenance_without_gpu_claim(self):
        result = self.inspect()
        self.assertEqual(result['receipt_path'], str(self.patchdir / 'receipt.json'))
        self.assertEqual(result['receipt_sha256'], launch.sha(self.patchdir / 'receipt.json'))
        self.assertEqual(result['diff_sha256'], launch.sha(self.patchdir / 'patch.diff'))
        self.assertIn('NOT_GPU_qualified', result['status'])
        self.assertEqual(result['stock_sha256'], launch.sha(self.backup))
        self.assertEqual(result['patched_sha256'], launch.sha(self.source))

    def test_receipt_missing_or_foreign_rank_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            launch.inspect_routing_compat(self.root, 1, self.site, self.pins, launch.ROUTING_ENV)
        (self.patchdir / 'receipt.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            self.inspect()

    def test_receipt_cannot_substitute_another_source_backup_or_identity(self):
        for key in self.receipt:
            with self.subTest(key=key):
                original = self.receipt[key]
                self.receipt[key] = 'unknown'
                self.write_receipt()
                with self.assertRaisesRegex(RuntimeError, 'receipt mismatch'):
                    self.inspect()
                self.receipt[key] = original
        self.write_receipt()

    def test_current_artifacts_mutation_rejected_even_with_unchanged_receipt(self):
        for path in (self.source, self.backup, self.tool, self.manifest, self.weights / 'config.json'):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(old + b'# changed')
                with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                    self.inspect()
                path.write_bytes(old)

    def test_fake_diff_or_symlink_source_rejected(self):
        diff = self.patchdir / 'patch.diff'
        old = diff.read_text()
        diff.write_text('not the applied diff')
        with self.assertRaisesRegex(RuntimeError, 'saved diff mismatch'):
            self.inspect()
        diff.write_text(old)
        target = self.source.with_suffix('.copy')
        self.source.rename(target)
        self.source.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            self.inspect()

    def test_unknown_or_stock_source_pin_never_accepted(self):
        for digest in ('unknown', launch.ROUTING_STOCK_SHA):
            self.pins['files'][launch.ROUTING_SOURCE] = digest
            with self.assertRaisesRegex(RuntimeError, 'pins changed'):
                self.inspect()

    def test_disabled_compat_flag_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'activation environment missing'):
            launch.inspect_routing_compat(self.root, 0, self.site, self.pins,
                                         {'BATCHGEN_SGLANG_R1_ROUTING_COMPAT': '0'})


class RoutingProductionPinsTest(unittest.TestCase):
    def test_exact_production_tool_and_model_pins(self):
        pins = json.loads((ROOT / 'scripts/r1_reproduction/sglang_diagnostic_source_pins.json').read_text())
        self.assertEqual(pins['files'][launch.ROUTING_SOURCE],
                         '6fa332b6aad5017d336c50826ecac5e1aa8ff446821c5a860ba83f930234f835')
        self.assertEqual(pins['routing_compat']['stock_sha256'],
                         '620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49')
        self.assertEqual(launch.sha(ROOT / 'scripts/r1_reproduction/patch_sglang_r1_routing.py'),
                         pins['routing_compat']['patch_tool_sha256'])
        self.assertEqual(launch.routing_compat_plan(launch.ROOT, 0)['receipt_path'],
                         str(launch.ROOT / 'sglang-routing-compat-node0/patch/receipt.json'))
        self.assertEqual(launch.routing_compat_plan(launch.ROOT, 1)['receipt_path'],
                         str(launch.ROOT / 'sglang-routing-compat-v2/patch/receipt.json'))


if __name__ == '__main__':
    unittest.main()
