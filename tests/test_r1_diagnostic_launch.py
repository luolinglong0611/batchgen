"""CPU-only launch-plan contract checks; no torch or runtime imports."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r1_diagnostic_launch', ROOT / 'scripts/r1_reproduction/launch_r1_diagnostic.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class DiagnosticLaunchTest(unittest.TestCase):
    def test_both_nodes_same_topology_and_direct_converted_directory(self):
        for rank in (0, 1):
            env, cmd = launch.plan(launch.ROOT, rank, 'contract-v1')
            get = lambda key: cmd[cmd.index(key) + 1]
            self.assertEqual(get('--converted-ckpt-dir'), str(launch.ROOT / 'converted'))
            self.assertEqual(get('--model'), 'deepseek-ai/DeepSeek-R1')
            self.assertEqual((get('--world-size'), get('--nnodes'), get('--node-rank')), ('16', '2', str(rank)))
            self.assertEqual(get('--host-kv-cache-size'), '64')
            self.assertEqual(get('--input-ids-pool-size-gb'), '1')
            self.assertEqual(get('--kv-dtype'), 'bfloat16')
            self.assertIn('--disable-cuda-graphs', cmd)
            for flag in ('--enable-cuda-graph', '--enable-prefix-cache', '--enable-hugetlbfs', '--fast-init', '--enable-ep-with-offloading'):
                self.assertNotIn(flag, cmd)
            self.assertEqual(env['BATCHGEN_R1_SM103_ATTENTION'], 'triton_precision')
            self.assertEqual(env['BATCHGEN_W8A16_DEQUANT'], '0')
            self.assertEqual(env['BATCHGEN_ENABLE_ALL_TO_ALL'], '0')
            self.assertEqual(env['GLOO_SOCKET_IFNAME'], 'eth0')
            self.assertEqual(env['BATCHGEN_HOST_REGISTER_CHUNK_GIB'], '16')

    def test_only_current_http_cli_arguments(self):
        tree = ast.parse((ROOT / 'batchgen/server/server_args.py').read_text())
        accepted = {arg.value for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        _, cmd = launch.plan(launch.ROOT, 0, 'contract-v1')
        self.assertTrue(set(token for token in cmd if token.startswith('--')) <= accepted)

    def test_fast_plan_requires_explicit_choice_without_changing_weights_or_topology(self):
        default_env, default_command = launch.plan(launch.ROOT, 0, 'contract-v1')
        fast_env, fast_command = launch.plan(launch.ROOT, 0, 'contract-v1', 'fa4_flashinfer')
        self.assertEqual(fast_command, default_command)
        expected = dict(default_env, BATCHGEN_R1_SM103_ATTENTION='fa4_flashinfer',
                        BATCHGEN_R1_SM103_FAST_DIAGNOSTIC='1')
        self.assertEqual(fast_env, expected)
        self.assertNotIn('BATCHGEN_R1_SM103_FAST_DIAGNOSTIC', default_env)
        with self.assertRaises(ValueError):
            launch.plan(launch.ROOT, 0, 'contract-v1', 'unreviewed_backend')


class RegistrationEvidenceGateTest(unittest.TestCase):
    """Ephemeral fake files test validation only; never valid GPU evidence."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'repo/batchgen').mkdir(parents=True)
        (self.root / 'converted').mkdir()
        (self.root / 'converted.receipts').mkdir()
        self.core = self.root / 'repo/batchgen/core_engine.unit_test.so'
        self.core.write_bytes(b'Not a native library: CPU validator fixture only')
        self.path = self.root / 'registration-unit-test.json'
        self.metadata = {'total_byte_size': 8 * 1024**2 + 272, 'state_dict': {}}
        offset = 0
        for i in range(270):
            size, dtype = (8 * 1024**2, 'bfloat16') if i == 0 else ((4, 'float32') if i == 1 else (1, 'float8_e4m3fn'))
            self.metadata['state_dict'][f'tensor{i}'] = {'offset': offset, 'byte_size': size, 'dtype': dtype,
                                                       'shape': [size // (2 if i == 0 else 4 if i == 1 else 1)]}
            offset += size
        self.metadata['total_byte_size'] = offset
        stem = 'model-00001-of-000163'
        meta_path = self.root / 'converted' / (stem + '.json')
        meta_path.write_text(json.dumps(self.metadata))
        fake_source_sha = hashlib.sha256(b'unit-test source placeholder').hexdigest()
        receipt = {'validation': {'binary_sha256': fake_source_sha, 'metadata_sha256': launch.digest(meta_path)}}
        (self.root / 'converted.receipts' / (stem + '.safetensors.converted.json')).write_text(json.dumps(receipt))
        prefix = 16 * 1024**3 - 2 * 1024**2
        derived = json.loads(json.dumps(self.metadata))
        for item in derived['state_dict'].values():
            item['offset'] += prefix
        derived['total_byte_size'] += prefix
        self.data = {'status': 'passed', 'hostname': launch.HOSTS[0], 'registration_mode': 'native',
                     'native_register_chunk_gib': 16, 'reservation_bytes': 675 * 1024**3,
                     'worker_count': 8, 'hugetlbfs': False, 'thp': False, 'clean_native_release': True,
                     'core_engine_path': str(self.core), 'core_engine_sha256': launch.digest(self.core),
                     'derived_fixture': {'kind': 'shifted original tensor bytes with sparse zero prefix',
                         'prefix_bytes': prefix, 'nominal_boundary': 16 * 1024**3,
                         'crossing_tensor': 'tensor0', 'tensor_start': prefix, 'tensor_bytes': 8 * 1024**2,
                         'copied_source_sha256': fake_source_sha,
                         'derived_metadata_sha256': hashlib.sha256(json.dumps(derived).encode()).hexdigest()},
                     'workers': []}
        for gpu in range(8):
            child_path = self.path.with_suffix(f'.gpu{gpu}.json')
            tensors = [{'name': name, 'shape': m['shape'], 'bytes': m['byte_size'], 'dtype': m['dtype'],
                        'sha256': hashlib.sha256(name.encode()).hexdigest(), 'host_registered': True,
                        'host_exact': True, 'roundtrip_exact': True} for name, m in self.metadata['state_dict'].items()]
            child = {'pid': 1000 + gpu, 'physical_gpu': gpu, 'status': 'passed',
                     'simultaneous_registration_barrier_passed': True, 'tensor_count': 270,
                     'core_engine': str(self.core), 'core_engine_sha256': launch.digest(self.core), 'tensors': tensors}
            child_path.write_text(json.dumps(child))
            self.data['workers'].append({'pid': 1000 + gpu, 'report': str(child_path), 'status': 'passed',
                                        'exit_code': 0, 'simultaneous_registration_barrier_passed': True,
                                        'tensor_count': 270})

    def validate(self):
        self.path.write_text(json.dumps(self.data))
        return launch.validate_registration_qualification(self.root, 0, self.path)

    def test_complete_evidence_and_core_fingerprint(self):
        result = self.validate()
        self.assertEqual(result['core_engine_sha256'], launch.digest(self.core))
        self.assertEqual(len(result['worker_report_sha256']), 8)
        self.core.write_bytes(b'different build')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            self.validate()

    def test_missing_or_weaker_owner_evidence_rejected(self):
        for key, wrong in [('status', 'failed'), ('registration_mode', 'chunked_driver_probe'),
                           ('reservation_bytes', 32 * 1024**3), ('worker_count', 1),
                           ('native_register_chunk_gib', 0), ('clean_native_release', False),
                           ('hostname', launch.HOSTS[1]), ('hugetlbfs', True), ('thp', True)]:
            with self.subTest(field=key):
                old = self.data[key]
                self.data[key] = wrong
                with self.assertRaises(RuntimeError):
                    self.validate()
                self.data[key] = old
        del self.data['clean_native_release']
        with self.assertRaisesRegex(RuntimeError, 'postvalidation'):
            self.validate()

    def test_shifted_fixture_required_and_bound_to_original(self):
        for key, wrong in [('tensor_start', 16 * 1024**3), ('copied_source_sha256', '0' * 64),
                           ('derived_metadata_sha256', '0' * 64), ('prefix_bytes', 1),
                           ('crossing_tensor', 'absent')]:
            with self.subTest(field=key):
                old = self.data['derived_fixture'][key]
                self.data['derived_fixture'][key] = wrong
                with self.assertRaises(RuntimeError):
                    self.validate()
                self.data['derived_fixture'][key] = old

    def test_each_worker_must_have_real_dma_and_matching_core(self):
        child_path = self.path.with_suffix('.gpu7.json')
        original = json.loads(child_path.read_text())
        for key, wrong in [('status', 'failed'), ('simultaneous_registration_barrier_passed', False),
                           ('physical_gpu', 0), ('tensor_count', 269), ('core_engine_sha256', '0' * 64)]:
            with self.subTest(field=key):
                child = dict(original, **{key: wrong})
                child_path.write_text(json.dumps(child))
                with self.assertRaises(RuntimeError):
                    self.validate()
        child = json.loads(json.dumps(original))
        child['tensors'][0]['roundtrip_exact'] = False
        child_path.write_text(json.dumps(child))
        with self.assertRaisesRegex(RuntimeError, 'byte comparison'):
            self.validate()
        child_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_execute_requires_explicit_evidence_before_any_runtime_import(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/r1_reproduction/launch_r1_diagnostic.py'),
                                 '--node-rank', '0', '--execute'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--execute requires --registration-qualification', result.stderr)


class WeightValidationReuseTest(unittest.TestCase):
    """Tiny CPU fixture only: no actual model hashes or launch qualification."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ('repo/scripts/r1_reproduction', 'repo/batchgen/ckpt_converter',
                       'converted', 'converted.receipts', 'runs/old/node0'):
            (self.root / folder).mkdir(parents=True)
        converter = self.root / 'repo/batchgen/ckpt_converter/ckpt_converter.py'
        converter.write_text('# CPU test placeholder, not a real converter')
        self.converter_sha = launch.digest(converter)
        entries = [{'Path': f'model-{i:05d}-of-000163.safetensors',
                    'Sha256': hashlib.sha256(f'fake source{i}'.encode()).hexdigest(), 'Size': i}
                   for i in range(1, 164)]
        manifest = self.root / 'repo/scripts/r1_reproduction/r1_model_manifest.json'
        manifest.write_text(json.dumps(entries))
        patcher = mock.patch.object(launch, 'MANIFEST_SHA', launch.digest(manifest))
        patcher.start()
        self.addCleanup(patcher.stop)
        quiet = mock.patch('builtins.print')
        quiet.start()
        self.addCleanup(quiet.stop)
        for entry in entries:
            stem = Path(entry['Path']).stem
            binary = self.root / 'converted' / (stem + '.bin')
            binary.write_bytes(entry['Path'].encode())
            metadata = binary.with_suffix('.json')
            metadata.write_text(json.dumps({'unit_test': stem}))
            receipt = {'source_sha256': entry['Sha256'], 'source_size': entry['Size'],
                       'source_revision': launch.REVISION, 'manifest_sha256': launch.MANIFEST_SHA,
                       'converter_sha256': self.converter_sha, 'marlin': False,
                       'validation': {'binary_size': binary.stat().st_size,
                                      'binary_sha256': launch.digest(binary),
                                      'metadata_sha256': launch.digest(metadata)}}
            (self.root / 'converted.receipts' / (entry['Path'] + '.converted.json')).write_text(json.dumps(receipt))
        self.records, _ = launch.validate_weights(self.root)
        self.binary = self.root / 'converted/model-00001-of-000163.bin'
        self.receipt = self.root / 'converted.receipts/model-00001-of-000163.safetensors.converted.json'
        self.path = self.root / 'runs/old/node0/launch.json'
        env, cmd = launch.plan(self.root, 0, 'old')
        self.report = {'status': 'exited', 'exit_code': 1, 'hostname': launch.HOSTS[0],
                       'utc': datetime.now(timezone.utc).isoformat(),
                       'environment': env, 'command': cmd, 'weights': self.records,
                       'source_sha256': {'batchgen/ckpt_converter/ckpt_converter.py': self.converter_sha}}
        self.write_report()

    def write_report(self):
        self.path.write_text(json.dumps(self.report))

    def validate(self):
        return launch.validate_weights(self.root, self.path, 0)

    def test_full_hash_records_identity_and_exact_reuse_avoids_binary_reads(self):
        self.assertTrue(all(r['binary_verification']['mode'] == 'full_sha256' for r in self.records))
        real_digest = launch.stable_digest
        def no_bin(path):
            self.assertNotEqual(path.suffix, '.bin', 'unchanged binary was unexpectedly read')
            return real_digest(path)
        with mock.patch.object(launch, 'stable_digest', side_effect=no_bin):
            records, evidence = self.validate()
        self.assertTrue(all(r['binary_verification']['mode'] == 'reused_exact_identity' for r in records))
        self.assertEqual(evidence['prior_status'], 'exited')
        self.assertEqual(evidence['prior_exit_code'], 1)
        self.assertEqual(evidence['report_sha256'], launch.digest(self.path))
        launch.check_weight_identities(records)

    def test_legacy_timestamps_allow_reuse_but_old_start_requires_real_hash(self):
        self.report['weights'] = [{k: v for k, v in r.items() if k in
                                 ('shard', 'binary_size', 'binary_sha256', 'metadata_sha256')}
                                 for r in self.records]
        self.write_report()
        records, _ = self.validate()
        self.assertTrue(all(r['binary_verification']['mode'] == 'reused_legacy_prevalidation_timestamps' for r in records))
        self.report['utc'] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.write_report()
        records, _ = self.validate()
        self.assertTrue(all(r['binary_verification']['mode'] == 'full_sha256' for r in records))

    def test_prior_fast_diagnostic_reuses_only_exact_recorded_plan(self):
        env, cmd = launch.plan(self.root, 0, 'old', 'fa4_flashinfer')
        self.report.update(environment=env, command=cmd)
        self.write_report()
        records, evidence = self.validate()
        self.assertTrue(all(r['binary_verification']['mode'] == 'reused_exact_identity' for r in records))
        self.assertEqual(evidence['prior_exit_code'], 1)
        # Reuse is a weight identity cache, not a quality claim, and a missing
        # explicit fast opt-in must not be accepted as a previously valid plan.
        del self.report['environment']['BATCHGEN_R1_SM103_FAST_DIAGNOSTIC']
        self.write_report()
        with self.assertRaisesRegex(RuntimeError, 'environment'):
            self.validate()

    def test_replaced_identical_binary_rehashes_and_corruption_fails(self):
        replacement = self.root / 'same-bytes'
        replacement.write_bytes(self.binary.read_bytes())
        replacement.replace(self.binary)
        records, _ = self.validate()
        self.assertEqual(records[0]['binary_verification']['mode'], 'full_sha256')
        self.assertEqual(records[1]['binary_verification']['mode'], 'reused_exact_identity')
        self.binary.write_bytes(b'x' * self.binary.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            self.validate()

    def test_current_receipts_metadata_and_source_are_always_validated(self):
        original = self.receipt.read_text()
        self.receipt.write_text(json.dumps(json.loads(original), indent=2))
        records, _ = self.validate()
        self.assertEqual(records[0]['binary_verification']['mode'], 'full_sha256')
        receipt = json.loads(original)
        receipt['source_sha256'] = '0' * 64
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, 'source_sha256 mismatch'):
            self.validate()
        self.receipt.write_text(original)
        self.binary.with_suffix('.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            self.validate()

    def test_arbitrary_manifest_incomplete_live_or_other_node_report_rejected(self):
        arbitrary = self.root / 'checksums.json'
        arbitrary.write_text(self.path.read_text())
        with self.assertRaisesRegex(RuntimeError, 'local runs'):
            launch.validate_weights(self.root, arbitrary, 0)
        for key, wrong in [('weights', self.records[:-1]), ('status', 'launching'),
                           ('hostname', launch.HOSTS[1]), ('command', []),
                           ('source_sha256', {}),
                           ('utc', (datetime.now(timezone.utc) + timedelta(days=1)).isoformat())]:
            with self.subTest(field=key):
                old = self.report[key]
                self.report[key] = wrong
                self.write_report()
                with self.assertRaises(RuntimeError):
                    self.validate()
                self.report[key] = old

    def test_symlink_and_postvalidation_change_are_rejected(self):
        self.receipt.write_text(self.receipt.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'changed before launch'):
            launch.check_weight_identities(self.records)
        target = self.root / 'binary-target'
        self.binary.replace(target)
        self.binary.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'nonregular'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
