"""CPU contract tests; the integration test uses two spawned CPU-only processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/skeleton_mmap_qualification.py'
spec = importlib.util.spec_from_file_location('skeleton_mmap_qualification', SCRIPT)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
HAS_TORCH = importlib.util.find_spec('torch') is not None


class SafetyContract(unittest.TestCase):
    def test_requires_explicit_hidden_gpu_before_import(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(RuntimeError, 'explicitly'):
                qualification.qualify()

    def test_child_error_is_not_success(self):
        class Connection:
            def poll(self, _): return True
            def recv(self): return {'event': 'error', 'traceback': 'deliberate failure'}
        import time
        with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
            qualification._receive(Connection(), 'loaded', time.monotonic() + 1)


@unittest.skipUnless(HAS_TORCH, 'Requires BatchGen CPU-capable Torch environment')
class TensorContract(unittest.TestCase):
    def test_manifest_observes_logical_and_outside_view_storage_bytes(self):
        import torch
        original = torch.arange(16, dtype=torch.float32)
        tensors = {'view': original[4:8]}
        before = qualification.tensor_manifest(tensors)['view']
        original[0] = 100
        after = qualification.tensor_manifest(tensors)['view']
        self.assertEqual(before['logical_sha256'], after['logical_sha256'])
        self.assertNotEqual(before['storage_sha256'], after['storage_sha256'])
        original[5] = 200
        final = qualification.tensor_manifest(tensors)['view']
        self.assertNotEqual(after['logical_sha256'], final['logical_sha256'])

    def test_fixture_contains_real_alias_and_noncontiguous_views(self):
        tensors = qualification.fixture()
        self.assertTrue(qualification.alias_contract(tensors))
        self.assertFalse(tensors['noncontiguous'].is_contiguous())
        self.assertGreater(tensors['offset_view'].storage_offset(), 0)
        tensors['alias'] = tensors['alias'].clone()
        self.assertFalse(qualification.alias_contract(tensors))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux memfd required')
    def test_two_process_mmap_cow_and_close_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'qualification.json'
            result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output)],
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertLess(report['serialized_bytes'], 8 * 1024**2)
            self.assertEqual(report['child_exit_codes'], [0, 0])
            self.assertTrue(report['cow']['backing_unchanged'])
            self.assertFalse(report['cuda_initialized'])
            self.assertTrue(report['child_processes_absent'])
            for row in report['after_owner_close']:
                self.assertEqual(row['new_load_error'], 'FileNotFoundError')
            for row in report['released']:
                self.assertEqual(row['mapping']['mappings'], 0)
                self.assertFalse(row['cuda_initialized'])


if __name__ == '__main__':
    unittest.main()
