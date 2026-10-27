"""CPU checks for recovery provenance, bounded downloads and child timeout."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/restore_sglang_alignment_env.py'
SPEC = importlib.util.spec_from_file_location('restore_sg_alignment_test_target', FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Read1Response:
    """A network-free stream that rejects the old accumulating read API."""
    def __init__(self, chunks):
        self.headers = {}
        self.chunks = iter(chunks)
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        raise AssertionError('Accumulating response.read must not be used')

    def read1(self, size):
        self.read_sizes.append(size)
        chunk = next(self.chunks, b'')
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


class RestoreEnvironmentTests(unittest.TestCase):
    def download_record(self, data):
        return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'url': 'https://example.invalid/pinned.whl'}

    def lock_fixture(self, extra=''):
        names = list(M.BINARY_WHEELS)
        data = ('# keep this comment and spacing\n'
                'sglang==0.5.5.post3 \\\n    --hash=sha256:unchanged\n'
                'sgl-kernel @ file://' + M.OLD_WHEELS + names[0].replace('+', '%2B') + '#sha256=a \\\n    --hash=sha256:b\n'
                'triton @ file://' + M.OLD_WHEELS + names[1] + '#sha256=c\n' + extra).encode()
        return data, {'resolved-wheels.lock': {'sha256': hashlib.sha256(data).hexdigest()}}

    def test_default_plan_has_no_side_effect_calls(self):
        with patch.object(M, 'execute') as execute, patch.object(M, 'save_json') as save, \
             patch.object(M, 'run_step') as run, patch.object(M, 'download_verified') as download:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(M.main([]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result['mode'], 'plan_only')
            self.assertFalse(result['gpu_operations'])
            for method in (execute, save, run, download):
                method.assert_not_called()

    def test_lock_rewrites_only_two_known_paths(self):
        data, artifacts = self.lock_fixture()
        with patch.object(M, 'ARTIFACTS', artifacts):
            mapped, replacements = M.remap_lock(data, Path('/new/wheels'))
        self.assertEqual(len(replacements), 2)
        restored = mapped.decode()
        for item in replacements:
            restored = restored.replace(item['new'], item['old'])
        self.assertEqual(restored.encode(), data)
        self.assertIn(b'file:///new/wheels/sgl_kernel-', mapped)
        self.assertIn(b'%2Bcu130', mapped)
        self.assertNotIn(M.OLD_WHEELS.encode(), mapped)

    def test_lock_rejects_original_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Original dependency lock'):
            M.remap_lock(b'not the pinned lock', Path('/new/wheels'))

    def test_lock_rejects_unexpected_file_dependency(self):
        data, artifacts = self.lock_fixture('other @ file:///arbitrary/other.whl\n')
        with patch.object(M, 'ARTIFACTS', artifacts), self.assertRaisesRegex(ValueError, 'Unexpected local'):
            M.remap_lock(data, Path('/new/wheels'))

    def test_lock_requires_both_expected_dependencies(self):
        data = b'sglang==0.5.5.post3\n'
        artifacts = {'resolved-wheels.lock': {'sha256': hashlib.sha256(data).hexdigest()}}
        with patch.object(M, 'ARTIFACTS', artifacts), self.assertRaisesRegex(ValueError, 'exactly one'):
            M.remap_lock(data, Path('/new/wheels'))

    def test_verified_copy_never_publishes_bad_bytes_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming, outgoing = root / 'source', root / 'output'
            incoming.write_bytes(b'payload')
            wrong = {'bytes': 7, 'sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                M.verified_copy(incoming, outgoing, wrong)
            self.assertFalse(outgoing.exists())
            self.assertEqual((root / 'output.partial').read_bytes(), b'payload')
            outgoing.write_bytes(b'owned by another run')
            with self.assertRaises(FileExistsError):
                M.verified_copy(incoming, outgoing, wrong)
            self.assertEqual(outgoing.read_bytes(), b'owned by another run')

    def test_download_uses_read1_chunks_and_caps_socket_timeout(self):
        for timeout in (0.25, 60):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                outgoing = Path(directory) / 'artifact.whl'
                response = Read1Response([b'pay', b'load', b''])
                with patch.object(M.urllib.request, 'urlopen', return_value=response) as open_url, \
                     patch.object(M.time, 'monotonic', return_value=0):
                    result = M.download_verified(outgoing, self.download_record(b'payload'), timeout)
                self.assertEqual(open_url.call_args.kwargs['timeout'], min(10, timeout))
                self.assertEqual(response.read_sizes, [64 * 1024] * 3)
                self.assertEqual(outgoing.read_bytes(), b'payload')
                self.assertFalse(outgoing.with_name('artifact.whl.partial').exists())
                self.assertEqual(result['bytes'], 7)
                self.assertEqual(result['sha256'], hashlib.sha256(b'payload').hexdigest())

    def test_download_checks_deadline_before_first_read(self):
        with tempfile.TemporaryDirectory() as directory:
            outgoing = Path(directory) / 'artifact.whl'
            response = Read1Response([b'payload'])
            with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                 patch.object(M.time, 'monotonic', side_effect=[0, 10]), \
                 self.assertRaisesRegex(TimeoutError, 'deadline exceeded'):
                M.download_verified(outgoing, self.download_record(b'payload'), 10)
            self.assertEqual(response.read_sizes, [])
            self.assertFalse(outgoing.exists())
            self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'')

    def test_download_checks_deadline_after_each_read_and_retains_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            outgoing = Path(directory) / 'artifact.whl'
            response = Read1Response([b'pay', b'load'])
            with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                 patch.object(M.time, 'monotonic', side_effect=[0, 1, 2, 3, 10]), \
                 self.assertRaisesRegex(TimeoutError, 'deadline exceeded'):
                M.download_verified(outgoing, self.download_record(b'payload'), 10)
            self.assertEqual(response.read_sizes, [64 * 1024] * 2)
            self.assertFalse(outgoing.exists())
            self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'pay')

    def test_download_does_not_publish_when_final_flush_exceeds_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            outgoing = Path(directory) / 'artifact.whl'
            response = Read1Response([b'payload', b''])
            clock = SimpleNamespace(now=0)

            def slow_flush(_fd):
                clock.now = 10

            with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                 patch.object(M.time, 'monotonic', side_effect=lambda: clock.now), \
                 patch.object(M.os, 'fsync', side_effect=slow_flush), \
                 self.assertRaisesRegex(TimeoutError, 'deadline exceeded'):
                M.download_verified(outgoing, self.download_record(b'payload'), 10)
            self.assertFalse(outgoing.exists())
            self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'payload')

    def test_download_retains_partial_on_short_eof_or_stream_exception(self):
        for ending, error in ((b'', EOFError), (EOFError('broken stream'), EOFError),
                              (TimeoutError('socket stalled'), TimeoutError)):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                outgoing = Path(directory) / 'artifact.whl'
                response = Read1Response([b'pay', ending])
                with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                     patch.object(M.time, 'monotonic', return_value=0), \
                     self.assertRaises(error):
                    M.download_verified(outgoing, self.download_record(b'payload'), 10)
                self.assertFalse(outgoing.exists())
                self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'pay')

    def test_download_rejects_stream_without_read1(self):
        with tempfile.TemporaryDirectory() as directory:
            outgoing = Path(directory) / 'artifact.whl'
            response = Read1Response([b'payload'])
            response.read1 = None
            with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                 self.assertRaisesRegex(TypeError, 'requires a response.read1 stream'):
                M.download_verified(outgoing, self.download_record(b'payload'), 10)
            self.assertEqual(response.read_sizes, [])
            self.assertFalse(outgoing.exists())
            self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'')

    def test_download_does_not_publish_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            outgoing = Path(directory) / 'artifact.whl'
            response = Read1Response([b'changed', b''])
            with patch.object(M.urllib.request, 'urlopen', return_value=response), \
                 patch.object(M.time, 'monotonic', return_value=0), \
                 self.assertRaisesRegex(ValueError, 'size/SHA256 mismatch'):
                M.download_verified(outgoing, self.download_record(b'payload'), 10)
            self.assertFalse(outgoing.exists())
            self.assertEqual(outgoing.with_name('artifact.whl.partial').read_bytes(), b'changed')

    def test_children_hide_gpu_and_clear_foreign_runtime(self):
        args = SimpleNamespace(cache_dir=Path('/new/cache'))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'LD_PRELOAD': '/foreign.so',
                                    'PYTHONPATH': '/foreign', 'UV_DEFAULT_INDEX': 'https://invalid/'}):
            env = M.child_environment(args, Path('/new/cxx'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'], 'none')
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('UV_DEFAULT_INDEX', env)
        self.assertEqual(env['LD_LIBRARY_PATH'], '/new/cxx/lib')

    def test_owned_child_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'CPU recovery step failed'):
                M.run_step([sys.executable, '-c', 'import time; time.sleep(10)'],
                           'timeout', job, os.environ.copy(), 0.05)
            record = json.loads((job / 'timeout.run.json').read_text())
            self.assertTrue(record['timed_out'])
            self.assertIsNotNone(record['exit_code'])
            self.assertNotEqual(record['exit_code'], 0)

    def test_rejects_overlapping_or_relative_roots(self):
        args = M.parser().parse_args(['--env-root', '/new', '--recovery-root', '/new/sub'])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            M.validate_args(args)
        args = M.parser().parse_args(['--env-root', 'relative'])
        with self.assertRaisesRegex(ValueError, 'absolute'):
            M.validate_args(args)


if __name__ == '__main__':
    unittest.main()
