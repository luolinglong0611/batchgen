"""Stdlib CPU checks; these do not substitute for the Linux CUDA build."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/fmha_precompile_qualification.py'
spec = importlib.util.spec_from_file_location('fmha_precompile_qualification', SOURCE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class QualificationPreparationTests(unittest.TestCase):
    def test_byte_hash_and_detect_real_input_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'input'
            p.write_bytes(bytes(range(256)) * 4097)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(q.checked_files({'input': sha}, root)[0]['sha256'], sha)
            with p.open('r+b') as f:
                f.seek(523)
                f.write(b'altered')
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                q.checked_files({'input': sha}, root)

    def test_exact_aux_copies_and_private_runtime_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'env/lib/python3.11/site-packages/nvidia/cu13/include'
            src.mkdir(parents=True)
            for i, name in enumerate(('cublasLt.h', 'cublas_api.h', 'cublas_v2.h', 'nvrtc.h')):
                (src / name).write_bytes(bytes([i, 0, 255]) * 99)
            lib = root / 'cuda130-packages/nvidia/cu13/lib/libcudart.so.13'
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b'fake_runtime_only_for_file_test')
            stage = root / 'newstage'
            stage.mkdir()
            receipt = q.make_overlay(stage, root)
            self.assertEqual(len(receipt['headers']), 4)
            for row in receipt['headers']:
                a, b = Path(row['source']['path']), Path(row['copy']['path'])
                self.assertFalse(b.is_symlink())
                self.assertEqual(a.read_bytes(), b.read_bytes())
                a.write_bytes(b'later unrelated source change')
                self.assertNotEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual((stage / 'lib/libcudart.so').resolve(), lib.resolve())
            with self.assertRaises(FileExistsError):
                q.make_overlay(stage, root)

    def test_environment_ignores_ambient_overrides_and_excludes_BG_CRT(self):
        old = dict(os.environ)
        try:
            os.environ.update({'CUDA_HOME': '/bad', 'CPATH': '/bad',
                'CPLUS_INCLUDE_PATH': '/bad', 'FLASHINFER_DISABLE_JIT': '1',
                'FLASHINFER_WORKSPACE_BASE': '/old-cache'})
            env = q.environment(Path('/private/new'), Path('/private/root'))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'], 'none')
            self.assertEqual(env['MAX_JOBS'], '1')
            self.assertEqual(env['LD_PRELOAD'], '/private/root/cuda130-packages/nvidia/cu13/lib/libcudart.so.13')
            self.assertEqual(env['FLASHINFER_CUDA_ARCH_LIST'], '10.3a')
            self.assertNotIn('FLASHINFER_DISABLE_JIT', env)
            self.assertNotIn('CPLUS_INCLUDE_PATH', env)
            self.assertNotIn('/env/lib/python3.11/site-packages/nvidia/cu13/include', env['CPATH'])
            self.assertNotIn('/bad', json.dumps(env))
            self.assertEqual(env['FLASHINFER_WORKSPACE_BASE'], '/private/new/workspace')
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_real_python_socket_lookup_denied_before_network(self):
        code = (f'import runpy,sys,socket; q=runpy.run_path({str(SOURCE)!r}); '
                'sys.addaudithook(q["deny_network"]); socket.getaddrinfo("example.invalid",443)')
        p = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Network operation forbidden', p.stderr)

    def test_include_trace_resolves_bin_parent_and_deduplicates(self):
        t = '. /tool/bin/../include/crt/host_runtime.h\n.. /tool/include/crt/host_runtime.h\n'
        self.assertEqual(q.header_trace_paths(t), ['/tool/include/crt/host_runtime.h'])
        self.assertEqual(q.header_trace_paths('ordinary compiler text'), [])

    def test_explicit_hidden_required(self):
        old = dict(os.environ)
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['NVIDIA_VISIBLE_DEVICES'] = 'none'
            q.check_cuda_hidden()
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            with self.assertRaises(RuntimeError):
                q.check_cuda_hidden()
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == '__main__':
    unittest.main()
