"""CPU-only transaction/rollback tests; no torch import or native execution."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/deploy_mgn_cub_fix.py'
spec = importlib.util.spec_from_file_location('mgn_deploy_test_module', SOURCE)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old, self.new = b'OLD MGN\0' * 8, b'NEW MGN\0' * 16
        self.core, self.wrapper = b'qualified core', b'qualified wrapper'
        self.site, self.packages, self.targets = deploy.layout(self.root)
        for package in self.packages.values():
            (package / 'common').mkdir(parents=True)
            for relative in deploy.WRAPPERS:
                (package / relative).write_bytes(self.wrapper)
            (package / 'common/other.so').write_bytes(b'other native unchanged')
        for path in self.targets.values():
            path.write_bytes(self.old)
            path.chmod(0o751)
        core = self.root / 'repo/batchgen/core_engine.test.so'
        core.parent.mkdir(parents=True)
        core.write_bytes(self.core)
        self.wheel = self.root / 'candidate.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr(deploy.MEMBER, self.new)
            archive.writestr('batchgen_kernels/common/other.so', b'NEVER INSTALL THIS')
        patches = mock.patch.multiple(deploy, OLD_SHA=sha(self.old), NEW_SHA=sha(self.new),
                                      NEW_SIZE=len(self.new), CORE_SHA=sha(self.core),
                                      WHEEL_SHA=deploy.digest(self.wheel),
                                      WRAPPERS={key: sha(self.wrapper) for key in deploy.WRAPPERS})
        patches.start()
        self.addCleanup(patches.stop)
        self.audit = mock.patch.object(deploy, 'audit_no_batchgen', return_value={'blockers': []})
        self.audit_mock = self.audit.start()
        self.addCleanup(self.audit.stop)
        self.probes = mock.patch.object(deploy, 'verify_imports', side_effect=lambda root, expected: [
            {'mode': 'repo', 'sha256': expected, 'cuda_initialized': False},
            {'mode': 'env', 'sha256': expected, 'cuda_initialized': False}])
        self.probe_mock = self.probes.start()
        self.addCleanup(self.probes.stop)

    def apply(self, name='transaction'):
        return deploy.apply(self.root, 0, self.wheel, self.root / name)

    def assert_old(self):
        self.assertTrue(all(path.read_bytes() == self.old for path in self.targets.values()))

    def test_success_only_replaces_two_targets_and_preserves_other_artifacts(self):
        original_protected = deploy.protected_inventory(self.root)
        result = self.apply()
        self.assertEqual(result['status'], 'applied_cpu_import_verified')
        self.assertTrue(all(path.read_bytes() == self.new for path in self.targets.values()))
        self.assertEqual(deploy.protected_inventory(self.root), original_protected)
        for key, target in self.targets.items():
            self.assertEqual((self.root / f'transaction/{key}.original.so').read_bytes(), self.old)
            self.assertEqual(target.stat().st_mode & 0o777, 0o751)
        manifest = json.loads((self.root / 'transaction/transaction.json').read_text())
        self.assertEqual(manifest['source_commit'], deploy.SOURCE)
        self.assertEqual(manifest['protected'], original_protected)
        self.assertGreaterEqual(self.audit_mock.call_count, 3)

    def test_failed_second_replace_rolls_back_both_targets(self):
        actual = deploy.atomic_replace
        calls = []
        def fail_once(path, data, mode):
            calls.append((path, data))
            if len(calls) == 2:
                raise OSError('Simulated second-target write failure')
            return actual(path, data, mode)
        with mock.patch.object(deploy, 'atomic_replace', side_effect=fail_once):
            result = self.apply()
        self.assertEqual(result['status'], 'apply_failed')
        self.assertEqual(result['rollback']['status'], 'rolled_back')
        self.assert_old()

    def test_interrupt_between_replace_and_journal_list_append_still_restores(self):
        actual = deploy.atomic_replace
        interrupted = False
        def interrupt_after_replace(path, data, mode):
            nonlocal interrupted
            actual(path, data, mode)
            if data == self.new and not interrupted:
                interrupted = True
                raise KeyboardInterrupt('Simulated interrupt immediately after os.replace')
        with mock.patch.object(deploy, 'atomic_replace', side_effect=interrupt_after_replace):
            result = self.apply()
        self.assertEqual(result['status'], 'apply_failed')
        self.assertEqual(result['changed'], [])
        self.assertEqual(result['rollback']['status'], 'rolled_back')
        self.assert_old()

    def test_rollback_evidence_and_parent_directory_are_durable_before_first_replace(self):
        actual_replace, actual_fsync = deploy.atomic_replace, deploy.os.fsync
        synced_inodes = []
        def fsync(fd):
            synced_inodes.append(os.fstat(fd).st_ino)
            actual_fsync(fd)
        def replace(path, data, mode):
            transaction = self.root / 'transaction'
            self.assertTrue((transaction / 'transaction.json').exists())
            self.assertEqual((transaction / 'repo.original.so').read_bytes(), self.old)
            self.assertEqual((transaction / 'env.original.so').read_bytes(), self.old)
            self.assertIn(transaction.stat().st_ino, synced_inodes)
            self.assertIn(self.root.stat().st_ino, synced_inodes)
            return actual_replace(path, data, mode)
        with mock.patch.object(deploy.os, 'fsync', side_effect=fsync), \
             mock.patch.object(deploy, 'atomic_replace', side_effect=replace):
            self.assertEqual(self.apply()['status'], 'applied_cpu_import_verified')

    def test_failed_new_import_rolls_back_and_validates_original_imports(self):
        def probe(root, expected):
            if expected == deploy.NEW_SHA:
                raise RuntimeError('Native ABI failure')
            return [{'sha256': expected}]
        self.probe_mock.side_effect = probe
        result = self.apply()
        self.assertEqual(result['status'], 'apply_failed')
        self.assertEqual(result['rollback']['status'], 'rolled_back')
        self.assert_old()
        self.assertEqual(self.probe_mock.call_args.args[1], deploy.OLD_SHA)

    def test_backup_tamper_refuses_manual_rollback_without_changing_targets(self):
        self.apply()
        (self.root / 'transaction/env.original.so').write_bytes(b'corrupt backup')
        with self.assertRaisesRegex(RuntimeError, 'backup SHA'):
            deploy.restore(self.root, self.root / 'transaction')
        self.assertTrue(all(path.read_bytes() == self.new for path in self.targets.values()))

    def test_rollback_can_recover_partial_apply_after_uncatchable_interruption(self):
        self.apply()
        self.targets['repo'].write_bytes(self.old)  # One target already restored.
        result = deploy.restore(self.root, self.root / 'transaction')
        self.assertEqual(result['status'], 'rolled_back')
        self.assert_old()

    def test_unexpected_target_after_interruption_is_not_overwritten(self):
        self.apply()
        self.targets['env'].write_bytes(b'unrelated later binary')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected current target'):
            deploy.restore(self.root, self.root / 'transaction')
        self.assertEqual(self.targets['repo'].read_bytes(), self.new)

    def test_wrong_wheel_is_rejected_before_output_or_mutation(self):
        self.wheel.write_bytes(b'changed wheel')
        with self.assertRaisesRegex(RuntimeError, 'wheel SHA'):
            self.apply()
        self.assertFalse((self.root / 'transaction').exists())
        self.assert_old()

    def test_wrong_member_sha_rejected_even_if_outer_wheel_pin_matches(self):
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr(deploy.MEMBER, b'?' * len(self.new))
        with mock.patch.object(deploy, 'WHEEL_SHA', deploy.digest(self.wheel)):
            with self.assertRaisesRegex(RuntimeError, 'MGN SHA'):
                self.apply()
        self.assert_old()

    def test_symlink_target_rejected(self):
        path = self.targets['env']
        path.unlink()
        path.symlink_to(self.targets['repo'])
        with self.assertRaisesRegex(RuntimeError, 'Symlink'):
            self.apply()
        self.assert_old()

    def test_unknown_original_sha_does_not_get_backed_up_as_qualified(self):
        self.targets['env'].write_bytes(b'unknown binary')
        with self.assertRaisesRegex(RuntimeError, 'Expected original'):
            self.apply()
        self.assertFalse((self.root / 'transaction').exists())

    def test_protected_change_is_detected_and_not_hidden_as_success(self):
        original = deploy.atomic_replace
        def changed_core(path, data, mode):
            original(path, data, mode)
            if data == self.new:
                (self.root / 'repo/batchgen/core_engine.test.so').write_bytes(b'external change')
        with mock.patch.object(deploy, 'atomic_replace', side_effect=changed_core):
            result = self.apply()
        self.assertEqual(result['status'], 'apply_failed')
        self.assertTrue(result['parent_cleanup_required'])
        self.assert_old()

    def test_reusing_transaction_output_is_refused(self):
        (self.root / 'transaction').mkdir()
        with self.assertRaises(FileExistsError):
            self.apply()
        self.assert_old()

    def test_active_batchgen_refuses_before_output(self):
        self.audit_mock.side_effect = RuntimeError('Active BatchGen process')
        with self.assertRaisesRegex(RuntimeError, 'Active BatchGen'):
            self.apply()
        self.assertFalse((self.root / 'transaction').exists())
        self.assert_old()


class SafetyTests(unittest.TestCase):
    def test_default_plan_does_not_inspect_import_or_mutate(self):
        with mock.patch('sys.argv', [str(SOURCE), '--node-rank', '0']), \
             mock.patch.object(deploy, 'protected_inventory', side_effect=AssertionError), \
             mock.patch.object(deploy, 'verify_imports', side_effect=AssertionError), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(deploy.main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['status'], 'plan_only')
        self.assertEqual(len(report['targets']), 2)

    def test_mapped_deleted_library_and_spawn_are_blockers_but_sglang_is_not(self):
        root = Path('/example')
        self.assertTrue(deploy.process_reasons(['python', '-m', 'batchgen.launch_http_server'], '', '', root))
        self.assertTrue(deploy.process_reasons(['python'], '/x/_C_mgn_ops.so (deleted)', '', root))
        self.assertTrue(deploy.process_reasons(['python'], '/x/batchgen/core_engine.test.so', '', root))
        self.assertTrue(deploy.process_reasons(['python', '-c', 'multiprocessing.spawn.main()'], '', '/example/repo', root))
        self.assertFalse(deploy.process_reasons(['python', '-m', 'sglang.launch_server'], '/lib/libtorch.so', '', root))

    def test_cpu_probe_has_fresh_isolated_processes_and_no_device_query(self):
        root = Path('/example')
        reports = []
        def child(command, **kwargs):
            env = kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['BATCHGEN_KERNELS_DEV'], '0')
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)
            self.assertNotIn('LD_PRELOAD', env)
            self.assertEqual(command[1:3], ['-I', '-B'])
            reports.append(command[-3])
            return subprocess.CompletedProcess(command, 0, json.dumps({'sha256': deploy.NEW_SHA, 'cuda_initialized': False}), '')
        with mock.patch.dict(os.environ, {'PYTHONPATH': '/bad', 'PYTHONHOME': '/bad', 'LD_PRELOAD': '/bad'}), \
             mock.patch.object(deploy.subprocess, 'run', side_effect=child):
            deploy.verify_imports(root, deploy.NEW_SHA)
        self.assertEqual(reports, ['repo', 'env'])
        self.assertNotIn('get_device', deploy.PROBE)
        self.assertNotIn('is_available', deploy.PROBE)

    def test_proc_audit_detects_mapped_library_without_command_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            pid = proc / '12345678'
            pid.mkdir()
            (pid / 'stat').write_text('12345678 (python) ' + ' '.join(['S'] + ['0'] * 18 + ['99']))
            (pid / 'status').write_text(f'Uid:\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\n')
            (pid / 'cmdline').write_bytes(b'python\0')
            (pid / 'maps').write_text('/example/repo/batchgen/core_engine.test.so (deleted)')
            (pid / 'cwd').symlink_to('/tmp')
            with self.assertRaisesRegex(RuntimeError, 'Active BatchGen'):
                deploy.audit_no_batchgen(Path('/example'), proc)

    def test_unreadable_live_process_map_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            pid = proc / '12345678'
            pid.mkdir()
            (pid / 'stat').write_text('12345678 (python) ' + ' '.join(['S'] + ['0'] * 18 + ['99']))
            (pid / 'status').write_text(f'Uid:\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\t{os.geteuid()}\n')
            (pid / 'cmdline').write_bytes(b'python\0')
            original = Path.read_text
            def read(path, *args, **kwargs):
                if path == pid / 'maps':
                    raise PermissionError('Cannot inspect live process')
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, 'read_text', read):
                with self.assertRaisesRegex(RuntimeError, 'Cannot audit live process'):
                    deploy.audit_no_batchgen(Path('/example'), proc)

    def test_other_uid_permission_limits_are_reported_without_reading_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            pid = proc / '12345678'
            pid.mkdir()
            (pid / 'stat').write_text('12345678 (python) ' + ' '.join(['S'] + ['0'] * 18 + ['99']))
            outside_uid = os.geteuid() + 10000
            (pid / 'status').write_text(f'Uid:\t{outside_uid}\t{outside_uid}\t{outside_uid}\t{outside_uid}\n')
            original = Path.open
            def open_path(path, *args, **kwargs):
                if path == pid / 'cmdline':
                    raise AssertionError('Other UID command must not be inspected')
                if path == pid / 'maps':
                    raise PermissionError(13, 'Permission denied')
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, 'open', open_path):
                report = deploy.audit_no_batchgen(Path('/example'), proc)
            self.assertTrue(report['same_uid_audit_complete'])
            self.assertFalse(report['namespace_complete'])
            self.assertEqual(report['processes_scanned'], 0)
            self.assertEqual(report['uninspected_other_uid'][0]['effective_uid'], outside_uid)
            self.assertEqual(report['uninspected_other_uid'][0]['map_access'], 'permission_denied')


if __name__ == '__main__':
    unittest.main()
