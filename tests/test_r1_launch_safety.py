"""CPU-only own-process core filtering and local quota-preflight contracts."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/r1_launch_safety.py'
spec = importlib.util.spec_from_file_location('r1_launch_safety_tested', SOURCE)
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def filesystem(bytes_available=64 * 1024**3, inodes=999999):
    return SimpleNamespace(f_bavail=bytes_available // 4096, f_frsize=4096, f_favail=inodes)


class CorePolicyTests(unittest.TestCase):
    def fake_files(self, root):
        own = root/'own-filter'; own.write_text('00000033\n')
        pattern = root/'pattern'; pattern.write_text('|/platform/collector %p\n')
        return own, pattern

    def test_only_own_filter_written_and_cpu_exec_reads_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            own, pattern = self.fake_files(Path(tmp))
            inherited = {'pid': os.getpid() + 1000000, 'coredump_filter': '00000000', 'rlimit_core_soft': 0}
            with patch.object(safety, 'SELF_FILTER', own), patch.object(safety, 'CORE_PATTERN', pattern), \
                 patch.object(safety.resource, 'getrlimit', side_effect=[(1234, 5678), (0, 5678)]), \
                 patch.object(safety.resource, 'setrlimit') as limits, \
                 patch.object(safety.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(inherited))) as run:
                record = safety.configure_self_core_policy()
            self.assertEqual(own.read_text(), '0x0\n')
            self.assertEqual(pattern.read_text(), '|/platform/collector %p\n')
            limits.assert_called_once_with(resource.RLIMIT_CORE, (0, 5678))
            self.assertEqual(run.call_args.args[0], [sys.executable, '-I', '-S', '-c', safety.INHERITANCE_PROBE])
            self.assertEqual(run.call_args.kwargs['timeout'], 10)
            self.assertEqual(record['cpu_exec_probe'], inherited)
            self.assertTrue(record['core_pattern_is_pipe'])
            self.assertFalse(record['hard_core_byte_limit_guaranteed'])
            self.assertFalse(record['host_sysctl_modified'])

    def test_write_denied_readback_drift_or_child_noninheritance_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            own, pattern = self.fake_files(Path(tmp))
            fake = SimpleNamespace(read_text=lambda: '33',
                write_text=lambda _: (_ for _ in ()).throw(PermissionError(errno.EACCES, 'denied')))
            with patch.object(safety, 'SELF_FILTER', fake), patch.object(safety, 'CORE_PATTERN', pattern), \
                 patch.object(safety.resource, 'getrlimit', return_value=(123, 456)), \
                 patch.object(safety.subprocess, 'run') as run:
                with self.assertRaises(safety.LaunchSafetyError) as raised:
                    safety.configure_self_core_policy()
                self.assertEqual(raised.exception.record['errno'], errno.EACCES)
                run.assert_not_called()
            fake = SimpleNamespace(read_text=lambda: '33', write_text=lambda _: None)
            with patch.object(safety, 'SELF_FILTER', fake), patch.object(safety, 'CORE_PATTERN', pattern), \
                 patch.object(safety.resource, 'getrlimit', return_value=(0, 456)), \
                 patch.object(safety.resource, 'setrlimit'), patch.object(safety.subprocess, 'run') as run:
                with self.assertRaisesRegex(safety.LaunchSafetyError, 'changed before model launch'):
                    safety.configure_self_core_policy()
                run.assert_not_called()
            for inherited in ({'pid': 99999, 'coredump_filter': '33', 'rlimit_core_soft': 0},
                              {'pid': 99999, 'coredump_filter': '0', 'rlimit_core_soft': 123},
                              {'pid': os.getpid(), 'coredump_filter': '0', 'rlimit_core_soft': 0}):
                with self.subTest(inherited=inherited), patch.object(safety, 'SELF_FILTER', own), \
                     patch.object(safety, 'CORE_PATTERN', pattern), \
                     patch.object(safety.resource, 'getrlimit', return_value=(0, 456)), \
                     patch.object(safety.resource, 'setrlimit'), \
                     patch.object(safety.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(inherited))):
                    with self.assertRaisesRegex(safety.LaunchSafetyError, 'did not inherit'):
                        safety.configure_self_core_policy()

    def test_pre_popen_assertion_is_read_only_and_rejects_rlimit_drift(self):
        own = SimpleNamespace(read_text=lambda: '00000000', write_text=lambda _: self.fail('Read-only assertion wrote filter'))
        with patch.object(safety, 'SELF_FILTER', own), patch.object(safety.resource, 'getrlimit', return_value=(0, -1)):
            self.assertEqual(safety.assert_self_core_policy()['rlimit_core_soft'], 0)
        with patch.object(safety, 'SELF_FILTER', own), patch.object(safety.resource, 'getrlimit', return_value=(1, -1)):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                safety.assert_self_core_policy()

    def test_missing_or_malformed_proc_file_is_not_silently_accepted(self):
        for value in ('', '-1', 'not-a-mask', '0xZZ'):
            own = SimpleNamespace(read_text=lambda: value)
            with self.subTest(value=value), patch.object(safety, 'SELF_FILTER', own):
                with self.assertRaises(safety.LaunchSafetyError):
                    safety.configure_self_core_policy()


class LocalDiskTests(unittest.TestCase):
    def test_real_small_write_fsync_cleanup_does_not_touch_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); preserved=root/'existing';preserved.write_bytes(b'keep existing bytes')
            with patch.object(safety.os, 'statvfs', return_value=filesystem()):
                record=safety.preflight_local_disk(root)
            self.assertEqual(record['probe_written_bytes'], 1024**2)
            self.assertTrue(record['probe_removed'])
            self.assertFalse(record['project_quota_headroom_proven'])
            self.assertEqual(list(root.iterdir()), [preserved])
            self.assertEqual(preserved.read_bytes(), b'keep existing bytes')

    def test_edquot_despite_plentiful_statvfs_is_rejected_and_private_probe_removed(self):
        for operation in ('write', 'fsync'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                with patch.object(safety.os,'statvfs',return_value=filesystem()), \
                     patch.object(safety.os,operation,side_effect=OSError(errno.EDQUOT,'project quota exceeded')):
                    with self.assertRaises(safety.LaunchSafetyError) as raised:
                        safety.preflight_local_disk(root)
                self.assertEqual(raised.exception.record['errno'], errno.EDQUOT)
                self.assertEqual(raised.exception.record['status'], 'failed_NOT_admitted')
                self.assertTrue(raised.exception.record['probe_removed'])
                self.assertEqual(list(root.iterdir()), [])

    def test_filesystem_low_bytes_or_inodes_rejected_before_write(self):
        for available in (filesystem(bytes_available=1024),filesystem(inodes=0)):
            with tempfile.TemporaryDirectory() as tmp, patch.object(safety.os,'statvfs',return_value=available), \
                 patch.object(safety.tempfile,'mkstemp') as create:
                with self.assertRaisesRegex(safety.LaunchSafetyError,'byte/inode headroom'):
                    safety.preflight_local_disk(Path(tmp))
                create.assert_not_called()

    def test_fuse_and_resolved_fuse_are_rejected_before_statfs_or_writes(self):
        with patch.object(safety.os,'statvfs') as statfs, patch.object(safety.tempfile,'mkstemp') as create:
            with self.assertRaisesRegex(safety.LaunchSafetyError,'Never statfs'):
                safety.preflight_local_disk(Path('/mnt/fuse/oss/must-not-be-probed'))
            statfs.assert_not_called();create.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp, patch.object(safety.Path,'resolve',return_value=Path('/mnt/fuse/oss/alias')), \
             patch.object(safety.os,'statvfs') as statfs:
            with self.assertRaisesRegex(safety.LaunchSafetyError,'resolves into'):
                safety.preflight_local_disk(Path(tmp))
            statfs.assert_not_called()

    def test_combined_receipt_keeps_failed_check_and_never_admits(self):
        failure={'status':'failed_NOT_admitted','errno':errno.EDQUOT}
        with patch.object(safety,'configure_self_core_policy',return_value={'status':'passed'}), \
             patch.object(safety,'preflight_local_disk',side_effect=safety.LaunchSafetyError('quota',failure)):
            with self.assertRaises(safety.LaunchSafetyError) as raised:
                safety.prepare_launch_safety(Path('/a/local/output'))
        self.assertEqual(raised.exception.record['status'],'failed_NOT_admitted')
        self.assertEqual(raised.exception.record['failed_check'],failure)
        self.assertEqual(raised.exception.record['core_policy']['status'],'passed')


@unittest.skipUnless(Path('/proc/self/coredump_filter').is_file(), 'Real inheritance check requires Linux; macOS runs mocks/local disk tests')
class LinuxInheritanceTests(unittest.TestCase):
    def test_real_cpu_child_and_grandchild_inheritance_do_not_modify_test_parent(self):
        before=Path('/proc/self/coredump_filter').read_text()
        pattern=Path('/proc/sys/kernel/core_pattern').read_text()
        limits=resource.getrlimit(resource.RLIMIT_CORE)
        code=(f'import sys,json;sys.path.insert(0,{str(SOURCE.parent)!r});'
              'import r1_launch_safety as s;print(json.dumps(s.configure_self_core_policy()))')
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none'),timeout=15,check=True)
        record=json.loads(result.stdout)
        self.assertEqual(int(record['self_readback']['coredump_filter_observed'],16),0)
        self.assertEqual(int(record['cpu_exec_probe']['coredump_filter'],16),0)
        self.assertEqual(record['cpu_exec_probe']['rlimit_core_soft'],0)
        self.assertEqual(Path('/proc/self/coredump_filter').read_text(),before)
        self.assertEqual(Path('/proc/sys/kernel/core_pattern').read_text(),pattern)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CORE),limits)


if __name__=='__main__':
    unittest.main()
