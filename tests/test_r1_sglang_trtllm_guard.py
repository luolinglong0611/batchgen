"""Exercise resource guard decisions and owned-process cleanup without a GPU."""
import contextlib
import importlib.util
import io
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest import mock

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/r1_reproduction'
sys.path.insert(0,str(SCRIPTS))
try:
    spec=importlib.util.spec_from_file_location('sg_guard',SCRIPTS/'run_sglang_trtllm_guard.py')
    guard=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
finally:
    sys.path.pop(0)


class GuardTest(unittest.TestCase):
    def test_default_plan_never_inspects_gpu_or_spawns(self):
        with mock.patch.object(sys,'argv',['guard','--node-rank','0','--stage','mla','--run-id','cpu-unit-plan']):
            with mock.patch.object(guard,'gpu_apps',side_effect=AssertionError('GPU query forbidden')):
                with mock.patch.object(guard.subprocess,'Popen',side_effect=AssertionError('Spawn forbidden')):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(guard.main(),0)

    def test_each_limit_is_fail_closed(self):
        sample={'MemAvailable_bytes':128*1024**3,'session_RSS_bytes':1024**3,
                'target_GPU_bytes':1024**3,'unexpected_GPU_pids':[]}
        self.assertIsNone(guard.violation(sample,1))
        self.assertEqual(guard.violation(sample,901),'wall_time_limit')
        for key,value,reason in [
            ('MemAvailable_bytes',0,'host_available_memory_limit'),
            ('session_RSS_bytes',8*1024**3,'session_RSS_limit'),
            ('target_GPU_bytes',4*1024**3,'target_GPU_memory_limit'),
            ('unexpected_GPU_pids',[999],'unexpected_process_on_assigned_GPU'),
        ]:
            self.assertEqual(guard.violation({**sample,key:value},1),reason)

    def setUp(self):
        patcher = mock.patch.object(guard, 'session_processes', return_value=[])
        self.session_scan = patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def row(pid=123, ticks=100, pgid=123):
        return dict(pid=pid, pgid=pgid, session=pgid, start_ticks=ticks,
                    state='S', RSS_bytes=4096)

    def test_reused_leader_and_group_are_never_signalled(self):
        old, reused = self.row(), self.row(ticks=999)
        proc = mock.Mock(pid=123)
        with mock.patch.object(guard, 'process_identity', return_value=reused), \
             mock.patch.object(guard, 'group_processes', return_value=[reused]), \
             mock.patch.object(guard.os, 'killpg') as send:
            audit = guard.stop_own_group(proc, old, grace_seconds=0)
        send.assert_not_called()
        proc.wait.assert_not_called()
        self.assertEqual(audit['unverified_group_members'], [reused])

    def test_compiler_after_leader_exit_cleaned_before_reaping(self):
        leader, compiler = {**self.row(), 'state': 'Z'}, self.row(pid=124)
        alive = [compiler]
        proc = mock.Mock(pid=123)
        events = []
        def send(pgid, sig):
            events.append(('signal', pgid, sig))
            alive.clear()
        proc.wait.side_effect = lambda **kw: events.append(('wait',)) or 0
        with mock.patch.object(guard, 'process_identity', return_value=leader), \
             mock.patch.object(guard, 'group_processes', side_effect=lambda pg: list(alive)), \
             mock.patch.object(guard.os, 'killpg', side_effect=send):
            audit = guard.stop_own_group(proc, self.row(), grace_seconds=0)
        self.assertEqual(events, [('signal', 123, signal.SIGTERM), ('wait',)])
        self.assertEqual(audit['remaining_owned'], [])

    def test_missing_leader_means_orphan_audit_only(self):
        orphan = self.row(pid=124)
        with mock.patch.object(guard, 'process_identity', return_value=None), \
             mock.patch.object(guard, 'group_processes', return_value=[orphan]), \
             mock.patch.object(guard.os, 'killpg') as send:
            audit = guard.stop_own_group(mock.Mock(pid=123), self.row(), grace_seconds=0)
        send.assert_not_called()
        self.assertEqual(audit['unverified_group_members'], [orphan])

    def test_only_live_group_escalates_and_never_signals_after_wait(self):
        row = self.row()
        alive = [row]
        events = []
        proc = mock.Mock(pid=123)
        proc.wait.side_effect = lambda **kw: events.append('wait') or -9
        def send(pgid, sig):
            events.append(sig)
            if sig == signal.SIGKILL: alive.clear()
        with mock.patch.object(guard, 'process_identity', return_value=row), \
             mock.patch.object(guard, 'group_processes', side_effect=lambda pg: list(alive)), \
             mock.patch.object(guard.os, 'killpg', side_effect=send):
            guard.stop_own_group(proc, row, grace_seconds=0)
        self.assertEqual(events, [signal.SIGTERM, signal.SIGKILL, 'wait'])

    def test_identity_rechecked_immediately_before_signal(self):
        with mock.patch.object(guard, 'process_identity', return_value=self.row(ticks=999)), \
             mock.patch.object(guard.os, 'killpg') as send:
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                guard.signal_owned_group(self.row(), signal.SIGTERM)
        send.assert_not_called()

    def test_exit_detection_uses_wnowait_not_poll_or_wait(self):
        proc = mock.Mock(pid=123)
        with mock.patch.object(guard.os, 'waitid', return_value=object()):
            self.assertTrue(guard.child_exited(proc))
        proc.poll.assert_not_called()
        proc.wait.assert_not_called()
        import ast
        tree = ast.parse((SCRIPTS/'run_sglang_trtllm_guard.py').read_text())
        attr_calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('poll', attr_calls)
        self.assertEqual(attr_calls.count('wait'), 1)

    def test_separate_compiler_group_remains_explicit_audit(self):
        compiler = {**self.row(pid=124), 'pgid': 124}
        self.session_scan.return_value = [compiler]
        with mock.patch.object(guard, 'process_identity', return_value=self.row()), \
             mock.patch.object(guard, 'group_processes', return_value=[]), \
             mock.patch.object(guard.os, 'killpg') as send:
            audit = guard.stop_own_group(mock.Mock(pid=123), self.row(), grace_seconds=0)
        send.assert_not_called()
        self.assertEqual(audit['remaining_owned'], [])
        self.assertEqual(audit['remaining_session_members'], [compiler])

    def test_gpu_query_has_five_second_deadline(self):
        with mock.patch.object(guard.subprocess, 'run', return_value=mock.Mock(stdout='')) as run:
            self.assertEqual(guard.gpu_apps(1), [])
        self.assertEqual(run.call_args.kwargs['timeout'], 5)


if __name__=='__main__':
    unittest.main()
