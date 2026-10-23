"""CPU ownership/cleanup regressions, with Linux local-process integration."""
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/r1_launch_process.py'
spec = importlib.util.spec_from_file_location('r1_launch_process_under_test', SOURCE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def identity(pid=54321, state='Z', pgid=None):
    return {'pid': pid, 'ppid': os.getpid(), 'pgid': pid if pgid is None else pgid,
            'session': 54321, 'start_ticks': pid + 1234, 'state': state}


class FakeChild:
    def __init__(self, events):
        self.pid = 54321
        self.returncode = None
        self.events = events

    def wait(self, timeout):
        self.events.append('reap')
        self.returncode = 0
        return 0


class MockedOwnershipTests(unittest.TestCase):
    def mock_environment(self, stack, child, current, observe):
        stack.enter_context(patch.object(launch.Path, 'is_file', return_value=True))
        stack.enter_context(patch.object(launch, 'process_identity', return_value=current))
        stack.enter_context(patch.object(launch, 'assert_owned', return_value=current))
        stack.enter_context(patch.object(launch, 'child_exited', return_value=True))
        stack.enter_context(patch.object(launch, 'snapshot', side_effect=observe))

    def test_normal_exit_audit_then_reap_without_signals(self):
        from contextlib import ExitStack
        events = []; child = FakeChild(events)
        state = {'leader_exited_unreaped': True, 'primary_group_live': [],
                 'other_group_live_audit_only': [], 'session_members_including_zombies': [identity()]}
        def observe(*_):
            events.append('audit')
            self.assertIsNone(child.returncode)
            return state
        with ExitStack() as stack:
            self.mock_environment(stack, child, identity(), observe)
            kill = stack.enter_context(patch.object(launch.os, 'kill', side_effect=AssertionError('Unexpected signal')))
            killpg = stack.enter_context(patch.object(launch.os, 'killpg', side_effect=AssertionError('Unexpected group signal')))
            record = launch.supervise_child(child)
        self.assertEqual(record['status'], 'exited')
        self.assertTrue(record['child_reaped'])
        self.assertEqual(events[-1], 'reap')
        self.assertIn('audit', events[:-1])
        kill.assert_not_called(); killpg.assert_not_called()

    def test_other_pgid_is_only_audited_and_requires_parent(self):
        from contextlib import ExitStack
        events = []; child = FakeChild(events)
        other = identity(55555, 'S', pgid=55555)
        state = {'leader_exited_unreaped': True, 'primary_group_live': [],
                 'other_group_live_audit_only': [other], 'session_members_including_zombies': [identity(), other]}
        with ExitStack() as stack:
            self.mock_environment(stack, child, identity(), lambda *_: state)
            kill = stack.enter_context(patch.object(launch.os, 'kill', side_effect=AssertionError('Wrong PID signal')))
            killpg = stack.enter_context(patch.object(launch.os, 'killpg', side_effect=AssertionError('Wrong PGID signal')))
            record = launch.supervise_child(child)
        self.assertTrue(record['parent_cleanup_required'])
        self.assertEqual(record['pre_reap_snapshot']['other_group_live_audit_only'], [other])
        self.assertEqual(events, ['reap'])
        kill.assert_not_called(); killpg.assert_not_called()

    def test_stop_once_leader_then_primary_escalation_shields_repeat_signals(self):
        from contextlib import ExitStack
        events = []; child = FakeChild(events)
        dead = [False]
        def observe(*_):
            events.append('audit')
            self.assertIsNone(child.returncode)
            return {'leader_exited_unreaped': dead[0],
                    'primary_group_live': [] if dead[0] else [identity(state='S')],
                    'other_group_live_audit_only': [], 'session_members_including_zombies': []}
        def update(record):
            if record['status'] == 'waiting':
                handler = signal.getsignal(signal.SIGTERM)
                handler(signal.SIGTERM, None)
                handler(signal.SIGINT, None)
            elif record['status'] == 'cleaning':
                self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
                self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        def leader(*_):
            events.append('leader TERM')
            return True
        def group(_child, _identity, sig):
            events.append('group ' + str(int(sig)))
            if sig == signal.SIGKILL:
                dead[0] = True
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with ExitStack() as stack:
            self.mock_environment(stack, child, identity(state='S'), observe)
            stack.enter_context(patch.object(launch, 'child_exited', return_value=False))
            stack.enter_context(patch.object(launch, 'wait_for_release', side_effect=lambda *_: observe()))
            stack.enter_context(patch.object(launch, 'signal_leader', side_effect=leader))
            stack.enter_context(patch.object(launch, 'signal_primary_group', side_effect=group))
            record = launch.supervise_child(child, on_update=update)
        self.assertEqual(record['requested_stop_signal'], signal.SIGTERM)
        self.assertEqual(record['signals'], [{'target': 'leader', 'signal': int(signal.SIGTERM)},
            {'target': 'primary_group', 'signal': int(signal.SIGTERM)},
            {'target': 'primary_group', 'signal': int(signal.SIGKILL)}])
        self.assertEqual(events.count('leader TERM'), 1)
        self.assertEqual(events[-1], 'reap')
        self.assertEqual(record['release_grace_seconds'], 300)
        self.assertFalse(record['parent_cleanup_required'])
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})

    def test_identity_change_or_preexisting_reap_refuses_numeric_signals(self):
        child = FakeChild([]); saved = identity(state='S')
        for changed in (dict(saved, start_ticks=saved['start_ticks'] + 1), dict(saved, pgid=999)):
            with patch.object(launch, 'process_identity', return_value=changed), patch.object(launch.os, 'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError, 'identity no longer matches'):
                    launch.signal_primary_group(child, saved, signal.SIGTERM)
                kill.assert_not_called()
        child.returncode = 0
        with patch.object(launch.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'reaped'):
                launch.signal_leader(child, saved, signal.SIGTERM)
            kill.assert_not_called()

    def test_callback_error_cannot_prevent_reap_and_cleanup_failure_is_explicit(self):
        from contextlib import ExitStack
        events = []; child = FakeChild(events)
        state = {'leader_exited_unreaped': True, 'primary_group_live': [],
                 'other_group_live_audit_only': [], 'session_members_including_zombies': []}
        with ExitStack() as stack:
            self.mock_environment(stack, child, identity(), lambda *_: state)
            record = launch.supervise_child(child, on_update=lambda _: (_ for _ in ()).throw(OSError('disk full')))
        self.assertTrue(record['child_reaped'])
        self.assertTrue(record['callback_errors'])
        events = []; child = FakeChild(events)
        with ExitStack() as stack:
            self.mock_environment(stack, child, identity(), lambda *_: (_ for _ in ()).throw(RuntimeError('lost ownership')))
            killpg = stack.enter_context(patch.object(launch.os, 'killpg'))
            record = launch.supervise_child(child)
        self.assertTrue(record['parent_cleanup_required'])
        self.assertIn('lost ownership', record['cleanup_error'])
        self.assertEqual(events, [])
        killpg.assert_not_called()

    def test_short_native_release_grace_is_rejected(self):
        with patch.object(launch.Path, 'is_file', return_value=True):
            with self.assertRaisesRegex(ValueError, 'at least 180'):
                launch.supervise_child(FakeChild([]), release_grace_seconds=30)


LINUX = Path('/proc/self/stat').is_file()


@unittest.skipUnless(LINUX, 'Real local-process integration requires Linux /proc; mocks run on macOS')
class LinuxProcessTests(unittest.TestCase):
    def spawn(self, script):
        return subprocess.Popen([sys.executable, '-u', '-c', script], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)

    def emergency_cleanup(self, child):
        # Test-only emergency cleanup still pins/rechecks the original unreaped
        # child. Never kill a numeric group once supervise_child has reaped it.
        if child.returncode is None:
            saved = launch.process_identity(child.pid)
            if saved is not None:
                try:
                    launch.assert_owned(child, saved)
                    os.killpg(child.pid, signal.SIGKILL)
                except (RuntimeError, ProcessLookupError, ChildProcessError):
                    pass
            child.wait(timeout=10)
        if child.stdout:
            child.stdout.close()
        if child.stderr:
            child.stderr.close()

    def test_real_normal_exit_code_and_no_signal(self):
        child = self.spawn('raise SystemExit(7)')
        try:
            record = launch.supervise_child(child, poll_seconds=0.01)
            self.assertEqual(record['exit_code'], 7)
            self.assertEqual(record['signals'], [])
            self.assertTrue(record['child_reaped'])
            self.assertFalse(record['parent_cleanup_required'])
        finally:
            self.emergency_cleanup(child)

    def test_real_leader_exits_before_primary_descendant_then_safe_group_cleanup(self):
        child = self.spawn("import subprocess,sys\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\nprint(p.pid,flush=True)")
        descendant = int(child.stdout.readline())
        actual_wait = launch.wait_for_release
        calls = []
        def fast_release(proc, saved, seconds, interval, update):
            calls.append(seconds)
            # Speed up only the test's first release wait; production admission
            # still enforces >=180 and records the default 300 seconds.
            return actual_wait(proc, saved, 0 if len(calls) == 1 else seconds, interval, update)
        try:
            with patch.object(launch, 'wait_for_release', side_effect=fast_release):
                record = launch.supervise_child(child, group_term_grace_seconds=2,
                                               kill_release_grace_seconds=2, poll_seconds=0.01)
            self.assertEqual(calls[0], 300)
            self.assertTrue(record['cleanup_entry_snapshot']['leader_exited_unreaped'])
            self.assertIn(descendant, [p['pid'] for p in record['cleanup_entry_snapshot']['primary_group_live']])
            self.assertEqual(record['signals'][0], {'target': 'primary_group', 'signal': int(signal.SIGTERM)})
            self.assertTrue(record['child_reaped'])
            self.assertFalse(record['pre_reap_snapshot']['primary_group_live'])
            self.assertFalse(record['parent_cleanup_required'])
        finally:
            self.emergency_cleanup(child)

    def test_real_repeated_parent_stop_delivers_only_one_leader_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = Path(tmp) / 'terms'
            script = ("import signal,time\nfrom pathlib import Path\n"
                f"p=Path({str(count)!r})\n"
                "def stop(sig,frame):\n p.write_text(p.read_text()+'TERM\\n' if p.exists() else 'TERM\\n')\n time.sleep(.15)\n raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM,stop)\nprint('ready',flush=True)\nwhile True: time.sleep(.1)")
            child = self.spawn(script)
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            sent = set()
            def update(record):
                if record['status'] == 'waiting' and 'first' not in sent:
                    sent.add('first'); os.kill(os.getpid(), signal.SIGTERM)
                if record['status'] == 'cleaning' and 'repeat' not in sent:
                    sent.add('repeat')
                    os.kill(os.getpid(), signal.SIGTERM)
                    os.kill(os.getpid(), signal.SIGINT)
            try:
                record = launch.supervise_child(child, poll_seconds=.01, on_update=update)
                self.assertEqual(count.read_text(), 'TERM\n')
                self.assertEqual(record['signals'], [{'target': 'leader', 'signal': int(signal.SIGTERM)}])
                self.assertTrue(record['child_reaped'])
                self.assertFalse(record['parent_cleanup_required'])
            finally:
                self.emergency_cleanup(child)

    def test_real_other_pgid_survives_and_is_audit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / 'owned-test-stop'
            descendant_script = ("import time\nfrom pathlib import Path\n"
                f"p=Path({str(stop)!r})\ndeadline=time.monotonic()+10\n"
                "while not p.exists() and time.monotonic()<deadline: time.sleep(.02)")
            child = self.spawn("import subprocess,sys,os\n"
                f"p=subprocess.Popen([sys.executable,'-c',{descendant_script!r}],preexec_fn=os.setpgrp)\n"
                "print(p.pid,flush=True)")
            other = int(child.stdout.readline())
            try:
                record = launch.supervise_child(child, poll_seconds=.01)
                self.assertTrue(record['child_reaped'])
                self.assertEqual(record['signals'], [])
                self.assertTrue(record['parent_cleanup_required'])
                self.assertIn(other, [p['pid'] for p in record['pre_reap_snapshot']['other_group_live_audit_only']])
                self.assertNotIn(launch.process_identity(other)['state'], ('Z', 'X'))
            finally:
                # Cooperatively stop the test-owned process via its unique file;
                # no numeric PID/PGID signal is needed after leader reap.
                stop.touch()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    current = launch.process_identity(other)
                    if current is None or current['state'] in ('Z', 'X'):
                        break
                    time.sleep(.02)
                else:
                    self.fail('Test-owned independent PGID did not exit cooperatively')
                self.emergency_cleanup(child)


if __name__ == '__main__':
    unittest.main()
