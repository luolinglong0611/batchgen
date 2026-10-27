"""CPU lifecycle tests for bounded local process launch and rank closure."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/run_sglang_r1_shared_probe.py'
SPEC = importlib.util.spec_from_file_location('shared_probe_launcher_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
LINUX_PIDFD = M.sys.platform == 'linux' and callable(getattr(M.os, 'pidfd_open', None)) and callable(getattr(M.signal, 'pidfd_send_signal', None))


def process(pid, ppid, start, sid=None, state='S'):
    return {'pid': pid, 'ppid': ppid, 'start_ticks': start, 'sid': sid or pid,
            'pgrp': sid or pid, 'state': state}


class BoundedSharedProbeLifecycle(unittest.TestCase):
    def test_default_launch_plan_starts_no_process_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(nnodes=2, nproc_per_node=8, node_rank=0,
                wall_seconds=840, probe_args=['--', '--weights-sha256=explicit'], python=Path(sys.executable),
                master_addr='example-owner-address', master_port=43121, job_dir=Path(directory) / 'not-created',
                execute=False)
            with patch.object(M.subprocess, 'Popen') as process:
                result = M.launch(args)
            process.assert_not_called()
            self.assertEqual(result['status'], 'plan_only_NOT_executed')
            self.assertFalse(args.job_dir.exists())
            args.probe_args = ['--execute']
            with self.assertRaisesRegex(ValueError, 'owned by this bounded launcher'):
                M.launch(args)

    @unittest.skipUnless(LINUX_PIDFD, 'Real guarded execution requires Linux /proc + pidfd')
    def test_bounded_owned_child_success_closes_log_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'child.log'
            result = M.run_bounded([sys.executable, '-c', 'print("closed")'], directory, log, 5)
            self.assertTrue(result['closed'])
            self.assertEqual(result['returncode'], 0)
            self.assertFalse(result['timed_out'])
            self.assertTrue(result['clean_exit'])
            self.assertTrue(result['all_owned_identities_absent'])
            self.assertGreaterEqual(result['stable_empty_scans'], 2)
            self.assertEqual(log.read_text().strip(), 'closed')

    @unittest.skipUnless(LINUX_PIDFD, 'Real guarded execution requires Linux /proc + pidfd')
    def test_bounded_owned_child_timeout_is_closed_and_never_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = M.run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   directory, Path(directory) / 'child.log', 0.05, 1)
            self.assertTrue(result['closed'])
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'], 0)
            self.assertIn('SIGTERM', result['signals'])
            self.assertFalse(result['clean_exit'])

    @unittest.skipUnless(LINUX_PIDFD, 'Real guarded execution requires Linux /proc + pidfd')
    def test_new_session_grandchild_is_adopted_signalled_by_identity_and_closed(self):
        code = ('import subprocess,sys; subprocess.Popen([sys.executable,"-c",'
                '"import time; time.sleep(30)"],start_new_session=True)')
        with tempfile.TemporaryDirectory() as directory:
            result = M.run_bounded([sys.executable, '-c', code], directory,
                                   Path(directory) / 'child.log', 0.3, 0.3)
        self.assertTrue(result['closed'])
        self.assertFalse(result['clean_exit'])
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['new_session_descendants'])
        child_ids = {(p['pid'], p['start_ticks']) for p in result['new_session_descendants']}
        self.assertTrue(any((p['pid'], p['start_ticks']) in child_ids for p in result['signal_records']))

    def test_unsupported_platform_fails_before_any_process_or_log(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(M.sys, 'platform', 'darwin'), patch.object(M.subprocess, 'Popen') as popen:
            log = Path(directory) / 'never-created'
            with self.assertRaisesRegex(RuntimeError, 'no killpg fallback'):
                M.run_bounded([sys.executable, '-c', 'print(1)'], directory, log, 1)
            popen.assert_not_called()
            self.assertFalse(log.exists())

    def test_proc_stat_comm_may_contain_spaces_and_parentheses(self):
        fields = ['S', '41', '42', '43'] + ['0'] * 15 + ['987654'] + ['0'] * 4
        record = M.parse_proc_stat(44, '44 (name (with) spaces) ' + ' '.join(fields))
        self.assertEqual((record['pid'], record['ppid'], record['sid'], record['start_ticks']), (44, 41, 43, 987654))

    def test_new_SID_and_subreaper_orphan_are_owned_but_reused_PID_is_not(self):
        known = {(10, 100): process(10, 1, 100), (12, 120): process(12, 10, 120)}
        snapshot = {10: process(10, 1, 999), 12: process(12, 10, 120, sid=55),
                    13: process(13, 12, 130, sid=56), 14: process(14, 1, 140, sid=57),
                    15: process(15, 10, 150), 99: process(99, 98, 990, sid=55)}
        owned = M.discover_owned(snapshot, known, 1)
        self.assertEqual(set(owned), {(10, 999), (12, 120), (13, 130), (14, 140), (15, 150)})
        # An exclusive subreaper owns all new direct children, including PID10
        # in that synthetic case. Reuse under an unrelated parent is excluded.
        snapshot[10]['ppid'] = 98
        owned = M.discover_owned(snapshot, known, 1)
        self.assertEqual(set(owned), {(12, 120), (13, 130), (14, 140)})

    def test_pidfd_signal_requires_fresh_matching_start_ticks(self):
        tree = M.OwnedTree(1)
        tree.known[(20, 200)] = process(20, 1, 200)
        tree.fds[(20, 200)] = 321
        with patch.object(M, 'read_proc', return_value=process(20, 98, 999)), patch.object(M.signal, 'pidfd_send_signal', create=True) as send:
            self.assertFalse(tree.send((20, 200), M.signal.SIGTERM, {}))
            send.assert_not_called()
        with patch.object(M, 'read_proc', return_value=process(20, 1, 200)), patch.object(M.signal, 'pidfd_send_signal', create=True) as send:
            self.assertTrue(tree.send((20, 200), M.signal.SIGTERM, {}))
            send.assert_called_once_with(321, M.signal.SIGTERM, None, 0)

    def test_existing_child_prevents_ambiguous_ownership_before_Popen(self):
        with patch.object(M, 'require_linux_tracking'), patch.object(M, 'proc_snapshot', return_value={9: process(9, M.os.getpid(), 90)}), patch.object(M.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'pre-existing direct children'):
                M.run_bounded(['unused'], '.', 'unused.log', 1)
            popen.assert_not_called()

    def test_synthetic_success_requires_reaped_leader_and_two_complete_empty_scans(self):
        owner, pid = M.os.getpid(), 4242
        first = process(pid, owner, 424200)
        zombie = dict(first, state='Z')
        child = SimpleNamespace(pid=pid, poll=lambda: 0)
        snapshots = [{}, {pid: zombie}, {}, {}]
        with tempfile.TemporaryDirectory() as directory, patch.object(M, 'require_linux_tracking'), patch.object(M, 'subreaper_mode', return_value=0), patch.object(M, 'proc_snapshot', side_effect=snapshots), patch.object(M, 'read_proc', return_value=first), patch.object(M.os, 'pidfd_open', return_value=777, create=True), patch.object(M.os, 'close') as closed, patch.object(M.subprocess, 'Popen', return_value=child), patch.object(M.time, 'sleep'):
            result = M.run_bounded(['synthetic-child'], directory, Path(directory) / 'out.log', 2)
        self.assertTrue(result['closed'])
        self.assertTrue(result['clean_exit'])
        self.assertEqual(result['stable_empty_scans'], 2)
        self.assertEqual(result['tracking_method'], M.TRACKING_METHOD)
        closed.assert_called_once_with(777)

    def test_unknown_process_scan_error_can_never_claim_closed(self):
        owner, pid = M.os.getpid(), 4242
        first = process(pid, owner, 424200)
        child = SimpleNamespace(pid=pid, poll=lambda: 0)
        ticks = iter(range(100))
        with tempfile.TemporaryDirectory() as directory, patch.object(M, 'require_linux_tracking'), patch.object(M, 'subreaper_mode', return_value=0), patch.object(M, 'proc_snapshot', side_effect=[{}, PermissionError('unobservable ancestry'), PermissionError('still unobservable')]), patch.object(M, 'read_proc', return_value=first), patch.object(M.os, 'pidfd_open', return_value=777, create=True), patch.object(M.signal, 'pidfd_send_signal', create=True), patch.object(M.os, 'close'), patch.object(M.subprocess, 'Popen', return_value=child), patch.object(M.time, 'sleep'), patch.object(M.time, 'monotonic', side_effect=lambda: next(ticks)):
            result = M.run_bounded(['synthetic-child'], directory, Path(directory) / 'out.log', 1, 0.01)
        self.assertFalse(result['closed'])
        self.assertFalse(result['clean_exit'])
        self.assertTrue(result['closure_errors'])

    def test_nonfinite_timeout_cannot_launch_an_unbounded_child(self):
        for wall in (float('inf'), float('nan'), -1):
            with self.subTest(wall=wall), patch.object(M.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'Finite wall'):
                    M.run_bounded(['unused'], '.', 'unused.log', wall)
                popen.assert_not_called()

    def fake_closed_nodes(self, root):
        probe = M.probe_module()
        directories = []
        for node in (0, 1):
            folder = root / str(node)
            (folder / 'ranks').mkdir(parents=True)
            hashes = {}
            expected = list(range(node * 8, (node + 1) * 8))
            for rank in expected:
                record = {'rank': rank, 'world_size': 16,
                    'status': 'passed_real_shared_branch_NOT_full_model_qualified',
                    'source_sha256': M.sha(M.PROBE), 'tp_members': list(range(16)),
                    'attention_tp_size': 1, 'attention_dp_size': 16,
                    'expert_parallel_size': 16, 'actual_shared_tp_size': 1,
                    'numerical_contract': None, 'numerical_contract_sha256': None,
                    'sources': {'fake_fixture': {'sha256': 'same-test-source'}},
                    'weights': {'fake_fixture': 'same-test-tensors'},
                    'cases': [{'case': name, 'padding': mode, 'routed_SUM_calls': 1,
                        'local_shared_calls': 1, 'shared_added_once_bitwise': True,
                        'separate_shared_oracle_bitwise': True, 'native_gather_matches_independent_rows': True}
                        for name, _ in probe.case_counts(16) for mode in ('SUM_LEN', 'MAX_LEN')]}
                path = folder / f'ranks/rank-{rank}.json'
                path.write_text(json.dumps(record))
                hashes[str(rank)] = M.sha(path)
            receipt = {'status': 'closed_passed_local_rank_set_NOT_final16', 'closed': True,
                'timed_out': False, 'returncode': 0, 'source_sha256': M.sha(PATH),
                'clean_exit': True, 'tracking_method': M.TRACKING_METHOD,
                'all_owned_identities_absent': True, 'leader_reaped': True,
                'stable_empty_scans': 2, 'closure_errors': [],
                'probe_sha256': M.sha(M.PROBE), 'nnodes': 2, 'nproc_per_node': 8, 'world_size': 16,
                'node_rank': node, 'expected_ranks': expected, 'rank_result_sha256': hashes}
            (folder / 'launcher-result.json').write_text(json.dumps(receipt))
            directories.append(folder)
        return directories

    def test_only_both_closed_launchers_with_all_unchanged_rank_receipts_can_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = self.fake_closed_nodes(Path(directory))
            result = M.aggregate(folders)
            self.assertTrue(result['all16_rank_receipts_closed'])
            self.assertTrue(result['final16_layout_qualified'])
            self.assertFalse(result['full_model_qualified'])
            with self.assertRaisesRegex(ValueError, 'Exactly two'):
                M.aggregate(folders[:1])
            with self.assertRaisesRegex(ValueError, 'duplicate real node'):
                M.aggregate([folders[0], folders[0]])
            path = folders[1] / 'ranks/rank-15.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'changed after launcher exit'):
                M.aggregate(folders)

    def test_legacy_leader_only_closed_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = self.fake_closed_nodes(Path(directory))
            path = folders[0] / 'launcher-result.json'
            receipt = json.loads(path.read_text())
            del receipt['all_owned_identities_absent']
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'did not close successfully'):
                M.aggregate(folders)


if __name__ == '__main__':
    unittest.main()
