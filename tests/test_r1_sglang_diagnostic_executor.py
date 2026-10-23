"""Local-only executor tests: no GPU, model import, remote host or service."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/execute_sglang_diagnostic.py'
spec = importlib.util.spec_from_file_location('executor', SCRIPT)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def gpu_xml(used=4, utilization=0, process='', count=8):
    return '<nvidia_smi_log><attached_gpus>8</attached_gpus>' + ''.join(
        f'<gpu><uuid>GPU-{i}</uuid><product_name>NVIDIA L20D</product_name>'
        f'<fb_memory_usage><used>{used} MiB</used></fb_memory_usage>'
        f'<utilization><gpu_util>{utilization} %</gpu_util></utilization>'
        f'<processes>{process}</processes></gpu>' for i in range(count)) + '</nvidia_smi_log>'


class ExecutorTests(unittest.TestCase):
    def test_plan_default_does_no_preflight_or_process(self):
        with patch.object(e, 'prepare', side_effect=AssertionError), \
             patch.object(e, 'execute', side_effect=AssertionError), \
             patch.object(e.subprocess, 'run', side_effect=AssertionError), \
             patch.object(e.subprocess, 'Popen', side_effect=AssertionError), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(e.main(['--node-rank', '0', '--cohort-id', 'test',
                                     '--run-dir', str(e.ROOT / 'dryrun')]), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'plan_only_NOT_executed')

    def test_gpu_snapshot_requires_all_eight_idle_known_devices(self):
        self.assertEqual(len(e.parse_idle_gpus(gpu_xml())), 8)
        for xml in (gpu_xml(used=65), gpu_xml(utilization=1), gpu_xml(count=7),
                    gpu_xml(process='<process_info><pid>123</pid></process_info>'),
                    gpu_xml().replace('4 MiB', 'N/A'),
                    gpu_xml().replace('GPU-7', 'GPU-6')):
            with self.subTest(xml=xml[:80]), self.assertRaises(RuntimeError):
                e.parse_idle_gpus(xml)

    def test_environment_exactly_follows_planner_cleanup(self):
        planner = e.load_planner()
        inherited = {'BATCHGEN_FOO': 'x', 'SGLANG_OLD': 'x', 'NCCL_OLD': 'x',
                     'TRITON_PTXAS_PATH': '/wrong', 'NORMAL': 'kept'}
        env, unset, command = planner.plan(e.ROOT, 0, inherited)
        actual = e.child_environment({'environment': env, 'unset_environment': unset}, inherited)
        self.assertEqual(actual, {'NORMAL': 'kept', **env})
        self.assertNotIn('TRITON_PTXAS_PATH', actual)

    def test_real_planner_merge_pins_RAS_and_isolates_batchgen_CUDA_overrides(self):
        planner = e.load_planner()
        poisoned = {
            'LD_PRELOAD': '/batchgen/libcudart.so.13', 'CUDA_PATH': '/batchgen/cuda',
            'CUDA_LIB_PATH': '/batchgen/cuda/lib', 'CUDA_HOME': '/batchgen/cuda',
            'LD_LIBRARY_PATH': '/batchgen/cuda/lib', 'CPATH': '/batchgen/include',
            'LIBRARY_PATH': '/batchgen/link', 'CUDACXX': '/batchgen/nvcc',
            'NVCC': '/batchgen/nvcc', 'NVCC_PREPEND_FLAGS': '-I/batchgen/include',
            'NVCC_APPEND_FLAGS': '--device-debug', 'CC': '/batchgen/cc',
            'CXX': '/batchgen/cxx', 'CPPFLAGS': '-I/batchgen/include',
            'CFLAGS': '-O0', 'CXXFLAGS': '-O0', 'LDFLAGS': '-L/batchgen/lib',
            'FLASHINFER_NVCC': '/batchgen/nvcc',
            'FLASHINFER_WORKSPACE_BASE': '/batchgen/flashinfer',
            'TRITON_PTXAS_PATH': '/batchgen/ptxas', 'DISABLE_MMA_V5': '1',
        }
        for rank in (0, 1):
            for RAS in (None, '0', '1'):
                with self.subTest(rank=rank, inherited_RAS=RAS):
                    inherited = dict(poisoned)
                    if RAS is not None:
                        inherited['NCCL_RAS_ENABLE'] = RAS
                    env, unset, command = planner.plan(e.ROOT, rank, inherited)
                    actual = e.child_environment({'environment': env, 'unset_environment': unset}, inherited)
                    self.assertEqual(actual, env)
                    self.assertEqual(actual['NCCL_RAS_ENABLE'], '0')
                    for key in poisoned:
                        if key not in env:
                            self.assertNotIn(key, actual)
                    base = e.ROOT / 'sglang-baseline-v8'
                    toolkit = e.ROOT / 'cuda130-packages/nvidia/cu13'
                    self.assertEqual(actual['CUDA_HOME'], str(toolkit))
                    self.assertEqual(actual['PATH'].split(':')[:2],
                                     [str(base / 'baseline-env/bin'), str(toolkit / 'bin')])
                    self.assertEqual(Path(actual['CUDA_HOME']) / 'bin/nvcc', toolkit / 'bin/nvcc')
                    self.assertEqual(command[0], str(base / 'baseline-env/bin/python'))
                    self.assertEqual(actual['CPATH'], str(base / 'baseline-env/lib/python3.11/site-packages/nvidia/cu13/include'))
                    self.assertEqual(actual['LD_LIBRARY_PATH'],
                        f'{base}/private-cxx-runtime/lib:{base}/baseline-env/lib/python3.11/site-packages/nvidia/cu13/lib')

    def test_historical_launcher_source_receipt_cannot_authorize_new_plan(self):
        local, peer, config = self.receipts()
        sources = e.source_hashes()
        local['source_sha256'] = dict(sources)
        peer['source_sha256'] = dict(sources)
        e.validate_receipts(local, peer, 0, 'same', 1001, sources, config)
        historical = dict(sources)
        historical['launch_sglang_diagnostic.py'] = '6decad7c8113cc9c68ad2c8e81c532ddf25c1594138b6c31471d9dc8fd59e361'
        self.assertNotEqual(historical['launch_sglang_diagnostic.py'], sources['launch_sglang_diagnostic.py'])
        peer['source_sha256'] = historical
        with self.assertRaisesRegex(RuntimeError, 'identity/cohort/source mismatch'):
            e.validate_receipts(local, peer, 0, 'same', 1001, sources, config)

    def receipts(self):
        config = {'environment': {}, 'unset_environment': [], 'command': ['model']}
        local = {'status': 'prepared_NOT_started', 'node_rank': 0, 'hostname': e.HOSTS[0],
                 'cohort_id': 'same', 'root': str(e.ROOT), 'source_sha256': {'x': 'hash'},
                 'readiness_source_sha256': e.READINESS_PINS, 'created_unix': 1000,
                 'ttl_seconds': 600, 'memory_after_bytes': 850 * e.GIB,
                 'gpus': [{}] * 8, 'plan_sha256': e.digest(config)}
        peer = dict(local, node_rank=1, hostname=e.HOSTS[1])
        return local, peer, config

    def test_receipts_reject_expiry_future_wrong_cohort_source_and_plan(self):
        local, peer, config = self.receipts()
        e.validate_receipts(local, peer, 0, 'same', 1600, {'x': 'hash'}, config)
        for now in (1600.1, 999):
            with self.assertRaises(RuntimeError):
                e.validate_receipts(local, peer, 0, 'same', now, {'x': 'hash'}, config)
        for key, value in [('cohort_id', 'other'), ('node_rank', 0), ('source_sha256', {}),
                           ('memory_after_bytes', 849 * e.GIB), ('hostname', 'other')]:
            bad = dict(peer, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                e.validate_receipts(local, bad, 0, 'same', 1001, {'x': 'hash'}, config)
        with self.assertRaises(RuntimeError):
            e.validate_receipts(local, peer, 0, 'same', 1001, {'x': 'hash'}, {'changed': True})

    def test_prepare_cpu_first_single_gpu_query_and_fresh_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new'
            args = types.SimpleNamespace(node_rank=0, root=e.ROOT, run_dir=path, cohort_id='cohort')
            events = []
            planner = Mock()
            planner.plan.return_value = ({}, [], ['model'])
            planner.inspect.side_effect = lambda *a: events.append('cpu') or {
                'status': 'CPU_artifact_inspection_passed_NOT_GPU_qualified'}
            def gpu(*a, **kw):
                events.append('gpu')
                self.assertEqual(kw['timeout'], 5)
                self.assertEqual(a[0], ['nvidia-smi', '-q', '-x'])
                return types.SimpleNamespace(stdout=gpu_xml())
            with patch.object(e, 'require_container'), patch.object(e, 'check_readiness_sources', return_value=e.READINESS_PINS), \
                 patch.object(e, 'require_memory', return_value=900*e.GIB), \
                 patch.object(e.subprocess, 'run', side_effect=gpu) as query:
                receipt = e.prepare(args, planner)
                self.assertEqual(events, ['cpu', 'gpu'])
                self.assertEqual(query.call_count, 1)
                self.assertEqual(receipt['gpu_query_count'], 1)
                with self.assertRaises(FileExistsError):
                    e.prepare(args, planner)
                self.assertEqual(query.call_count, 1)

    def test_failed_cpu_inspection_never_queries_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(node_rank=0, root=e.ROOT, run_dir=Path(directory)/'new', cohort_id='x')
            planner = Mock()
            planner.plan.return_value = ({}, [], ['model'])
            planner.inspect.side_effect = RuntimeError('CPU failed')
            with patch.object(e, 'require_container'), patch.object(e.subprocess, 'run') as gpu:
                with self.assertRaisesRegex(RuntimeError, 'CPU failed'):
                    e.prepare(args, planner)
                gpu.assert_not_called()
            self.assertTrue((args.run_dir / 'prepare-failed.json').exists())

    def test_group_signal_rejects_pid_reuse(self):
        identity = {'pid': 42, 'pgid': 42, 'sid': 42, 'starttime_ticks': 100}
        with patch.object(e, 'process_identity', return_value=dict(identity, starttime_ticks=101)), \
             patch.object(e.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'starttime'):
                e.signal_owned_group(identity, signal.SIGTERM)
            kill.assert_not_called()
        with patch.object(e, 'process_identity', return_value=identity), patch.object(e.os, 'killpg') as kill:
            e.signal_owned_group(identity, signal.SIGTERM)
            kill.assert_called_once_with(42, signal.SIGTERM)

    def test_cleanup_term_then_kill_only_owned_group(self):
        child = Mock()
        child.wait.return_value = -9
        identity = {'pid': 42, 'pgid': 42, 'sid': 42, 'starttime_ticks': 100}
        with patch.object(e, 'signal_owned_group') as send, \
             patch.object(e, 'live_group_members', side_effect=[[identity], [identity]]), \
             patch.object(e, 'child_exited', return_value=True), \
             patch.object(e, 'live_session_members', return_value=[]):
            report = e.stop_owned_group(child, identity, grace=0)
        self.assertEqual([call.args for call in send.call_args_list],
                         [(identity, signal.SIGTERM), (identity, signal.SIGKILL)])
        self.assertTrue(report['cleanup_complete'])

    def test_other_pgid_in_same_session_prevents_false_cleanup_success(self):
        child = Mock()
        child.wait.return_value = -15
        identity = {'pid': 42, 'pgid': 42, 'sid': 42, 'starttime_ticks': 100}
        compiler = {'pid': 84, 'pgid': 80, 'sid': 42, 'starttime_ticks': 110, 'state': 'S'}
        events = []
        def audit(unreaped):
            self.assertEqual(unreaped, identity)
            child.wait.assert_not_called()
            events.append('session_audit')
            return [compiler]
        def reap(**kwargs):
            events.append('reap')
            self.assertEqual(kwargs, {'timeout': 0})
            return -15
        child.wait.side_effect = reap
        with patch.object(e, 'signal_owned_group') as send, \
             patch.object(e, 'live_group_members', return_value=[]), \
             patch.object(e, 'child_exited', return_value=True), \
             patch.object(e, 'live_session_members', side_effect=audit):
            result = e.stop_owned_group(child, identity, grace=0)
        self.assertFalse(result['cleanup_complete'])
        self.assertTrue(result['parent_cleanup_required'])
        self.assertEqual(result['remaining_live_members'], [])
        self.assertEqual(result['remaining_live_other_pgid_members'], [compiler])
        self.assertEqual(result['remaining_live_session_members'], [compiler])
        self.assertEqual(events, ['session_audit', 'reap'])
        send.assert_called_once_with(identity, signal.SIGTERM)

    def test_session_scan_includes_other_pgid_excludes_other_sid_and_zombies(self):
        identity = {'pid': 42, 'pgid': 42, 'sid': 42, 'starttime_ticks': 100, 'state': 'Z'}
        compiler = {'pid': 84, 'pgid': 80, 'sid': 42, 'starttime_ticks': 110, 'state': 'S'}
        foreign = dict(compiler, pid=90, sid=90)
        zombie = dict(compiler, pid=85, state='Z')
        table = {item['pid']: item for item in (identity, compiler, foreign, zombie)}
        with patch.object(e.Path, 'iterdir', return_value=[Path('/proc')/str(pid) for pid in table]), \
             patch.object(e, 'process_identity', side_effect=table.__getitem__):
            self.assertEqual(e.live_session_members(identity), [compiler])

    def test_failed_session_audit_is_never_cleanup_complete(self):
        child = Mock()
        child.wait.return_value = 0
        identity = {'pid': 42, 'pgid': 42, 'sid': 42, 'starttime_ticks': 100}
        with patch.object(e, 'signal_owned_group'), \
             patch.object(e, 'live_group_members', return_value=[]), \
             patch.object(e, 'child_exited', return_value=True), \
             patch.object(e, 'live_session_members', side_effect=PermissionError('proc read denied')):
            result = e.stop_owned_group(child, identity, grace=0)
        self.assertFalse(result['cleanup_complete'])
        self.assertFalse(result['session_audit_complete'])
        self.assertIn('proc read denied', result['session_audit_error'])

    def test_marker_crosses_read_boundary_and_is_remembered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'log'
            marker = e.MARKERS[1]
            path.write_text(marker[:20])
            scan = e.LogMarker(path, marker)
            self.assertFalse(scan.scan())
            with path.open('a') as f:
                f.write(marker[20:] + '\n')
            self.assertTrue(scan.scan())
            self.assertTrue(scan.scan())

    def test_rank1_requires_local_dummy_and_real_rank0_model(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return json.dumps({'data': [{'id': e.MODEL}]}).encode() if url.endswith('models') else b''
        with patch.object(e, 'http_get', side_effect=fetch):
            report = e.probe_ready(1)
        self.assertEqual(calls, ['http://10.66.9.163:30000/health',
                                'http://10.66.9.163:30000/v1/models', 'http://127.0.0.1:30000/health'])
        self.assertEqual(report['rank1_local_dummy_health_http'], 200)
        with patch.object(e, 'http_get', return_value=b'{"data":[{"id":"wrong-model"}]}'):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                e.probe_ready(0)

    def test_memory_guard_stops_without_gpu_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'service.log').touch()
            with patch.object(e, 'memory_available', return_value=127*e.GIB), \
                 patch.object(e, 'stop_owned_group', return_value={'exit_code': -15, 'cleanup_complete': True}), \
                 patch.object(e.subprocess, 'run', side_effect=AssertionError):
                result = e.supervise(Mock(), {'pid': 1}, path, 0, lambda: False)
            self.assertEqual(result['stop_reason'], 'host_memory_below_128GiB')
            self.assertEqual(result['status'], 'stopped')

    def test_ready_does_not_timeout_evaluation_and_waits_for_parent(self):
        class DoneFuture:
            def done(self):
                return True
            def result(self):
                return {'rank0_health_http': 200}
        class Pool:
            def submit(self, *args):
                return DoneFuture()
            def shutdown(self, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'service.log').write_text(e.MARKERS[0])
            # Four guard iterations: submit, mark ready, >startup deadline,
            # then parent stop. Passing 7200 after READY must not stop the run.
            tick = [0]
            stopping = [False]
            def monotonic():
                tick[0] += 1
                return tick[0] if tick[0] <= 8 else 9000 + tick[0]
            def memory():
                count[0] += 1
                if count[0] == 4:
                    stopping[0] = True
                return 900 * e.GIB
            count = [0]
            with patch.object(e, 'ThreadPoolExecutor', Pool), \
                 patch.object(e, 'memory_available', side_effect=memory), \
                 patch.object(e, 'child_exited', return_value=False), \
                 patch.object(e, 'POLL_SECONDS', 0), \
                 patch.object(e.time, 'monotonic', side_effect=monotonic), \
                 patch.object(e, 'stop_owned_group', return_value={'exit_code': 0, 'cleanup_complete': True}):
                # Pool's initializer takes max_workers in production.
                with patch.object(e, 'ThreadPoolExecutor', side_effect=lambda **kw: Pool()):
                    result = e.supervise(Mock(), {'pid': 1}, path, 0, lambda: stopping[0])
            self.assertTrue(result['ready'])
            self.assertEqual(result['stop_reason'], 'parent_stop_signal')
            self.assertEqual(count[0], 4)

    def test_log_write_failure_still_cleans_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'service.log').touch()
            with patch.object(e, 'memory_available', return_value=900*e.GIB), \
                 patch.object(e, 'write_json', side_effect=OSError('disk full')), \
                 patch.object(e, 'stop_owned_group', return_value={'exit_code': -15, 'cleanup_complete': True}) as stop:
                with self.assertRaises(OSError):
                    e.supervise(Mock(), {'pid': 1}, path, 0, lambda: False)
                stop.assert_called_once()

    def test_startup_timeout_always_cleans_own_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'service.log').touch()
            with patch.object(e, 'memory_available', return_value=900*e.GIB), \
                 patch.object(e, 'child_exited', return_value=False), \
                 patch.object(e.time, 'monotonic', side_effect=[0, 7201]), \
                 patch.object(e, 'stop_owned_group', return_value={'exit_code': -15, 'cleanup_complete': True}) as stop:
                result = e.supervise(Mock(), {'pid': 1}, path, 0, lambda: False)
            self.assertEqual(result['stop_reason'], 'startup_timeout_7200s')
            stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
