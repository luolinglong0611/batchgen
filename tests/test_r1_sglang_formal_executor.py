"""CPU fixtures only: no remote, GPU query, HTTP server or model import."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/r1_reproduction/execute_sglang_formal_candidate.py'
spec = importlib.util.spec_from_file_location('formal_executor_tests', SCRIPT)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FormalExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.formal = f.pinned_module('plan_sglang_formal_candidate.py')
        self.diag, self.help = self.formal.load_inputs()
        self.args = types.SimpleNamespace(
            candidate_plan=self.directory / 'candidate.json', expected_plan_sha256=None,
            source_audit=self.formal.DEFAULT_AUDIT, node_rank=0, root=self.diag.ROOT,
            cohort_id='new-formal', run_dir=self.directory / 'run',
            peer_receipt=self.directory / 'peer.json', expected_peer_receipt_sha256=None)
        self.set_candidate('A')

    def set_candidate(self, topology, mutation=None):
        candidate = self.formal.build_candidate(self.diag, self.help, topology, 16, 4096, '0.85')
        if mutation:
            mutation(candidate)
        self.args.candidate_plan.write_text(json.dumps(candidate))
        self.args.expected_plan_sha256 = hashlib.sha256(self.args.candidate_plan.read_bytes()).hexdigest()
        return candidate

    def context(self):
        return f.FormalContext(self.args)

    def receipts(self, context):
        e = context.base
        values = []
        for rank in (0, 1):
            values.append({'status': 'prepared_NOT_started', 'node_rank': rank,
                'hostname': e.HOSTS[rank], 'cohort_id': self.args.cohort_id, 'root': str(e.ROOT),
                'run_dir': str(self.args.run_dir.resolve()), 'created_unix': 1000, 'ttl_seconds': 600,
                'source_sha256': context.current_sources(), 'readiness_source_sha256': e.READINESS_PINS,
                'plan_sha256': e.digest(context.configs[rank]), 'gpu_query_count': 1,
                'memory_before_bytes': 850 * e.GIB, 'memory_after_bytes': 850 * e.GIB,
                'formal_receipt_schema': f.FORMAL_SCHEMA, 'formal_binding': copy.deepcopy(context.binding),
                'physical_node': rank, 'physical_node_role': context.configs[rank]['physical_node_role'],
                'gpus': [{'uuid': f'node{rank}-gpu{i}', 'memory_used_mib': 4,
                          'utilization_percent': 0} for i in range(8)]})
        return values

    def validate(self, context, local, peer, now=1001):
        context.base.validate_receipts(local, peer, 0, self.args.cohort_id, now,
                                      context.current_sources(), context.configs[0])

    def test_pinned_whole_plan_reconstruction_A_and_B_keeps_capacity(self):
        for topology in ('A', 'B'):
            self.set_candidate(topology)
            context = self.context()
            for rank, config in enumerate(context.configs):
                command = config['command']
                dp = 16 if topology == 'A' else 8
                self.assertEqual(command[command.index('--max-running-requests') + 1], str(16 * dp))
                self.assertEqual(command[command.index('--chunked-prefill-size') + 1], str(4096 * dp))
                self.assertNotIn('--max-total-tokens', command)
                self.assertEqual(config['environment']['BATCHGEN_SGLANG_R1_ROUTING_COMPAT'], '1')
                self.assertIn(f'node{rank}' if rank == 0 else 'compat-v2',
                              context.candidate['nodes'][rank]['routing_compat']['receipt_path'])

    def test_graph128_whole_plan_reconstruction_and_qualification_tamper(self):
        candidate = self.formal.build_candidate(self.diag, self.help, 'A', 128, 2048, '0.85', 'graph', 128)
        for tamper in (False, True):
            value = copy.deepcopy(candidate)
            if tamper: value['cuda_graph_policy']['actual_capture_replay_qualified'] = True
            self.args.candidate_plan.write_text(json.dumps(value))
            self.args.expected_plan_sha256 = hashlib.sha256(self.args.candidate_plan.read_bytes()).hexdigest()
            if tamper:
                with self.assertRaisesRegex(ValueError, 'Whole candidate'): self.context()
            else:
                context = self.context()
                self.assertEqual(context.candidate, candidate)
                self.assertIn('source-audit/sglang/srt/model_executor/cuda_graph_runner.py', context.sources)
                self.assertIn('source-audit/sglang/srt/layers/attention/trtllm_mla_backend.py', context.sources)
                self.assertTrue(all('--disable-cuda-graph' not in row['command'] for row in context.configs))

    def test_whole_candidate_rejects_edited_argv_env_capacity_identity_and_extra_fields(self):
        mutations = [lambda c: c['nodes'][0]['argv'].append('--foo'),
            lambda c: c['nodes'][1]['environment'].update(LD_PRELOAD='/wrong'),
            lambda c: c.update(launch_authorized=True), lambda c: c.update(total_gpus=16.0),
            lambda c: c.update(total_gpus=True), lambda c: c.update(extra='unreviewed'),
            lambda c: c['nodes'][1].update(actual_kv_tokens_per_rank=1000000)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.set_candidate('A', mutate)
                with self.assertRaisesRegex(ValueError, 'Whole candidate'):
                    self.context()

    def test_raw_plan_SHA_duplicate_keys_symlink_and_mutation_rejected(self):
        self.args.expected_plan_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.context()
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            f.strict_json('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            f.strict_json('{"x":NaN}')
        link = self.directory / 'link'
        link.symlink_to(self.args.candidate_plan)
        with self.assertRaises((ValueError, OSError)):
            f.stable_read(link)
        self.set_candidate('A')
        context = self.context()
        self.args.candidate_plan.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            context.current_sources()

    def test_default_plan_has_no_subprocess_resource_probe_or_HTTP(self):
        root = self.diag.ROOT
        with patch.object(f.subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(f.subprocess, 'Popen', side_effect=AssertionError('Popen')), \
             patch.object(f, 'check_ports', side_effect=AssertionError('ports')), \
             patch.object(f.socket, 'gethostname', side_effect=AssertionError('hostname')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(f.main(['--candidate-plan', str(self.args.candidate_plan),
                '--expected-plan-sha256', self.args.expected_plan_sha256, '--node-rank', '0',
                '--cohort-id', 'new', '--run-dir', str(root / 'new')]), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'plan_only_NOT_executed')

    def test_fresh_paired_formal_receipts_and_both_plan_digests(self):
        context = self.context()
        local, peer = self.receipts(context)
        self.validate(context, local, peer)
        for key, value in [('formal_receipt_schema', None), ('formal_binding', {}),
                           ('plan_sha256', 'wrong'), ('physical_node', 0),
                           ('source_sha256', {}), ('hostname', 'wrong'),
                           ('node_rank', 0), ('cohort_id', 'old'),
                           ('memory_before_bytes', 849 * context.base.GIB),
                           ('memory_after_bytes', 849 * context.base.GIB), ('gpu_query_count', 2)]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate(context, local, dict(peer, **{key: value}))
        with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
            self.validate(context, dict(local, plan_sha256='wrong'), peer)
        for now in (999, 1600.1):
            with self.assertRaises(RuntimeError):
                self.validate(context, local, peer, now)
        self.validate(context, local, peer, 1600)

    def test_idle_inventories_are_distinct_complete_and_finite(self):
        context = self.context()
        local, peer = self.receipts(context)
        for mutate in (lambda p: p['gpus'].pop(),
                lambda p: p['gpus'][1].update(uuid=p['gpus'][0]['uuid']),
                lambda p: p['gpus'][0].update(uuid=local['gpus'][0]['uuid']),
                lambda p: p['gpus'][0].update(memory_used_mib=65),
                lambda p: p['gpus'][0].update(utilization_percent=1),
                lambda p: p['gpus'][0].update(uuid=None)):
            value = copy.deepcopy(peer)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                self.validate(context, local, value)

    def test_service_environment_exact_replacement_RAS0_SG_own_CUDA(self):
        context = self.context()
        polluted = {'NCCL_RAS_ENABLE': '1', 'LD_PRELOAD': '/BG/libcudart.so.13',
            'FLASHINFER_WORKSPACE_BASE': '/BG/cache', 'TRITON_PTXAS_PATH': '/BG/ptxas',
            'CUDA_HOME': '/BG/cuda', 'AWS_SECRET_ACCESS_KEY': 'test-secret', 'PATH': '/wrong'}
        with patch.dict(os.environ, polluted):
            for config in context.configs:
                actual = context.base.child_environment(config)
                self.assertEqual(actual, config['environment'])
                self.assertEqual(actual['NCCL_RAS_ENABLE'], '0')
                for key in ('LD_PRELOAD', 'AWS_SECRET_ACCESS_KEY', 'FLASHINFER_WORKSPACE_BASE'):
                    self.assertNotIn(key, actual)
                self.assertEqual(actual['CUDA_HOME'], str(self.diag.ROOT / 'cuda130-packages/nvidia/cu13'))
                self.assertIn('sglang-baseline-v8', actual['LD_LIBRARY_PATH'])

    def test_A_dummy_peer_and_B_both_local_real_server_readiness(self):
        models = json.dumps({'data': [{'id': self.diag.MODEL if hasattr(self.diag, 'MODEL')
                                     else 'deepseek-ai/DeepSeek-R1'}]}).encode()
        for topology in ('A', 'B'):
            self.set_candidate(topology)
            context = self.context()
            calls = []
            def response(url):
                calls.append(url)
                return models if url.endswith('/v1/models') else b'OK'
            with patch.object(context.base, 'http_get', side_effect=response):
                context.base.probe_ready(1)
            if topology == 'A':
                self.assertIn('Dummy health check', context.base.MARKERS[1])
                self.assertIn('http://10.66.9.163:30000/v1/models', calls)
                self.assertIn('http://127.0.0.1:30000/health', calls)
            else:
                self.assertEqual(context.base.MARKERS[0], context.base.MARKERS[1])
                self.assertEqual(calls, ['http://127.0.0.1:30000/health', 'http://127.0.0.1:30000/v1/models'])
                with patch.object(context.base, 'http_get', return_value=b'{"data":[]}'):
                    with self.assertRaisesRegex(RuntimeError, 'Local independent'):
                        context.base.probe_ready(1)

    def test_writer_binding_keeps_plan_digest_and_returned_receipt_identical(self):
        context = self.context()
        config = context.base.bundle(context, self.args.root, 0)
        before = context.base.digest(config)
        path = self.directory / 'plan.json'
        context.base.write_json(path, config, exclusive=True)
        self.assertEqual(before, context.base.digest(config))
        self.assertEqual(config, json.loads(path.read_text()))
        ready = {}
        context.base.write_json(self.directory / 'ready.json', ready, exclusive=True)
        self.assertIn('NOT quality, capacity or performance', ready['meaning'])

    def test_copied_nonrepository_bundle_with_explicit_source_audit(self):
        bundle = self.directory / 'new-tools'
        bundle.mkdir()
        for name in (*f.PINS, SCRIPT.name):
            shutil.copyfile(f.HERE / name, bundle / name)
        audit = bundle / 'source-audit'
        shutil.copytree(self.args.source_audit, audit)
        copied = bundle / SCRIPT.name
        spec = importlib.util.spec_from_file_location('copied_formal_executor', copied)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = copy.copy(self.args)
        args.source_audit = audit
        context = module.FormalContext(args)
        self.assertEqual(context.candidate['candidate_id'], 'A-c16-m085-k4096-eager')
        self.assertEqual(Path(context.base.__file__), (bundle / 'execute_sglang_diagnostic.py').resolve())
        self.assertEqual(Path(context.provider.__file__), (bundle / 'launch_sglang_diagnostic.py').resolve())

    def test_actual_base_prepare_each_node_one_snapshot_and_stable_expected_plan_digest(self):
        contexts, receipts = [], []
        for rank in (0, 1):
            args = copy.copy(self.args)
            args.node_rank, args.run_dir = rank, self.directory / f'prepare-node{rank}'
            context = f.FormalContext(args)
            e = context.base
            before = e.digest(context.configs[rank])
            xml = '<nvidia_smi_log><attached_gpus>8</attached_gpus>' + ''.join(
                f'<gpu><uuid>node{rank}-GPU{i}</uuid><fb_memory_usage><used>4 MiB</used></fb_memory_usage>'
                '<utilization><gpu_util>0 %</gpu_util></utilization><processes/></gpu>' for i in range(8)) + '</nvidia_smi_log>'
            with patch.object(e, 'require_container'), \
                 patch.object(e.socket, 'gethostname', return_value=e.HOSTS[rank]), \
                 patch.object(e, 'check_readiness_sources', return_value=e.READINESS_PINS), \
                 patch.object(context, 'inspect', return_value={'status': 'CPU_artifact_inspection_passed_NOT_GPU_qualified'}), \
                 patch.object(e, 'require_memory', return_value=900 * e.GIB), \
                 patch.object(e.subprocess, 'run', return_value=types.SimpleNamespace(stdout=xml)) as query:
                receipt = e.prepare(args, context)
                query.assert_called_once()
                self.assertEqual(query.call_args.args[0], ['nvidia-smi', '-q', '-x'])
                self.assertEqual(query.call_args.kwargs['timeout'], 5)
            self.assertEqual(before, e.digest(context.configs[rank]))
            self.assertEqual(receipt['plan_sha256'], before)
            written_plan = json.loads((args.run_dir / 'plan.json').read_text())
            self.assertEqual(e.digest(written_plan), before)
            self.assertEqual(receipt, json.loads((args.run_dir / 'receipt.json').read_text()))
            contexts.append(context)
            receipts.append(receipt)
        self.validate(contexts[0], *receipts, now=contexts[0].base.time.time())

    def test_inspection_clears_all_ambient_keys_and_checks_live_help_semantics(self):
        context = self.context()
        root = self.directory / 'installed-root'
        site = root / 'sglang-baseline-v8/baseline-env/lib/python3.11/site-packages'
        source = site / 'sglang/srt/server_args.py'
        source.parent.mkdir(parents=True)
        source.write_text('# pinned fixture')
        help_sha = 'a' * 64
        context.formal.SOURCE_PINS = {'sglang/srt/server_args.py': context.base.sha(source),
                                    'launch-server-help.stdout': help_sha}
        self.args.run_dir.mkdir()
        with patch.object(context, 'safety_preflight') as safety, \
             patch.object(f, 'check_ports', return_value={}), \
             patch.dict(os.environ, {'LD_PRELOAD': '/BG/runtime', 'SECRET_FIXTURE': 'private'}), \
             patch.object(context.provider, 'inspect', return_value={'launch_help_sha256': help_sha}) as inspect:
            result = context.inspect(root, 0, context.configs[0]['environment'], [], context.configs[0]['command'])
            safety.assert_called_once_with('prepare')
            self.assertEqual(set(inspect.call_args.args[3]), set(os.environ))
            self.assertIn('SECRET_FIXTURE', inspect.call_args.args[3])
            self.assertEqual(inspect.call_args.args[2], context.configs[0]['environment'])
            self.assertEqual(result['inspection_environment_policy'], 'exact_SG_allowlist_plus_CUDA_hidden')
        with patch.object(context, 'safety_preflight'), patch.object(f, 'check_ports', return_value={}), \
             patch.object(context.provider, 'inspect', return_value={'launch_help_sha256': 'wrong'}), \
             patch.object(context.base, 'write_json'):
            with self.assertRaisesRegex(RuntimeError, 'installed help differs'):
                context.inspect(root, 0, {}, [], [])

    def test_safety_inheritance_probe_has_exact_CPU_hidden_environment(self):
        context = self.context()
        self.args.run_dir.mkdir()
        def preflight(root):
            context.safety.subprocess.run(['fixture-cpu-probe'], timeout=10)
            return {'status': 'fixture'}
        with patch.object(context.safety, 'prepare_launch_safety', side_effect=preflight), \
             patch.object(f.subprocess, 'run') as run, \
             patch.dict(os.environ, {'LD_PRELOAD': '/wrong', 'SECRET': 'private'}):
            context.safety_preflight('prepare')
            env = run.call_args.kwargs['env']
            self.assertEqual(env, dict(context.configs[0]['environment'],
                                      CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none'))
            self.assertNotIn('LD_PRELOAD', env)
            self.assertNotIn('SECRET', env)

    def prepare_launch_fixture(self, context):
        e = context.base
        self.args.run_dir.mkdir()
        xml = '<nvidia_smi_log><attached_gpus>8</attached_gpus>' + ''.join(
            f'<gpu><uuid>GPU{i}</uuid><fb_memory_usage><used>4 MiB</used></fb_memory_usage>'
            '<utilization><gpu_util>0 %</gpu_util></utilization><processes/></gpu>' for i in range(8)) + '</nvidia_smi_log>'
        (self.args.run_dir / 'gpu-preflight.xml').write_text(xml)
        (self.args.run_dir / 'cpu-inspection.json').write_text('{}')
        local, peer = self.receipts(context)
        local['created_unix'] = peer['created_unix'] = e.time.time()
        local.update(inspection_sha256=e.sha(self.args.run_dir / 'cpu-inspection.json'),
                     gpu_snapshot_sha256=e.sha(self.args.run_dir / 'gpu-preflight.xml'))
        (self.args.run_dir / 'receipt.json').write_text(json.dumps(local))
        self.args.peer_receipt.write_text(json.dumps(peer))
        self.args.expected_peer_receipt_sha256 = e.sha(self.args.peer_receipt)
        for target, name, value in ((e, 'require_container', None),
                (e, 'check_readiness_sources', e.READINESS_PINS), (e, 'require_memory', 900 * e.GIB),
                (e, 'process_identity', {'pid': os.getpid(), 'pgid': 1, 'sid': 1, 'starttime_ticks': 1}),
                (context, 'safety_preflight', {}), (context.safety, 'assert_self_core_policy', {}),
                (f, 'check_ports', {})):
            mock = patch.object(target, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        return e

    def test_peer_explicit_SHA_and_core_failure_before_Popen(self):
        context = self.context()
        self.prepare_launch_fixture(context)
        self.args.expected_peer_receipt_sha256 = '0' * 64
        with patch.object(f.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                context.execute()
            spawn.assert_not_called()
        self.args.expected_peer_receipt_sha256 = context.base.sha(self.args.peer_receipt)
        with patch.object(context.safety, 'assert_self_core_policy', side_effect=RuntimeError('core drift')), \
             patch.object(f.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'core drift'):
                context.execute()
            spawn.assert_not_called()

    def test_first_identity_capture_failure_cleans_unreaped_owned_child(self):
        context = self.context()
        e = self.prepare_launch_fixture(context)
        identity = {'pid': 4242, 'pgid': 4242, 'sid': 4242, 'starttime_ticks': 10}
        child = Mock(pid=4242, returncode=None)
        original_identity = e.process_identity
        with patch.object(f.subprocess, 'Popen', return_value=child), \
             patch.object(context, 'capture_child_identity', side_effect=OSError('first child capture failed')), \
             patch.object(e, 'process_identity', side_effect=lambda pid: identity if pid == 4242 else original_identity(pid)), \
             patch.object(e, 'same_group_leader', return_value=True), \
             patch.object(e, 'stop_owned_group', return_value={'cleanup_complete': True}) as cleanup:
            with self.assertRaisesRegex(OSError, 'first child capture failed'):
                context.execute()
            cleanup.assert_called_once_with(child, identity)
        self.assertTrue((self.args.run_dir / 'launch-error-cleanup.json').exists())

    def test_child_receipt_failure_cleans_and_preserves_original_write_error(self):
        context = self.context()
        e = self.prepare_launch_fixture(context)
        identity = {'pid': 4242, 'pgid': 4242, 'sid': 4242, 'starttime_ticks': 10}
        child = Mock(pid=4242, returncode=None)
        writer = e.write_json
        def write(path, value, exclusive=False):
            if Path(path).name == 'child.json':
                raise OSError('child identity receipt full disk')
            return writer(path, value, exclusive)
        with patch.object(f.subprocess, 'Popen', return_value=child), \
             patch.object(context, 'capture_child_identity', return_value=identity), \
             patch.object(e, 'write_json', side_effect=write), \
             patch.object(e, 'same_group_leader', return_value=True), \
             patch.object(e, 'stop_owned_group', return_value={'cleanup_complete': True}) as cleanup:
            with self.assertRaisesRegex(OSError, 'child identity receipt full disk'):
                context.execute()
            cleanup.assert_called_once_with(child, identity)
        self.assertTrue((self.args.run_dir / 'launch-error-cleanup.json').exists())

    def test_supervisor_reaped_then_final_write_error_never_rescans_or_signals(self):
        context = self.context()
        e = self.prepare_launch_fixture(context)
        pid = 4242  # Fixture only; never spawn or signal a real process.
        child = Mock(pid=pid, returncode=None)
        identity = {'pid': pid, 'pgid': pid, 'sid': pid, 'starttime_ticks': 10}
        def reaped(*a):
            child.returncode = -15
            raise OSError('final status write failed after reap')
        with patch.object(f.subprocess, 'Popen', return_value=child), \
             patch.object(context, 'capture_child_identity', return_value=identity.copy()), \
             patch.object(e, 'supervise', side_effect=reaped), \
             patch.object(e, 'same_group_leader') as scan, patch.object(e, 'stop_owned_group') as cleanup:
            with self.assertRaisesRegex(OSError, 'after reap'):
                context.execute()
            scan.assert_not_called()
            cleanup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
