"""CPU-only union/dispatch contracts; synthetic traces never qualify a model."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'r1_causal_finalizer_test', ROOT / 'scripts/r1_reproduction/finalize_r1_cross_node_trace.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
HSPEC = importlib.util.spec_from_file_location(
    'r1_causal_finalizer_helper_test', ROOT / 'scripts/r1_reproduction/r1_causal_evidence.py')
H = importlib.util.module_from_spec(HSPEC)
HSPEC.loader.exec_module(H)


def synthetic_identity(node, pid):
    return {'hostname': f'actual-node{node}', 'boot_id': f'00000000-0000-0000-0000-{node + 1:012d}',
            'pid': pid, 'start_ticks': pid + 123,
            'time_namespace': {'device': 4, 'inode': 100 + node, 'link': f'time:[{100 + node}]'},
            'pid_namespace': {'device': 4, 'inode': 200 + node, 'link': f'pid:[{200 + node}]'},
            'timens_offsets': {'monotonic': [0, 0], 'boottime': [0, 0]},
            'helper_sha256': M.sha(H.__file__)}


class PinnedCoreFixture:
    """Expose the frozen interface without importing model or network modules."""
    RUN_SCHEMA = 'synthetic-run-v1'

    def __init__(self):
        self.intervals = []
        self.collections = []
        self.closed_calls = []
        self.closures = []
        self.helper = SimpleNamespace(validate_binding=H.validate_binding, validate_closure=H.validate_closure,
                                      closed_workers=self.closed_workers)

    def causal_module(self):
        return self.helper

    def closed_workers(self, identities):
        self.closed_calls.append(copy.deepcopy(identities))
        node = 0 if identities[0]['hostname'] == 'actual-node0' else 1
        return H.validate_closure({
            'schema': H.CLOSURE_SCHEMA, 'observer_identity': synthetic_identity(node, 9000),
            'workers': [{'identity': copy.deepcopy(identity), 'status': 'original_exited_no_proc',
                         'observed_process': None} for identity in identities],
            'observed_monotonic_ns': 3000 + len(self.closed_calls), 'source_sha256': M.sha(H.__file__),
        }, identities)

    def validate_causal_closures(self, run, events):
        closures = run['trace'].get('causal_worker_closures')
        if closures is None:
            return False
        self.closures.append(copy.deepcopy(closures))
        if [row['node_rank'] for row in closures] != [0, 1]:
            raise ValueError('Both ordered original-node worker closures required')
        for node, row in enumerate(closures):
            starts = sorted((e for e in events if e['event'] == 'trace_start' and e['rank'] // 8 == node),
                            key=lambda e: e['rank'])
            H.validate_closure(row['closure'], [e['clock_identity'] for e in starts])
            if any(e['observed_monotonic_ns'] > row['closure']['observed_monotonic_ns']
                   for e in events if e['rank'] // 8 == node):
                raise ValueError('Synthetic closure predates final trace event')
        return True

    def validate_suite(self, suite):
        return suite

    def validate_deployment(self, deployment, backend, suite):
        return deployment

    def is_causal_suite(self, suite):
        return suite.get('collection_protocol', {}).get('time_basis') == M.CAUSAL_TIME_BASIS

    def run_time_interval(self, run, suite):
        self.intervals.append(copy.deepcopy(run))
        if run.get('collection_protocol') != suite.get('collection_protocol'):
            raise ValueError('Exact causal suite/run protocol differs')
        a, b = run.get('started_monotonic_ns'), run.get('finished_monotonic_ns')
        if type(a) is not int or type(b) is not int or not 0 < a <= b:
            raise ValueError('Invalid collector monotonic envelope')
        return a, b

    def validate_collection_evidence(self, run, suite, suite_sha, long_stage, events):
        self.collections.append(copy.deepcopy(events))
        # A deliberately minimal stand-in for the separately tested core ACK
        # gate: the finalizer must not hide this late native request event.
        if any(e.get('event') == 'completion' and e.get('observed_monotonic_ns', 0) > 1500
               for e in events):
            raise ValueError('Synthetic native event crossed its causal ACK boundary')
        return {'mode': 'synthetic_causal_core_gate'}

    def validate_evidence(self, run_path, suite, suite_sha, long_stage, **kwargs):
        run = json.loads(run_path.read_text())
        events = json.loads((run_path.parent / run['trace']['events_file']).read_text())
        self.validate_collection_evidence(run, suite, suite_sha, long_stage, events)
        return {'hard_failures': [], 'missing_evidence': ['synthetic evidence is never qualification'],
                'actual_coverage_count': 0, 'required_coverage_count': 1952}


class CausalFinalizer(unittest.TestCase):
    def fixture(self, root, *, causal=True, late=False):
        core = PinnedCoreFixture()
        protocol = {'version': M.CAUSAL_VERSION, 'time_basis': M.CAUSAL_TIME_BASIS,
                    'cohort_size': 16}
        suite = {'schema': 'synthetic-suite-v1'}
        if causal:
            suite['collection_protocol'] = protocol
        suite_path = root / 'suite.json'
        M.write_new(suite_path, suite)
        qualifier = root / 'pinned-core.py'
        qualifier.write_text('# synthetic CPU interface fixture\n')
        deployment_path = root / 'deployment.json'
        M.write_new(deployment_path, {'run_id': 'synthetic-causal-run', 'backend': 'sglang'})
        deployment = dict(json.loads(deployment_path.read_text()), artifact_sha256=M.sha(deployment_path))
        original = root / 'original'
        (original / 'trace').mkdir(parents=True)
        raw_dirs, node0_events = [], []
        for node in (0, 1):
            folder = root / f'node{node}'
            folder.mkdir()
            raw_dirs.append(folder)
            for rank in range(node * 8, node * 8 + 8):
                envelope = {'schema': 'r1-full-model-trace-v1', 'rank': rank, 'world_size': 16,
                            'run_id': deployment['run_id'], 'suite_sha256': M.sha(suite_path),
                            'deployment_sha256': M.sha(deployment_path)}
                events = [
                    dict(envelope, event='trace_start', ordinal=1, observed_unix_ns=1,
                         observed_monotonic_ns=100, actual_pid=1000 + rank,
                         clock_identity=synthetic_identity(node, 1000 + rank)),
                    dict(envelope, event='model_hooks_registered', ordinal=2, observed_unix_ns=99999 if causal else 2,
                         observed_monotonic_ns=200),
                    dict(envelope, event='input_tokens', ordinal=3, observed_unix_ns=10 if causal else 150,
                         observed_monotonic_ns=1100, request_id='one', native_request_id='one'),
                    dict(envelope, event='completion', ordinal=4, observed_unix_ns=99999 if causal else 250,
                         observed_monotonic_ns=1600 if late else 1400,
                         request_id='one', native_request_id='one'),
                ]
                (folder / f'rank-{rank}-pid-{1000 + rank}.jsonl').write_text(
                    ''.join(M.canonical(e) + '\n' for e in events))
                if node == 0:
                    node0_events.extend(events if causal else events[:3])
        M.write_new(original / 'trace/events.json', node0_events)
        run = {'schema': core.RUN_SCHEMA, 'scope': 'full_model',
               'status': 'evidence_collected_NOT_pair_qualified', 'backend': 'sglang',
               'suite_sha256': M.sha(suite_path), 'tool_sha256': M.sha(qualifier),
               'deployment': deployment, 'long_correctness': False,
               'started_unix_ns': 200 if causal else 100,
               'finished_unix_ns': 100 if causal else 200,
               'trace': {'events_file': 'trace/events.json',
                         'events_sha256': M.sha(original / 'trace/events.json'),
                         'event_count': len(node0_events)}}
        if causal:
            run.update(collection_protocol=copy.deepcopy(protocol), started_monotonic_ns=1000,
                       finished_monotonic_ns=2000,
                       causal_binding={'schema': H.CLOCK_SCHEMA, 'run_id': deployment['run_id'],
                           'suite_sha256': M.sha(suite_path), 'deployment_sha256': M.sha(deployment_path),
                           'helper_sha256': M.sha(H.__file__), 'client_ip': '10.0.0.1',
                           'nodes': [{'node_rank': node, 'ip': f'10.0.0.{node + 1}', 'port': 7000,
                                      'hostname': f'actual-node{node}'} for node in (0, 1)]})
        M.write_new(original / 'run.json', run)
        common = dict(original_run=original / 'run.json', original_run_sha256=M.sha(original / 'run.json'),
                      suite=suite_path, suite_sha256=M.sha(suite_path),
                      deployment=deployment_path, deployment_sha256=M.sha(deployment_path),
                      qualification_tool=qualifier, qualification_tool_sha256=M.sha(qualifier))
        return core, suite, run, common, raw_dirs

    def seal(self, root, common, raw_dirs):
        manifests = []
        for node in (0, 1):
            out = root / f'sealed{node}'
            with patch.object(M.socket, 'gethostname', return_value=f'actual-node{node}'):
                M.seal(SimpleNamespace(**common, node_rank=node, trace_directory=raw_dirs[node],
                                      origin_hostname=f'actual-node{node}', output=out, execute=True))
            manifests.append(out / 'manifest.json')
        return SimpleNamespace(**common, node_manifest=manifests,
                               node_manifest_sha256=[M.sha(p) for p in manifests],
                               output=root / 'derived', execute=True)

    def test_explicit_causal_union_keeps_all_raw_and_original_backward_wall_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, suite, run, common, dirs = self.fixture(root)
            frozen = common['original_run'].read_bytes()
            with patch.object(M, 'load_tool', return_value=core), patch.dict(
                    'sys.modules', {'torch': None, 'triton': None, 'tokenizers': None}):
                args = self.seal(root, common, dirs)
                result = M.finalize(args)
            self.assertTrue(core.intervals)
            self.assertEqual(len(core.collections[-1]), 64)
            self.assertEqual({e['rank'] for e in core.collections[-1]}, set(range(16)))
            derived = json.loads((args.output / 'run.json').read_text())
            for key, value in run.items():
                if key != 'trace':
                    self.assertEqual(derived[key], value)
            self.assertEqual(common['original_run'].read_bytes(), frozen)
            manifest = json.loads(args.node_manifest[0].read_text())
            self.assertEqual(manifest['collection_protocol'], suite['collection_protocol'])
            self.assertEqual(manifest['started_monotonic_ns'], 1000)
            self.assertEqual(manifest['finished_monotonic_ns'], 2000)
            self.assertEqual(len(core.closed_calls), 4)
            self.assertEqual(derived['trace']['causal_worker_closures'], core.closures[-1])
            self.assertEqual(manifest['causal_worker_closure'], core.closures[-1][0]['closure'])
            self.assertFalse(result['native_evidence_passed'])
            self.assertFalse(result['formal_performance_permitted'])

    def test_late_native_event_is_preserved_and_reaches_causal_core_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root, late=True)
            with patch.object(M, 'load_tool', return_value=core):
                args = self.seal(root, common, dirs)
                with self.assertRaisesRegex(ValueError, 'causal ACK boundary'):
                    M.finalize(args)
            events = json.loads((args.output / 'trace/events.json').read_text())
            self.assertEqual(len(events), 64)
            self.assertEqual(sum(e['event'] == 'completion' for e in events), 16)
            self.assertEqual(core.collections[-1], events)

    def test_legacy_still_filters_by_wall_and_does_not_emit_causal_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root, causal=False)
            with patch.object(M, 'load_tool', return_value=core):
                args = self.seal(root, common, dirs)
                result = M.finalize(args)
            self.assertEqual(result['event_count'], 48)
            self.assertFalse(core.intervals)
            self.assertFalse(core.closed_calls)
            self.assertFalse(core.closures)
            manifest = json.loads(args.node_manifest[0].read_text())
            for key in ('collection_protocol', 'started_monotonic_ns', 'finished_monotonic_ns',
                        'causal_worker_closure'):
                self.assertNotIn(key, manifest)

    def test_suite_run_protocol_mismatch_and_bad_monotonic_envelope_fail_closed(self):
        for mutation in ('cohort', 'missing_protocol', 'missing_mono', 'backward_mono', 'bool_mono', 'unknown_basis'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                core, _suite, run, common, _dirs = self.fixture(root)
                if mutation == 'cohort': run['collection_protocol']['cohort_size'] = 1
                elif mutation == 'missing_protocol': del run['collection_protocol']
                elif mutation == 'missing_mono': del run['started_monotonic_ns']
                elif mutation == 'backward_mono': run['started_monotonic_ns'] = 3000
                elif mutation == 'bool_mono': run['started_monotonic_ns'] = True
                else: run['collection_protocol']['time_basis'] = 'unknown'
                common['original_run'].write_text(json.dumps(run))
                common['original_run_sha256'] = M.sha(common['original_run'])
                with patch.object(M, 'load_tool', return_value=core), self.assertRaises(ValueError):
                    M.context(SimpleNamespace(**common))

    def test_new_run_cannot_enable_causal_filtering_for_legacy_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, run, common, _dirs = self.fixture(root, causal=False)
            run['collection_protocol'] = {'version': M.CAUSAL_VERSION, 'time_basis': M.CAUSAL_TIME_BASIS}
            common['original_run'].write_text(json.dumps(run))
            common['original_run_sha256'] = M.sha(common['original_run'])
            with patch.object(M, 'load_tool', return_value=core), self.assertRaisesRegex(ValueError, 'explicitly admitted'):
                M.context(SimpleNamespace(**common))

    def test_actual_pinned_core_admits_only_matching_causal_envelope(self):
        spec = importlib.util.spec_from_file_location(
            'r1_causal_finalizer_existing_fixture', ROOT / 'tests/test_r1_cross_node_trace_finalizer.py')
        existing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(existing)
        core = existing.FIXTURES.tool
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = existing.FIXTURES.fixed_suite()
            suite.update(collection_protocol=core.collection_protocol(16, M.CAUSAL_TIME_BASIS),
                         collection_source_sha256=M.sha(core.__file__), tool_sha256=M.sha(core.__file__))
            common, _dirs, run = existing.CrossNodeTraceSealing().fixture(root, suite=suite)
            run.update(collection_protocol=copy.deepcopy(suite['collection_protocol']),
                       started_monotonic_ns=1000, finished_monotonic_ns=2000,
                       started_unix_ns=200, finished_unix_ns=100)
            common['original_run'].write_text(json.dumps(run))
            common['original_run_sha256'] = M.sha(common['original_run'])
            admitted_core, admitted_suite, admitted_run, _deployment = M.context(SimpleNamespace(**common))
            self.assertTrue(admitted_core.is_causal_suite(admitted_suite))
            self.assertEqual(admitted_core.run_time_interval(admitted_run, admitted_suite), (1000, 2000))
            run['finished_monotonic_ns'] = 999
            common['original_run'].write_text(json.dumps(run))
            common['original_run_sha256'] = M.sha(common['original_run'])
            with self.assertRaisesRegex(ValueError, 'monotonic'):
                M.context(SimpleNamespace(**common))

    def test_causal_manifest_tampering_does_not_rebind_original_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root)
            with patch.object(M, 'load_tool', return_value=core):
                args = self.seal(root, common, dirs)
                manifest = json.loads(args.node_manifest[1].read_text())
                manifest['finished_monotonic_ns'] += 1
                args.node_manifest[1].write_text(json.dumps(manifest))
                args.node_manifest_sha256[1] = M.sha(args.node_manifest[1])
                with self.assertRaisesRegex(ValueError, 'original time interval differs'):
                    M.finalize(args)

    def test_causal_raw_binding_and_partial_lines_remain_strict(self):
        for field, value in (('rank', 1), ('ordinal', 8), ('run_id', 'wrong'),
                             ('suite_sha256', 'wrong'), ('deployment_sha256', 'wrong'), ('actual_pid', 5)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _core, _suite, run, common, dirs = self.fixture(root)
                path = dirs[0] / 'rank-0-pid-1000.jsonl'
                rows = path.read_text().splitlines()
                first = json.loads(rows[0]); first[field] = value
                rows[0] = M.canonical(first)
                path.write_text('\n'.join(rows) + '\n')
                with self.assertRaises(ValueError):
                    M.read_raw(path, 0, run, run['deployment'], common['suite_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            _core, _suite, run, common, dirs = self.fixture(Path(directory))
            path = dirs[0] / 'rank-0-pid-1000.jsonl'
            path.write_text(path.read_text().rstrip('\n'))
            with self.assertRaisesRegex(ValueError, 'unclosed partial'):
                M.read_raw(path, 0, run, run['deployment'], common['suite_sha256'])

    def test_causal_seal_requires_local_origin_and_closed_worker_before_copy(self):
        for failure in ('remote_sealer', 'wrong_origin', 'live_before', 'live_after'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                core, _suite, _run, common, dirs = self.fixture(root)
                original_closed = core.helper.closed_workers
                if failure.startswith('live_'):
                    call_count = [0]
                    def live(identities):
                        call_count[0] += 1
                        if failure == 'live_before' or call_count[0] == 2:
                            raise ValueError('Original worker is still live')
                        return original_closed(identities)
                    core.helper.closed_workers = live
                args = SimpleNamespace(**common, node_rank=0, trace_directory=dirs[0],
                    origin_hostname='actual-node1' if failure == 'wrong_origin' else 'actual-node0',
                    output=root / 'sealed', execute=True)
                with patch.object(M, 'load_tool', return_value=core), patch.object(M.socket, 'gethostname',
                        return_value='remote-copy-host' if failure == 'remote_sealer' else 'actual-node0'):
                    with self.assertRaises(ValueError):
                        M.seal(args)
                self.assertFalse((args.output / 'manifest.json').exists())
                if failure != 'live_after':
                    self.assertFalse(args.output.exists())

    def test_closure_may_advance_absent_to_pid_reused_without_changing_original_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root)
            original_closed = core.helper.closed_workers
            def progressing(identities):
                result = original_closed(identities)
                if len(core.closed_calls) % 2 == 0:
                    for row in result['workers']:
                        row.update(status='original_exited_PID_reused', observed_process={
                            'pid': row['identity']['pid'], 'start_ticks': row['identity']['start_ticks'] + 1, 'state': 'R'})
                return result
            core.helper.closed_workers = progressing
            with patch.object(M, 'load_tool', return_value=core):
                args = self.seal(root, common, dirs)
                M.finalize(args)
            for path in args.node_manifest:
                closure = json.loads(path.read_text())['causal_worker_closure']
                self.assertEqual({row['status'] for row in closure['workers']}, {'original_exited_PID_reused'})

    def test_manifest_closure_missing_or_forged_fails_offline_before_output(self):
        for mutation in ('missing', 'live', 'helper_sha', 'worker_identity', 'missing_worker',
                         'observer_pidns', 'wrong_origin', 'predates_trace'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                core, _suite, _run, common, dirs = self.fixture(root)
                with patch.object(M, 'load_tool', return_value=core):
                    args = self.seal(root, common, dirs)
                    manifest = json.loads(args.node_manifest[1].read_text())
                    closure = manifest['causal_worker_closure']
                    if mutation == 'missing': del manifest['causal_worker_closure']
                    elif mutation == 'live': closure['workers'][0]['status'] = 'live'
                    elif mutation == 'helper_sha': closure['source_sha256'] = 'f' * 64
                    elif mutation == 'worker_identity': closure['workers'][0]['identity']['start_ticks'] += 1
                    elif mutation == 'missing_worker': closure['workers'].pop()
                    elif mutation == 'observer_pidns':
                        closure['observer_identity']['pid_namespace'] = {'device': 4, 'inode': 999, 'link': 'pid:[999]'}
                    elif mutation == 'wrong_origin': manifest['sealing_hostname'] = 'remote-copy-host'
                    else: closure['observed_monotonic_ns'] = 1000
                    args.node_manifest[1].write_text(json.dumps(manifest))
                    args.node_manifest_sha256[1] = M.sha(args.node_manifest[1])
                    with self.assertRaises(ValueError):
                        M.finalize(args)
                    self.assertFalse(args.output.exists())

    def test_finalizer_requires_core_to_accept_both_closures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root)
            with patch.object(M, 'load_tool', return_value=core):
                args = self.seal(root, common, dirs)
                core.validate_causal_closures = lambda *_args: False
                with self.assertRaisesRegex(ValueError, 'Both original-node'):
                    M.finalize(args)
            self.assertFalse(args.output.exists())

    def test_causal_plan_does_not_read_live_processes_or_produce_a_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, _suite, _run, common, dirs = self.fixture(root)
            with patch.object(M, 'load_tool', return_value=core):
                result = M.seal(SimpleNamespace(**common, node_rank=0, trace_directory=dirs[0],
                    origin_hostname='actual-node0', output=root / 'sealed', execute=False))
            self.assertEqual(result['status'], 'plan_only_NOT_executed')
            self.assertNotIn('causal_worker_closure', result)
            self.assertFalse(core.closed_calls)


if __name__ == '__main__':
    unittest.main()
