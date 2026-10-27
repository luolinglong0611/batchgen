"""Client-only cohort contracts; synthetic records never qualify a model."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = load('r1_cohort_versions', ROOT / 'tests/test_r1_full_model_suite_versions.py')
C = load('r1_cohort_crossnode_fixtures', ROOT / 'tests/test_r1_cross_node_trace_finalizer.py')
tool = V.tool
semantic = load('r1_cohort_semantic', ROOT / 'scripts/r1_reproduction/r1_independent_semantic.py')


class CollectionCohorts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = V.prepare_fixture(self.root, 'v2')
        self.suite = copy.deepcopy(self.original)
        self.suite['collection_protocol'] = tool.collection_protocol(16)
        self.suite['collection_source_sha256'] = tool.sha(tool.__file__)
        self.suite_sha = tool.value_sha(self.suite)
        self.plan = tool.collection_schedule(self.suite, self.suite_sha, True)

    def tearDown(self):
        self.temp.cleanup()

    def fixture_run(self, backend='batchgen', suite=None, long_stage=True):
        suite = suite or self.suite
        suite_sha = tool.value_sha(suite)
        plan = tool.collection_schedule(suite, suite_sha, long_stage)
        run = {'backend': backend, 'tool_sha256': suite['tool_sha256'],
               'collection_source_sha256': tool.sha(tool.__file__),
               'collection_schedule': plan, 'collection_schedule_sha256': tool.value_sha(plan),
               'collection_receipts': [], 'records': [], 'started_unix_ns': 1,
               'finished_unix_ns': 10000}
        events = []
        for index, group in enumerate(plan['cohorts']):
            start, end = 100 * (index + 1), 100 * (index + 1) + 90
            batch = 'batch-' + str(index)
            receipt = {'index': index, 'kind': group['kind'], 'planned_ids': group['ids'],
                'received_ids': list(reversed(group['ids'])), 'started_unix_ns': start,
                'finished_unix_ns': end, 'transport': backend, 'status': 'all_HTTP_results_received'}
            if backend == 'batchgen':
                receipt['batch_id'] = batch
            else:
                receipt['requests'] = [{'id': key, 'started_unix_ns': start + 1,
                                        'finished_unix_ns': end - 1} for key in group['ids']]
            run['collection_receipts'].append(receipt)
            for key in group['ids']:
                record = {'id': key}
                if backend == 'batchgen':
                    record['batch_id'] = batch
                run['records'].append(record)
                for kind in ('input_tokens', 'logits', 'completion'):
                    events.append({'event': kind, 'request_id': key,
                        'native_request_id': key + '@' + batch if backend == 'batchgen' else key,
                        'observed_unix_ns': start + 40})
        return run, events

    def validate(self, run, events=(), suite=None, long_stage=True):
        suite = suite or self.suite
        return tool.validate_collection_evidence(run, suite, tool.value_sha(suite), long_stage, events)

    def test_actual_prepare_changes_only_collection_metadata_and_schedule_is_exact84(self):
        args = SimpleNamespace(suite_version='v2', cohort_size=16, tokenizer_json=self.root / 'tokenizer.json',
            tokenizer_sha256=tool.sha(self.root / 'tokenizer.json'), execute=True, output=self.root / 'cohort16.json')
        with patch.object(tool, 'reference_tokenizer', return_value=V.TinyTokenizer()), \
             patch.object(tool, 'exact_length_prompt', side_effect=lambda tok, length: ('original fixed boundary', [0] + [2] * (length - 1))), \
             patch.object(tool, 'exact_text', side_effect=lambda tok, ids: 'native-prefix:' + tool.value_sha(ids)), \
             contextlib.redirect_stdout(io.StringIO()):
            tool.prepare(args)
        actual = json.loads(args.output.read_text())
        self.assertEqual(actual['jobs'], self.original['jobs'])
        self.assertEqual(actual['numeric_contract'], self.original['numeric_contract'])
        self.assertEqual(actual['sampling'], self.original['sampling'])
        self.assertEqual(actual['budget_protocol'], self.original['budget_protocol'])
        self.assertEqual([len(g['ids']) for g in self.plan['cohorts']], [16, 16, 16, 16, 16, 2, 1, 1])
        self.assertEqual(self.plan['cohorts'][0]['kind'], 'coverage')
        self.assertEqual([g['kind'] for g in self.plan['cohorts'][-2:]], ['long', 'long'])
        self.assertEqual({key for g in self.plan['cohorts'] for key in g['ids']}, {j['id'] for j in actual['jobs']})

    def test_independent_96_stays_full_context_and_ordered_six_groups_of16(self):
        suite = C.CrossNodeTraceSealing.independent_suite()
        original_jobs = copy.deepcopy(suite['jobs'])
        suite.update(collection_protocol=tool.collection_protocol(16), collection_source_sha256=tool.sha(tool.__file__))
        semantic.validate_suite(suite)
        plan = tool.collection_schedule(suite, tool.value_sha(suite), False)
        self.assertEqual([len(g['ids']) for g in plan['cohorts']], [16] * 6)
        self.assertEqual([key for g in plan['cohorts'] for key in g['ids']], [j['id'] for j in original_jobs])
        self.assertEqual(suite['jobs'], original_jobs)
        run, events = self.fixture_run('sglang', suite, False)
        self.assertEqual(self.validate(run, events, suite, False)['cohort_count'], 6)

    def test_legacy_serial_records_remain_explicitly_legacy_and_cannot_be_upgraded(self):
        legacy = tool.validate_collection_evidence({}, self.original, tool.value_sha(self.original), True, [])
        self.assertEqual(legacy['mode'], 'legacy_serial_receipts_unavailable')
        self.assertEqual(legacy['cohort_count'], 69)
        with self.assertRaisesRegex(ValueError, 'schedule is missing'):
            self.validate({})
        with self.assertRaisesRegex(ValueError, 'differs from frozen suite'):
            tool.requested_collection_schedule(SimpleNamespace(cohort_size=1, suite_sha256=self.suite_sha), self.suite, True)
        old_independent = C.CrossNodeTraceSealing.independent_suite()
        self.assertEqual(len(tool.collection_schedule(old_independent, 'b' * 64, False)['cohorts']), 81)

    def test_bg_collection_uses_actual_batch_ids_and_accepts_reordered_output_without_zip(self):
        calls = []
        def fake(_transport, jobs, _model, _timeout, context_limit):
            batch = 'actual-' + str(len(calls))
            calls.append(([j['id'] for j in jobs], context_limit))
            return [{'id': j['id'], 'batch_id': batch} for j in reversed(jobs)]
        run, _ = self.fixture_run()
        run.update(records=[], collection_receipts=[], started_unix_ns=time.time_ns())
        with patch.object(tool, 'collect_bg_batch', side_effect=fake):
            tool.collect_cohorts(None, self.suite, self.plan, run, 'model', 7200, self.root / 'progress.json', 16384)
        run['finished_unix_ns'] = time.time_ns()
        self.assertEqual([ids for ids, _ in calls], [g['ids'] for g in self.plan['cohorts']])
        self.assertEqual(run['records'][0]['id'], self.plan['cohorts'][0]['ids'][-1])
        self.assertEqual(self.validate(run)['cohort_count'], 8)
        self.assertTrue(all(context == 16384 for _, context in calls))

    def test_sg_submits_each_group_concurrently_and_completes_barrier_before_next(self):
        membership = {key: index for index, group in enumerate(self.plan['cohorts']) for key in group['ids']}
        barriers = [threading.Barrier(len(g['ids'])) for g in self.plan['cohorts']]
        finished = [0] * len(barriers)
        lock = threading.Lock()
        def fake(_transport, job):
            index = membership[job['id']]
            with lock:
                if index:
                    self.assertEqual(finished[index - 1], len(self.plan['cohorts'][index - 1]['ids']))
            barriers[index].wait(timeout=5)
            with lock:
                finished[index] += 1
            return {'id': job['id']}
        run, _ = self.fixture_run('sglang')
        run.update(records=[], collection_receipts=[], started_unix_ns=time.time_ns())
        with patch.object(tool, 'collect_sg', side_effect=fake):
            tool.collect_cohorts(None, self.suite, self.plan, run, 'model', 7200, self.root / 'progress.json', 16384)
        run['finished_unix_ns'] = time.time_ns()
        self.assertEqual(finished, [len(g['ids']) for g in self.plan['cohorts']])
        self.assertEqual(self.validate(run)['cohort_count'], 8)

    def test_failed_cohort_never_submits_a_later_group_or_retries(self):
        calls = []
        def fake(_transport, jobs, *_args):
            calls.append([j['id'] for j in jobs])
            if len(calls) == 2:
                raise TimeoutError('synthetic failed group')
            return [{'id': j['id'], 'batch_id': 'only-success'} for j in jobs]
        run, _ = self.fixture_run()
        run.update(records=[], collection_receipts=[])
        with patch.object(tool, 'collect_bg_batch', side_effect=fake), self.assertRaises(TimeoutError):
            tool.collect_cohorts(None, self.suite, self.plan, run, 'model', 7200, self.root / 'progress.json', 16384)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(run['collection_receipts']), 1)
        self.assertEqual(len(run['records']), 16)

    def test_mixed_teacher_and_generation_batch_preserves_each_request_payload(self):
        jobs = [next(j for j in self.suite['jobs'] if j['kind'] == 'teacher'),
                next(j for j in self.suite['jobs'] if j['kind'] == 'generation' and j['semantic'])]
        captured = []
        class Transport:
            def call(self, method, path, payload=None, content_type=None, binary=False):
                captured.append((method, path, payload))
                if path == '/v1/files': return {'id': 'input-file'}
                if path == '/v1/batches': return {'id': 'actual-batch', 'status': 'completed', 'output_file_id': 'output-file'}
                rows = [{'custom_id': job['id'], 'response': {'status_code': 200, 'body': {
                    'choices': [{'text': 'synthetic', 'finish_reason': 'length' if job['ignore_eos'] else 'stop'}],
                    'usage': {'prompt_tokens': len(job['input_ids']), 'completion_tokens': 1}}}} for job in reversed(jobs)]
                return ('\n'.join(json.dumps(row) for row in rows) + '\n').encode()
        records = tool.collect_bg_batch(Transport(), jobs, 'model', 7200)
        payload = captured[0][2].decode()
        requests = [json.loads(line) for line in payload.splitlines() if line.startswith('{')]
        self.assertEqual([row['custom_id'] for row in requests], [j['id'] for j in jobs])
        self.assertEqual([row['body']['max_tokens'] for row in requests], [1, 8192])
        self.assertEqual([row['body']['ignore_eos'] for row in requests], [True, False])
        self.assertEqual([row['body']['prompt'] for row in requests], [j['prompt'] for j in jobs])
        self.assertEqual([row['id'] for row in records], [j['id'] for j in reversed(jobs)])
        self.assertTrue(all(row['batch_id'] == 'actual-batch' for row in records))
        for job in jobs:
            request = tool.sg_request(job)
            self.assertEqual(request['rid'], job['id'])
            self.assertEqual(request['input_ids'], job['input_ids'])
            self.assertEqual(request['sampling_params']['ignore_eos'], job['ignore_eos'])
            self.assertEqual(request['sampling_params']['max_new_tokens'], job['max_new_tokens'])

    def test_missing_duplicate_different_or_reordered_groups_fail(self):
        run, events = self.fixture_run()
        mutations = []
        bad = copy.deepcopy(run); bad['collection_receipts'].pop(); mutations.append(bad)
        bad = copy.deepcopy(run); bad['collection_receipts'][1] = copy.deepcopy(bad['collection_receipts'][0]); mutations.append(bad)
        bad = copy.deepcopy(run); bad['collection_receipts'][0]['received_ids'][0] = 'unknown'; mutations.append(bad)
        bad = copy.deepcopy(run); bad['collection_receipts'][0]['received_ids'][0] = bad['collection_receipts'][0]['received_ids'][1]; mutations.append(bad)
        bad = copy.deepcopy(run); bad['collection_receipts'][0]['planned_ids'].reverse(); mutations.append(bad)
        bad = copy.deepcopy(run); bad['collection_schedule']['cohorts'][0]['ids'].reverse(); bad['collection_schedule_sha256'] = tool.value_sha(bad['collection_schedule']); mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):
                self.validate(bad, events)

    def test_overlapping_group_and_cross_barrier_native_events_fail(self):
        run, events = self.fixture_run()
        bad = copy.deepcopy(run)
        bad['collection_receipts'][1]['started_unix_ns'] = bad['collection_receipts'][0]['finished_unix_ns'] - 1
        with self.assertRaisesRegex(ValueError, 'barrier overlaps'):
            self.validate(bad, events)
        bad_events = copy.deepcopy(events)
        bad_events[0]['observed_unix_ns'] = run['collection_receipts'][1]['started_unix_ns']
        with self.assertRaisesRegex(ValueError, 'evidence crossed'):
            self.validate(run, bad_events)
        sg, sg_events = self.fixture_run('sglang')
        sg_events[0]['observed_unix_ns'] = sg['collection_receipts'][0]['started_unix_ns']
        with self.assertRaisesRegex(ValueError, 'evidence crossed'):
            self.validate(sg, sg_events)  # Inside cohort, outside this HTTP request.

    def test_wrong_native_id_batch_mapping_and_source_binding_fail(self):
        run, events = self.fixture_run()
        self.assertEqual(self.validate(run, events)['cohort_count'], 8)
        wrong = copy.deepcopy(events)
        wrong[0]['native_request_id'] = wrong[0]['request_id'] + '@different-batch'
        with self.assertRaisesRegex(ValueError, 'Native request identity'):
            self.validate(run, wrong)
        for field in ('tool_sha256', 'collection_source_sha256'):
            bad = copy.deepcopy(run); bad[field] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'collector source identities differ'):
                self.validate(bad, events)
        bad = copy.deepcopy(run); bad['records'][0]['batch_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'wrong actual cohort batch'):
            self.validate(bad, events)
        bad_suite = copy.deepcopy(self.suite); bad_suite['collection_source_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original frozen collection source'):
            tool.validate_suite(bad_suite)

    def test_mixed_engine_schedules_cannot_be_compared(self):
        run, events = self.fixture_run()
        cohort = self.validate(run, events)
        legacy = tool.validate_collection_evidence({}, self.original, tool.value_sha(self.original), True, [])
        with self.assertRaisesRegex(ValueError, 'different collection schedules'):
            tool.require_same_collection_schedule({'collection': cohort}, {'collection': legacy})
        tool.require_same_collection_schedule({'collection': cohort}, {'collection': copy.deepcopy(cohort)})

    def test_existing_native_evidence_gate_rejects_cross_barrier_and_never_qualifies_missing_traces(self):
        fixture_root = self.root / 'evidence'
        fixture_root.mkdir()
        common, _, original = C.CrossNodeTraceSealing().fixture(fixture_root, suite=self.suite)
        schedule_run, _ = self.fixture_run('sglang', long_stage=False)
        for field in ('collection_schedule', 'collection_schedule_sha256', 'collection_source_sha256', 'collection_receipts'):
            original[field] = schedule_run[field]
        # This fixture uses byte SHA, while fixture_run uses canonical value SHA.
        plan = tool.collection_schedule(self.suite, common['suite_sha256'], False)
        original.update(collection_schedule=plan, collection_schedule_sha256=tool.value_sha(plan),
                        started_unix_ns=1, finished_unix_ns=10000)
        common['original_run'].write_text(json.dumps(original))
        evidence = tool.validate_evidence(common['original_run'], self.suite, common['suite_sha256'], False)
        self.assertTrue(evidence['missing_evidence'])
        self.assertEqual(evidence['collection']['cohort_count'], 6)
        original['collection_receipts'].pop()
        common['original_run'].write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'Missing/extra collection cohort'):
            tool.validate_evidence(common['original_run'], self.suite, common['suite_sha256'], False)


if __name__ == '__main__':
    unittest.main()
