"""CPU-only causal evidence checks; synthetic records never qualify a model."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.r1_reproduction import qualify_r1_full_model as core
from scripts.r1_reproduction import r1_causal_evidence as causal
from scripts.r1_reproduction import r1_full_model_trace as trace


def identity(node, pid):
    return {'hostname': 'node' + str(node), 'boot_id': ('0' if node == 0 else '1') * 8 + '-1111-2222-3333-444444444444',
            'pid': pid, 'start_ticks': pid + 1, 'helper_sha256': causal.digest(causal.__file__),
            'time_namespace': {'device': 4, 'inode': 1234, 'link': 'time:[1234]'},
            'pid_namespace': {'device': 4, 'inode': 4321, 'link': 'pid:[4321]'},
            'timens_offsets': {'monotonic': [0, 0], 'boottime': [0, 0]}}


def stamp(value):
    return {'unix_ns': 500, 'monotonic_before_ns': value, 'monotonic_after_ns': value + 1}


def fixture(backend='batchgen'):
    suite = {'schema': core.SCHEMA, 'tool_sha256': core.sha(core.__file__),
             'collection_source_sha256': core.sha(core.__file__),
             'collection_protocol': core.collection_protocol(1, 'node_monotonic_causal_v2'),
             'jobs': [{'id': 'job' + str(i), 'kind': 'generation', 'stage': 'short'} for i in range(2)]}
    suite_sha = core.value_sha(suite)
    schedule = core.collection_schedule(suite, suite_sha, False)
    binding = {'schema': causal.CLOCK_SCHEMA, 'run_id': 'fresh-run', 'suite_sha256': suite_sha,
               'deployment_sha256': 'd' * 64, 'helper_sha256': causal.digest(causal.__file__), 'client_ip': '10.0.0.1',
               'nodes': [{'node_rank': i, 'ip': '10.0.0.' + str(i + 1), 'port': 18973, 'hostname': 'node' + str(i)} for i in range(2)]}
    run = {'backend': backend, 'deployment': {'run_id': 'fresh-run', 'artifact_sha256': 'd' * 64},
           'collection_protocol': suite['collection_protocol'], 'causal_binding': binding,
           'causal_collector_identity': identity(0, 99), 'tool_sha256': core.sha(core.__file__),
           'collection_source_sha256': core.sha(core.__file__), 'collection_schedule': schedule,
           'collection_schedule_sha256': core.value_sha(schedule), 'collection_receipts': [], 'records': [],
           'started_monotonic_ns': 1, 'finished_monotonic_ns': 5000,
           'started_unix_ns': 9999999, 'finished_unix_ns': 1}  # Wall step is irrelevant only in explicit v2.
    for index, group in enumerate(schedule['cohorts']):
        base = 1000 + index * 1000
        receipt = {'index': index, 'kind': group['kind'], 'planned_ids': group['ids'], 'received_ids': group['ids'],
                   'started_monotonic_ns': base + 100, 'finished_monotonic_ns': base + 500,
                   'started_unix_ns': 900000, 'finished_unix_ns': 10,
                   'transport': backend, 'status': 'all_HTTP_results_received'}
        if backend == 'batchgen':
            receipt['batch_id'] = 'batch' + str(index)
        else:
            receipt['requests'] = [{'id': key, 'started_monotonic_ns': base + 101,
                                    'finished_monotonic_ns': base + 499,
                                    'started_unix_ns': 900000, 'finished_unix_ns': 10} for key in group['ids']]
        for phase in ('begin', 'end'):
            rows = []
            for node in (0, 1):
                value = base + node * 20 + (600 if phase == 'end' else 0)
                request = causal.make_request(binding, node, index, phase,
                    format(index * 4 + (2 if phase == 'end' else 0) + node, '032x'))
                row = dict(request, schema=causal.ACK_SCHEMA, client_identity=run['causal_collector_identity'],
                    client_send=stamp(value), client_receive=stamp(value + 10),
                    server_identity=identity(node, 100 + node), server_receive=stamp(value + 2 + node * 1000000),
                    server_send=stamp(value + 5 + node * 1000000))
                rows.append(row)
            receipt['causal_' + phase] = rows
        run['collection_receipts'].append(receipt)
        run['records'].extend(dict(id=key, **({'batch_id': receipt['batch_id']} if backend == 'batchgen' else {})) for key in group['ids'])
    events = []
    for rank in range(16):
        actual = identity(rank // 8, 200 + rank)
        envelope = {'rank': rank, 'clock_identity_sha256': core.value_sha(actual)}
        events.append(dict(envelope, event='trace_start', ordinal=1, observed_unix_ns=999999,
            observed_monotonic_ns=500 + (rank // 8) * 1000000, actual_pid=actual['pid'], clock_identity=actual))
        for index, group in enumerate(schedule['cohorts']):
            key = group['ids'][0]
            events.append(dict(envelope, event='completion', ordinal=index + 2, request_id=key,
                native_request_id=key + '@batch' + str(index) if backend == 'batchgen' else key,
                observed_unix_ns=999999, observed_monotonic_ns=1300 + index * 1000 + (rank // 8) * 1000000))
    return suite, run, events


def validate(suite, run, events):
    return core.validate_collection_evidence(run, suite, core.value_sha(suite), False, events)


def add_closed_workers(run, events):
    rows = []
    for node in (0, 1):
        identities = [event['clock_identity'] for event in events
                      if event['event'] == 'trace_start' and event['rank'] // 8 == node]
        closure = {'schema': causal.CLOSURE_SCHEMA, 'observer_identity': identity(node, 900 + node),
                   'observed_monotonic_ns': 3000 + node * 1000000,
                   'source_sha256': causal.digest(causal.__file__),
                   'workers': [{'identity': actual, 'status': 'original_exited_no_proc',
                                'observed_process': None} for actual in identities]}
        rows.append({'node_rank': node, 'closure': closure})
    run['trace'] = {'causal_worker_closures': rows}


class CausalCollection(unittest.TestCase):
    def test_both_native_APIs_use_same_explicit_local_domains_not_wall_conversion(self):
        for backend in ('batchgen', 'sglang'):
            suite, run, events = fixture(backend)
            result = validate(suite, run, events)
            self.assertEqual(result['time_basis'], 'node_monotonic_causal_v2')
            self.assertFalse(result['clock_drift_assumption_used'])
            self.assertEqual(result['node_cohort_intervals']['1'][0], [1001026, 1001622])

    def test_missing_replayed_reordered_early_late_and_changed_domain_fail_closed(self):
        cases = ('missing_end', 'missing_node', 'replay_nonce', 'wrong_cohort', 'HTTP_before_ACK',
                 'next_begin_before_previous_end', 'server_restarted', 'client_restarted', 'worker_domain',
                 'missing_rank', 'same_process_two_ranks', 'endpoint_claims_worker', 'missing_mono',
                 'late', 'early', 'wrong_native', 'missing_ordinal', 'unsubmitted')
        for mode in cases:
            suite, run, events = fixture()
            receipt = run['collection_receipts'][0]
            if mode == 'missing_end': del receipt['causal_end']
            elif mode == 'missing_node': receipt['causal_begin'].pop()
            elif mode == 'replay_nonce': receipt['causal_end'][0]['nonce'] = receipt['causal_begin'][0]['nonce']
            elif mode == 'wrong_cohort': receipt['causal_end'][0]['cohort_index'] = 1
            elif mode == 'HTTP_before_ACK': receipt['started_monotonic_ns'] = 1001
            elif mode == 'next_begin_before_previous_end': run['collection_receipts'][1]['causal_begin'][0]['client_send'] = stamp(1200)
            elif mode == 'server_restarted': receipt['causal_end'][0]['server_identity']['pid'] += 10
            elif mode == 'client_restarted': receipt['causal_end'][0]['client_identity'] = identity(0, 109)
            elif mode == 'worker_domain': events[0]['clock_identity']['timens_offsets']['monotonic'] = [1, 0]
            elif mode == 'missing_rank': events = [e for e in events if e['rank'] != 15]
            elif mode in ('same_process_two_ranks', 'endpoint_claims_worker'):
                actual = identity(0, 200 if mode == 'same_process_two_ranks' else 100)
                for event in events:
                    if event['rank'] == 1:
                        event['clock_identity_sha256'] = core.value_sha(actual)
                        if event['event'] == 'trace_start':
                            event.update(clock_identity=actual, actual_pid=actual['pid'])
            elif mode == 'missing_mono': del events[1]['observed_monotonic_ns']
            elif mode == 'late': events[2]['observed_monotonic_ns'] = 2700
            elif mode == 'early': events[1]['observed_monotonic_ns'] = 999
            elif mode == 'wrong_native': events[1]['native_request_id'] = 'job0@another-batch'
            elif mode == 'missing_ordinal': events[1]['ordinal'] = 3
            else: events[1]['request_id'] = 'unsubmitted'
            with self.subTest(mode=mode), self.assertRaises((ValueError, KeyError)):
                validate(suite, run, events)

    def test_legacy_requires_legacy_wall_and_cannot_receive_new_run(self):
        suite, run, events = fixture()
        del suite['collection_protocol']; del suite['collection_source_sha256']
        with self.assertRaises(ValueError): core.run_time_interval(run, suite)
        run.pop('collection_protocol')
        with self.assertRaises(ValueError): core.run_time_interval(run, suite)
        suite, run, events = fixture()
        del run['collection_protocol']
        with self.assertRaises(ValueError): validate(suite, run, events)

    def test_snapshot_keeps_late_raw_events_only_under_explicit_causal_option(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / 'source'; source.mkdir()
            event = {'run_id': 'r', 'suite_sha256': 's', 'deployment_sha256': 'd', 'event': 'completion', 'observed_unix_ns': 999}
            (source / 'rank-0-pid-1.jsonl').write_text(json.dumps(event) + '\n')
            old = core.snapshot_trace(source, base / 'old', {'run_id': 'r', 'artifact_sha256': 'd'}, 's', 10, 20)
            new = core.snapshot_trace(source, base / 'new', {'run_id': 'r', 'artifact_sha256': 'd'}, 's', 10, 20, causal=True)
            self.assertEqual(old['event_count'], 0); self.assertEqual(new['event_count'], 1)

    def test_collector_RPCs_causally_surround_each_HTTP_cohort_without_retry(self):
        suite, run, _ = fixture()
        run['records'], run['collection_receipts'] = [], []
        order = []
        def exchange(binding, node, index, phase, timeout):
            order.append((phase, index, node))
            return {'node_rank': node}
        def collect(_transport, jobs, *_args):
            index = int(jobs[0]['id'][-1]); order.append(('HTTP', index))
            return [{'id': j['id'], 'batch_id': 'batch' + str(index)} for j in jobs]
        transport = SimpleNamespace(deadline=time.monotonic() + 60)
        with tempfile.TemporaryDirectory() as directory, patch.object(causal, 'exchange', side_effect=exchange), patch.object(core, 'collect_bg_batch', side_effect=collect):
            core.collect_cohorts(transport, suite, run['collection_schedule'], run, 'r1', 60, Path(directory) / 'progress.json', 16384)
        self.assertEqual(order, [(phase, index, node) if phase != 'HTTP' else (phase, index)
            for index in range(2) for phase, node in [('begin', 0), ('begin', 1), ('HTTP', 0), ('end', 0), ('end', 1)]])
        self.assertTrue(all(r['started_monotonic_ns'] <= r['finished_monotonic_ns'] for r in run['collection_receipts']))

    def test_late_summary_cannot_manufacture_in_window_forward_coverage(self):
        suite, run, events = fixture(); collection = validate(suite, run, events)
        summary = {'rank': 8, 'observed_monotonic_ns': 9999999, 'last_nonempty_observed_monotonic_ns': 1001300}
        self.assertTrue(core.observation_inside_collection(summary, collection, run, 'last_nonempty_observed'))
        summary['last_nonempty_observed_monotonic_ns'] = 1001700
        self.assertFalse(core.observation_inside_collection(summary, collection, run, 'last_nonempty_observed'))

    def test_trace_emit_stamps_each_event_and_does_not_change_legacy_schema(self):
        import io
        r = trace.Recorder.__new__(trace.Recorder)
        r.ordinal=0; r.rank=0; r.world_size=16; r.run_id='r'; r.suite_sha='s'; r.deployment_sha='d'
        r.clock_identity=identity(0,200); r.stream=io.StringIO()
        first=r.emit('completion'); second=r.emit('completion')
        self.assertLessEqual(first['observed_monotonic_ns'],second['observed_monotonic_ns'])
        self.assertEqual(first['clock_identity_sha256'],core.value_sha(r.clock_identity))
        r.clock_identity=None
        self.assertNotIn('observed_monotonic_ns',r.emit('legacy'))

    def test_live_prefix_is_not_closed_and_both_complete_original_node_seals_are_required(self):
        suite, run, events = fixture()
        self.assertFalse(core.validate_causal_closures(run, events))
        add_closed_workers(run, events)
        self.assertTrue(core.validate_causal_closures(run, events))

    def test_closed_seals_cannot_omit_writers_or_precede_complete_raw_events(self):
        for mode in ('missing_node', 'reversed_nodes', 'missing_worker', 'wrong_worker',
                     'wrong_pid_namespace', 'wrong_clock_domain', 'before_last_ack', 'late_raw_event'):
            suite, run, events = fixture()
            add_closed_workers(run, events)
            wrappers = run['trace']['causal_worker_closures']
            closure = wrappers[0]['closure']
            if mode == 'missing_node': wrappers.pop()
            elif mode == 'reversed_nodes': wrappers.reverse()
            elif mode == 'missing_worker': closure['workers'].pop()
            elif mode == 'wrong_worker': closure['workers'][0]['identity'] = identity(0, 400)
            elif mode == 'wrong_pid_namespace':
                closure['observer_identity']['pid_namespace'] = {'device': 4, 'inode': 5555, 'link': 'pid:[5555]'}
            elif mode == 'wrong_clock_domain': closure['observer_identity']['timens_offsets']['monotonic'] = [1, 0]
            elif mode == 'before_last_ack': closure['observed_monotonic_ns'] = 2601
            else: events.append(dict(events[0], event='layer_summary', ordinal=4, observed_monotonic_ns=3001))
            with self.subTest(mode=mode), self.assertRaises((ValueError, KeyError)):
                core.validate_causal_closures(run, events)


if __name__ == '__main__': unittest.main()
