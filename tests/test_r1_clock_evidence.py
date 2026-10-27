"""No sockets, clock changes, GPUs or remote execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r1_clock_test', ROOT / 'scripts/r1_reproduction/r1_clock_evidence.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def stamp(wall, mono=None):
    mono = wall if mono is None else mono
    return {'unix_ns': wall, 'monotonic_before_ns': mono - 1, 'monotonic_after_ns': mono + 1}


def sample(index, base, offset=100, outward=10, inward=30):
    return {'schema': tool.SCHEMA, 'session_id': 'run1', 'ordinal': index,
        'client': {'address': '10.66.9.163', 'hostname': 'client', 'boot_id': 'client-boot', 'source_sha256': tool.digest(tool.__file__)},
        'server': {'address': '10.66.9.166', 'hostname': 'server', 'boot_id': 'server-boot', 'source_sha256': tool.digest(tool.__file__)},
        'client_send': stamp(base), 'server_receive': stamp(base + outward + offset, base + outward),
        'server_send': stamp(base + outward + offset + 10, base + outward + 10),
        'client_receive': stamp(base + outward + 10 + inward)}


class ClockEvidence(unittest.TestCase):
    def samples(self, offset=100):
        return [sample(0, 1000, offset), sample(1, 2000, offset)]

    def test_asymmetric_network_bounds_not_half_rtt_guess(self):
        row = sample(0, 1000)
        self.assertEqual(tool.exchange_bounds(row, 0), (70, 110))
        interval = tool.event_client_interval(1600, self.samples(), 0, 1000, 0)
        self.assertEqual(interval, (1490, 1530))
        self.assertLessEqual(interval[0], 1500)
        self.assertGreaterEqual(interval[1], 1500)

    def test_negative_offset_sign(self):
        self.assertEqual(tool.event_client_interval(1300, self.samples(-200), 0, 1000, 0), (1490, 1530))

    def test_boundary_uncertain_and_outside_are_not_supported(self):
        self.assertEqual(tool.classify((1490, 1530), 1495, 1600), 'uncertain_at_original_window_boundary')
        self.assertEqual(tool.classify((1490, 1530), 1600, 1700), 'outside_original_window')
        self.assertEqual(tool.classify((1490, 1530), 1400, 1600), 'supported_under_declared_clock_assumption')

    def test_wall_step_within_exchange_rejected(self):
        rows = self.samples()
        rows[0]['client_receive']['unix_ns'] += 10
        with self.assertRaisesRegex(ValueError, 'Clock step'):
            tool.validate_samples(rows, 0)

    def test_wall_step_between_probes_rejected(self):
        rows = self.samples()
        for key in ('server_receive', 'server_send'):
            rows[1][key]['unix_ns'] += 20
        with self.assertRaisesRegex(ValueError, 'Clock step'):
            tool.validate_samples(rows, 0)

    def test_identity_boot_source_order_changes_rejected(self):
        for action in ('boot', 'source', 'ordinal', 'session'):
            with self.subTest(action=action):
                rows = self.samples()
                if action == 'boot': rows[1]['server']['boot_id'] = 'rebooted'
                if action == 'source': rows[1]['server']['source_sha256'] = '0' * 64
                if action == 'ordinal': rows[1]['ordinal'] = 0
                if action == 'session': rows[1]['session_id'] = 'run2'
                with self.assertRaises(ValueError): tool.validate_samples(rows, 0)

    def test_brackets_and_gap_required_no_extrapolation(self):
        with self.assertRaisesRegex(ValueError, 'pre/post'):
            tool.event_client_interval(2300, self.samples(), 0, 1000, 0)
        with self.assertRaisesRegex(ValueError, 'pre/post'):
            tool.event_client_interval(1600, self.samples(), 0, 900, 0)

    def test_conflicting_offsets_rejected(self):
        rows = [sample(0, 1000), sample(1, 2000, 200)]
        with self.assertRaisesRegex(ValueError, 'conflict'):
            tool.event_client_interval(1600, rows, 0, 1000, 0)

    def test_drift_enlarges_intervals_never_shrinks(self):
        base = tool.event_client_interval(1600, self.samples(), 0, 1000, 0)
        expanded = tool.event_client_interval(1600, self.samples(), 1_000_000, 1000, 0)
        self.assertLess(expanded[0], base[0])
        self.assertGreater(expanded[1], base[1])

    def test_invalid_policies_and_boolean_timestamps_rejected(self):
        for drift in (-1, True, 1_000_000_000):
            with self.assertRaises(ValueError): tool.event_client_interval(1600, self.samples(), drift, 1000, 0)
        with self.assertRaises(ValueError): tool.event_client_interval(True, self.samples(), 0, 1000, 0)

    def fixture(self, root, window=(1400, 1700), event_ns=1600):
        run = {'backend': 'batchgen', 'suite_sha256': 'suite-sha',
            'deployment': {'run_id': 'run1', 'artifact_sha256': 'deployment-sha'},
            'collection_receipts': [{'planned_ids': ['req1'], 'started_unix_ns': window[0], 'finished_unix_ns': window[1]}]}
        event = {'rank': 8, 'ordinal': 2, 'event': 'logits', 'request_id': 'req1', 'observed_unix_ns': event_ns,
            'run_id': 'run1', 'suite_sha256': 'suite-sha', 'deployment_sha256': 'deployment-sha'}
        args = SimpleNamespace(output=root / 'audit.json', relative_drift_ppb=0, max_probe_gap_ns=1000, step_tolerance_ns=0)
        for key, value in (('samples', '\n'.join(json.dumps(row) for row in self.samples()) + '\n'),
                           ('run', json.dumps(run)), ('events', json.dumps([event]))):
            path = root / (key + '.json')
            path.write_text(value)
            setattr(args, key, path)
            setattr(args, key + '_sha256', tool.digest(path))
        return args

    def test_audit_supports_only_pinned_input_and_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            old = [path.read_bytes() for path in (args.run, args.events, args.samples)]
            result = tool.audit(args)
            self.assertEqual(result['status'], 'supplementary_windows_supported_NOT_qualified')
            self.assertEqual(old, [path.read_bytes() for path in (args.run, args.events, args.samples)])
            with self.assertRaises(FileExistsError): tool.audit(args)
            args.events.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Pinned'): tool.audit(args)

    def test_audit_does_not_repair_failed_original_gate_by_offset(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td), window=(1400, 1550))
            result = tool.audit(args)
            self.assertEqual(result['status'], 'clock_window_evidence_NOT_supported')
            event = json.loads(args.output.read_text())['events'][0]
            self.assertEqual(event['status'], 'supported_under_declared_clock_assumption')
            self.assertIs(event['original_unadjusted_gate'], False)

    def test_audit_rejects_wrong_session_and_raw_run_bindings(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            value = json.loads(args.events.read_text()); value[0]['suite_sha256'] = 'other'
            args.events.write_text(json.dumps(value)); args.events_sha256 = tool.digest(args.events)
            with self.assertRaisesRegex(ValueError, 'different original'): tool.audit(args)
            value = json.loads(args.run.read_text()); value['deployment']['run_id'] = 'run2'
            args.run.write_text(json.dumps(value)); args.run_sha256 = tool.digest(args.run)
            with self.assertRaisesRegex(ValueError, 'different run session'): tool.audit(args)

    def test_unclosed_clock_journal_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            args.samples.write_bytes(args.samples.read_bytes().rstrip(b'\n'))
            args.samples_sha256 = tool.digest(args.samples)
            with self.assertRaisesRegex(ValueError, 'unclosed'): tool.audit(args)


if __name__ == '__main__':
    unittest.main()
