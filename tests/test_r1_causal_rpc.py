"""Pure CPU causal metadata/RPC mocks; no sockets, GPU or remote execution."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/r1_causal_evidence.py'
SPEC = importlib.util.spec_from_file_location('r1_causal_rpc_test_tool', PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def own(node=0, pid=None):
    return {'hostname': 'actual-node' + str(node), 'boot_id': ('a' if node == 0 else 'b') * 8 + '-1234-5678-9012-123456789abc',
            'pid': pid or 100 + node, 'start_ticks': 1000 + node,
            'time_namespace': {'device': 4, 'inode': 4026531834, 'link': 'time:[4026531834]'},
            'pid_namespace': {'device': 4, 'inode': 4026532000, 'link': 'pid:[4026532000]'},
            'timens_offsets': {'monotonic': [0, 0], 'boottime': [0, 0]},
            'helper_sha256': tool.digest(tool.__file__)}


def binding():
    return {'schema': tool.CLOCK_SCHEMA, 'run_id': 'new-causal-run', 'suite_sha256': 'a' * 64,
            'deployment_sha256': 'b' * 64, 'helper_sha256': tool.digest(tool.__file__),
            'client_ip': '10.1.0.1', 'nodes': [
                {'node_rank': i, 'ip': '10.1.0.' + str(i + 1), 'port': 18879, 'hostname': 'actual-node' + str(i)}
                for i in range(2)]}


def stamp(value, wall=99999999):
    return {'unix_ns': wall, 'monotonic_before_ns': value, 'monotonic_after_ns': value + 1}


def ack(node=1, index=0, phase='begin'):
    b = binding()
    row = tool.make_request(b, node, index, phase, 'c' * 32)
    return dict(row, schema=tool.ACK_SCHEMA, client_identity=own(0), client_send=stamp(10),
                client_receive=stamp(30), server_identity=own(node), server_receive=stamp(1000),
                server_send=stamp(1010))


class CausalMetadata(unittest.TestCase):
    def test_binding_is_exact_frozen_two_node_contract(self):
        b = binding()
        self.assertIs(tool.validate_binding(b, b['run_id'], b['suite_sha256'], b['deployment_sha256']), b)
        mutations = (
            lambda x: x.update(command='sh'), lambda x: x.update(client_ip='10.1.0.2'),
            lambda x: x.update(helper_sha256='c' * 64), lambda x: x.update(run_id='../other'),
            lambda x: x['nodes'][0].update(node_rank=False), lambda x: x['nodes'][1].update(port=True),
            lambda x: x['nodes'][1].update(ip='10.1.0.1'), lambda x: x['nodes'][1].update(ip='0.0.0.0'),
            lambda x: x['nodes'][1].update(hostname='actual-node0'), lambda x: x['nodes'].reverse(),
            lambda x: x['nodes'][0].update(hostname=123), lambda x: x.update(nodes=x['nodes'][:1]),
        )
        for mutation in mutations:
            candidate = copy.deepcopy(b)
            mutation(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                tool.validate_binding(candidate, b['run_id'], b['suite_sha256'], b['deployment_sha256'])

    def test_same_domain_allows_distinct_processes_but_not_clock_domain_changes(self):
        a, b = own(), own(pid=999)
        b['start_ticks'] = 9000
        self.assertTrue(tool.same_domain(a, b))
        b['pid_namespace'] = {'device': 4, 'inode': 4026532001, 'link': 'pid:[4026532001]'}
        self.assertTrue(tool.same_domain(a, b))  # A PID namespace is not a clock domain.
        self.assertFalse(tool.same_domain(own(0), own(1)))
        for field in ('hostname', 'boot_id', 'time_namespace', 'timens_offsets'):
            b = copy.deepcopy(a)
            if field == 'hostname': b[field] = 'other-node'
            elif field == 'boot_id': b[field] = own(1)[field]
            elif field == 'time_namespace': b[field]['device'] += 1
            else: b[field]['monotonic'][0] += 1
            self.assertFalse(tool.same_domain(a, b))
        for field in a:
            b = copy.deepcopy(a)
            del b[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.validate_identity(b)
        for field, value in (('pid', True), ('start_ticks', 1.0), ('hostname', 123)):
            b = dict(a, **{field: value})
            with self.assertRaises(ValueError): tool.validate_identity(b)

    def test_identity_reads_actual_process_namespace_and_offsets(self):
        reads = {'/proc/self/timens_offsets': 'monotonic 0 0\nboottime -2 123\n',
                 '/proc/self/stat': '321 (space and ) name) ' + ' '.join(['S'] + ['0'] * 18 + ['444'] + ['0'] * 3),
                 '/proc/sys/kernel/random/boot_id': own()['boot_id'] + '\n'}
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            return reads[str(path)] if str(path) in reads else original_read(path, *args, **kwargs)
        def namespace_stat(path, *args, **kwargs):
            return SimpleNamespace(st_dev=4, st_ino=4026532000 if str(path).endswith('/pid') else 4026531834)
        def namespace_link(path):
            return 'pid:[4026532000]' if str(path).endswith('/pid') else 'time:[4026531834]'
        with patch.object(Path, 'read_text', read), patch.object(Path, 'stat', namespace_stat), \
                patch.object(tool.os, 'readlink', namespace_link), \
                patch.object(tool.os, 'getpid', return_value=321), patch.object(tool.socket, 'gethostname', return_value='actual-node0'):
            actual = tool.identity()
        self.assertEqual(actual['start_ticks'], 444)
        self.assertEqual(actual['pid'], 321)
        self.assertEqual(actual['timens_offsets']['boottime'], [-2, 123])
        self.assertEqual(actual['pid_namespace'], own()['pid_namespace'])

    def test_no_cross_node_wall_or_monotonic_conversion_is_implied(self):
        value = ack()
        value['client_receive']['unix_ns'] = 3  # wall step: retained, not used for the causal order.
        self.assertEqual(tool.validate_exchange(value, binding()), value)
        value['server_send'] = stamp(999)
        with self.assertRaisesRegex(ValueError, 'order reversed'): tool.validate_exchange(value, binding())
        value = ack()
        value['client_receive']['monotonic_before_ns'] = 9
        with self.assertRaises(ValueError): tool.validate_exchange(value, binding())

    def test_ack_rejects_missing_forged_bool_and_nonfinite_metadata(self):
        for field in ack():
            candidate = ack()
            del candidate[field]
            with self.subTest(field=field), self.assertRaises(ValueError): tool.validate_exchange(candidate, binding())
        for field, value in (('cohort_index', True), ('node_rank', False), ('nonce', int('1' * 32)),
                             ('suite_sha256', 'd' * 64), ('binding_sha256', 'd' * 64)):
            candidate = ack()
            candidate[field] = value
            with self.assertRaises(ValueError): tool.validate_exchange(candidate, binding())
        for raw in ('{"phase":"begin","phase":"end"}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError): tool.decode_json(raw)

    def test_session_rejects_replay_reordering_and_changed_identity(self):
        own_server = own(1)
        session = tool.Session(binding(), 1, own_server)
        with patch.object(tool, 'identity', return_value=own_server), patch.object(tool, 'stamp', return_value=stamp(100)):
            begin = tool.make_request(binding(), 1, 0, 'begin', 'a' * 32)
            self.assertEqual(session.answer(begin, stamp(90))['nonce'], 'a' * 32)
            for request in (begin, tool.make_request(binding(), 1, 1, 'begin', 'b' * 32),
                            tool.make_request(binding(), 1, 0, 'end', 'a' * 32)):
                with self.assertRaises(ValueError): session.answer(request, stamp(90))
            session.answer(tool.make_request(binding(), 1, 0, 'end', 'b' * 32), stamp(90))
            session.answer(tool.make_request(binding(), 1, 1, 'begin', 'c' * 32), stamp(90))
        self.assertEqual((session.next_index, session.next_phase), (1, 'end'))
        with patch.object(tool, 'identity', return_value=own(1, pid=999)), self.assertRaisesRegex(ValueError, 'identity changed'):
            session.answer(tool.make_request(binding(), 1, 1, 'end', 'd' * 32), stamp(90))

    def test_plan_does_not_observe_process_or_open_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'binding.json'
            path.write_text(json.dumps(binding()))
            with patch.object(tool, 'identity', side_effect=AssertionError('no process observation')), \
                    patch.object(tool, 'Server', side_effect=AssertionError('no sockets')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(tool.main(['serve', '--binding', str(path), '--binding-sha256', tool.digest(path),
                                           '--node-rank', '1', '--duration-seconds', '30']), 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'plan_only_NOT_executed')


class MockRPC(unittest.TestCase):
    def run_exchange(self, mutate=None, status=200):
        observed = {}
        class Connection:
            def __init__(self, *args, **kwargs): observed['connection'] = (args, kwargs)
            def request(self, method, path, body, headers):
                observed.update(method=method, path=path, request=json.loads(body), headers=headers)
            def getresponse(self):
                payload = dict(observed['request'], schema=tool.ACK_SCHEMA, server_identity=own(1),
                               server_receive=stamp(1000), server_send=stamp(1005))
                if mutate: mutate(payload)
                raw = json.dumps(payload).encode()
                return SimpleNamespace(status=status, read=lambda size: raw[:size])
            def close(self): observed['closed'] = True
        with patch.object(tool.http.client, 'HTTPConnection', Connection), \
                patch.object(tool, 'identity', return_value=own(0)), \
                patch.object(tool, 'stamp', side_effect=[stamp(10), stamp(30)]):
            result = tool.exchange(binding(), 1, 0, 'begin', 10)
        return result, observed

    def test_client_preserves_exact_nonce_and_actual_rpc_stamps(self):
        result, seen = self.run_exchange()
        self.assertTrue(seen['closed'])
        self.assertEqual(seen['method'], 'POST')
        self.assertEqual(seen['path'], '/r1-causal/new-causal-run')
        self.assertEqual(seen['connection'][1]['source_address'], ('10.1.0.1', 0))
        self.assertEqual(result['nonce'], seen['request']['nonce'])
        self.assertEqual(result['client_receive'], stamp(30))

    def test_client_rejects_mismatched_ack_without_retry(self):
        for mutate in (lambda x: x.update(nonce='d' * 32), lambda x: x.update(cohort_index=1),
                       lambda x: x.update(command='sh'), lambda x: x['server_identity'].update(hostname='other')):
            with self.assertRaises(ValueError): self.run_exchange(mutate)
        with self.assertRaises(ValueError): self.run_exchange(status=302)

    def test_server_rejects_peer_path_body_and_wrong_metadata_without_answering(self):
        request = tool.make_request(binding(), 1, 0, 'begin', 'c' * 32)
        def handler(raw, peer='10.1.0.1', path='/r1-causal/new-causal-run', extra=None):
            obj = tool.Handler.__new__(tool.Handler)
            from email.message import Message
            obj.headers = Message()
            obj.headers['Content-Length'] = str(len(raw))
            obj.headers['Content-Type'] = 'application/json'
            for key, value in (extra or {}).items(): obj.headers[key] = value
            obj.client_address, obj.path = (peer, 1234), path
            obj.server = SimpleNamespace(session=tool.Session(binding(), 1, own(1)))
            obj.rfile, obj.wfile, obj.errors = io.BytesIO(raw), io.BytesIO(), []
            obj.send_error = lambda code: obj.errors.append(code)
            obj.send_response = lambda code: setattr(obj, 'code', code)
            obj.send_header = lambda *_: None
            obj.end_headers = lambda: None
            return obj
        raw = json.dumps(request).encode()
        cases = [handler(raw, peer='10.1.0.99'), handler(raw, path='/any/file'),
                 handler(raw, extra={'Transfer-Encoding': 'chunked'}), handler(b'{}'),
                 handler(raw, extra={'Content-Length': str(len(raw))})]
        for candidate in cases:
            with patch.object(tool, 'stamp', return_value=stamp(10)):
                candidate.do_POST()
            self.assertIn(candidate.errors[0], (400, 403))
            self.assertFalse(candidate.wfile.getvalue())
        valid = handler(raw)
        with patch.object(tool, 'identity', return_value=own(1)), patch.object(tool, 'stamp', side_effect=[stamp(10), stamp(20)]):
            valid.do_POST()
        self.assertEqual(valid.code, 200)
        self.assertEqual(json.loads(valid.wfile.getvalue())['nonce'], request['nonce'])


class ClosedWorkers(unittest.TestCase):
    @staticmethod
    def proc_stat(pid, start, state):
        return str(pid) + ' (known worker) ' + ' '.join([state] + ['0'] * 18 + [str(start)] + ['0'] * 3)

    def capture(self, processes, workers=None):
        workers = workers or [own(0, 101), own(0, 102), own(0, 103)]
        reads = []
        def read(path, *args, **kwargs):
            reads.append(str(path))
            if str(path) not in processes:
                raise AssertionError('Unexpected process/path lookup: ' + str(path))
            value = processes[str(path)]
            if isinstance(value, BaseException): raise value
            return value
        with patch.object(tool, 'identity', return_value=own(0, 999)), patch.object(Path, 'read_text', read), \
                patch.object(Path, 'iterdir', side_effect=AssertionError('No process scanning')), \
                patch.object(tool.time, 'monotonic_ns', return_value=100_000):
            receipt = tool.closed_workers(workers)
        return receipt, reads

    def test_only_absent_and_reused_known_workers_are_closed_without_signals(self):
        receipt, reads = self.capture({
            '/proc/101/stat': FileNotFoundError(),
            '/proc/102/stat': self.proc_stat(102, 2000, 'R'),
            '/proc/103/stat': FileNotFoundError(),
        })
        self.assertEqual(reads, ['/proc/101/stat', '/proc/102/stat', '/proc/103/stat'])
        self.assertEqual([r['status'] for r in receipt['workers']], [
            'original_exited_no_proc', 'original_exited_PID_reused', 'original_exited_no_proc'])
        self.assertIsNone(receipt['workers'][0]['observed_process'])
        self.assertEqual(receipt['workers'][1]['observed_process'], {'pid': 102, 'start_ticks': 2000, 'state': 'R'})
        self.assertEqual(receipt['observed_monotonic_ns'], 100_000)
        with patch.object(Path, 'read_text', side_effect=AssertionError('Offline validator must not inspect /proc')):
            self.assertIs(tool.validate_closure(receipt, [own(0, 101), own(0, 102), own(0, 103)]), receipt)

    def test_live_or_unreadable_worker_fails_closed(self):
        for value in (self.proc_stat(101, 1000, 'R'), self.proc_stat(101, 1000, 'S'),
                      self.proc_stat(101, 1000, 'Z'), self.proc_stat(101, 1000, 'X'), PermissionError(),
                      self.proc_stat(102, 1000, 'Z'), 'malformed stat'):
            with self.subTest(value=value), self.assertRaises((ValueError, PermissionError, IndexError)):
                self.capture({'/proc/101/stat': value}, [own(0, 101)])

    def test_identity_mismatch_self_or_duplicate_rejected_before_proc_lookup(self):
        different_pid_ns = own(0, 101)
        different_pid_ns['pid_namespace'] = {'device': 4, 'inode': 50, 'link': 'pid:[50]'}
        for workers in ([], [own(0, 999)], [own(0, 101), own(0, 101)], [own(1, 101)], [different_pid_ns]):
            with self.subTest(workers=workers), patch.object(tool, 'identity', return_value=own(0, 999)), \
                    patch.object(Path, 'read_text', side_effect=AssertionError('No unknown PID access')), \
                    self.assertRaises(ValueError):
                tool.closed_workers(workers)

    def test_offline_closure_rejects_incomplete_or_inconsistent_proofs(self):
        receipt, _ = self.capture({'/proc/101/stat': self.proc_stat(101, 2000, 'R')}, [own(0, 101)])
        mutations = (
            lambda x: x.update(source_sha256='c' * 64), lambda x: x.update(observed_monotonic_ns=True),
            lambda x: x.update(workers=[]), lambda x: x['workers'].append(copy.deepcopy(x['workers'][0])),
            lambda x: x['workers'][0].update(status='still_live'),
            lambda x: x['workers'][0].update(status='original_exited_no_proc'),
            lambda x: x['workers'][0].update(status='original_exited_terminal_state'),
            lambda x: x['workers'][0]['observed_process'].update(start_ticks=1000, state='Z'),
            lambda x: x['workers'][0]['observed_process'].update(start_ticks=1000, state='X'),
            lambda x: x['workers'][0]['observed_process'].update(start_ticks=True),
            lambda x: x['workers'][0]['identity'].update(pid=102),
            lambda x: x['observer_identity'].update(pid=101),
            lambda x: x['workers'][0].pop('observed_process'),
        )
        for mutation in mutations:
            bad = copy.deepcopy(receipt)
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                tool.validate_closure(bad, [own(0, 101)])


if __name__ == '__main__':
    unittest.main()
