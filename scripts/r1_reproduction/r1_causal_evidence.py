#!/usr/bin/env python3
"""Single-run, metadata-only causal ACKs; no clock adjustment or drift premise.

The service admits one fixed peer/path and exact begin/end metadata. It cannot
execute commands, accept paths, read remote-selected files, or import GPU code.
Import and CLI plan mode do not open sockets. RPC ACKs are not worker drains.
"""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import time

CLOCK_SCHEMA = 'r1-causal-clock-binding-v1'
REQUEST_SCHEMA = 'r1-causal-cohort-request-v1'
ACK_SCHEMA = 'r1-causal-cohort-ack-v1'
SHA = re.compile(r'^[0-9a-f]{64}$')
RUN = re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
HOST = re.compile(r'^[A-Za-z0-9_.-]{1,253}$')
UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
NONCE = re.compile(r'^[0-9a-f]{32}$')
MAX_BODY = 4096
DOMAIN_FIELDS = ('hostname', 'boot_id', 'time_namespace', 'timens_offsets')
IDENTITY_FIELDS = set(DOMAIN_FIELDS) | {'pid', 'start_ticks', 'pid_namespace', 'helper_sha256'}
CLOSURE_SCHEMA = 'r1-causal-worker-closure-v1'
CLOSED_STATUSES = {'original_exited_no_proc', 'original_exited_PID_reused'}
REQUEST_FIELDS = {'schema', 'run_id', 'suite_sha256', 'deployment_sha256', 'binding_sha256',
                  'node_rank', 'cohort_index', 'phase', 'nonce'}
SERVER_FIELDS = (REQUEST_FIELDS - {'schema'}) | {
    'schema', 'server_identity', 'server_receive', 'server_send'}
EXCHANGE_FIELDS = SERVER_FIELDS | {'client_identity', 'client_send', 'client_receive'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha_values(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value):
    return type(value) is int and value > 0


def _matches(pattern, value):
    return type(value) is str and pattern.fullmatch(value) is not None


def decode_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON metadata key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON metadata')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _ipv4(value):
    if type(value) is not str:
        raise ValueError('Explicit literal IPv4 address required')
    address = ipaddress.IPv4Address(value)
    if str(address) != value or address.is_unspecified or address.is_multicast or value == '255.255.255.255':
        raise ValueError('Explicit unicast IPv4 address required')
    return value


def validate_identity(value):
    if type(value) is not dict or set(value) != IDENTITY_FIELDS:
        raise ValueError('Complete actual process/clock identity required')
    namespace, offsets = value['time_namespace'], value['timens_offsets']
    if (not _matches(HOST, value['hostname']) or not _matches(UUID, value['boot_id'])
            or not _positive(value['pid']) or not _positive(value['start_ticks'])
            or value['helper_sha256'] != digest(__file__)
            or type(namespace) is not dict or set(namespace) != {'device', 'inode', 'link'}
            or type(namespace['device']) is not int or namespace['device'] < 0
            or not _positive(namespace['inode'])
            or namespace['link'] != 'time:[' + str(namespace['inode']) + ']'
            or type(offsets) is not dict or set(offsets) != {'monotonic', 'boottime'}):
        raise ValueError('Invalid actual process/clock identity')
    namespace = value['pid_namespace']
    if (type(namespace) is not dict or set(namespace) != {'device', 'inode', 'link'}
            or type(namespace['device']) is not int or namespace['device'] < 0
            or not _positive(namespace['inode'])
            or namespace['link'] != 'pid:[' + str(namespace['inode']) + ']'):
        raise ValueError('Actual PID namespace identity unavailable')
    for pair in offsets.values():
        if (type(pair) is not list or len(pair) != 2 or any(type(x) is not int for x in pair)
                or not 0 <= pair[1] < 1_000_000_000):
            raise ValueError('Actual time namespace offsets unavailable')
    return value


def identity():
    """Read this process only; namespace numbers are never compared across nodes."""
    namespace = Path('/proc/self/ns/time')
    actual_stat = namespace.stat()
    pid_namespace = Path('/proc/self/ns/pid')
    pid_stat = pid_namespace.stat()
    offsets = {}
    for line in Path('/proc/self/timens_offsets').read_text().splitlines():
        name, seconds, nanoseconds = line.split()
        if name in offsets:
            raise ValueError('Duplicate actual time namespace offset')
        offsets[name] = [int(seconds), int(nanoseconds)]
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return validate_identity({
        'hostname': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'pid': os.getpid(), 'start_ticks': int(fields[19]),
        'time_namespace': {'device': actual_stat.st_dev, 'inode': actual_stat.st_ino,
                           'link': os.readlink(namespace)},
        'pid_namespace': {'device': pid_stat.st_dev, 'inode': pid_stat.st_ino,
                          'link': os.readlink(pid_namespace)},
        'timens_offsets': offsets, 'helper_sha256': digest(__file__),
    })


def same_domain(a, b):
    validate_identity(a)
    validate_identity(b)
    return all(canonical(a[key]) == canonical(b[key]) for key in DOMAIN_FIELDS)


def _closure_identities(identities, observer):
    validate_identity(observer)
    if type(identities) is not list or not 1 <= len(identities) <= 16:
        raise ValueError('Explicit nonempty bounded worker identity list required')
    expected = {}
    for worker in identities:
        validate_identity(worker)
        key = (worker['pid'], worker['start_ticks'])
        if (key in expected or worker['pid'] == observer['pid'] or not same_domain(worker, observer)
                or canonical(worker['pid_namespace']) != canonical(observer['pid_namespace'])):
            raise ValueError('Worker closure requires unique other processes in this clock/PID domain')
        expected[key] = worker
    return expected


def validate_closure(closure, identities):
    """Validate sealed closure data offline; do not inspect /proc or a live PID."""
    fields = {'schema', 'observer_identity', 'workers', 'observed_monotonic_ns', 'source_sha256'}
    if (type(closure) is not dict or set(closure) != fields or closure['schema'] != CLOSURE_SCHEMA
            or closure['source_sha256'] != digest(__file__) or not _positive(closure['observed_monotonic_ns'])
            or type(closure['workers']) is not list):
        raise ValueError('Complete source-bound worker closure required')
    expected = _closure_identities(identities, closure['observer_identity'])
    seen = set()
    for row in closure['workers']:
        if type(row) is not dict or set(row) != {'identity', 'status', 'observed_process'}:
            raise ValueError('Exact closed worker observation required')
        worker = validate_identity(row['identity'])
        key = (worker['pid'], worker['start_ticks'])
        if key in seen or key not in expected or canonical(worker) != canonical(expected[key]):
            raise ValueError('Closure worker identity missing, duplicated or changed')
        seen.add(key)
        status, observed = row['status'], row['observed_process']
        if type(status) is not str or status not in CLOSED_STATUSES:
            raise ValueError('Unproven worker termination status')
        if status == 'original_exited_no_proc':
            if observed is not None:
                raise ValueError('Absent process cannot have an observed live stat')
            continue
        if (type(observed) is not dict or set(observed) != {'pid', 'start_ticks', 'state'}
                or type(observed['pid']) is not int or observed['pid'] != worker['pid']
                or not _positive(observed['start_ticks'])
                or type(observed['state']) is not str or len(observed['state']) != 1
                or observed['state'] not in 'RSDZTWXIKPN'):
            raise ValueError('Invalid actual observed process stat')
        if observed['start_ticks'] == worker['start_ticks']:
            raise ValueError('PID reuse requires a different actual start time')
    if seen != set(expected):
        raise ValueError('Closure is missing an expected original worker')
    return closure


def closed_workers(identities):
    """Read only known worker PIDs after owned cleanup; never scan or signal."""
    observer = identity()
    expected = _closure_identities(identities, observer)
    rows = []
    for (pid, _start), worker in expected.items():
        try:
            raw = Path('/proc/' + str(pid) + '/stat').read_text()
        except FileNotFoundError:
            status, observed = 'original_exited_no_proc', None
        else:
            actual_pid = int(raw.split(' ', 1)[0])
            fields = raw.rsplit(')', 1)[1].split()
            observed = {'pid': actual_pid, 'start_ticks': int(fields[19]), 'state': fields[0]}
            if actual_pid != pid:
                raise ValueError('Known /proc path returned another process')
            if observed['start_ticks'] != worker['start_ticks']:
                status = 'original_exited_PID_reused'
            else:
                # A zombie group leader can retain live threads sharing its
                # files after pthread_exit. Z/X is not whole-writer closure.
                raise ValueError('Original worker PID/start still exists; owned cleanup must finish before sealing')
        rows.append({'identity': json.loads(canonical(worker)), 'status': status, 'observed_process': observed})
    result = {'schema': CLOSURE_SCHEMA, 'observer_identity': observer, 'workers': rows,
              'observed_monotonic_ns': time.monotonic_ns(), 'source_sha256': digest(__file__)}
    return validate_closure(result, identities)


def stamp():
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {'unix_ns': wall, 'monotonic_before_ns': before, 'monotonic_after_ns': after}


def check_stamp(value):
    if (type(value) is not dict or set(value) != {'unix_ns', 'monotonic_before_ns', 'monotonic_after_ns'}
            or any(not _positive(x) for x in value.values())
            or value['monotonic_before_ns'] > value['monotonic_after_ns']):
        raise ValueError('Invalid original clock stamp')
    return value


def validate_binding(value, run_id, suite_sha, deployment_sha):
    fields = {'schema', 'run_id', 'suite_sha256', 'deployment_sha256', 'helper_sha256', 'client_ip', 'nodes'}
    if (type(value) is not dict or set(value) != fields or value['schema'] != CLOCK_SCHEMA
            or not _matches(RUN, value['run_id']) or value['run_id'] != run_id
            or value['suite_sha256'] != suite_sha or value['deployment_sha256'] != deployment_sha
            or any(not _matches(SHA, value[key]) for key in ('suite_sha256', 'deployment_sha256'))
            or value['helper_sha256'] != digest(__file__)
            or type(value['nodes']) is not list or len(value['nodes']) != 2):
        raise ValueError('Explicit frozen causal binding differs')
    _ipv4(value['client_ip'])
    for rank, node in enumerate(value['nodes']):
        if (type(node) is not dict or set(node) != {'node_rank', 'ip', 'port', 'hostname'}
                or type(node['node_rank']) is not int or node['node_rank'] != rank
                or type(node['port']) is not int or not 1 <= node['port'] <= 65535
                or not _matches(HOST, node['hostname'])):
            raise ValueError('Exactly ordered node0/node1 endpoints and actual hostnames required')
        _ipv4(node['ip'])
    if (value['client_ip'] != value['nodes'][0]['ip']
            or len({node['ip'] for node in value['nodes']}) != 2
            or len({node['hostname'] for node in value['nodes']}) != 2):
        raise ValueError('Collector must be node0; actual node endpoints must differ')
    return value


def _binding(value):
    if type(value) is not dict:
        raise ValueError('Missing explicit causal binding')
    return validate_binding(value, value.get('run_id'), value.get('suite_sha256'), value.get('deployment_sha256'))


def make_request(binding, node_rank, cohort_index, phase, nonce):
    _binding(binding)
    result = {'schema': REQUEST_SCHEMA, 'run_id': binding['run_id'], 'suite_sha256': binding['suite_sha256'],
              'deployment_sha256': binding['deployment_sha256'], 'binding_sha256': sha_values(binding),
              'node_rank': node_rank, 'cohort_index': cohort_index, 'phase': phase, 'nonce': nonce}
    validate_request(result, binding, node_rank)
    return result


def validate_request(value, binding, node_rank):
    if (type(value) is not dict or set(value) != REQUEST_FIELDS or value['schema'] != REQUEST_SCHEMA
            or type(node_rank) is not int or node_rank not in (0, 1)
            or type(value['node_rank']) is not int or value['node_rank'] != node_rank
            or type(value['cohort_index']) is not int or not 0 <= value['cohort_index'] < 10000
            or value['phase'] not in ('begin', 'end') or not _matches(NONCE, value['nonce'])
            or any(value[key] != binding[key] for key in ('run_id', 'suite_sha256', 'deployment_sha256'))
            or value['binding_sha256'] != sha_values(binding)):
        raise ValueError('Unexpected causal request metadata')
    return value


def validate_exchange(value, binding):
    _binding(binding)
    if type(value) is not dict or set(value) != EXCHANGE_FIELDS or value['schema'] != ACK_SCHEMA:
        raise ValueError('Complete original causal ACK required')
    request = {key: value[key] for key in REQUEST_FIELDS}
    request['schema'] = REQUEST_SCHEMA
    validate_request(request, binding, value['node_rank'])
    for field in ('client_identity', 'server_identity'):
        validate_identity(value[field])
    if (value['client_identity']['hostname'] != binding['nodes'][0]['hostname']
            or value['server_identity']['hostname'] != binding['nodes'][value['node_rank']]['hostname']):
        raise ValueError('Actual ACK process is on the wrong declared node')
    for field in ('client_send', 'client_receive', 'server_receive', 'server_send'):
        check_stamp(value[field])
    for a, b in (('client_send', 'client_receive'), ('server_receive', 'server_send')):
        if value[a]['monotonic_after_ns'] > value[b]['monotonic_before_ns']:
            raise ValueError('Same-process causal stamp order reversed')
    return value


class Session:
    """Only begin0/end0/begin1/end1...; accepted requests are never retried."""
    def __init__(self, binding, node_rank, own):
        self.binding = _binding(binding)
        if type(node_rank) is not int or node_rank not in (0, 1):
            raise ValueError('Actual node rank required')
        validate_identity(own)
        if own['hostname'] != binding['nodes'][node_rank]['hostname']:
            raise ValueError('Service must run on its exact declared hostname')
        self.node_rank, self.own = node_rank, own
        self.next_index, self.next_phase, self.seen_nonces = 0, 'begin', set()

    def answer(self, request, received):
        validate_request(request, self.binding, self.node_rank)
        check_stamp(received)
        if (request['cohort_index'] != self.next_index or request['phase'] != self.next_phase
                or request['nonce'] in self.seen_nonces):
            raise ValueError('Reused nonce, out-of-order or repeated cohort ACK')
        # Re-read our own identity: a changed source/clock domain fails closed.
        if canonical(identity()) != canonical(self.own):
            raise ValueError('Serving process/clock/source identity changed')
        response = dict(request, schema=ACK_SCHEMA, server_identity=self.own,
                        server_receive=received, server_send=stamp())
        check_stamp(response['server_send'])
        if received['monotonic_after_ns'] > response['server_send']['monotonic_before_ns']:
            raise ValueError('Service monotonic order reversed')
        self.seen_nonces.add(request['nonce'])
        if self.next_phase == 'begin':
            self.next_phase = 'end'
        else:
            self.next_phase = 'begin'
            self.next_index += 1
        return response


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        received = stamp()
        session = self.server.session
        if (self.client_address[0] != session.binding['client_ip']
                or self.path != '/r1-causal/' + session.binding['run_id']
                or self.headers.get('Transfer-Encoding') is not None):
            self.send_error(403)
            return
        try:
            lengths = self.headers.get_all('Content-Length') or []
            if len(lengths) != 1 or not re.fullmatch(r'[1-9][0-9]{0,3}', lengths[0]):
                raise ValueError('One bounded Content-Length required')
            length = int(lengths[0])
            if length > MAX_BODY or self.headers.get('Content-Type') != 'application/json':
                raise ValueError('Only bounded JSON metadata is accepted')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request')
            response = session.answer(decode_json(raw), received)
            body = canonical(response).encode()
            if len(body) > MAX_BODY:
                raise ValueError('Bounded ACK size exceeded')
        except (ValueError, TypeError, KeyError, OSError):
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


def exchange(binding, node_rank, cohort_index, phase, timeout):
    """One synchronous ACK; caller's before/after ordering creates the fence."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError('Causal RPC timeout must be finite in (0,60] seconds')
    request = make_request(binding, node_rank, cohort_index, phase, secrets.token_hex(16))
    own = identity()
    if own['hostname'] != binding['nodes'][0]['hostname']:
        raise ValueError('Causal collector must execute on actual node0')
    node = binding['nodes'][node_rank]
    connection = http.client.HTTPConnection(node['ip'], node['port'], timeout=timeout,
                                            source_address=(binding['client_ip'], 0))
    try:
        sent = stamp()
        connection.request('POST', '/r1-causal/' + binding['run_id'], body=canonical(request).encode(),
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read(MAX_BODY + 1)
        received = stamp()
        if response.status != 200 or len(raw) > MAX_BODY:
            raise ValueError('Missing or oversized causal ACK; no retry')
        result = decode_json(raw)
        if type(result) is not dict or set(result) != SERVER_FIELDS or result.get('schema') != ACK_SCHEMA:
            raise ValueError('Unexpected causal ACK structure')
        if any(canonical(result[key]) != canonical(request[key]) for key in REQUEST_FIELDS - {'schema'}):
            raise ValueError('Causal ACK nonce/run/cohort/binding differs')
        if canonical(identity()) != canonical(own):
            raise ValueError('Collector process/clock/source identity changed during RPC')
        result.update(client_identity=own, client_send=sent, client_receive=received)
        return validate_exchange(result, binding)
    finally:
        connection.close()


def serve(binding, node_rank, duration_seconds):
    if (type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds)
            or not 0 < duration_seconds <= 86400):
        raise ValueError('Explicit finite single-run endpoint lifetime required')
    session = Session(binding, node_rank, identity())
    node = binding['nodes'][node_rank]
    with Server((node['ip'], node['port']), Handler) as server:
        server.session, server.timeout = session, 1
        deadline = time.monotonic() + duration_seconds
        while time.monotonic() < deadline:
            server.handle_request()
    return {'status': 'causal_endpoint_closed_NOT_qualification', 'node_rank': node_rank,
            'binding_sha256': sha_values(binding), 'complete_cohorts': session.next_index,
            'next_phase': session.next_phase}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve',))
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--duration-seconds', type=float, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if digest(args.binding) != args.binding_sha256:
        raise ValueError('Explicit causal binding file SHA differs')
    binding = _binding(decode_json(args.binding.read_text()))
    if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 86400:
        raise ValueError('Explicit finite endpoint lifetime required')
    if not args.execute:
        print(canonical({'status': 'plan_only_NOT_executed', 'binding_sha256': sha_values(binding),
                         'binding_file_sha256': args.binding_sha256, 'node_rank': args.node_rank,
                         'path': '/r1-causal/' + binding['run_id'], 'helper_sha256': digest(__file__)}))
        return 0
    print(canonical(serve(binding, args.node_rank, args.duration_seconds)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
