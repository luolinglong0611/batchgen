#!/usr/bin/env python3
"""Seal raw node traces and derive complete offline evidence without reruns.

Both commands default to plan-only. Original collector records, timestamps,
deployment, suite and trace events are immutable. This utility performs no
network/GPU action and never grants full-model or performance qualification.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import socket
import time


SCHEMA = 'r1-cross-node-trace-seal-v1'
RAW_NAME = re.compile(r'rank-(\d+)-pid-(\d+)\.jsonl')
BOOTSTRAP = {'trace_start', 'model_hooks_registered'}
CAUSAL_VERSION = 'r1-correctness-collection-cohorts-v2'
CAUSAL_TIME_BASIS = 'node_monotonic_causal_v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError('Explicit artifact SHA mismatch: ' + str(path))
    return json.loads(path.read_text())


def safe_file(directory, name):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Trace files must be single safe basenames')
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise ValueError('Trace file must be an ordinary in-directory file')
    return path


def copy_verified(source, destination, expected):
    if sha(source) != expected:
        raise ValueError('Sealed source changed before copy')
    with source.open('rb') as stream, destination.open('xb') as output:
        shutil.copyfileobj(stream, output, 1024 * 1024)
    if sha(source) != expected or sha(destination) != expected:
        raise ValueError('Sealed source changed during copy')


def evidence_mode(args):
    # Keep programmatic fixed-suite callers compatible; semantic mode is opt-in.
    mode = getattr(args, 'mode', 'fixed')
    if mode not in ('fixed', 'independent'):
        raise ValueError('Unknown evidence mode')
    semantic_path = getattr(args, 'semantic_tool', None)
    semantic_sha = getattr(args, 'semantic_tool_sha256', None)
    if mode == 'independent' and (semantic_path is None or not semantic_sha):
        raise ValueError('Independent mode requires the original semantic tool and its explicit SHA')
    if mode == 'fixed' and (semantic_path is not None or semantic_sha is not None):
        raise ValueError('Semantic source pins require explicit independent mode')
    return mode


def mode_binding(args, suite):
    return {'mode': evidence_mode(args), 'suite_schema': suite['schema'],
            'qualification_tool_sha256': args.qualification_tool_sha256,
            'semantic_tool_sha256': getattr(args, 'semantic_tool_sha256', None)}


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def causal_run(run):
    """Only an explicit new run protocol can disable legacy wall filtering."""
    protocol = run.get('collection_protocol')
    if protocol is None:
        return False
    if (type(protocol) is not dict or protocol.get('version') != CAUSAL_VERSION
            or protocol.get('time_basis') != CAUSAL_TIME_BASIS):
        raise ValueError('Unsupported explicit trace collection time protocol')
    return True


def time_binding(run):
    if not causal_run(run):
        return {}
    return {'collection_protocol': copy.deepcopy(run['collection_protocol']),
            'started_monotonic_ns': run['started_monotonic_ns'],
            'finished_monotonic_ns': run['finished_monotonic_ns']}


def context(args):
    mode = evidence_mode(args)
    if sha(args.qualification_tool) != args.qualification_tool_sha256:
        raise ValueError('Pinned original qualification tool differs')
    tool = load_tool('r1_original_cross_node_qualifier', args.qualification_tool)
    suite_validator, collector_sha = tool, args.qualification_tool_sha256
    if mode == 'independent':
        if sha(args.semantic_tool) != args.semantic_tool_sha256:
            raise ValueError('Pinned original independent semantic tool differs')
        # The unchanged semantic module imports its adjacent qualifier. Check
        # that dependency before import, then verify the actual loaded module.
        adjacent_core = args.semantic_tool.with_name('qualify_r1_full_model.py')
        if sha(adjacent_core) != args.qualification_tool_sha256:
            raise ValueError('Independent semantic tool imports a different qualification source')
        suite_validator = load_tool('r1_original_cross_node_semantic', args.semantic_tool)
        if (Path(suite_validator.core.__file__).resolve() != adjacent_core.resolve()
                or sha(suite_validator.core.__file__) != args.qualification_tool_sha256):
            raise ValueError('Loaded independent semantic core differs from the pinned qualifier')
        tool, collector_sha = suite_validator.core, args.semantic_tool_sha256
    suite = suite_validator.validate_suite(checked_json(args.suite, args.suite_sha256))
    if mode == 'independent' and suite.get('tool_sha256') != collector_sha:
        raise ValueError('Independent suite semantic source identity differs')
    run = checked_json(args.original_run, args.original_run_sha256)
    deployment = checked_json(args.deployment, args.deployment_sha256)
    tool.validate_deployment(deployment, run.get('backend'), suite)
    expected_deployment = dict(deployment, artifact_sha256=args.deployment_sha256)
    if (run.get('schema') != tool.RUN_SCHEMA or run.get('scope') != 'full_model'
            or run.get('status') != 'evidence_collected_NOT_pair_qualified'
            or run.get('suite_sha256') != args.suite_sha256
            or run.get('tool_sha256') != collector_sha
            or run.get('deployment') != expected_deployment):
        raise ValueError('Original HTTP capture/source/deployment is incomplete or changed; offline union cannot repair it')
    if (mode == 'independent' and (run.get('independent_semantic') is not True or run.get('long_correctness') is not False)
            or mode == 'fixed' and run.get('independent_semantic') is True):
        raise ValueError('Original collector evidence mode differs')
    declared_causal = (type(suite.get('collection_protocol')) is dict
                      and suite['collection_protocol'].get('time_basis') == CAUSAL_TIME_BASIS)
    if declared_causal or causal_run(run):
        if (not callable(getattr(tool, 'is_causal_suite', None))
                or not callable(getattr(tool, 'run_time_interval', None))
                or not tool.is_causal_suite(suite) or not causal_run(run)):
            raise ValueError('Causal trace union requires an explicitly admitted suite and collector protocol')
        # The pinned core binds the exact suite/run protocol and validates the
        # same collector's monotonic envelope. Wall observations are preserved,
        # including backward steps, and never substituted for causal evidence.
        tool.run_time_interval(run, suite)
    else:
        start, finish = run.get('started_unix_ns'), run.get('finished_unix_ns')
        if type(start) is not int or type(finish) is not int or not 0 < start <= finish:
            raise ValueError('Original collector time interval is invalid')
    return tool, suite, run, expected_deployment


def event_selected(event, started, finished):
    timestamp = event.get('observed_unix_ns')
    if type(timestamp) is not int or timestamp <= 0:
        raise ValueError('Actual worker event lacks a real timestamp')
    if event.get('event') in BOOTSTRAP:
        if timestamp > finished:
            raise ValueError('Model/bootstrap event postdates the original collection interval')
        return True
    return started <= timestamp <= finished


def read_raw(path, node, run, deployment, suite_sha):
    match = RAW_NAME.fullmatch(path.name)
    if not match:
        raise ValueError('Unknown raw worker filename')
    rank, pid = map(int, match.groups())
    if rank not in range(node * 8, (node + 1) * 8):
        raise ValueError('Raw worker rank does not belong to its actual node partition')
    raw = path.read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise ValueError('Worker JSONL is empty or has an unclosed partial final line')
    events, artifacts = [], {}
    for ordinal, line in enumerate(raw.decode().splitlines(), 1):
        event = json.loads(line)
        if (event.get('schema') != 'r1-full-model-trace-v1' or event.get('rank') != rank
                or event.get('world_size') != 16 or event.get('ordinal') != ordinal
                or event.get('run_id') != deployment['run_id']
                or event.get('suite_sha256') != suite_sha
                or event.get('deployment_sha256') != deployment['artifact_sha256']):
            raise ValueError('Raw event source/rank/ordinal/run/suite/deployment binding differs')
        if ordinal == 1 and (event.get('event') != 'trace_start' or event.get('actual_pid') != pid):
            raise ValueError('Actual PID/worker start does not match the raw filename')
        if ordinal == 1 and causal_run(run):
            identity = event.get('clock_identity')
            if type(identity) is not dict or type(identity.get('pid')) is not int or identity['pid'] != pid:
                raise ValueError('Causal worker identity PID differs from its raw writer filename')
        # Causal runs retain the complete raw stream. In particular, native
        # events outside an ACK window must reach the core gate and fail there;
        # no cross-node wall-clock filtering may hide them or trim bootstrap.
        selected = causal_run(run) or event_selected(event, run['started_unix_ns'], run['finished_unix_ns'])
        if event.get('full_logits'):
            artifact = event['full_logits']
            if (artifact.get('dtype') != 'float32_le' or artifact.get('elements') != 129280
                    or artifact.get('bytes') != 129280 * 4):
                raise ValueError('Native full vocabulary shape/dtype metadata mismatch')
            name = artifact['file']
            if name in artifacts:
                raise ValueError('Duplicate native full logits filename')
            artifacts[name] = artifact
        if selected:
            events.append(event)
    return rank, events, artifacts


def causal_helper(tool, run, deployment, suite_sha):
    factory = getattr(tool, 'causal_module', None)
    if not callable(factory):
        raise ValueError('Pinned causal helper is required for closed-worker sealing')
    helper = factory()
    if any(not callable(getattr(helper, name, None)) for name in
           ('validate_binding', 'validate_closure', 'closed_workers')):
        raise ValueError('Pinned causal helper lacks closed-worker proof support')
    binding = helper.validate_binding(run.get('causal_binding'), deployment['run_id'],
                                      suite_sha, deployment['artifact_sha256'])
    return helper, binding


def checked_closure(helper, closure, identities, origin_hostname, sealing_hostname, expected_hostname):
    result = helper.validate_closure(closure, identities)
    if not (origin_hostname == sealing_hostname == expected_hostname
            == result['observer_identity']['hostname']):
        raise ValueError('Causal raw trace must be sealed on its bound original node')
    return result


def seal(args):
    tool, suite, run, deployment = context(args)
    causal = causal_run(run)
    files, ranks, identities = {}, set(), {}
    for path in sorted(args.trace_directory.glob('rank-*-pid-*.jsonl')):
        safe_file(args.trace_directory, path.name)
        before = sha(path)
        rank, _events, artifacts = read_raw(path, args.node_rank, run, deployment, args.suite_sha256)
        if rank in ranks:
            raise ValueError('Multiple worker PID files for one rank; restarted/mixed workers cannot be silently merged')
        ranks.add(rank)
        if causal:
            identities[rank] = _events[0].get('clock_identity')
        files[path.name] = {'sha256': before, 'bytes': path.stat().st_size, 'kind': 'raw_worker_JSONL'}
        for name, artifact in artifacts.items():
            original = safe_file(args.trace_directory, name)
            if name in files or original.stat().st_size != 129280 * 4 or sha(original) != artifact['sha256']:
                raise ValueError('Native full logits source collision/shape/SHA mismatch')
            files[name] = {'sha256': artifact['sha256'], 'bytes': 129280 * 4, 'kind': 'native_full_logits_f32'}
        if sha(path) != before:
            raise ValueError('Raw worker file changed while sealing; collect a stable snapshot')
    expected = list(range(args.node_rank * 8, (args.node_rank + 1) * 8))
    if sorted(ranks) != expected:
        raise ValueError('All eight real node ranks are required before sealing')
    if causal:
        helper, binding = causal_helper(tool, run, deployment, args.suite_sha256)
        worker_identities = [identities[rank] for rank in expected]
    receipt = {'schema': SCHEMA, 'status': 'plan_only_NOT_executed', 'node_rank': args.node_rank,
        'expected_ranks': expected, 'origin_hostname': args.origin_hostname,
        'sealing_hostname': socket.gethostname(), 'origin_trace_directory': str(args.trace_directory.resolve()),
        'source_sha256': sha(__file__), 'original_run_sha256': args.original_run_sha256,
        'suite_sha256': args.suite_sha256, 'deployment_sha256': args.deployment_sha256,
        **mode_binding(args, suite),
        'run_id': deployment['run_id'], 'started_unix_ns': run['started_unix_ns'],
        'finished_unix_ns': run['finished_unix_ns'], 'files': files,
        **time_binding(run),
        'full_model_qualified': False, 'formal_performance_permitted': False}
    if args.execute:
        if causal:
            closure_before = checked_closure(helper, helper.closed_workers(worker_identities), worker_identities,
                args.origin_hostname, receipt['sealing_hostname'], binding['nodes'][args.node_rank]['hostname'])
        args.output.mkdir(parents=True, exist_ok=False)
        raw = args.output / 'raw'
        raw.mkdir()
        for name, data in files.items():
            copy_verified(safe_file(args.trace_directory, name), raw / name, data['sha256'])
        if causal:
            closure_after = checked_closure(helper, helper.closed_workers(worker_identities), worker_identities,
                args.origin_hostname, receipt['sealing_hostname'], binding['nodes'][args.node_rank]['hostname'])
            if (canonical(closure_after['observer_identity']) != canonical(closure_before['observer_identity'])
                    or closure_after['observed_monotonic_ns'] < closure_before['observed_monotonic_ns']):
                raise ValueError('Worker closure observer identity or monotonic order changed while sealing')
            # Closed status may advance from absent to PID reuse; both
            # observations must still prove the exact original writer exited.
            receipt['causal_worker_closure'] = closure_after
        receipt.update(status='sealed_raw_node_trace_NOT_qualified', sealed_unix_ns=time.time_ns())
        write_new(args.output / 'manifest.json', receipt)
    return receipt


def finalize(args):
    tool, suite, original, deployment = context(args)
    causal = causal_run(original)
    if causal:
        helper, binding = causal_helper(tool, original, deployment, args.suite_sha256)
        if not callable(getattr(tool, 'validate_causal_closures', None)):
            raise ValueError('Pinned core lacks the closed-worker evidence gate')
    external_paths = getattr(args, 'external_evidence', None) or []
    external_pins = getattr(args, 'external_evidence_sha256', None) or []
    if len(external_paths) != len(external_pins):
        raise ValueError('Every optional original launch/runtime evidence artifact needs an explicit SHA')
    external = []
    for source, pin in zip(external_paths, external_pins):
        if sha(source) != pin:
            raise ValueError('Original external launch/runtime evidence SHA mismatch')
        external.append({'path': str(source), 'sha256': pin, 'bytes': source.stat().st_size})
    if len(args.node_manifest) != 2 or len(args.node_manifest_sha256) != 2:
        raise ValueError('Exactly two explicit node manifests and SHA pins are required')
    events, inputs, nodes, hosts, names, closures = [], [], set(), set(), {}, []
    for manifest_path, expected_sha in zip(args.node_manifest, args.node_manifest_sha256):
        manifest = checked_json(manifest_path, expected_sha)
        node = manifest.get('node_rank')
        if node not in (0, 1) or node in nodes:
            raise ValueError('Missing/duplicate actual node partition')
        nodes.add(node)
        if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'sealed_raw_node_trace_NOT_qualified'
                or manifest.get('source_sha256') != sha(__file__)
                or manifest.get('original_run_sha256') != args.original_run_sha256
                or manifest.get('suite_sha256') != args.suite_sha256
                or manifest.get('deployment_sha256') != args.deployment_sha256
                or any(manifest.get(key) != value for key, value in mode_binding(args, suite).items())
                or manifest.get('run_id') != deployment['run_id']
                or manifest.get('started_unix_ns') != original['started_unix_ns']
                or manifest.get('finished_unix_ns') != original['finished_unix_ns']
                or any(manifest.get(key) != value for key, value in time_binding(original).items())
                or (not causal_run(original) and any(key in manifest for key in
                    ('collection_protocol', 'started_monotonic_ns', 'finished_monotonic_ns')))):
            raise ValueError('Sealed node source/deployment/original time interval differs')
        host = manifest.get('origin_hostname')
        if not isinstance(host, str) or not host or host in hosts:
            raise ValueError('Distinct actual origin hostnames required')
        hosts.add(host)
        raw = manifest_path.parent / 'raw'
        rank_set, referenced, identities = set(), set(), {}
        for name, data in manifest['files'].items():
            path = safe_file(raw, name)
            if sha(path) != data['sha256'] or path.stat().st_size != data['bytes']:
                raise ValueError('Sealed raw file changed')
            if data['kind'] == 'raw_worker_JSONL':
                rank, selected, artifacts = read_raw(path, node, original, deployment, args.suite_sha256)
                if rank in rank_set:
                    raise ValueError('Multiple raw workers for one rank')
                rank_set.add(rank)
                if causal:
                    identities[rank] = selected[0].get('clock_identity')
                events.extend(selected)
                referenced.update(artifacts)
        expected = list(range(node * 8, (node + 1) * 8))
        if sorted(rank_set) != expected or manifest.get('expected_ranks') != expected:
            raise ValueError('Sealed node does not contain all eight required actual ranks')
        if causal:
            closure = checked_closure(helper, manifest.get('causal_worker_closure'),
                [identities[rank] for rank in expected], host, manifest.get('sealing_hostname'),
                binding['nodes'][node]['hostname'])
            closures.append({'node_rank': node, 'closure': copy.deepcopy(closure)})
        actual_blobs = {name for name, data in manifest['files'].items() if data['kind'] == 'native_full_logits_f32'}
        if referenced != actual_blobs:
            raise ValueError('Native full logits references differ from sealed artifact set')
        for event in events:
            artifact = event.get('full_logits')
            if artifact and event['rank'] in rank_set:
                if artifact['file'] in names:
                    raise ValueError('Cross-node full logits collision')
                data = manifest['files'].get(artifact['file'])
                if data is None or artifact['sha256'] != data['sha256']:
                    raise ValueError('Event/logits sealed SHA differs')
                names[artifact['file']] = (safe_file(raw, artifact['file']), artifact)
        inputs.append({'manifest_path': str(manifest_path), 'manifest_sha256': expected_sha,
                       'node_rank': node, 'origin_hostname': host, 'raw_file_count': len(manifest['files'])})
    # Every already-saved collector event must survive, with no rewriting/dedup.
    original_event_path = args.original_run.parent / original['trace']['events_file']
    original_events = checked_json(original_event_path, original['trace']['events_sha256'])
    if Counter(map(canonical, original_events)) - Counter(map(canonical, events)):
        raise ValueError('Union would drop or modify an already-frozen original collector event')
    events.sort(key=lambda e: (e['rank'], e['ordinal']))
    if causal:
        closures.sort(key=lambda row: row['node_rank'])
        closure_run = copy.deepcopy(original)
        closure_run['trace']['causal_worker_closures'] = closures
        if tool.validate_causal_closures(closure_run, events) is not True:
            raise ValueError('Both original-node closed-worker proofs are required')
    receipt = {'schema': 'r1-cross-node-trace-finalization-v1', 'status': 'plan_only_NOT_executed',
        'source_sha256': sha(__file__), 'original_run_sha256': args.original_run_sha256,
        **mode_binding(args, suite),
        'nodes': inputs, 'observed_ranks': sorted({e['rank'] for e in events}),
        'event_count': len(events), 'original_started_unix_ns': original['started_unix_ns'],
        'original_finished_unix_ns': original['finished_unix_ns'],
        **time_binding(original),
        'timestamps_rewritten': False, 'HTTP_records_rewritten': False,
        'actual_launch_argv_verified_by_this_utility': False,
        'original_external_evidence': external,
        'actual_runtime_parameter_and_launch_receipt_review_still_required': True,
        'full_model_qualified': False, 'formal_performance_permitted': False}
    if not args.execute:
        return receipt
    args.output.mkdir(parents=True, exist_ok=False)
    trace = args.output / 'trace'
    trace.mkdir()
    provenance = args.output / 'provenance'
    provenance.mkdir()
    for source, name, expected in ((args.suite, 'original-suite.json', args.suite_sha256),
            (args.deployment, 'original-deployment.json', args.deployment_sha256),
            (args.qualification_tool, 'original-qualification-tool.py', args.qualification_tool_sha256),
            (original_event_path, 'original-collector-events.json', original['trace']['events_sha256'])):
        copy_verified(source, provenance / name, expected)
    if evidence_mode(args) == 'independent':
        copy_verified(args.semantic_tool, provenance / 'original-semantic-tool.py', args.semantic_tool_sha256)
    for item in inputs:
        copy_verified(Path(item['manifest_path']), provenance / f'node{item["node_rank"]}-manifest.json', item['manifest_sha256'])
    for number, item in enumerate(external):
        source = Path(item['path'])
        copy_verified(source, provenance / f'external-{number}-{source.name}', item['sha256'])
    for name, (source, artifact) in names.items():
        copy_verified(source, trace / name, artifact['sha256'])
        tool.read_logits(args.output, artifact)  # All129280 native FP32 values finite.
    write_new(trace / 'events.json', events)
    derived = copy.deepcopy(original)
    derived['trace'] = {'events_file': 'trace/events.json', 'events_sha256': sha(trace / 'events.json'),
                        'event_count': len(events)}
    if causal:
        derived['trace']['causal_worker_closures'] = copy.deepcopy(closures)
    derived['cross_node_derivation'] = dict(receipt, status='derived_trace_union_NOT_qualified')
    copy_verified(args.original_run, args.output / 'original-collector-run.json', args.original_run_sha256)
    write_new(args.output / 'run.json', derived)
    if evidence_mode(args) == 'independent':
        # Exactly the existing independent score() evidence gate. The separate
        # paired semantic scoring/long-smoke gate is not run or bypassed here.
        checked = tool.validate_evidence(args.output / 'run.json', suite, args.suite_sha256,
                                         False, semantic_smoke=False, context_limit=32768)
    else:
        checked = tool.validate_evidence(args.output / 'run.json', suite, args.suite_sha256,
                                         original['long_correctness'])
    hard, missing = checked['hard_failures'], checked['missing_evidence']
    passed = not hard and not missing
    receipt.update(status='sealed_native_evidence_passed_PARAMETERS_AND_PAIR_PENDING' if passed else
                   'sealed_native_evidence_FAILED_OR_INCOMPLETE', native_evidence_passed=passed,
                   hard_failures=hard, missing_evidence=missing,
                   actual_coverage_count=checked['actual_coverage_count'],
                   required_coverage_count=checked['required_coverage_count'],
                   derived_run_path=str(args.output / 'run.json'), derived_run_sha256=sha(args.output / 'run.json'))
    # Recheck explicit inputs after every read/copy; original artifacts stay intact.
    context(args)
    for item in inputs:
        path = Path(item['manifest_path'])
        manifest = checked_json(path, item['manifest_sha256'])
        for name, data in manifest['files'].items():
            if sha(safe_file(path.parent / 'raw', name)) != data['sha256']:
                raise ValueError('A sealed raw input changed during finalization')
    write_new(args.output / 'finalization.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('seal', 'finalize'):
        command = sub.add_parser(name)
        command.add_argument('--execute', action='store_true')
        command.add_argument('--mode', choices=('fixed', 'independent'), default='fixed',
                             help='Fixed correctness suite (default), or the separate independent 48x2 semantic suite')
        command.add_argument('--original-run', type=Path, required=True)
        command.add_argument('--original-run-sha256', required=True)
        command.add_argument('--suite', type=Path, required=True)
        command.add_argument('--suite-sha256', required=True)
        command.add_argument('--deployment', type=Path, required=True)
        command.add_argument('--deployment-sha256', required=True)
        command.add_argument('--qualification-tool', type=Path, required=True)
        command.add_argument('--qualification-tool-sha256', required=True)
        command.add_argument('--semantic-tool', type=Path,
                             help='Original independent collector source; requires --mode independent')
        command.add_argument('--semantic-tool-sha256')
        command.add_argument('--output', type=Path, required=True)
        if name == 'seal':
            command.add_argument('--trace-directory', type=Path, required=True)
            command.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
            command.add_argument('--origin-hostname', required=True)
        else:
            command.add_argument('--node-manifest', type=Path, action='append', required=True)
            command.add_argument('--node-manifest-sha256', action='append', required=True)
            command.add_argument('--external-evidence', type=Path, action='append',
                                 help='Optional original actual-launch/runtime receipt, archived verbatim')
            command.add_argument('--external-evidence-sha256', action='append')
    args = parser.parse_args(argv)
    if args.execute and not args.output.is_absolute():
        parser.error('--execute requires an absolute fresh output directory')
    result = {'seal': seal, 'finalize': finalize}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['status'] == 'sealed_native_evidence_FAILED_OR_INCOMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
