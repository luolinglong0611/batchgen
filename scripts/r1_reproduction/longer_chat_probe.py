#!/usr/bin/env python3
"""Freeze or collect independent R1 chat probes; never a quality/performance gate.

No arguments prints help. prepare/plan do not make HTTP calls. Only explicit run
uses HTTP, and never launches, stops, resets, or queries a model service/GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, build_opener

import full_model_correctness as fixed

SOURCE_FIXTURE_SHA256 = 'f7ed0434a2894d604115affeed28ba279e59acda0794932d6a21359fa649d53d'
SEED = 20261023
TAGS = {'longer-chat': 'r1-longer-chat-2048-v1',
        'identical-arithmetic': 'r1-identical-arithmetic-16x128-v1',
        'identical-arithmetic-wide-pool': 'r1-identical-arithmetic-16x128-context12288-v1'}
SOURCE_NAMES = ('longer_chat_probe.py', 'full_model_correctness.py',
                'batch_benchmark_client.py', 'benchmark_client.py')
LIMITATIONS = [
    'Collection success is not model quality, numerical correctness, or performance qualification.',
    'Original six-case fixture, 6000 workloads, 64-QA fixture, and earlier gates remain unchanged.',
    'The two copies of each long-chat case share one mixed batch; differences do not establish determinism or its failure.',
    'BatchGen API returns text only. No text re-tokenization is performed or called native token evidence.',
    'R1_NATIVE_TOKENS worker traces are separate, service-run-specific evidence; absent traces mean unavailable IDs.',
    'Sixteen identical requests do not prove sixteen-rank coverage. Only actual worker traces can establish assigned ranks.',
    'SGLang uses native /generate input_ids; HTTP concurrency does not guarantee the same internal batch/rank assignment as BatchGen.',
    'Teacher text/IDs are frozen references only; this probe sends no teacher-forced requests or extra discovery HTTP.',
    'EOS is allowed; capped, empty, and thinking-only completions must remain visible and are not quality passes.',
    'Protocol seed is sent to SGLang; BatchGen exposes no request seed and uses greedy argmax.',
    'Deployment JSON is an operator declaration pinned by SHA, not independently verified against the running service.',
    'No retries, automatic request cancellation, warmup, model/service operations, or performance claims.',
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require_sha(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(label + ' requires an explicit lowercase SHA256')
    if sha(path) != expected:
        raise ValueError(label + ' SHA mismatch')


def source_pins():
    return {name: sha(Path(__file__).resolve().with_name(name)) for name in SOURCE_NAMES}


def generation(probe):
    return {'max_new_tokens': 2048 if probe == 'longer-chat' else 128,
            'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'ignore_eos': False,
            'protocol_seed': SEED, 'skip_special_tokens': True,
            'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}


def pool_context(probe):
    return 12288 if probe == 'identical-arithmetic-wide-pool' else 4096


def frozen_requests(fixture, probe):
    if probe == 'longer-chat':
        selected = [(repeat, case) for repeat in range(2) for case in fixture['cases']]
    elif probe in ('identical-arithmetic', 'identical-arithmetic-wide-pool'):
        arithmetic = next(case for case in fixture['cases'] if case['id'] == 'arithmetic')
        selected = [(repeat, arithmetic) for repeat in range(16)]
    else:
        raise ValueError('Unknown probe')
    return [{'custom_id': f'{probe}-{index:02d}-{case["id"]}',
             'submission_ordinal': index, 'copy_index': repeat,
             'case_id': case['id'], 'case': copy.deepcopy(case)}
            for index, (repeat, case) in enumerate(selected)]


def prepare(fixture_path, output, probe):
    fixture_path, output = Path(fixture_path), Path(output)
    require_sha(fixture_path, SOURCE_FIXTURE_SHA256, 'Original fixture')
    original = fixture_path.read_bytes()
    fixture = fixed.validate_fixture(json.loads(original))
    requests = frozen_requests(fixture, probe)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source-fixture.json').write_bytes(original)
    write_new(output / 'requests.json', requests)
    manifest = {'schema': 'r1-independent-chat-probe-manifest-v1', 'protocol_tag': TAGS[probe],
        'probe': probe, 'source_fixture': 'source-fixture.json',
        'source_fixture_sha256': SOURCE_FIXTURE_SHA256, 'source_schema': fixture['schema'],
        'conditions': fixture['conditions'], 'generation': generation(probe),
        'requests': 'requests.json', 'requests_sha256': sha(output / 'requests.json'),
        'request_count': len(requests), 'source_pins': source_pins(),
        'batchgen_batches': 1, 'batchgen_max_context_length': pool_context(probe),
        'sglang_http_concurrency': len(requests),
        'teacher_reference_only': True, 'rank_coverage': 'unobserved; requires actual service-run worker traces',
        'intended_execution': ('fresh admitted engine; this batch is the first submitted wave'
                               if probe != 'longer-chat' else 'one mixed chat batch'),
        'scope': 'paired chat diagnostics; not quality, determinism, or performance qualification',
        'limitations': LIMITATIONS}
    require_sha(fixture_path, SOURCE_FIXTURE_SHA256, 'Original fixture after prepare')
    write_new(output / 'manifest.json', manifest)
    return {'manifest': str(output / 'manifest.json'), 'manifest_sha256': sha(output / 'manifest.json'),
            'requests_sha256': manifest['requests_sha256'], 'request_count': len(requests),
            'status': 'prepared_NOT_executed', 'protocol_tag': manifest['protocol_tag']}


def load_manifest(path, expected_sha):
    path = Path(path)
    require_sha(path, expected_sha, 'Manifest')
    manifest = json.loads(path.read_bytes())
    probe = manifest.get('probe')
    if (probe not in TAGS or manifest.get('schema') != 'r1-independent-chat-probe-manifest-v1'
            or manifest.get('protocol_tag') != TAGS[probe]
            or manifest.get('generation') != generation(probe)
            or manifest.get('source_fixture') != 'source-fixture.json'
            or manifest.get('requests') != 'requests.json'
            or manifest.get('source_fixture_sha256') != SOURCE_FIXTURE_SHA256
            or manifest.get('source_pins') != source_pins()):
        raise ValueError('Frozen protocol or client source pins changed')
    require_sha(path.parent / 'source-fixture.json', SOURCE_FIXTURE_SHA256, 'Copied original fixture')
    fixture = fixed.validate_fixture(json.loads((path.parent / 'source-fixture.json').read_bytes()))
    require_sha(path.parent / 'requests.json', manifest['requests_sha256'], 'Requests')
    requests = json.loads((path.parent / 'requests.json').read_bytes())
    if (requests != frozen_requests(fixture, probe) or manifest.get('conditions') != fixture['conditions']
            or manifest.get('request_count') != len(requests)
            or manifest.get('batchgen_batches') != 1
            or manifest.get('batchgen_max_context_length') != pool_context(probe)
            or manifest.get('sglang_http_concurrency') != len(requests)
            or manifest.get('teacher_reference_only') is not True):
        raise ValueError('Frozen request/input/teacher/order contract changed')
    return manifest, fixture, requests


def sglang_payload(item, config):
    sampling = {key: value for key, value in config.items() if key != 'protocol_seed'}
    sampling['sampling_seed'] = config['protocol_seed']
    return {'input_ids': item['case']['input_ids'], 'sampling_params': sampling,
            'stream': False, 'return_logprob': True, 'logprob_start_len': -1,
            'return_text_in_logprobs': False}


def batchgen_requests(requests, config, model_id, tag):
    return [{'custom_id': item['custom_id'], 'original_id': item['case_id'], 'dataset': tag,
             'expected_input_tokens': len(item['case']['input_ids']),
             'max_output_tokens': config['max_new_tokens'],
             'body': {'model': model_id, 'prompt': item['case']['prompt'],
                'temperature': config['temperature'], 'top_p': config['top_p'], 'top_k': config['top_k'],
                'max_tokens': config['max_new_tokens'], 'n': 1, 'stream': False,
                'ignore_eos': config['ignore_eos'], 'presence_penalty': config['presence_penalty'],
                'frequency_penalty': config['frequency_penalty']}}
            for item in requests]


def make_plan(manifest_path, manifest_sha, backend, deployment_path, deployment_sha, endpoints):
    import batch_benchmark_client as batch_api
    manifest, fixture, requests = load_manifest(manifest_path, manifest_sha)
    require_sha(deployment_path, deployment_sha, 'Deployment source JSON')
    source = json.loads(Path(deployment_path).read_bytes())
    deployment = source.get('deployment', source)
    fixed.validate_deployment(deployment, backend, fixture)
    if not endpoints or len(endpoints) != 1:
        raise ValueError('Each diagnostic uses exactly one explicit coordinator/native endpoint')
    endpoints = [batch_api.base_url(endpoint) for endpoint in endpoints]
    payloads = (batchgen_requests(requests, manifest['generation'], deployment['model_id'], manifest['protocol_tag'])
                if backend == 'batchgen' else [sglang_payload(item, manifest['generation']) for item in requests])
    return {'schema': 'r1-independent-chat-probe-plan-v1', 'status': 'planned_NOT_executed',
            'backend': backend, 'manifest_sha256': manifest_sha, 'protocol_tag': manifest['protocol_tag'],
            'source_fixture_sha256': SOURCE_FIXTURE_SHA256, 'source_pins': source_pins(),
            'deployment_source_sha256': deployment_sha, 'deployment': deployment, 'endpoints': endpoints,
            'request_count': len(requests), 'payloads': payloads, 'payloads_sha256': digest(payloads),
            'request_ids': [item['custom_id'] for item in requests],
            'submission': 'one files/batches batch' if backend == 'batchgen' else 'closed-loop HTTP /generate',
            'http_concurrency': 1 if backend == 'batchgen' else len(requests),
            'batch_count': 1 if backend == 'batchgen' else None,
            'batchgen_max_context_length': manifest['batchgen_max_context_length'] if backend == 'batchgen' else None,
            'timeout_seconds': 1200, 'http_timeout_seconds': 1200 if backend == 'sglang' else 60,
            'retries': 0, 'warmup_requests': 0, 'teacher_forced_requests': 0,
            'native_ids_source': 'separate R1_NATIVE_TOKENS worker logs only' if backend == 'batchgen' else 'native response output_ids if returned',
            'rank_coverage': 'unobserved', 'performance_measurement': False, 'quality_gate': None,
            'limitations': LIMITATIONS}


def sglang_record(response, item, cap):
    """Keep native metadata distinct from the text and from unavailable evidence."""
    errors = []
    meta = response.get('meta_info', {}) if isinstance(response, dict) else {}
    text = response.get('text') if isinstance(response, dict) else None
    native_ids = response.get('output_ids') if isinstance(response, dict) else None
    if not isinstance(text, str):
        errors.append('Missing response text')
    if meta.get('prompt_tokens') != len(item['case']['input_ids']):
        errors.append('prompt_tokens differs from frozen input IDs')
    count = meta.get('completion_tokens')
    if type(count) is not int or not 0 <= count <= cap:
        errors.append('Missing/invalid completion_tokens')
    finish = meta.get('finish_reason')
    if not isinstance(finish, dict) or finish.get('type') not in ('stop', 'length'):
        errors.append('Missing/non-successful finish_reason')
    if native_ids is not None:
        if not isinstance(native_ids, list) or any(type(x) is not int or x < 0 for x in native_ids):
            errors.append('Invalid native output_ids')
        elif len(native_ids) != count:
            errors.append('Native output ID count differs from completion_tokens')
    logps = meta.get('output_token_logprobs')
    nonfinite = 0
    if logps is not None:
        if not isinstance(logps, list):
            errors.append('Invalid output_token_logprobs')
        else:
            for row in logps:
                if (not isinstance(row, (list, tuple)) or len(row) < 2
                        or type(row[0]) not in (float, int) or not math.isfinite(row[0])):
                    nonfinite += 1
            if nonfinite:
                errors.append('Nonfinite/malformed output_token_logprobs')
    return {'custom_id': item['custom_id'], 'case_id': item['case_id'], 'copy_index': item['copy_index'],
            'success': not errors, 'errors': errors, 'text': text,
            'native_output_ids': native_ids, 'native_output_ids_available': native_ids is not None,
            'finish_reason': finish, 'usage': {key: meta.get(key) for key in ('prompt_tokens', 'completion_tokens', 'cached_tokens')},
            'output_token_logprobs': logps, 'nonfinite_or_malformed_output_logprobs': nonfinite,
            'response': response}


def native_call(endpoint, payload, item, output, cap):
    """One standard-library HTTP attempt; preserve every received body verbatim."""
    import batch_benchmark_client as batch_api
    body = canonical(payload).encode()
    prefix = output / item['custom_id']
    prefix.with_suffix('.request.json').write_bytes(body)
    request = Request(endpoint + '/generate', data=body, headers={'Content-Type': 'application/json'}, method='POST')
    started = time.monotonic()
    try:
        try:
            response = build_opener(batch_api._helper.NoRedirect()).open(request, timeout=1200)
        except HTTPError as error:
            response = error
        with response:
            status = response.status
            raw = response.read(64 * 1024**2 + 1)
            prefix.with_suffix('.http-body').write_bytes(raw)
            if len(raw) > 64 * 1024**2:
                raise ValueError('HTTP response exceeds 64 MiB; bounded prefix preserved')
            if not 200 <= status < 300:
                raise RuntimeError(f'HTTP {status}; not retried')
            def finite_float(text):
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError('Nonfinite response JSON number; raw bytes preserved')
                return value
            def reject_constant(text):
                raise ValueError('Nonfinite response JSON constant ' + text + '; raw bytes preserved')
            value = json.loads(raw, parse_float=finite_float, parse_constant=reject_constant)
        record = sglang_record(value, item, cap)
        record.update({'http_status': status, 'raw_response_file': prefix.name + '.http-body',
                       'raw_response_sha256': hashlib.sha256(raw).hexdigest()})
    except Exception as error:
        record = {'custom_id': item['custom_id'], 'case_id': item['case_id'],
                  'copy_index': item['copy_index'], 'success': False,
                  'errors': [f'{type(error).__name__}: {error}'], 'server_request_may_still_be_active': True}
    record['elapsed_seconds_diagnostic_only'] = time.monotonic() - started
    write_new(prefix.with_suffix('.record.json'), record)
    return record


def execute(args):
    import batch_benchmark_client as batch_api
    plan = make_plan(args.manifest, args.expected_manifest_sha256, args.backend,
                     args.deployment_source, args.expected_deployment_sha256, args.endpoint)
    manifest, _, requests = load_manifest(args.manifest, args.expected_manifest_sha256)
    # Check every input/source pin before creating output or making any HTTP call.
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'plan.json', plan)
    (output / 'manifest.json').write_bytes(args.manifest.read_bytes())
    for name in ('source-fixture.json', 'requests.json'):
        (output / name).write_bytes((args.manifest.parent / name).read_bytes())
    (output / 'deployment-source.json').write_bytes(args.deployment_source.read_bytes())
    write_new(output / 'started.json', {'status': 'HTTP_collection_started', 'source_pins': source_pins(),
                                      'protocol_tag': manifest['protocol_tag']})
    report = {'schema': 'r1-independent-chat-probe-run-v1', 'status': 'collection_failed',
              'backend': args.backend, 'manifest_sha256': args.expected_manifest_sha256,
              'plan_sha256': sha(output / 'plan.json'), 'records': [],
              'rank_coverage': 'unobserved', 'quality_gate': None, 'limitations': LIMITATIONS}
    try:
        if args.backend == 'batchgen':
            config = SimpleNamespace(timeout_seconds=1200, http_timeout_seconds=60, poll_seconds=1,
                max_context_length=plan['batchgen_max_context_length'],
                ignore_eos=False)
            result = batch_api.run_phase(config, 'probe', plan['payloads'], output, plan['endpoints'][0], None)
            report['batch'] = result
            rows_path = output / 'probe.requests.jsonl'
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
            for row in rows:
                report['records'].append({**row, 'native_output_ids': None,
                    'native_output_ids_available': False,
                    'native_ids_source': 'separate service-run R1_NATIVE_TOKENS worker logs required'})
            if not result['all_requests_valid']:
                raise RuntimeError('Batch or response validation failed; inspect preserved artifacts')
        else:
            # Exactly N requests with N workers; no warmup/discovery/teacher calls.
            # All work is joined so ordinary HTTP failures do not discard peers' responses.
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                pending = {executor.submit(native_call, plan['endpoints'][0], payload, item, output,
                                           manifest['generation']['max_new_tokens']): item
                           for item, payload in zip(requests, plan['payloads'])}
                records = {pending[future]['custom_id']: future.result() for future in as_completed(pending)}
            report['records'] = [records[item['custom_id']] for item in requests]
        if (len(report['records']) != len(requests)
                or {row['custom_id'] for row in report['records']} != set(plan['request_ids'])
                or any(not row['success'] for row in report['records'])):
            raise RuntimeError('Missing/invalid diagnostic response(s)')
        report['status'] = 'diagnostic_collected_NOT_quality_determinism_or_performance_pass'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['server_requests_may_remain'] = True
    finally:
        report['artifacts_sha256'] = {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
        write_new(output / 'run.json', report)
    return {'status': report['status'], 'run': str(output / 'run.json'), 'run_sha256': sha(output / 'run.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    prepare_parser = commands.add_parser('prepare', help='CPU-only freeze; no HTTP')
    prepare_parser.add_argument('--fixture', type=Path, required=True)
    prepare_parser.add_argument('--probe', choices=tuple(TAGS), required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    for name in ('plan', 'run'):
        command = commands.add_parser(name, help='No HTTP' if name == 'plan' else 'Explicit HTTP collection only')
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--expected-manifest-sha256', required=True)
        command.add_argument('--backend', choices=('batchgen', 'sglang'), required=True)
        command.add_argument('--deployment-source', type=Path, required=True)
        command.add_argument('--expected-deployment-sha256', required=True)
        command.add_argument('--endpoint', action='append', required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'prepare':
        result = prepare(args.fixture, args.output, args.probe)
    elif args.command == 'plan':
        result = make_plan(args.manifest, args.expected_manifest_sha256, args.backend,
                           args.deployment_source, args.expected_deployment_sha256, args.endpoint)
        write_new(args.output, result)
        result = {'status': result['status'], 'plan': str(args.output), 'plan_sha256': sha(args.output)}
    else:
        result = execute(args)
    print(canonical(result))
    return int(result['status'] == 'collection_failed')


if __name__ == '__main__':
    raise SystemExit(main())
