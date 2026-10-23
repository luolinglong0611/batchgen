#!/usr/bin/env python3
"""Exercise16 simultaneously submitted R1 requests; operational smoke, not a benchmark.

Requires the completed fixed serial diagnostic. Keeps identical prompts/128-token
caps and records text agreement, without claiming native-token or quality equality.
No retries, service launch, cancellation, or changes to the prior serial report.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import batch_benchmark_client as client
import full_model_correctness as fixed


def requests(fixture, model):
    result = []
    for i in range(16):
        case = fixture['cases'][i % len(fixture['cases'])]
        result.append({'custom_id': f'parallel-{i:02d}-{case["id"]}',
            'original_id': case['id'], 'dataset': 'r1-fixed-concurrency-smoke-v1',
            'expected_input_tokens': len(case['input_ids']), 'max_output_tokens': 128,
            'body': {'model': model, 'prompt': case['prompt'], 'temperature': 0.0,
                     'top_p': 1.0, 'top_k': 1, 'max_tokens': 128, 'n': 1,
                     'stream': False, 'ignore_eos': False,
                     'presence_penalty': 0.0, 'frequency_penalty': 0.0}})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--serial-run', type=Path, required=True)
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    fixture = fixed.validate_fixture(json.loads(args.fixture.read_text()))
    serial = json.loads(args.serial_run.read_text())
    deployment = fixed.validate_deployment(serial['deployment'], 'batchgen', fixture)
    expected = {(repeat, c['id']) for repeat in range(2) for c in fixture['cases']}
    if (serial['status'] != 'diagnostic_collected_not_quality_pass' or
            len(serial['records']) != 12 or
            {(x['repeat'], x['id']) for x in serial['records']} != expected or
            serial['fixture_sha256'] != fixed.sha(args.fixture)):
        raise ValueError('A complete matching serial diagnostic is required')
    items = requests(fixture, deployment['model_id'])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'scope': '16ready requests in one batch; operational smoke NOT formal performance or quality qualification',
              'fixture_sha256': fixed.sha(args.fixture), 'serial_run_sha256': fixed.sha(args.serial_run),
              'source_sha256': fixed.sha(__file__), 'deployment': deployment,
              'expected_requests': 16, 'results': [], 'limitations': fixed.LIMITATIONS}
    fixed.save_new(args.output/'requests.json', items)
    config = SimpleNamespace(timeout_seconds=900, http_timeout_seconds=60,
                             poll_seconds=1, max_context_length=4096, ignore_eos=False)
    try:
        outcome = client.run_phase(config, 'concurrent16', items, args.output,
                                   client.base_url(args.endpoint), None)
        report['batch'] = outcome
        if not outcome['all_requests_valid']:
            raise RuntimeError('Concurrency smoke failed; preserve batch artifacts and inspect server before further work')
        rows = [json.loads(line) for line in (args.output/'concurrent16.requests.jsonl').read_text().splitlines()]
        if len(rows) != 16 or {x['custom_id'] for x in rows} != {x['custom_id'] for x in items}:
            raise ValueError('Missing or duplicate concurrent results')
        baseline = {x['id']: x['text'] for x in serial['records'] if x['repeat'] == 0}
        for row in rows:
            report['results'].append({'custom_id': row['custom_id'], 'id': row['original_id'],
                'success': row['success'], 'usage': row['usage'], 'finish_reason': row['finish_reason'],
                'text_sha256': hashlib.sha256(row['output_text'].encode()).hexdigest(),
                'text_matches_serial_first_repeat': row['output_text'] == baseline[row['original_id']]})
        report['status'] = 'operational_smoke_completed_NOT_quality_or_performance_pass'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['server_request_may_still_be_active'] = True
    finally:
        fixed.save_new(args.output/'run.json', report)
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if report['status'].startswith('operational_smoke_completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
