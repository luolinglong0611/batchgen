#!/usr/bin/env python3
"""Freeze and score an offline 64-item paired QA sanity check; never send HTTP.

prepare derives a NEW workload from the pinned 8192/2048 campaign files.
plan only prints commands for the existing clients; a separate operator must
review the deployment and run those commands. score consumes preserved client
JSONL. No model/tokenizer imports, downloads, service operations or pass threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import string

SCOPE = 'paired clipped-context QA sanity, not original LongBench leaderboard or statistical pass'
SOURCE_MANIFEST_SHA = '8dc0c57dbe130c496b85cbe5563ca71df7d58638d7ad9ae5b7044d9e3945baa7'
TASKS = ('qasper', 'hotpotqa', '2wikimqa', 'musique', 'narrativeqa', 'triviaqa', 'multifieldqa_en')
SEED = 20261023
COUNT = 64
PROFILE = 'input8192-output2048'
GENERATION = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1,
              'max_new_tokens': 2048, 'ignore_eos': False, 'n': 1, 'stream': False}
CLIENT_NAMES = ('benchmark_client.py', 'batch_benchmark_client.py')
METRICS = {
    'A_raw_completion': 'EM/token-F1 of the complete returned text; not identified as a final answer',
    'B_observable_post_think_suffix': 'last </think> suffix; unextractable items score zero only as a conservative lower bound',
    'normalization': 'lowercase; delete ASCII string.punctuation; remove standalone a/an/the; collapse whitespace',
    'token_f1': 'whitespace token multiset overlap; best reference alias separately for EM and F1',
    'denominator': 'all 64 selected items, including failed, missing and unextractable outputs',
    'truncation': 'finish_reason length counts as length_capped; stop is reported_stop, not proof of semantic completion',
    'units': 'EM and F1 fractions in [0,1]; no statistical pass threshold',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def hash_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def reference_scores(prediction, answers):
    normalized = normalize(prediction)
    if not normalized:
        return {'em': 0.0, 'token_f1': 0.0}
    em, f1 = 0.0, 0.0
    predicted = normalized.split()
    for answer in answers:
        expected = normalize(answer)
        em = max(em, float(normalized == expected))
        target = expected.split()
        overlap = sum((Counter(predicted) & Counter(target)).values())
        if overlap:
            f1 = max(f1, 2 * overlap / (len(predicted) + len(target)))
    return {'em': em, 'token_f1': f1}


def post_think_suffix(text):
    closes = text.count('</think>')
    opens = text.count('<think>')
    result = {'observed_open_count': opens, 'observed_close_count': closes,
              'multiple_closes': closes > 1, 'extractable': False, 'suffix': ''}
    if not closes:
        result['status'] = 'missing_think_close' if opens else 'no_think_markup'
        return result
    suffix = text.rsplit('</think>', 1)[1].strip()
    if '<think>' in suffix:
        result['status'] = 'unclosed_think_after_last_close'
        return result
    result.update(status='observable_suffix' if suffix else 'empty_observable_suffix',
                  extractable=True, suffix=suffix)
    return result


def safe_artifact(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Artifact must be an existing file within its manifest directory')
    return path


def verified_sources(manifest_path, expected_sha):
    if sha(manifest_path) != expected_sha:
        raise ValueError('Source manifest SHA differs from the explicitly pinned campaign manifest')
    manifest = json.loads(manifest_path.read_text())
    artifacts = [a for a in manifest['artifacts']
                 if a['input_tokens'] == 8192 and a['max_output_tokens'] == 2048]
    if len(artifacts) != 1 or manifest.get('prompt_format') != 'raw':
        raise ValueError('Expected one existing raw-prompt 8192/2048 profile')
    artifact = artifacts[0]
    paths = {k: safe_artifact(manifest_path.parent, artifact[k]) for k in ('requests', 'provenance')}
    for key, path in paths.items():
        if sha(path) != artifact[key + '_sha256']:
            raise ValueError('Source artifact SHA mismatch: ' + key)
    return manifest, artifact, paths


def candidate_pool(manifest, paths):
    available = sorted(set(TASKS) & set(manifest['tasks']))
    pool, seen_custom, total = {}, set(), 0
    occurrences = Counter()
    with paths['requests'].open() as requests, paths['provenance'].open() as provenance:
        for request_line, metadata_line in zip_longest(requests, provenance):
            if request_line is None or metadata_line is None:
                raise ValueError('Source request/provenance line counts differ')
            request, metadata = json.loads(request_line), json.loads(metadata_line)
            total += 1
            custom_id = request['custom_id']
            if custom_id in seen_custom or custom_id != metadata['custom_id']:
                raise ValueError('Duplicate or misaligned source custom ID')
            seen_custom.add(custom_id)
            if metadata['dataset'] not in available or metadata.get('language') != 'en':
                continue
            if request.get('method') != 'POST' or request.get('url') != '/v1/completions':
                raise ValueError('Only existing completion envelopes are supported')
            body, ids, answers = request['body'], metadata['input_token_ids'], metadata['answers']
            if (not isinstance(body.get('prompt'), str) or not body['prompt']
                    or body.get('max_tokens') != 2048 or body.get('temperature') != 0
                    or body.get('top_p') != 1 or body.get('n') != 1 or body.get('stream') is not False
                    or metadata.get('input_tokens') != 8192 or metadata.get('max_output_tokens') != 2048
                    or len(ids) != 8192 or any(type(x) is not int or x < 0 for x in ids)
                    or hash_value(ids) != metadata['input_token_ids_sha256']):
                raise ValueError('Source prompt/generation/token-ID contract changed')
            if (not isinstance(answers, list) or not answers
                    or any(not isinstance(a, str) for a in answers)
                    or not any(normalize(a) for a in answers)):
                raise ValueError('Every QA item needs at least one nonempty normalized reference answer')
            key = (metadata['dataset'], metadata['original_id'])
            if not all(isinstance(x, str) and x for x in key):
                raise ValueError('Dataset/original ID must be nonempty strings')
            occurrences[key[0]] += 1
            identity = {'prompt_sha256': hash_value(body['prompt']),
                        'input_ids_sha256': metadata['input_token_ids_sha256'],
                        'answers': answers, 'source_row_sha256': metadata['source_row_sha256'],
                        'context_modified': metadata['context_modified']}
            if key in pool:
                if pool[key]['identity'] != identity:
                    raise ValueError('Conflicting duplicate original QA sample')
            else:
                pool[key] = {'request': request, 'metadata': metadata, 'identity': identity}
    if total != manifest['count_per_profile']:
        raise ValueError('Source count differs from manifest')
    return pool, available, occurrences


def prepare(manifest_path, output, expected_sha=SOURCE_MANIFEST_SHA):
    if output.exists():
        raise FileExistsError('Refuse to overwrite a prepared QA workload')
    manifest, artifact, paths = verified_sources(manifest_path, expected_sha)
    pool, tasks, occurrences = candidate_pool(manifest, paths)
    if len(pool) < COUNT:
        raise ValueError('Fewer than 64 unique eligible English QA samples')
    ordered = sorted(pool, key=lambda k: (hash_value([SEED, *k]), k))[:COUNT]
    # The preparation is a snapshot; recheck sources before creating NEW files.
    verified_sources(manifest_path, expected_sha)
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / (PROFILE + '.requests.jsonl')
    provenance_path = output / (PROFILE + '.provenance.jsonl')
    selected = []
    effective_hash = hashlib.sha256()
    with request_path.open('x') as requests, provenance_path.open('x') as provenance:
        for ordinal, key in enumerate(ordered):
            sample = pool[key]
            custom_id = f'qa-{ordinal:04d}'
            original = sample['request']
            body = dict(original['body'])
            body.update(temperature=0.0, top_p=1.0, top_k=1, max_tokens=2048, ignore_eos=False)
            requests.write(canonical({'custom_id': custom_id, 'method': 'POST',
                                      'url': '/v1/completions', 'body': body}) + '\n')
            effective_hash.update(custom_id.encode() + b'\0' + canonical(body).encode() + b'\n')
            metadata = {**sample['metadata'], 'custom_id': custom_id,
                        'source_custom_id': original['custom_id'],
                        'selection_rank_sha256': hash_value([SEED, *key]),
                        'prompt_sha256': sample['identity']['prompt_sha256']}
            provenance.write(canonical(metadata) + '\n')
            selected.append({k: metadata[k] for k in ('custom_id', 'dataset', 'original_id',
                'source_custom_id', 'selection_rank_sha256', 'prompt_sha256',
                'input_token_ids_sha256', 'source_row_sha256', 'context_modified')})
    result = {'schema_version': 1, 'qa_schema': 'r1-clipped-qa-sanity-v1', 'scope': SCOPE,
        'generator_sha256': sha(__file__), 'seed': SEED, 'count_per_profile': COUNT,
        'sampling': 'unique identity SHA256 rank without replacement',
        'selection_algorithm': 'deduplicate (dataset,original_id); sort SHA256(canonical JSON [seed,dataset,original_id]), then identity; first64',
        'source_manifest_sha256': expected_sha, 'source_manifest_path': str(manifest_path.resolve()),
        'source_artifact': artifact, 'source_count_per_profile': manifest['count_per_profile'],
        'source_ignore_eos': manifest['ignore_eos'], 'source_prompt_format': manifest['prompt_format'],
        'tasks': tasks, 'eligible_english_occurrences_by_task': dict(occurrences),
        'unique_candidates': len(pool), 'unique_candidates_by_task': dict(Counter(k[0] for k in pool)),
        'selected_task_counts': dict(Counter(k[0] for k in ordered)), 'selected_ids': selected,
        'generation': GENERATION, 'metrics': METRICS, 'ignore_eos': False,
        'effective_payloads_sha256': effective_hash.hexdigest(),
        'tokenizer_revision': manifest['tokenizer_revision'], 'tokenizer_files': manifest['tokenizer_files'],
        'tokenizer_file_pins': manifest['tokenizer_file_pins'],
        'tokenizer_file_manifest_sha256': manifest['tokenizer_file_manifest_sha256'],
        'dataset_revision': manifest['dataset_revision'], 'prompt_format': 'raw',
        'client_source_sha256': {n: sha(Path(__file__).with_name(n)) for n in CLIENT_NAMES},
        'artifacts': [{'input_tokens': 8192, 'max_output_tokens': 2048,
            'requests': request_path.name, 'requests_sha256': sha(request_path),
            'provenance': provenance_path.name, 'provenance_sha256': sha(provenance_path)}],
        'limitations': [SCOPE, 'Middle-clipped context may omit answer evidence.',
            'Raw prompts are unchanged; thinking markup is not guaranteed.',
            'A scores complete returned text and does not identify a final answer.',
            'B coverage is reported; unextractable zero scores are a lower bound, not proof of a wrong model answer.',
            'Generation EOS is now allowed; original 6000 performance files remain unchanged.',
            'Neither a model quality pass threshold nor a statistical equivalence claim is defined.',
            'Runtime revisions, disabled prefix/cache/graphs and matching tokenizer assets require separate deployment evidence.',
            'Runtime prompt token counts do not prove token-ID equality. Timing is not a performance comparison.']}
    write_new(output / 'manifest.json', result)
    return {'status': 'prepared_NOT_requested', 'manifest': str(output / 'manifest.json'),
            'manifest_sha256': sha(output / 'manifest.json'), 'selected_task_counts': result['selected_task_counts'],
            'unique_candidates': len(pool), 'scope': SCOPE}


def load_qa(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('qa_schema') != 'r1-clipped-qa-sanity-v1' or manifest.get('scope') != SCOPE
            or manifest.get('generation') != GENERATION or manifest.get('metrics') != METRICS
            or manifest.get('seed') != SEED or manifest.get('count_per_profile') != COUNT):
        raise ValueError('Frozen QA protocol changed')
    artifact = manifest['artifacts'][0]
    cases = []
    for key in ('requests', 'provenance'):
        path = safe_artifact(manifest_path.parent, artifact[key])
        if sha(path) != artifact[key + '_sha256']:
            raise ValueError('Frozen QA artifact changed')
        if key == 'provenance':
            cases = [json.loads(x) for x in path.read_text().splitlines()]
    if len(cases) != COUNT or len({(c['dataset'], c['original_id']) for c in cases}) != COUNT:
        raise ValueError('Expected exactly64 unique QA identities')
    if [c['custom_id'] for c in cases] != [x['custom_id'] for x in manifest['selected_ids']]:
        raise ValueError('Selected ID order differs from frozen provenance')
    return manifest, cases


def client_plan(manifest_path, backend, endpoint, output, server_revision=None):
    manifest, _ = load_qa(manifest_path)
    scripts = Path(__file__).parent
    for name in CLIENT_NAMES:
        if sha(scripts / name) != manifest['client_source_sha256'][name]:
            raise ValueError('Reviewed client source changed since QA preparation')
    script = scripts / ('benchmark_client.py' if backend == 'sglang' else 'batch_benchmark_client.py')
    command = ['python3', str(script), '--manifest', str(manifest_path), '--profile', PROFILE,
               '--endpoint', endpoint, '--output', str(output)]
    if backend == 'sglang':
        command += ['--submission-mode', 'closed-loop', '--concurrency', '8',
                    '--warmup-per-endpoint', '0', '--timeout-seconds', '1200']
        rows = output / 'requests.jsonl'
    else:
        if not server_revision:
            raise ValueError('An observed BatchGen server revision is required to print its client command')
        command += ['--server-revision', server_revision, '--warmup-count', '0',
                    '--max-context-length', '10240', '--timeout-seconds', '86400']
        rows = output / 'measurement.requests.jsonl'
    return {'status': 'plan_only_NOT_requested', 'scope': SCOPE, 'backend': backend,
            'manifest_sha256': sha(manifest_path), 'request_count': COUNT,
            'command': command, 'score_result_jsonl': str(rows),
            'client_logic': 'Existing pinned clients preserve raw response bodies; no retries or redirects.',
            'timing_scope': 'Zero warmup is deliberate for QA only; do not use these timings for BCT/speedup.',
            'deployment_required': 'Verify current R1 revision/tokenizer/precision/topology/cache and actual ready service separately before executing the command.'}


def score(manifest_path, result_path, config_path, backend, output):
    manifest, cases = load_qa(manifest_path)
    expected_name = 'requests.jsonl' if backend == 'sglang' else 'measurement.requests.jsonl'
    if result_path.resolve().parent != config_path.resolve().parent or result_path.name != expected_name:
        raise ValueError('Client config and backend result file must belong to the same run directory')
    config = json.loads(config_path.read_text())
    client_name = 'benchmark_client.py' if backend == 'sglang' else 'batch_benchmark_client.py'
    artifact = manifest['artifacts'][0]
    request_key = 'request_file_sha256' if backend == 'sglang' else 'source_requests_sha256'
    provenance_key = 'provenance_file_sha256' if backend == 'sglang' else 'source_provenance_sha256'
    if (config.get('manifest_sha256') != sha(manifest_path)
            or config.get('client_sha256') != manifest['client_source_sha256'][client_name]
            or config.get('effective_payloads_sha256') != manifest['effective_payloads_sha256']
            or config.get(request_key) != artifact['requests_sha256']
            or config.get(provenance_key) != artifact['provenance_sha256']
            or config.get('request_count') != COUNT or config.get('temperature') != 0
            or config.get('ignore_eos') is not False):
        raise ValueError('Client config does not bind these exact frozen inputs/generation/client bytes')
    expected = {c['custom_id'] for c in cases}
    responses = {}
    with result_path.open() as stream:
        for line in stream:
            response = json.loads(line)
            key = response.get('custom_id')
            if key not in expected or key in responses:
                raise ValueError('Unknown or duplicate response ID')
            responses[key] = response
    totals = Counter({key: 0 for key in ('successful', 'failed_or_missing', 'missing_response',
        'reported_stop', 'length_capped', 'completion_tokens_at_cap', 'unknown_finish_reason',
        'reported_length_capped_all_responses', 'reported_stop_all_responses',
        'B_extractable', 'B_nonempty_suffix', 'B_no_think_markup', 'B_missing_think_close',
        'B_unclosed_think_after_last_close', 'B_empty_observable_suffix', 'B_multiple_closes')})
    task_values = {}; records = []
    for case in cases:
        key = case['custom_id']; response = responses.get(key)
        text = response.get('output_text') if response else None
        usage = response.get('usage') or {} if response else {}
        if response and (response.get('dataset') != case['dataset']
                         or response.get('original_id') != case['original_id']):
            raise ValueError('Response source identity differs from the frozen QA case')
        totals['reported_length_capped_all_responses'] += int(bool(response and response.get('finish_reason') == 'length'))
        totals['reported_stop_all_responses'] += int(bool(response and response.get('finish_reason') == 'stop'))
        successful = bool(response and response.get('success') is True and isinstance(text, str)
            and usage.get('prompt_tokens') == 8192 and type(usage.get('completion_tokens')) is int
            and 0 <= usage['completion_tokens'] <= 2048 and response.get('finish_reason') in ('stop', 'length'))
        if successful:
            totals['successful'] += 1
            totals['reported_stop' if response['finish_reason'] == 'stop' else 'length_capped'] += 1
            totals['completion_tokens_at_cap'] += int(usage['completion_tokens'] == 2048)
            raw = reference_scores(text, case['answers'])
            suffix = post_think_suffix(text)
            post = reference_scores(suffix['suffix'], case['answers']) if suffix['extractable'] else {'em': 0.0, 'token_f1': 0.0}
            totals['B_extractable'] += int(suffix['extractable'])
            totals['B_nonempty_suffix'] += int(bool(suffix['suffix']))
            totals['B_' + suffix['status']] += 1
            totals['B_multiple_closes'] += int(suffix['multiple_closes'])
        else:
            totals['failed_or_missing'] += 1
            totals['missing_response'] += int(response is None)
            totals['unknown_finish_reason'] += int(bool(response and response.get('finish_reason') not in ('stop', 'length')))
            raw = post = {'em': 0.0, 'token_f1': 0.0}
            suffix = {'status': 'failed_or_missing', 'extractable': False, 'suffix': ''}
        record = {'custom_id': key, 'dataset': case['dataset'], 'original_id': case['original_id'],
            'successful': successful, 'finish_reason': response.get('finish_reason') if response else None,
            'completion_tokens': usage.get('completion_tokens'), 'A_raw_completion': raw,
            'B_observable_post_think_suffix_lower_bound': post,
            'suffix_extraction': {k: v for k, v in suffix.items() if k != 'suffix'}}
        records.append(record)
        values = task_values.setdefault(case['dataset'], [])
        values.append(record)
    def aggregate(rows):
        return {'count': len(rows), 'A_raw_completion': {k: sum(r['A_raw_completion'][k] for r in rows) / len(rows) for k in ('em', 'token_f1')},
            'B_observable_post_think_suffix_lower_bound': {k: sum(r['B_observable_post_think_suffix_lower_bound'][k] for r in rows) / len(rows) for k in ('em', 'token_f1')},
            'B_suffix_coverage': sum(r['suffix_extraction']['extractable'] for r in rows) / len(rows),
            'B_nonempty_suffix_coverage': sum(r['suffix_extraction']['status'] == 'observable_suffix' for r in rows) / len(rows)}
    report = {'status': 'descriptive_QA_sanity_NOT_quality_or_statistical_pass', 'scope': SCOPE,
        'manifest_sha256': sha(manifest_path), 'results_sha256': sha(result_path),
        'client_config_sha256': sha(config_path), 'backend': backend,
        'generation': manifest['generation'], 'metrics': METRICS, 'counts': dict(totals),
        **aggregate(records), 'per_task': {k: aggregate(v) for k, v in task_values.items()},
        'records': records, 'limitations': manifest['limitations']}
    write_new(output, report)
    return {'status': report['status'], 'scope': SCOPE, 'report': str(output), 'sha256': sha(output),
            'count': COUNT, 'counts': dict(totals), **aggregate(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--expected-source-manifest-sha', default=SOURCE_MANIFEST_SHA)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('plan'); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--backend', choices=('batchgen', 'sglang'), required=True)
    p.add_argument('--endpoint', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--server-revision')
    p = sub.add_parser('score'); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True); p.add_argument('--client-config', type=Path, required=True)
    p.add_argument('--backend', choices=('batchgen', 'sglang'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.source_manifest, args.output, args.expected_source_manifest_sha)
    elif args.action == 'plan':
        result = client_plan(args.manifest, args.backend, args.endpoint, args.output, args.server_revision)
    else:
        result = score(args.manifest, args.results, args.client_config, args.backend, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
