#!/usr/bin/env python3
"""Fixed R1 greedy/teacher-forced diagnostics; NEVER a quality/performance gate.

Stages may run with the two engines deployed at different times:
  python full_model_correctness.py prepare --tokenizer /models/R1 \
    --tokenizer-file-manifest r1_model_manifest.json --output fixture.json
  python full_model_correctness.py deployment-template --fixture fixture.json \
    --backend batchgen --output deployment.json
  # Fill null deployment fields with observed launch values before running.
  python full_model_correctness.py run --fixture fixture.json --backend batchgen \
    --endpoint http://node0:8000 --deployment deployment.json --output bg-run
  python full_model_correctness.py run --fixture fixture.json --backend sglang \
    --endpoint http://node0:30000 --deployment deployment.json --output sg-run
  python full_model_correctness.py compare --fixture fixture.json \
    --batchgen-run bg-run/run.json --sglang-run sg-run/run.json \
    --tokenizer /models/R1 --output comparison.json

prepare needs local transformers/tokenizers/PyTorch CPU; run uses stdlib only.
No service launches, model loading, remote-code trust, retries or GPU operations.
BatchGen files/batches output TEXT ONLY: token differences below are explicitly
retokenized text, NOT original sampled token equality. SGLang native /generate
returns output_ids and input_token_logprobs; the latter supports teacher-forced
NLL, NOT raw logits, and BatchGen currently has no matching API. Never infer KL,
full-vocabulary cosine, or cross-engine teacher-forced equality from this data.
Prior attention matrix v4 (1 passed / 8 failed) remains unchanged. Exit zero
means the diagnostic artifact was collected, not that model correctness passed.

API evidence pinned to SGLang commit 7b2fb3d47c9dc43367c6fd3bf9396f7ba45e6883:
python/sglang/srt/{managers/io_struct.py,managers/tokenizer_manager.py,
sampling/sampling_params.py,entrypoints/http_server.py}. Local BatchGen evidence:
server/io_struct.py CompletionRequest; server/batch_scheduler.py result writer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

SEED = 20261023
REVISION = '20f2865710d161dc4a32c4e8c3d7c42c7b9852f8'
SGLANG_COMMIT = '7b2fb3d47c9dc43367c6fd3bf9396f7ba45e6883'
FIXED_CASES = (
    ('arithmetic', 'Compute 17 multiplied by 19. Give the result.', '323'),
    ('algebra', 'Solve 3*x + 7 = 22 for x.', 'x = 5'),
    ('chinese', '盒子里有12个红球和8个蓝球。红球占全部球的比例是多少？', '红球占60%。'),
    ('recall', 'The access phrase in this fictional exercise is silver maple 47. Repeat the access phrase.', 'silver maple 47'),
    ('code', 'What does this Python expression evaluate to? sum(i*i for i in range(4))', '14'),
    ('logic', 'All glims are norks. All norks are tevs. Must every glim be a tev?', 'Yes.'),
)
CONDITIONS = {
    'model_revision': REVISION, 'weight_format': 'original_fp8_e4m3fn',
    'weight_scales': 'fp32_block128x128', 'activation_dtype': 'bfloat16',
    'kv_dtype': 'bfloat16', 'protocol_seed': SEED, 'prefix_cache': False,
    'cuda_graphs': False, 'speculative_decoding': False,
    'detokenization_include_special_tokens': False,
}
LIMITATIONS = [
    'No whole-model quality gate is defined or claimed by this tool.',
    'Earlier strict layer gate failures remain failures; this report cannot override them.',
    'BatchGen native output token IDs and teacher-forced logits/logprobs are unavailable.',
    'Text re-encoding may merge/split tokens or omit EOS/special tokens; it is not native token identity.',
    'Greedy does not guarantee bitwise deterministic kernels; two fixed repeats test observed stability.',
    'Seed 20261023 is sent to SGLang; BatchGen exposes no request seed and uses greedy argmax without sampling RNG.',
    'Operator deployment declarations are recorded, not independently verified by HTTP.',
    'Fixed short probes are diagnostics, not a representative task-quality dataset.',
    'Auxiliary batch-client timing/rate fields are transport diagnostics, not benchmark results.',
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_fixture(value):
    if value.get('schema') != 'r1-fixed-correctness-v1' or value.get('conditions') != CONDITIONS:
        raise ValueError('Fixture schema/precision conditions changed; create a new protocol version')
    if value.get('generation') != {'max_new_tokens': 128, 'temperature': 0.0, 'top_p': 1.0,
                                    'ignore_eos': False, 'repeats': 2, 'request_concurrency': 1}:
        raise ValueError('Fixed generation conditions changed')
    cases = value['cases']
    if [(c['id'], c['user_text'], c['teacher_text']) for c in cases] != list(FIXED_CASES):
        raise ValueError('Fixed prompt/teacher probe order changed')
    for case in cases:
        for key in ('input_ids', 'teacher_ids'):
            if not case[key] or any(type(x) is not int or x < 0 for x in case[key]):
                raise ValueError('Expected nonempty token ID vectors')
        if len(case['input_ids']) > 2048 or len(case['teacher_ids']) > 128:
            raise ValueError('Unexpectedly long fixed probe')
    return value


def prepare(args):
    from transformers import AutoTokenizer
    from prepare_workload import load_service_tokenizer, render_prompt, tokenizer_file_pins
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    service, service_provenance = load_service_tokenizer(args.tokenizer / 'tokenizer.json')
    files = [args.tokenizer / name for name in ('tokenizer.json', 'tokenizer_config.json')]
    pins = tokenizer_file_pins(files, args.tokenizer_file_manifest)
    cases = []
    for name, user, teacher in FIXED_CASES:
        prompt = render_prompt(tokenizer, '{input}', '', user, 'chat')
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        actual = service([prompt], return_tensors=None, padding=False, truncation=False,
                         return_attention_mask=False)['input_ids'][0]
        if actual != ids:
            raise ValueError('BatchGen service/HF tokenization differs for ' + name)
        cases.append({'id': name, 'user_text': user, 'prompt': prompt, 'input_ids': ids,
                      'teacher_text': teacher, 'teacher_ids': tokenizer.encode(teacher, add_special_tokens=False)})
    fixture = {'schema': 'r1-fixed-correctness-v1', 'conditions': CONDITIONS,
               'generation': {'max_new_tokens': 128, 'temperature': 0.0, 'top_p': 1.0,
                              'ignore_eos': False, 'repeats': 2, 'request_concurrency': 1},
               'cases': cases, 'tokenizer_files': pins, 'service_tokenizer': service_provenance,
               'teacher_probe_contract': 'Append fixed teacher_ids to exact prompt IDs; not tokenization of concatenated text or gold-answer accuracy',
               'limitations': LIMITATIONS, 'source_sha256': sha(__file__)}
    validate_fixture(fixture)
    save_new(args.output, fixture)
    print(canonical({'fixture': str(args.output), 'sha256': sha(args.output)}))


def validate_deployment(value, backend, fixture):
    for key, expected in CONDITIONS.items():
        if type(value.get(key)) is not type(expected) or value.get(key) != expected:
            raise ValueError('Deployment must declare fixed ' + key + '=' + repr(expected))
    if value.get('backend') != backend:
        raise ValueError('Deployment backend mismatch')
    for key in ('server_revision', 'model_id', 'attention_backend', 'activation_quantization',
                'launch_command_sha256', 'weight_verification_evidence'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError('Deployment must declare ' + key)
    for key in ('tp', 'ep', 'nodes', 'gpu_count'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError('Deployment must declare positive ' + key)
    launch_hash = value['launch_command_sha256']
    if len(launch_hash) != 64 or any(c not in '0123456789abcdef' for c in launch_hash):
        raise ValueError('launch_command_sha256 must be a lowercase SHA256')
    expected_assets = {name: item['sha256'] for name, item in fixture['tokenizer_files'].items()}
    if value.get('tokenizer_sha256') != expected_assets:
        raise ValueError('Deployment tokenizer hashes differ from frozen fixture')
    if backend == 'sglang' and value['server_revision'] != SGLANG_COMMIT:
        raise ValueError('SGLang API audit covers only the pinned source revision')
    return value


def deployment_template(args):
    fixture = validate_fixture(json.loads(args.fixture.read_text()))
    value = {**CONDITIONS, 'backend': args.backend, 'model_id': 'DeepSeek-R1',
             'server_revision': SGLANG_COMMIT if args.backend == 'sglang' else None,
             'tokenizer_sha256': {n: v['sha256'] for n, v in fixture['tokenizer_files'].items()},
             'attention_backend': None, 'activation_quantization': None, 'launch_command_sha256': None,
             'weight_verification_evidence': None, 'tp': None, 'ep': None, 'nodes': None, 'gpu_count': None,
             '_instructions': 'Replace every null with observed deployment values. Conditions are required protocol settings; verify the actual launch, do not blindly attest them.'}
    save_new(args.output, value)


def sglang_payload(case, teacher=False):
    sampling = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'max_new_tokens': 0 if teacher else 128,
                'ignore_eos': False, 'sampling_seed': SEED, 'skip_special_tokens': True,
                'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}
    return {'input_ids': case['input_ids'] + (case['teacher_ids'] if teacher else []),
            'sampling_params': sampling, 'stream': False, 'return_logprob': True,
            'logprob_start_len': 0 if teacher else -1, 'return_text_in_logprobs': False}


def teacher_metrics(response, case):
    rows = response.get('meta_info', {}).get('input_token_logprobs')
    if rows is None:
        return {'status': 'unavailable', 'reason': 'Server did not return input_token_logprobs'}
    ids = case['input_ids'] + case['teacher_ids']
    # logprob_start_len=0 returns positional rows, commonly with null BOS. Do
    # not search by matching repeated token IDs and silently misalign positions.
    offset = 0 if len(rows) == len(ids) else 1 if len(rows) == len(ids)-1 else None
    if offset is None or any(not isinstance(row, (list, tuple)) or len(row) < 2 or row[1] != ids[i+offset]
                             for i, row in enumerate(rows)):
        raise ValueError('Teacher-forced token alignment is ambiguous or incorrect')
    target = rows[len(case['input_ids'])-offset:]
    logps = [row[0] for row in target]
    if len(logps) != len(case['teacher_ids']) or any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in logps):
        raise ValueError('Missing/nonfinite/invalid teacher-token log probabilities')
    nll = -sum(logps) / len(logps)
    return {'status': 'collected_reference_only', 'teacher_token_ids': case['teacher_ids'],
            'teacher_token_logprobs': logps, 'mean_negative_log_likelihood': nll,
            'perplexity': math.exp(nll) if nll < 709 else None,
            'full_logits_available': False, 'cross_engine_comparison_available': False}


def run(args):
    import batch_benchmark_client as batch_api
    fixture = validate_fixture(json.loads(args.fixture.read_text()))
    deployment = validate_deployment(json.loads(args.deployment.read_text()), args.backend, fixture)
    endpoint = batch_api.base_url(args.endpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    save_new(args.output / 'fixture.json', fixture)
    save_new(args.output / 'deployment.json', deployment)
    summary = {'schema': 'r1-full-model-diagnostic-run-v1', 'status': 'collection_failed',
               'backend': args.backend, 'fixture_sha256': sha(args.fixture),
               'deployment': deployment, 'deployment_sha256': sha(args.deployment),
               'source_sha256': sha(__file__), 'endpoint': endpoint, 'records': [],
               'limitations': LIMITATIONS, 'teacher_forced_batchgen': 'unsupported by current API',
               'seed_control': {'protocol_seed': SEED, 'sglang': 'request sampling_seed',
                                'batchgen': 'no seed field; greedy argmax does not use sampling RNG'},
               'full_logits_comparison': 'unsupported by current APIs'}
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    started = time.perf_counter()
    # One submitted item at a time, identical fixed order, exactly two repeats.
    try:
        with (args.output / 'diagnostic.http.jsonl').open('x') as http_log:
            discovery = batch_api.Transport(endpoint, api_key, min(args.timeout, 60),
                                            time.perf_counter()+args.timeout, started, http_log)
            models = discovery.call('GET', '/v1/models')
            summary['models_response'] = models
            if deployment['model_id'] not in [item.get('id') for item in models.get('data', []) if isinstance(item, dict)]:
                raise ValueError('Declared model ID absent from endpoint /v1/models')
            for repeat in range(2):
                for case in fixture['cases']:
                    phase = f'r{repeat}-{case["id"]}'
                    record = {'id': case['id'], 'repeat': repeat, 'native_output_ids': None}
                    if args.backend == 'batchgen':
                        body = {'model': deployment['model_id'], 'prompt': case['prompt'], 'temperature': 0.0,
                                'top_p': 1.0, 'top_k': 1, 'max_tokens': 128, 'n': 1, 'stream': False,
                                'ignore_eos': False, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}
                        item = {'custom_id': phase, 'original_id': case['id'], 'dataset': 'r1-correctness-fixed-v1',
                                'body': body, 'expected_input_tokens': len(case['input_ids']), 'max_output_tokens': 128}
                        cfg = SimpleNamespace(timeout_seconds=args.timeout, http_timeout_seconds=min(args.timeout, 60),
                                              poll_seconds=1, max_context_length=4096, ignore_eos=False)
                        outcome = batch_api.run_phase(cfg, phase, [item], args.output, endpoint, api_key)
                        save_new(args.output / (phase + '.batch.json'), outcome)
                        if not outcome['all_requests_valid']:
                            raise RuntimeError('BatchGen diagnostic request failed; inspect preserved batch artifact')
                        rows = [json.loads(line) for line in (args.output/(phase+'.requests.jsonl')).read_text().splitlines()]
                        if len(rows) != 1 or not rows[0]['success']:
                            raise ValueError('Expected exactly one successful batch result')
                        row = rows[0]
                        record.update(text=row['output_text'], usage=row['usage'], finish_reason=row['finish_reason'],
                                      native_tokens_available=False)
                    else:
                        transport = batch_api.Transport(endpoint, api_key, min(args.timeout, 120),
                                                        time.perf_counter()+args.timeout, started, http_log)
                        response = transport.call('POST', '/generate', canonical(sglang_payload(case)).encode(), 'application/json')
                        save_new(args.output / (phase+'.response.json'), response)
                        meta = response.get('meta_info', {})
                        output_ids = response.get('output_ids')
                        if (not isinstance(response.get('text'), str) or not isinstance(output_ids, list) or
                            any(type(x) is not int or x < 0 for x in output_ids) or
                            meta.get('prompt_tokens') != len(case['input_ids']) or
                            meta.get('completion_tokens') != len(output_ids) or len(output_ids) > 128):
                            raise ValueError('SGLang response text/IDs/usage contract failed')
                        finish = meta.get('finish_reason')
                        if not isinstance(finish, dict) or finish.get('type') not in ('stop', 'length'):
                            raise ValueError('SGLang stopped with a non-successful/unknown reason')
                        record.update(text=response['text'], native_output_ids=output_ids, native_tokens_available=True,
                                      usage={'prompt_tokens': meta['prompt_tokens'], 'completion_tokens': meta['completion_tokens']},
                                      finish_reason=finish, output_token_logprobs=meta.get('output_token_logprobs'))
                        if repeat == 0:
                            # A new deadline for the separately auditable teacher request.
                            transport.deadline = time.perf_counter() + args.timeout
                            teacher = transport.call('POST', '/generate', canonical(sglang_payload(case, True)).encode(), 'application/json')
                            save_new(args.output / (phase+'.teacher-response.json'), teacher)
                            record['teacher_forced'] = teacher_metrics(teacher, case)
                    summary['records'].append(record)
        summary['status'] = 'diagnostic_collected_not_quality_pass'
    except Exception as exc:
        summary['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        summary['server_request_may_still_be_active'] = True
    finally:
        summary['elapsed_seconds_not_benchmark'] = time.perf_counter()-started
        save_new(args.output/'run.json', summary)
    print(canonical({'status': summary['status'], 'report': str(args.output/'run.json')}))
    return 0 if summary['status'] == 'diagnostic_collected_not_quality_pass' else 2


def sequence_metrics(a, b):
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    distance = previous[-1]
    return {'exact_match': a == b, 'length_a': len(a), 'length_b': len(b),
            'common_prefix_length': prefix, 'first_divergence_index': None if a == b else prefix,
            'positional_mismatches_in_overlap': sum(x != y for x, y in zip(a, b)),
            'edit_distance': distance, 'normalized_edit_distance': distance/max(1, len(a), len(b))}


def compare(args):
    from transformers import AutoTokenizer
    fixture = validate_fixture(json.loads(args.fixture.read_text()))
    for name, item in fixture['tokenizer_files'].items():
        if sha(args.tokenizer/name) != item['sha256']:
            raise ValueError('Comparison tokenizer differs from frozen fixture')
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    runs = {engine: json.loads(path.read_text()) for engine, path in
            (('batchgen', args.batchgen_run), ('sglang', args.sglang_run))}
    indexed = {}
    for engine, data in runs.items():
        if data.get('backend') != engine or data.get('fixture_sha256') != sha(args.fixture):
            raise ValueError('Engine or frozen fixture identity mismatch')
        if data.get('status') != 'diagnostic_collected_not_quality_pass':
            raise ValueError('Cannot compare incomplete/failed collection')
        validate_deployment(data['deployment'], engine, fixture)
        records = data['records']
        indexed[engine] = {(r['repeat'], r['id']): r for r in records}
        expected = {(repeat, c['id']) for repeat in range(2) for c in fixture['cases']}
        if len(indexed[engine]) != len(records) or set(indexed[engine]) != expected:
            raise ValueError('Missing or duplicate comparison records')
    result = {'status': 'diagnostic_comparison_only', 'fixture_sha256': sha(args.fixture),
              'input_runs_sha256': {'batchgen': sha(args.batchgen_run), 'sglang': sha(args.sglang_run)},
              'limitations': LIMITATIONS, 'cross_engine': [], 'repeatability': [],
              'teacher_forced_cross_engine_metrics': None, 'full_logits_metrics': None,
              'prior_attention_matrix_v4': {'passed': 1, 'failed': 8, 'overridden': False}}
    for repeat in range(2):
        for case in fixture['cases']:
            a, b = (indexed[engine][repeat, case['id']] for engine in ('batchgen', 'sglang'))
            at, bt = (tokenizer.encode(r['text'], add_special_tokens=False) for r in (a, b))
            result['cross_engine'].append({'id': case['id'], 'repeat': repeat,
                'batchgen_text': a['text'], 'sglang_text': b['text'], 'text_exact_match': a['text'] == b['text'],
                'batchgen_retokenized_ids': at, 'sglang_retokenized_ids': bt,
                'retokenized_text_tokens': sequence_metrics(at, bt), 'native_cross_engine_token_comparison': None,
                'sglang_native_output_ids': b['native_output_ids'],
                'sglang_native_vs_retokenized': sequence_metrics(b['native_output_ids'], bt),
                'completion_tokens_delta': a['usage']['completion_tokens']-b['usage']['completion_tokens'],
                'batchgen_finish_reason': a['finish_reason'], 'sglang_finish_reason': b['finish_reason'],
                'sglang_teacher_forced_reference_only': b.get('teacher_forced')})
    for engine in runs:
        for case in fixture['cases']:
            a, b = (indexed[engine][repeat, case['id']] for repeat in range(2))
            result['repeatability'].append({'engine': engine, 'id': case['id'],
                'text_exact_match': a['text'] == b['text'],
                'native_ids_exact_match': a['native_output_ids'] == b['native_output_ids'] if engine == 'sglang' else None})
    result['exact_text_matches'] = sum(r['text_exact_match'] for r in result['cross_engine'])
    result['comparison_count'] = len(result['cross_engine'])
    save_new(args.output, result)
    print(canonical({'status': result['status'], 'exact_text_matches': result['exact_text_matches'],
                     'comparison_count': result['comparison_count'], 'report': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare');p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--tokenizer-file-manifest', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('deployment-template');p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--backend', choices=('batchgen', 'sglang'), required=True);p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run');p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--backend', choices=('batchgen', 'sglang'), required=True);p.add_argument('--endpoint', required=True)
    p.add_argument('--deployment', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api-key-env');p.add_argument('--timeout', type=float, default=900)
    p = sub.add_parser('compare');p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--batchgen-run', type=Path, required=True);p.add_argument('--sglang-run', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, 'timeout', 1) <= 0 or not math.isfinite(getattr(args, 'timeout', 1)):
        parser.error('Timeout must be positive and finite')
    return {'prepare': prepare, 'deployment-template': deployment_template, 'run': run, 'compare': compare}[args.command](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
