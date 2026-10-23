"""CPU-only protocol freezing and mocked transports; no model or HTTP server."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('r1_longer_chat_probe_tested', DIRECTORY / 'longer_chat_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
import batch_benchmark_client as batch_api


def fixture_file(root):
    value = {'schema': 'r1-fixed-correctness-v1', 'conditions': copy.deepcopy(probe.fixed.CONDITIONS),
        'generation': {'max_new_tokens': 128, 'temperature': 0.0, 'top_p': 1.0,
                       'ignore_eos': False, 'repeats': 2, 'request_concurrency': 1},
        'cases': [{'id': case_id, 'user_text': user, 'teacher_text': teacher,
                   'prompt': '<User>' + user + '<Assistant><think>\n',
                   'input_ids': [0, index + 100, 42], 'teacher_ids': [index + 200]}
                  for index, (case_id, user, teacher) in enumerate(probe.fixed.FIXED_CASES)],
        'tokenizer_files': {'tokenizer.json': {'sha256': 'a' * 64}},
        'extra_provenance': {'retain': 'original fixture bytes'}}
    path = root / 'original.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=3) + '\n')
    return path, value


def deployment_file(root, fixture, backend):
    value = {**probe.fixed.CONDITIONS, 'backend': backend, 'model_id': 'deepseek-ai/DeepSeek-R1',
        'server_revision': probe.fixed.SGLANG_COMMIT if backend == 'sglang' else 'observed-bg-commit',
        'tokenizer_sha256': {name: item['sha256'] for name, item in fixture['tokenizer_files'].items()},
        'attention_backend': 'observed', 'activation_quantization': 'observed',
        'launch_command_sha256': 'b' * 64, 'weight_verification_evidence': 'pinned-weight-receipt',
        'tp': 16, 'ep': 1, 'nodes': 2, 'gpu_count': 16}
    path = root / (backend + '-deployment.json')
    probe.write_new(path, {'deployment': value, 'actual_launch_receipt': 'retained unchanged'})
    return path


class ProtocolTests(unittest.TestCase):
    def test_prepare_exact_inputs_teachers_order_source_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            before = source.read_bytes()
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                result = probe.prepare(source, root / 'long', 'longer-chat')
                manifest, actual_fixture, requests = probe.load_manifest(result['manifest'], result['manifest_sha256'])
                self.assertEqual(actual_fixture, fixture)
                self.assertEqual((root / 'long/source-fixture.json').read_bytes(), before)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(len(requests), 12)
                self.assertEqual(len({item['custom_id'] for item in requests}), 12)
                for index, item in enumerate(requests):
                    self.assertEqual(item['case'], fixture['cases'][index % 6])
                    self.assertEqual(item['copy_index'], index // 6)
                self.assertEqual(manifest['generation']['max_new_tokens'], 2048)
                self.assertFalse(manifest['generation']['ignore_eos'])
                self.assertEqual(manifest['generation']['protocol_seed'], 20261023)
                self.assertEqual(manifest['sglang_http_concurrency'], 12)
                with self.assertRaises(FileExistsError):
                    probe.prepare(source, root / 'long', 'longer-chat')

    def test_exact_original_pin_and_rehashed_tampering_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _ = fixture_file(root)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                probe.prepare(source, root / 'bad-pin', 'longer-chat')
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                result = probe.prepare(source, root / 'long', 'longer-chat')
                manifest_path = Path(result['manifest'])
                manifest = json.loads(manifest_path.read_bytes())
                requests_path = root / 'long/requests.json'
                requests = json.loads(requests_path.read_bytes())
                requests[0]['case']['input_ids'][0] += 1
                requests_path.write_text(probe.canonical(requests))
                manifest['requests_sha256'] = probe.sha(requests_path)
                manifest_path.write_text(probe.canonical(manifest))
                with self.assertRaisesRegex(ValueError, 'input/teacher/order'):
                    probe.load_manifest(manifest_path, probe.sha(manifest_path))

    def test_identical16_is_separate_protocol_without_assumed_rank_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                result = probe.prepare(source, root / 'same', 'identical-arithmetic')
                manifest, _, requests = probe.load_manifest(result['manifest'], result['manifest_sha256'])
                self.assertEqual({probe.canonical(item['case']) for item in requests},
                                 {probe.canonical(fixture['cases'][0])})
                self.assertEqual(len(requests), 16)
                self.assertEqual(len({item['custom_id'] for item in requests}), 16)
                self.assertEqual(manifest['generation']['max_new_tokens'], 128)
                self.assertEqual(manifest['protocol_tag'], 'r1-identical-arithmetic-16x128-v1')
                self.assertIn('unobserved', manifest['rank_coverage'])
                self.assertEqual(manifest['batchgen_batches'], 1)

    def test_both_plans_frozen_payloads_exactly12_no_teacher_http_and_source_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                frozen = probe.prepare(source, root / 'long', 'longer-chat')
                for backend in ('batchgen', 'sglang'):
                    deployment = deployment_file(root, fixture, backend)
                    with patch.object(probe, 'build_opener', side_effect=AssertionError('No HTTP in plan')):
                        plan = probe.make_plan(frozen['manifest'], frozen['manifest_sha256'], backend,
                            deployment, probe.sha(deployment), ['http://node0:30000/v1'])
                    self.assertEqual(len(plan['payloads']), 12)
                    self.assertEqual(plan['timeout_seconds'], 1200)
                    self.assertEqual(plan['retries'], 0)
                    self.assertEqual(plan['teacher_forced_requests'], 0)
                    self.assertEqual(plan['quality_gate'], None)
                    self.assertEqual(plan['endpoints'], ['http://node0:30000'])
                    if backend == 'sglang':
                        self.assertEqual(plan['http_concurrency'], 12)
                        for index, payload in enumerate(plan['payloads']):
                            self.assertEqual(payload['input_ids'], fixture['cases'][index % 6]['input_ids'])
                            self.assertEqual(payload['sampling_params']['max_new_tokens'], 2048)
                            self.assertEqual(payload['sampling_params']['sampling_seed'], 20261023)
                            self.assertEqual(payload['sampling_params']['top_k'], 1)
                            self.assertTrue(payload['return_logprob'])
                    else:
                        self.assertEqual(plan['batch_count'], 1)
                        self.assertIn('worker logs only', plan['native_ids_source'])
                        self.assertEqual(plan['payloads'][0]['body']['prompt'], fixture['cases'][0]['prompt'])
                        self.assertNotIn('seed', plan['payloads'][0]['body'])
                manifest_path = Path(frozen['manifest']); manifest = json.loads(manifest_path.read_bytes())
                manifest['source_pins']['benchmark_client.py'] = '0' * 64
                manifest_path.write_text(probe.canonical(manifest))
                with self.assertRaisesRegex(ValueError, 'source pins changed'):
                    probe.load_manifest(manifest_path, probe.sha(manifest_path))

    def test_plan_requires_matching_deployment_and_one_clean_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                frozen = probe.prepare(source, root / 'long', 'longer-chat')
                deployment = deployment_file(root, fixture, 'batchgen')
                for endpoints in ([], ['http://a', 'http://b'], ['http://u:p@a'], ['http://a?token=secret']):
                    with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                        probe.make_plan(frozen['manifest'], frozen['manifest_sha256'], 'batchgen',
                                        deployment, probe.sha(deployment), endpoints)
                with self.assertRaisesRegex(ValueError, 'Deployment source JSON SHA mismatch'):
                    probe.make_plan(frozen['manifest'], frozen['manifest_sha256'], 'batchgen',
                                    deployment, '0' * 64, ['http://node0'])
                with self.assertRaisesRegex(ValueError, 'backend mismatch'):
                    probe.make_plan(frozen['manifest'], frozen['manifest_sha256'], 'sglang',
                                    deployment, probe.sha(deployment), ['http://node0'])


class NativeResponseTests(unittest.TestCase):
    def test_empty_eos_response_remains_empty_native_evidence_is_explicit(self):
        item = {'custom_id': 'a', 'case_id': 'arithmetic', 'copy_index': 0, 'case': {'input_ids': [0, 7]}}
        response = {'text': '', 'output_ids': [1], 'meta_info': {'prompt_tokens': 2,
            'completion_tokens': 1, 'finish_reason': {'type': 'stop', 'matched': 1},
            'output_token_logprobs': [[-0.7, 1, None]]}}
        record = probe.sglang_record(response, item, 2048)
        self.assertTrue(record['success'])
        self.assertEqual(record['text'], '')
        self.assertEqual(record['native_output_ids'], [1])
        self.assertEqual(record['response'], response)
        del response['output_ids']
        self.assertFalse(probe.sglang_record(response, item, 2048)['native_output_ids_available'])
        response['output_ids'] = [1, 2]
        self.assertFalse(probe.sglang_record(response, item, 2048)['success'])

    def test_nonfinite_logprob_and_unexpected_finish_visible(self):
        item = {'custom_id': 'a', 'case_id': 'x', 'copy_index': 0, 'case': {'input_ids': [0]}}
        response = {'text': 'answer', 'output_ids': [3], 'meta_info': {'prompt_tokens': 1,
            'completion_tokens': 1, 'finish_reason': {'type': 'abort'},
            'output_token_logprobs': [[float('nan'), 3]]}}
        row = probe.sglang_record(response, item, 2048)
        self.assertFalse(row['success'])
        self.assertEqual(row['nonfinite_or_malformed_output_logprobs'], 1)
        self.assertEqual(row['finish_reason']['type'], 'abort')

    def test_http_bytes_preserved_and_failed_http_not_retried(self):
        class Response(io.BytesIO):
            status = 200
        item = {'custom_id': 'case-00', 'case_id': 'x', 'copy_index': 0, 'case': {'input_ids': [0]}}
        raw = b'{ "text": "Hi", "output_ids": [3], "meta_info": {"prompt_tokens":1,"completion_tokens":1,"finish_reason":{"type":"length"}} }\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opener = SimpleNamespace(open=lambda *_a, **_kw: Response(raw))
            with patch.object(probe, 'build_opener', return_value=opener):
                record = probe.native_call('http://node0', {'input_ids': [0]}, item, root, 128)
            self.assertTrue(record['success'])
            self.assertEqual((root / 'case-00.http-body').read_bytes(), raw)
            self.assertEqual(record['raw_response_sha256'], probe.hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(probe, 'build_opener', side_effect=TimeoutError('no retry')) as opener:
                record = probe.native_call('http://node0', {}, item, Path(tmp), 128)
            self.assertEqual(opener.call_count, 1)
            self.assertFalse(record['success'])
            self.assertTrue(record['server_request_may_still_be_active'])

    def test_nonstandard_nonfinite_response_keeps_raw_failure_record(self):
        class Response(io.BytesIO):
            status = 200
        item = {'custom_id': 'bad', 'case_id': 'x', 'copy_index': 0, 'case': {'input_ids': [0]}}
        for raw in (b'{"text":"x","score":NaN}', b'{"text":"x","score":1e999}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                opener = SimpleNamespace(open=lambda *_a, **_kw: Response(raw))
                with patch.object(probe, 'build_opener', return_value=opener):
                    record = probe.native_call('http://node0', {}, item, Path(tmp), 128)
                self.assertFalse(record['success'])
                self.assertIn('Nonfinite', record['errors'][0])
                self.assertEqual((Path(tmp) / 'bad.http-body').read_bytes(), raw)
                self.assertTrue((Path(tmp) / 'bad.record.json').exists())


class ExecutionTests(unittest.TestCase):
    def test_mocked_native_mode_12_inflight_and_results_restore_frozen_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                frozen = probe.prepare(source, root / 'long', 'longer-chat')
                deployment = deployment_file(root, fixture, 'sglang')
                barrier = threading.Barrier(12)
                seen, lock = [], threading.Lock()
                def native_call(endpoint, payload, item, output, cap):
                    with lock:
                        seen.append(item['custom_id'])
                    barrier.wait(timeout=5)
                    self.assertEqual(cap, 2048)
                    self.assertEqual(payload['input_ids'], item['case']['input_ids'])
                    return {'custom_id': item['custom_id'], 'success': True, 'native_output_ids': [1]}
                args = SimpleNamespace(manifest=Path(frozen['manifest']), expected_manifest_sha256=frozen['manifest_sha256'],
                    backend='sglang', deployment_source=deployment, expected_deployment_sha256=probe.sha(deployment),
                    endpoint=['http://node0:30000'], output=root / 'run')
                with patch.object(probe, 'native_call', side_effect=native_call) as client:
                    result = probe.execute(args)
                self.assertEqual(client.call_count, 12)
                self.assertEqual(len(seen), 12)
                self.assertIn('diagnostic_collected', result['status'])
                report = json.loads((root / 'run/run.json').read_bytes())
                plan = json.loads((root / 'run/plan.json').read_bytes())
                self.assertEqual([row['custom_id'] for row in report['records']], plan['request_ids'])

    def test_mocked_bg_executes_one_whole_batch_native_ids_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, fixture = fixture_file(root)
            with patch.object(probe, 'SOURCE_FIXTURE_SHA256', probe.sha(source)):
                frozen = probe.prepare(source, root / 'long', 'longer-chat')
                deployment = deployment_file(root, fixture, 'batchgen')
                def run_phase(config, phase, requests, directory, endpoint, api_key):
                    self.assertEqual(len(requests), 12)
                    self.assertFalse(config.ignore_eos)
                    self.assertEqual(config.max_context_length, 4096)
                    self.assertTrue(all(item['body']['max_tokens'] == 2048 for item in requests))
                    rows = [{'custom_id': item['custom_id'], 'success': True, 'output_text': 'text',
                             'usage': {'prompt_tokens': item['expected_input_tokens'], 'completion_tokens': 2},
                             'finish_reason': 'stop'} for item in requests]
                    (directory / 'probe.requests.jsonl').write_text(''.join(probe.canonical(row) + '\n' for row in rows))
                    return {'all_requests_valid': True, 'batch_id': 'one-batch'}
                args = SimpleNamespace(manifest=Path(frozen['manifest']), expected_manifest_sha256=frozen['manifest_sha256'],
                    backend='batchgen', deployment_source=deployment, expected_deployment_sha256=probe.sha(deployment),
                    endpoint=['http://node0:8000'], output=root / 'run')
                with patch.object(batch_api, 'run_phase', side_effect=run_phase) as client:
                    result = probe.execute(args)
                self.assertEqual(client.call_count, 1)
                self.assertIn('diagnostic_collected', result['status'])
                report = json.loads((root / 'run/run.json').read_text())
                self.assertTrue(all(row['native_output_ids'] is None for row in report['records']))
                self.assertEqual(report['rank_coverage'], 'unobserved')
                probe.load_manifest(root / 'run/manifest.json', frozen['manifest_sha256'])

    def test_no_arguments_prints_help_without_http(self):
        with patch.object(probe, 'build_opener', side_effect=AssertionError('No implicit HTTP')):
            with patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(probe.main([]), 0)


if __name__ == '__main__':
    unittest.main()
