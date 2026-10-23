"""CPU-only frozen-input, extraction, and all-item-denominator contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/qa_sanity.py'
spec = importlib.util.spec_from_file_location('r1_qa_sanity', SOURCE)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def source_fixture(root, unique=66, repeat=True):
    requests, provenance = [], []
    for i in range(unique + int(repeat)):
        identity = 0 if i == unique else i
        custom = 'source-' + str(i)
        ids = [0, identity + 1] + [42] * 8190
        requests.append({'custom_id': custom, 'method': 'POST', 'url': '/v1/completions',
            'body': {'model': 'deepseek-ai/DeepSeek-R1', 'prompt': 'Raw original ' + str(identity),
                'max_tokens': 2048, 'temperature': 0.0, 'top_p': 1.0, 'n': 1,
                'stream': False, 'ignore_eos': True}})
        provenance.append({'custom_id': custom, 'dataset': 'hotpotqa', 'original_id': str(identity),
            'language': 'en', 'answers': ['Paris', 'the Paris'], 'input_tokens': 8192,
            'max_output_tokens': 2048, 'input_token_ids': ids,
            'input_token_ids_sha256': qa.hash_value(ids), 'source_row_sha256': str(identity),
            'context_modified': True})
    for name, rows in [('requests.jsonl', requests), ('provenance.jsonl', provenance)]:
        (root / name).write_text(''.join(qa.canonical(row) + '\n' for row in rows))
    manifest = {'tasks': ['hotpotqa'], 'prompt_format': 'raw', 'ignore_eos': True,
        'count_per_profile': len(requests), 'dataset_revision': 'source-revision',
        'tokenizer_revision': 'checkpoint-revision', 'tokenizer_files': {'tokenizer.json': 'a' * 64},
        'tokenizer_file_pins': {'tokenizer.json': {'sha256': 'a' * 64, 'revision': 'checkpoint-revision'}},
        'tokenizer_file_manifest_sha256': 'b' * 64,
        'artifacts': [{'input_tokens': 8192, 'max_output_tokens': 2048,
            'requests': 'requests.jsonl', 'requests_sha256': qa.sha(root / 'requests.jsonl'),
            'provenance': 'provenance.jsonl', 'provenance_sha256': qa.sha(root / 'provenance.jsonl')}]}
    path = root / 'manifest.json'
    qa.write_new(path, manifest)
    return path, requests, provenance


class MetricTests(unittest.TestCase):
    def test_normalization_aliases_and_multiset_f1(self):
        self.assertEqual(qa.reference_scores('The QUICK, brown fox!', ['quick brown fox']),
                         {'em': 1.0, 'token_f1': 1.0})
        self.assertEqual(qa.reference_scores('USA', ['United States', 'USA'])['em'], 1)
        self.assertAlmostEqual(qa.reference_scores('alpha alpha beta', ['alpha beta beta'])['token_f1'], 2 / 3)
        self.assertEqual(qa.reference_scores('New-York', ['New York'])['em'], 0)
        self.assertEqual(qa.reference_scores('', ['Paris']), {'em': 0.0, 'token_f1': 0.0})

    def test_suffix_never_searches_reasoning_for_gold(self):
        suffix = qa.post_think_suffix('Reasoning says Paris</think>London')
        self.assertEqual(suffix['suffix'], 'London')
        self.assertEqual(qa.reference_scores(suffix['suffix'], ['Paris'])['em'], 0)
        suffix = qa.post_think_suffix('x</think>wrong</think>Paris')
        self.assertTrue(suffix['multiple_closes'])
        self.assertEqual(suffix['suffix'], 'Paris')

    def test_missing_markup_and_unclosed_thought_are_explicit(self):
        for text, status in [('Paris', 'no_think_markup'),
                             ('<think>Paris', 'missing_think_close'),
                             ('r</think><think>unfinished', 'unclosed_think_after_last_close')]:
            with self.subTest(text=text):
                value = qa.post_think_suffix(text)
                self.assertFalse(value['extractable'])
                self.assertEqual(value['status'], status)
                self.assertEqual(value['suffix'], '')
        empty = qa.post_think_suffix('r</think>')
        self.assertTrue(empty['extractable'])
        self.assertEqual(empty['status'], 'empty_observable_suffix')
        # Direct raw completion remains meaningful in metric A.
        self.assertEqual(qa.reference_scores('Paris', ['Paris'])['em'], 1)


class PreparationTests(unittest.TestCase):
    def test_freeze_preserves_raw_prompt_ids_answers_unique_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, requests, original = source_fixture(root)
            before = {p.name: qa.sha(p) for p in root.iterdir() if p.is_file()}
            for name in ('qa-a', 'qa-b'):
                qa.prepare(source, root / name, qa.sha(source))
            a, cases = qa.load_qa(root / 'qa-a/manifest.json')
            b, other = qa.load_qa(root / 'qa-b/manifest.json')
            self.assertEqual(a, b)
            self.assertEqual(cases, other)
            self.assertEqual(len({(c['dataset'], c['original_id']) for c in cases}), 64)
            frozen = [json.loads(x) for x in (root / 'qa-a/input8192-output2048.requests.jsonl').read_text().splitlines()]
            for request, meta in zip(frozen, cases):
                i = int(meta['original_id'])
                self.assertEqual(request['body']['prompt'], requests[i]['body']['prompt'])
                self.assertEqual(meta['input_token_ids'], original[i]['input_token_ids'])
                self.assertEqual(meta['answers'], original[i]['answers'])
                self.assertFalse(request['body']['ignore_eos'])
                self.assertEqual(request['body']['top_k'], 1)
            self.assertEqual(before, {name: qa.sha(root / name) for name in before})
            client_path = SOURCE.with_name('benchmark_client.py')
            client_spec = importlib.util.spec_from_file_location('qa_existing_client', client_path)
            client = importlib.util.module_from_spec(client_spec); client_spec.loader.exec_module(client)
            _, _, items, _, effective = client.load_workload(root / 'qa-a/manifest.json', qa.PROFILE, None, False)
            self.assertEqual(len(items), 64)
            self.assertEqual(effective, a['effective_payloads_sha256'])
            self.assertTrue(all(x['body']['top_k'] == 1 and x['body']['ignore_eos'] is False for x in items))
            with self.assertRaises(FileExistsError):
                qa.prepare(source, root / 'qa-a', qa.sha(source))

    def test_conflicting_duplicates_tamper_and_insufficient_unique_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, provenance = source_fixture(root)
            manifest = json.loads(source.read_text())
            paths = {k: root / manifest['artifacts'][0][k] for k in ('requests', 'provenance')}
            provenance[-1]['answers'] = ['Conflicting answer']
            paths['provenance'].write_text(''.join(qa.canonical(x) + '\n' for x in provenance))
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
                qa.candidate_pool(manifest, paths)
            with self.assertRaisesRegex(ValueError, 'artifact SHA mismatch'):
                qa.prepare(source, root / 'bad', qa.sha(source))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, _ = source_fixture(root, unique=63)
            with self.assertRaisesRegex(ValueError, 'Fewer than 64'):
                qa.prepare(source, root / 'bad', qa.sha(source))

    def test_client_plan_only_uses_frozen_inputs_and_eos_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, _ = source_fixture(root)
            qa.prepare(source, root / 'qa', qa.sha(source))
            manifest = root / 'qa/manifest.json'
            sg = qa.client_plan(manifest, 'sglang', 'http://node0:30000', root / 'sg')
            self.assertEqual(sg['status'], 'plan_only_NOT_requested')
            self.assertNotIn('--ignore-eos', sg['command'])
            self.assertEqual(sg['command'][sg['command'].index('--concurrency') + 1], '8')
            self.assertEqual(sg['command'][sg['command'].index('--timeout-seconds') + 1], '1200')
            with self.assertRaises(ValueError):
                qa.client_plan(manifest, 'batchgen', 'http://node0:10900', root / 'bg')
            bg = qa.client_plan(manifest, 'batchgen', 'http://node0:10900', root / 'bg', 'reviewed-revision')
            self.assertEqual(bg['command'][bg['command'].index('--warmup-count') + 1], '0')
            # Frozen artifact tampering is rejected even by plan, before any client can run.
            (root / 'qa/input8192-output2048.requests.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                qa.client_plan(manifest, 'sglang', 'http://node0:30000', root / 'sg')

    def test_all64_denominator_separates_direct_raw_and_suffix_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _, _ = source_fixture(root)
            qa.prepare(source, root / 'qa', qa.sha(source))
            manifest = root / 'qa/manifest.json'; _, cases = qa.load_qa(manifest)
            frozen = json.loads(manifest.read_text())
            artifact = frozen['artifacts'][0]
            config = {'manifest_sha256': qa.sha(manifest),
                'client_sha256': frozen['client_source_sha256']['benchmark_client.py'],
                'effective_payloads_sha256': frozen['effective_payloads_sha256'],
                'request_file_sha256': artifact['requests_sha256'],
                'provenance_file_sha256': artifact['provenance_sha256'],
                'request_count': 64, 'temperature': 0.0, 'ignore_eos': False}
            config_path = root / 'config.json'; qa.write_new(config_path, config)
            rows = []
            for i, (text, finish, success) in enumerate([
                    ('Paris', 'stop', True), ('reason</think>Paris', 'length', True),
                    ('<think>Paris', 'length', True), ('Paris', 'stop', False),
                    ('r</think>', 'stop', True), ('r</think>Paris', 'length', True)]):
                rows.append({'custom_id': cases[i]['custom_id'], 'output_text': text,
                    'dataset': cases[i]['dataset'], 'original_id': cases[i]['original_id'],
                    'success': success, 'finish_reason': finish,
                    'usage': {'prompt_tokens': 8192, 'completion_tokens': 2048 if finish == 'length' else 2}})
            rows[-1]['usage']['prompt_tokens'] = 999  # reported length, invalid usage; never silently drop it
            result_path = root / 'requests.jsonl'
            result_path.write_text(''.join(qa.canonical(x) + '\n' for x in rows))
            qa.score(manifest, result_path, config_path, 'sglang', root / 'score.json')
            report = json.loads((root / 'score.json').read_text())
            self.assertEqual(report['count'], 64)
            self.assertEqual(report['A_raw_completion']['em'], 1 / 64)
            self.assertEqual(report['B_observable_post_think_suffix_lower_bound']['em'], 1 / 64)
            self.assertEqual(report['B_suffix_coverage'], 2 / 64)
            self.assertEqual(report['B_nonempty_suffix_coverage'], 1 / 64)
            self.assertEqual(report['counts']['length_capped'], 2)
            self.assertEqual(report['counts']['reported_length_capped_all_responses'], 3)
            self.assertEqual(report['counts']['missing_response'], 58)
            self.assertEqual(report['counts']['failed_or_missing'], 60)
            self.assertNotIn('passed', report)
            result_path.write_text(result_path.read_text() + qa.canonical(rows[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate response'):
                qa.score(manifest, result_path, config_path, 'sglang', root / 'duplicate.json')
            for field in ['manifest_sha256', 'client_sha256', 'effective_payloads_sha256']:
                bad = {**config, field: 'f' * 64}; config_path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'Client config'):
                    qa.score(manifest, result_path, config_path, 'sglang', root / ('wrong-' + field + '.json'))
            config_path.write_text(json.dumps(config))
            bad_row = {**rows[0], 'original_id': 'wrong-source'}
            result_path.write_text(qa.canonical(bad_row) + '\n')
            with self.assertRaisesRegex(ValueError, 'source identity'):
                qa.score(manifest, result_path, config_path, 'sglang', root / 'wrong-source.json')


if __name__ == '__main__':
    unittest.main()
