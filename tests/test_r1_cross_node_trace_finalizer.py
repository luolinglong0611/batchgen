"""CPU-only sealing/union tests. Synthetic events never qualify a model."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/r1_reproduction/finalize_r1_cross_node_trace.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load('r1_cross_node_finalizer_test', PATH)
FIXTURES = load('r1_cross_node_existing_suite_fixture', ROOT / 'tests/test_r1_full_model_qualification_contract.py')
VERSIONS = load('r1_cross_node_versioned_suite_fixture', ROOT / 'tests/test_r1_full_model_suite_versions.py')


class CrossNodeTraceSealing(unittest.TestCase):
    def fixture(self, root, independent=False, suite=None):
        supplied_suite = suite is not None
        if suite is None:
            suite = self.independent_suite() if independent else FIXTURES.fixed_suite()
        suite_path = root / 'suite.json'
        M.write_new(suite_path, suite)
        qualifier = ROOT / 'scripts/r1_reproduction/qualify_r1_full_model.py'
        deployment = {'schema': 'r1-full-model-deployment-v1', 'backend': 'sglang', 'world_size': 16,
            'num_hidden_layers': 61, 'vocab_size': 129280, 'run_id': 'synthetic-offline-test',
            'tokenizer_json_sha256': suite['tokenizer_json_sha256'], 'graphs': False, 'prefix_cache': False,
            'speculative_decoding': False, 'checkpoint_identity': 'original-r1',
            'model_id': 'DeepSeek-R1', 'endpoint': 'http://never-contacted:30000',
            'actual_argv': ['synthetic-field-preserved-verbatim'],
            'source_files': {'full_model_trace': {'path': '/synthetic/deployed/trace.py', 'sha256': 'b' * 64}}}
        deployment_path = root / 'deployment.json'
        M.write_new(deployment_path, deployment)
        deployment = dict(deployment, artifact_sha256=M.sha(deployment_path))
        records = []
        for job in suite['jobs']:
            if job['stage'] != 'short':
                continue
            count = job['max_new_tokens'] if job['ignore_eos'] else 1
            records.append({'id': job['id'], 'native_output_ids': [42] * count,
                'prompt_tokens': len(job['input_ids']), 'completion_tokens': count,
                'text': 'FINAL: ' + job['answers'][0], 'finish_reason': 'length' if job['ignore_eos'] else 'stop'})
        raw_dirs, node0_events = [], []
        for node in (0, 1):
            folder = root / f'node{node}-raw'
            folder.mkdir()
            raw_dirs.append(folder)
            for rank in range(node * 8, (node + 1) * 8):
                envelope = {'schema': 'r1-full-model-trace-v1', 'rank': rank, 'world_size': 16,
                    'run_id': deployment['run_id'], 'suite_sha256': M.sha(suite_path),
                    'deployment_sha256': M.sha(deployment_path)}
                events = [dict(envelope, event='trace_start', ordinal=1, observed_unix_ns=1,
                               actual_pid=1000 + rank, source_files={'full_model_trace': 'b' * 64}, helper_sha256='b' * 64),
                    dict(envelope, event='model_hooks_registered', ordinal=2, observed_unix_ns=2,
                        actual_layer_count=61, actual_hook_count=183,
                        actual_config={'hidden_size': 7168, 'num_hidden_layers': 61, 'vocab_size': 129280, 'n_routed_experts': 256}),
                    dict(envelope, event='layer_summary', ordinal=3, observed_unix_ns=150,
                        last_nonempty_observed_unix_ns=150, component='attention', phase='prefill', layer_index=0,
                        nonfinite_elements=0, unobserved_activations=0, forward_calls=1, output_checked_elements=7168),
                    dict(envelope, event='layer_summary', ordinal=4, observed_unix_ns=250,
                        last_nonempty_observed_unix_ns=250, component='attention', phase='decode', layer_index=1,
                        nonfinite_elements=0, unobserved_activations=0, forward_calls=1, output_checked_elements=7168)]
                (folder / f'rank-{rank}-pid-{1000 + rank}.jsonl').write_text(''.join(M.canonical(e) + '\n' for e in events))
                if node == 0:
                    node0_events.extend(events[:3])
        original = root / 'original-collection'
        (original / 'trace').mkdir(parents=True)
        M.write_new(original / 'trace/events.json', node0_events)
        run = {'schema': FIXTURES.tool.RUN_SCHEMA, 'scope': 'full_model',
            'status': 'evidence_collected_NOT_pair_qualified', 'backend': 'sglang',
            'suite_sha256': M.sha(suite_path), 'tool_sha256': M.sha(qualifier), 'deployment': deployment,
            'started_unix_ns': 100, 'finished_unix_ns': 200, 'long_correctness': False,
            'teacher_method': 'original-method-must-not-change', 'records': records,
            'formal_performance_permitted': False,
            'trace': {'events_file': 'trace/events.json', 'events_sha256': M.sha(original / 'trace/events.json'), 'event_count': len(node0_events)}}
        if supplied_suite:
            run['suite_schema'] = suite['schema']
        M.write_new(original / 'run.json', run)
        common = dict(original_run=original / 'run.json', original_run_sha256=M.sha(original / 'run.json'),
            suite=suite_path, suite_sha256=M.sha(suite_path), deployment=deployment_path,
            deployment_sha256=M.sha(deployment_path), qualification_tool=qualifier,
            qualification_tool_sha256=M.sha(qualifier))
        if independent:
            semantic = qualifier.with_name('r1_independent_semantic.py')
            common.update(mode='independent', semantic_tool=semantic, semantic_tool_sha256=M.sha(semantic))
            run.update(independent_semantic=True, tool_sha256=M.sha(semantic))
            # Deliberately wrong answers cannot invoke the fixed-suite smoke
            # cutoff. Independent quality remains a separate paired score.
            for record in run['records']:
                record['text'] = 'FINAL: deliberately incorrect synthetic answer'
            common['original_run'].write_text(json.dumps(run))
            common['original_run_sha256'] = M.sha(common['original_run'])
        return common, raw_dirs, run

    @staticmethod
    def independent_suite():
        jobs = []
        for question in range(48):
            for repeat in (0, 1):
                ids = [0] + [2] * (16999 if question == 0 else 2)
                prompt = 'Synthetic full context ' + str(question)
                jobs.append({'id': f'independent-{question}-{repeat}', 'case_id': f'independent-q{question}',
                    'kind': 'generation', 'stage': 'short', 'question_index': question, 'repeat': repeat,
                    'dataset': 'synthetic', 'original_id': str(question), 'length_bin': 'test',
                    'prompt': prompt, 'prompt_sha256': FIXTURES.tool.value_sha(prompt),
                    'input_ids': ids, 'input_ids_sha256': FIXTURES.tool.value_sha(ids),
                    'max_new_tokens': 8192, 'ignore_eos': False, 'semantic': True, 'answers': ['expected']})
        semantic = ROOT / 'scripts/r1_reproduction/r1_independent_semantic.py'
        return {'schema': FIXTURES.semantic.SCHEMA, 'scope': 'full_model', 'jobs': jobs,
            'paired_rule': copy.deepcopy(FIXTURES.semantic.RULE),
            'source_files': {name: {'sha256': pin} for name, pin in FIXTURES.semantic.PINS.items()},
            'tool_sha256': M.sha(semantic), 'tokenizer_json_sha256': 'a' * 64,
            'world_size': 16, 'num_hidden_layers': 61, 'vocab_size': 129280,
            'full_context_preserved': True, 'prior_model_results_imported': False}

    def seal_fixture(self, root, common, raw_dirs, original):
        manifests = []
        for node in (0, 1):
            output = root / f'sealed-node{node}'
            M.seal(SimpleNamespace(**common, trace_directory=raw_dirs[node], node_rank=node,
                                  origin_hostname=f'actual-test-node{node}', output=output, execute=True))
            manifests.append(output / 'manifest.json')
        args = SimpleNamespace(**common, node_manifest=manifests,
            node_manifest_sha256=[M.sha(p) for p in manifests], output=root / 'derived', execute=True)
        return args, raw_dirs, original

    def sealed_nodes(self, root):
        return self.seal_fixture(root, *self.fixture(root))

    def test_union_preserves_original_HTTP_deployment_and_exact_time_window_but_cannot_invent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _raw, original = self.sealed_nodes(root)
            external = root / 'actual-launch-receipt.json'
            external.write_text('{"actual_argv":["original-owner-observation"]}\n')
            args.external_evidence = [external]
            args.external_evidence_sha256 = [M.sha(external)]
            frozen = args.original_run.read_bytes()
            result = M.finalize(args)
            derived = json.loads((args.output / 'run.json').read_text())
            self.assertEqual(args.original_run.read_bytes(), frozen)
            for key in original:
                if key != 'trace':
                    self.assertEqual(derived[key], original[key])
            events = json.loads((args.output / 'trace/events.json').read_text())
            self.assertEqual({e['rank'] for e in events}, set(range(16)))
            self.assertEqual(len(events), 48)
            self.assertFalse(any(e['observed_unix_ns'] == 250 for e in events))
            self.assertFalse(result['native_evidence_passed'])
            self.assertFalse(result['full_model_qualified'])
            self.assertTrue(result['missing_evidence'])
            self.assertEqual(result['actual_coverage_count'], 16)
            self.assertFalse(result['actual_launch_argv_verified_by_this_utility'])
            self.assertEqual((args.output / 'provenance/external-0-actual-launch-receipt.json').read_bytes(), external.read_bytes())

    def test_plan_creates_no_output_and_imports_no_accelerator(self):
        with tempfile.TemporaryDirectory() as directory:
            common, raw_dirs, _ = self.fixture(Path(directory))
            output = Path(directory) / 'never-created'
            with patch.dict('sys.modules', {'torch': None, 'triton': None, 'tokenizers': None}):
                result = M.seal(SimpleNamespace(**common, trace_directory=raw_dirs[0], node_rank=0,
                    origin_hostname='actual-test-node0', output=output, execute=False))
            self.assertEqual(result['status'], 'plan_only_NOT_executed')
            self.assertFalse(output.exists())

    def test_missing_node_wrong_partition_and_tampered_raw_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _raw, _ = self.sealed_nodes(Path(directory))
            bad = copy.copy(args)
            bad.node_manifest = args.node_manifest[:1]
            bad.node_manifest_sha256 = args.node_manifest_sha256[:1]
            with self.assertRaisesRegex(ValueError, 'Exactly two'):
                M.finalize(bad)
            bad.node_manifest = [args.node_manifest[0]] * 2
            bad.node_manifest_sha256 = [args.node_manifest_sha256[0]] * 2
            with self.assertRaisesRegex(ValueError, 'duplicate actual node'):
                M.finalize(bad)
            path = args.node_manifest[1].parent / 'raw/rank-15-pid-1015.jsonl'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Sealed raw file changed'):
                M.finalize(args)

    def test_partial_JSONL_and_worker_restart_cannot_be_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            common, raw_dirs, _ = self.fixture(Path(directory))
            args = SimpleNamespace(**common, trace_directory=raw_dirs[0], node_rank=0,
                origin_hostname='actual-test-node0', output=Path(directory) / 'sealed', execute=False)
            path = raw_dirs[0] / 'rank-0-pid-1000.jsonl'
            original = path.read_text()
            path.write_text(original.rstrip('\n'))
            with self.assertRaisesRegex(ValueError, 'unclosed partial'):
                M.seal(args)
            path.write_text(original)
            second = original.replace('"actual_pid":1000', '"actual_pid":2000')
            (raw_dirs[0] / 'rank-0-pid-2000.jsonl').write_text(second)
            with self.assertRaisesRegex(ValueError, 'Multiple worker PID'):
                M.seal(args)

    def test_original_clock_or_saved_event_cannot_be_rewritten_to_fill_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _raw, original = self.sealed_nodes(Path(directory))
            original['finished_unix_ns'] = 300
            args.original_run.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'Explicit artifact SHA mismatch'):
                M.finalize(args)
            args.original_run_sha256 = M.sha(args.original_run)
            with self.assertRaisesRegex(ValueError, 'original time interval differs'):
                M.finalize(args)

    def test_native_full_vocab_nonfinite_file_is_rejected_by_original_reader(self):
        from array import array
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _raw, _original = self.sealed_nodes(root)
            trace = args.node_manifest[0].parent / 'raw'
            values = array('f', [0.0] * 129280)
            values[129279] = float('nan')
            if sys.byteorder != 'little':
                values.byteswap()
            file = trace / 'native.f32'
            file.write_bytes(values.tobytes())
            artifact = {'file': file.name, 'sha256': M.sha(file), 'dtype': 'float32_le', 'elements': 129280, 'bytes': 129280 * 4}
            worker = trace / 'rank-0-pid-1000.jsonl'
            base = json.loads(worker.read_text().splitlines()[0])
            event = dict(base, event='logits', ordinal=5, observed_unix_ns=150,
                         request_id='tf-arithmetic-p0', full_logits=artifact)
            worker.write_text(worker.read_text() + M.canonical(event) + '\n')
            manifest = json.loads(args.node_manifest[0].read_text())
            manifest['files'][worker.name].update(sha256=M.sha(worker), bytes=worker.stat().st_size)
            manifest['files'][file.name] = {'sha256': M.sha(file), 'bytes': file.stat().st_size, 'kind': 'native_full_logits_f32'}
            args.node_manifest[0].write_text(json.dumps(manifest))
            args.node_manifest_sha256[0] = M.sha(args.node_manifest[0])
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                M.finalize(args)

    def test_independent_mode_requires_separate_source_and_its_actual_pinned_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common, raw_dirs, _ = self.fixture(root, independent=True)
            args = SimpleNamespace(**common, trace_directory=raw_dirs[0], node_rank=0,
                origin_hostname='actual-test-node0', output=root / 'plan-only', execute=False)
            with patch.dict('sys.modules', {'torch': None, 'triton': None, 'tokenizers': None}):
                result = M.seal(args)
            self.assertEqual(result['mode'], 'independent')
            self.assertEqual(result['suite_schema'], FIXTURES.semantic.SCHEMA)
            self.assertEqual(result['semantic_tool_sha256'], common['semantic_tool_sha256'])
            self.assertFalse(args.output.exists())
            bad = copy.copy(args)
            bad.semantic_tool_sha256 = None
            with self.assertRaisesRegex(ValueError, 'requires the original semantic tool'):
                M.seal(bad)
            bad.semantic_tool_sha256 = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Pinned original independent semantic tool differs'):
                M.seal(bad)
            copied = root / 'copied-source'
            copied.mkdir()
            shutil.copyfile(args.semantic_tool, copied / 'r1_independent_semantic.py')
            (copied / 'qualify_r1_full_model.py').write_text('raise AssertionError("must not import")\n')
            bad = copy.copy(args)
            bad.semantic_tool = copied / 'r1_independent_semantic.py'
            with self.assertRaisesRegex(ValueError, 'imports a different qualification source'):
                M.seal(bad)
            shutil.copyfile(args.qualification_tool, copied / 'qualify_r1_full_model.py')
            self.assertEqual(M.seal(bad)['mode'], 'independent')

    def test_independent_cannot_masquerade_as_fixed_or_rebind_collector_or_suite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common, raw_dirs, original = self.fixture(root, independent=True)
            args = SimpleNamespace(**common, trace_directory=raw_dirs[0], node_rank=0,
                origin_hostname='actual-test-node0', output=root / 'plan-only', execute=False)
            bad = copy.copy(args)
            bad.mode = 'fixed'
            with self.assertRaisesRegex(ValueError, 'require explicit independent mode'):
                M.seal(bad)
            bad.semantic_tool = bad.semantic_tool_sha256 = None
            with self.assertRaises(ValueError):
                M.seal(bad)  # The actual fixed validator must reject the 96 schema.
            original['tool_sha256'] = args.qualification_tool_sha256
            args.original_run.write_text(json.dumps(original))
            args.original_run_sha256 = M.sha(args.original_run)
            with self.assertRaisesRegex(ValueError, 'Original HTTP capture/source/deployment'):
                M.seal(args)
            original['tool_sha256'] = args.semantic_tool_sha256
            original['independent_semantic'] = False
            args.original_run.write_text(json.dumps(original))
            args.original_run_sha256 = M.sha(args.original_run)
            with self.assertRaisesRegex(ValueError, 'collector evidence mode differs'):
                M.seal(args)
            suite = json.loads(args.suite.read_text())
            suite['tool_sha256'] = args.qualification_tool_sha256
            args.suite.write_text(json.dumps(suite))
            args.suite_sha256 = M.sha(args.suite)
            with self.assertRaisesRegex(ValueError, 'suite semantic source identity differs'):
                M.seal(args)

    def test_independent_union_preserves_all_rank_prefix_logits_full_vectors_and_original_boundaries(self):
        from array import array
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common, raw_dirs, original = self.fixture(root, independent=True)
            suite = json.loads(common['suite'].read_text())
            added, vector_bytes = [], {}
            for rank in range(16):
                folder = raw_dirs[rank // 8]
                worker = folder / f'rank-{rank}-pid-{1000 + rank}.jsonl'
                envelope = json.loads(worker.read_text().splitlines()[0])
                job = suite['jobs'][rank]
                native_input = dict(envelope, event='input_tokens', ordinal=5, observed_unix_ns=100,
                    request_id=job['id'], input_count=len(job['input_ids']),
                    input_ids_sha256=job['input_ids_sha256'], input_ids_equal_fixture=True)
                logits = dict(envelope, event='logits', ordinal=6, observed_unix_ns=200,
                    request_id=job['id'], generated_before_step=0, phase='prefill',
                    input_ids_sha256=job['input_ids_sha256'], context_length=len(job['input_ids']),
                    vocab_size=129280, nonfinite_elements=0, greedy_token_id=42,
                    source_files_at_last_step={'full_model_trace': 'b' * 64})
                if rank in (0, 8):
                    values = array('f', [float(rank)] * 129280)
                    if sys.byteorder != 'little':
                        values.byteswap()
                    file = folder / f'rank-{rank}-native.f32'
                    file.write_bytes(values.tobytes())
                    vector_bytes[file.name] = file.read_bytes()
                    logits['full_logits'] = {'file': file.name, 'sha256': M.sha(file), 'dtype': 'float32_le',
                                            'elements': 129280, 'bytes': 129280 * 4}
                worker.write_text(worker.read_text() + M.canonical(native_input) + '\n' + M.canonical(logits) + '\n')
                added.extend((native_input, logits))
            args, _, _ = self.seal_fixture(root, common, raw_dirs, original)
            frozen = args.original_run.read_bytes()
            result = M.finalize(args)
            derived = json.loads((args.output / 'run.json').read_text())
            events = json.loads((args.output / 'trace/events.json').read_text())
            self.assertEqual(args.original_run.read_bytes(), frozen)
            for key in original:
                if key != 'trace':
                    self.assertEqual(derived[key], original[key])
            self.assertEqual(len(derived['records']), 96)
            self.assertTrue(derived['independent_semantic'])
            self.assertEqual(result['mode'], 'independent')
            self.assertEqual(result['suite_schema'], FIXTURES.semantic.SCHEMA)
            self.assertEqual(result['hard_failures'], [])
            self.assertTrue(result['missing_evidence'])
            self.assertFalse(result['native_evidence_passed'])
            self.assertFalse(result['formal_performance_permitted'])
            self.assertEqual(len(events), 80)
            for event in added:
                self.assertEqual(events.count(event), 1)
            self.assertEqual({e['rank'] for e in events if e['event'] == 'logits'}, set(range(16)))
            self.assertFalse(any(e['observed_unix_ns'] == 250 for e in events))
            for name, expected in vector_bytes.items():
                self.assertEqual((args.output / 'trace' / name).read_bytes(), expected)
            self.assertEqual((args.output / 'provenance/original-semantic-tool.py').read_bytes(), args.semantic_tool.read_bytes())

    def test_seals_bind_mode_schema_and_independent_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _ = self.seal_fixture(root, *self.fixture(root, independent=True))
            manifest_path = args.node_manifest[1]
            original = json.loads(manifest_path.read_text())
            for key, value in [('mode', 'fixed'), ('suite_schema', FIXTURES.tool.SCHEMA),
                               ('semantic_tool_sha256', args.qualification_tool_sha256)]:
                manifest = copy.deepcopy(original)
                manifest[key] = value
                manifest_path.write_text(json.dumps(manifest))
                args.node_manifest_sha256[1] = M.sha(manifest_path)
                with self.assertRaisesRegex(ValueError, 'source/deployment/original time interval differs'):
                    M.finalize(args)

    def test_actual_fixed84_v1_and_v2_validate_but_cannot_mix_node_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            prepared.mkdir()
            by_version = {}
            for version in ('v1', 'v2'):
                suite = VERSIONS.prepare_fixture(prepared, version)
                self.assertEqual(len(suite['jobs']), 84)
                version_root = root / version
                version_root.mkdir()
                args, _, original = self.seal_fixture(version_root, *self.fixture(version_root, suite=suite))
                by_version[version] = args
                result = M.finalize(args)
                derived = json.loads((args.output / 'run.json').read_text())
                self.assertEqual(result['mode'], 'fixed')
                self.assertEqual(result['suite_schema'], suite['schema'])
                self.assertEqual(derived['suite_schema'], suite['schema'])
                self.assertEqual(derived['records'], original['records'])
                self.assertFalse(result['native_evidence_passed'])
                self.assertFalse(result['formal_performance_permitted'])
            old, new = by_version['v1'], by_version['v2']
            mixed = copy.copy(new)
            mixed.execute = False
            mixed.node_manifest = [new.node_manifest[0], old.node_manifest[1]]
            mixed.node_manifest_sha256 = [new.node_manifest_sha256[0], old.node_manifest_sha256[1]]
            with self.assertRaisesRegex(ValueError, 'source/deployment/original time interval differs'):
                M.finalize(mixed)
            # Even if the claimed source/run/time identities are retained, an
            # explicit schema or suite-SHA substitution cannot pass the seal.
            manifest_path = new.node_manifest[1]
            frozen = json.loads(manifest_path.read_text())
            for key, value in [('suite_schema', FIXTURES.tool.SCHEMA), ('suite_sha256', old.suite_sha256)]:
                changed = dict(frozen, **{key: value})
                manifest_path.write_text(json.dumps(changed))
                mixed.node_manifest = new.node_manifest
                mixed.node_manifest_sha256 = [new.node_manifest_sha256[0], M.sha(manifest_path)]
                with self.assertRaisesRegex(ValueError, 'source/deployment/original time interval differs'):
                    M.finalize(mixed)


if __name__ == '__main__':
    unittest.main()
