import unittest, tempfile
import copy
import importlib.util
import json
from pathlib import Path
SPEC = importlib.util.spec_from_file_location('alignment_audit', Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/alignment_audit.py')
audit_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_tool)
AS_OF = '2026-10-26T04:00:00Z'
OBSERVED = '2026-10-26T03:59:00Z'
INPUT_HASH = 'a' * 64

def observations():
    result = {'schema': audit_tool.INPUT_SCHEMA, 'engines': {}, 'correctness': {}}
    for index, engine in enumerate(audit_tool.ENGINES):
        binding = {'source_sha256': str(index + 1) * 64, 'input_manifest_sha256': INPUT_HASH}
        parameters = {}
        for name in audit_tool.REQUIRED_PARAMETERS:
            value = 'fixed-' + name
            if name in audit_tool.HASH_PARAMETERS:
                value = INPUT_HASH
            elif name in audit_tool.POSITIVE_INT_PARAMETERS:
                value = 16
            elif name == 'graph_scope':
                value = {'framework': 'off', 'local': 'off'}
            elif name == 'prefill_budget':
                value = {'per_rank_tokens': 2048, 'chunking': True}
            elif name == 'gpu_uuids':
                value = [f'GPU-{number:02}' for number in range(16)]
            elif name == 'gpu_memory_budget':
                value = {'fraction': 0.85, 'cap_bytes_per_gpu': 240 * 1024**3}
            elif name == 'generation_policy':
                value = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'seed': 20261026}
            elif name == 'stop_policy':
                value = {'ignore_eos': True, 'output_length': {'short': 2048, 'long': 8192}}
            elif name in audit_tool.BOOLEAN_PARAMETERS:
                value = False
            parameters[name] = {'value': value, 'evidence_level': 'runtime', 'evidence': [f'/evidence/{engine}/{name}.json']}
        result['engines'][engine] = {'observed_at_utc': OBSERVED, 'binding': binding, 'parameters': parameters}
        result['correctness'][engine] = {name: {'status': 'passed', 'scope': 'full_model', 'evidence_level': 'runtime', 'observed_at_utc': OBSERVED, 'binding': copy.deepcopy(binding), 'evidence': [f'/evidence/{engine}/{name}.json']} for name in audit_tool.QUALIFICATIONS}
    return result

def run(data):
    return audit_tool.audit(data, as_of_utc=AS_OF)

def parameter(result, name):
    return next((row for row in result['parameters'] if row['parameter'] == name))

class AlignmentAuditTests(unittest.TestCase):

    def test_every_mandatory_parameter_missing_or_different_blocks(self):
        for name in audit_tool.REQUIRED_PARAMETERS:
            for mode in ('missing', 'different'):
                with self.subTest(parameter=name, mode=mode):
                    data = observations()
                    if mode == 'missing':
                        del data['engines']['SG']['parameters'][name]
                    else:
                        current = data['engines']['SG']['parameters'][name]['value']
                        if type(current) is bool:
                            replacement = not current
                        elif type(current) is int:
                            replacement = current + 1
                        elif type(current) is str:
                            replacement = ('b' * 64) if name in audit_tool.HASH_PARAMETERS else current + '-other'
                        elif type(current) is list:
                            replacement = list(current)
                            replacement[0] = 'GPU-other'
                        else:
                            replacement = copy.deepcopy(current)
                            replacement['different_policy'] = True
                        data['engines']['SG']['parameters'][name]['value'] = replacement
                    assert not run(data)['gate']['formal_test_permitted']

    def test_gpu_uuids_must_be_nonempty_unique_strings_and_match_count(self):
        for value in ([], ['GPU-one', 'GPU-one'], ['GPU-one', 2], [''], ['GPU-one']):
            with self.subTest(value=value):
                data = observations()
                for engine in audit_tool.ENGINES:
                    data['engines'][engine]['parameters']['gpu_uuids']['value'] = value
                assert not run(data)['gate']['formal_test_permitted']

    def test_boolean_policies_do_not_accept_numeric_lookalikes(self):
        for name in audit_tool.BOOLEAN_PARAMETERS:
            for value in (0, 1, 'false', None):
                with self.subTest(name=name, value=value):
                    data = observations()
                    for engine in audit_tool.ENGINES:
                        data['engines'][engine]['parameters'][name]['value'] = value
                    assert not run(data)['gate']['formal_test_permitted']

    def test_generation_and_stop_policies_require_semantic_keys_and_valid_types(self):
        invalid = [
            ('generation_policy', {}),
            ('generation_policy', {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1}),
            ('generation_policy', {'temperature': False, 'top_p': 1.0, 'top_k': 1, 'seed': 0}),
            ('generation_policy', {'temperature': 0.0, 'top_p': 1.1, 'top_k': 1, 'seed': 0}),
            ('generation_policy', {'temperature': 0.0, 'top_p': 1.0, 'top_k': True, 'seed': 0}),
            ('stop_policy', {}),
            ('stop_policy', {'ignore_eos': 1, 'output_length': 2048}),
            ('stop_policy', {'ignore_eos': True}),
            ('stop_policy', {'ignore_eos': True, 'output_length': {}}),
            ('stop_policy', {'ignore_eos': True, 'output_length': {'short': True}}),
            ('gpu_memory_budget', {}),
            ('gpu_memory_budget', 0.85),
            ('quantization_policy', {}),
            ('quantization_policy', ''),
        ]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                data = observations()
                for engine in audit_tool.ENGINES:
                    data['engines'][engine]['parameters'][name]['value'] = value
                assert not run(data)['gate']['formal_test_permitted']

    def test_passed_operator_fixture_or_missing_scope_cannot_qualify_full_model(self):
        for qualification in audit_tool.QUALIFICATIONS:
            for scope in ('operator', 'layer', None):
                with self.subTest(qualification=qualification, scope=scope):
                    data = observations()
                    data['correctness']['BG'][qualification]['scope'] = scope
                    result = run(data)
                    row = next(item for item in result['correctness'] if item['engine'] == 'BG' and item['qualification'] == qualification)
                    assert row['reported_status'] == 'passed'
                    assert not row['qualified']
                    assert 'qualification_scope_is_not_full_model' in row['reasons']
                    assert not result['gate']['formal_test_permitted']

    def test_additional_runtime_fields_support_false_zero_and_structured_values(self):
        data = observations()
        extra = {'prefix_reuse': False, 'retry_count': 0, 'GPU_UUIDs': ['GPU-one', 'GPU-two']}
        data['additional_required_parameters'] = list(extra)
        for engine in audit_tool.ENGINES:
            for name, value in extra.items():
                data['engines'][engine]['parameters'][name] = {
                    'value': value, 'evidence_level': 'runtime', 'evidence': ['/runtime.json']}
        assert run(data)['gate']['formal_test_permitted']

    def test_malformed_evidence_cannot_pass_or_break_report_rendering(self):
        data = observations()
        record = data['engines']['BG']['parameters']['active_cap']
        record['evidence_level'] = {'pretend': 'runtime'}
        record['evidence'] = 12
        data['correctness']['SG']['nan_tail']['evidence'] = [123]
        result = run(data)
        assert not result['gate']['formal_test_permitted']
        assert 'Formal measurement permitted: false' in audit_tool.markdown(result)

    def test_bound_current_equal_observations_and_all_qualifications_are_required(self):
        result = run(observations())
        assert result['gate']['formal_test_permitted']
        assert all((row['status'] == 'aligned' for row in result['parameters']))
        data = observations()
        del data['correctness']['SG']['semantic']
        assert not run(data)['gate']['formal_test_permitted']

    def test_nonruntime_equal_values_never_establish_alignment(self):
        for level in ['historical_runtime', 'source', 'planned', 'missing']:
            with self.subTest(level=level):
                data = observations()
                data['engines']['SG']['parameters']['kv_dtype']['evidence_level'] = level
                result = run(data)
                assert parameter(result, 'kv_dtype')['status'] == 'unverified'
                assert not result['gate']['formal_test_permitted']

    def test_stale_future_naive_and_missing_runtime_times_fail_closed(self):
        for time in ['2026-10-25T04:00:00Z', '2026-10-26T04:01:00Z', '2026-10-26T03:59:00', None]:
            with self.subTest(time=time):
                data = observations()
                data['engines']['BG']['observed_at_utc'] = time
                result = run(data)
                assert not result['gate']['formal_test_permitted']
                assert not any((row['status'] == 'aligned' for row in result['parameters']))

    def test_integer_configuration_does_not_accept_bool_float_or_invalid_domain(self):
        for value in [True, 16.0, False, 0, -1]:
            with self.subTest(value=value):
                data = observations()
                data['engines']['SG']['parameters']['attention_dp']['value'] = value
                result = run(data)
                assert parameter(result, 'attention_dp')['status'] != 'aligned'
                assert not result['gate']['formal_test_permitted']

    def test_nested_bool_and_numeric_policy_values_are_not_equal(self):
        data = observations()
        data['engines']['SG']['parameters']['prefill_budget']['value']['chunking'] = 1
        assert parameter(run(data), 'prefill_budget')['status'] == 'different'

    def test_shared_expert_or_graph_differences_are_visible_and_block_measurement(self):
        data = observations()
        data['engines']['SG']['parameters']['shared_expert_tp']['value'] = 1
        data['engines']['SG']['parameters']['graph_scope']['value'] = {'framework': 'on', 'local': 'off'}
        result = run(data)
        assert parameter(result, 'shared_expert_tp')['status'] == 'different'
        assert parameter(result, 'graph_scope')['status'] == 'different'
        assert not result['gate']['formal_test_permitted']

    def test_historical_differences_are_preserved_without_becoming_fresh(self):
        data = observations()
        record = data['engines']['SG']['parameters']['moe_kernel']
        record.update(value='other-kernel', evidence_level='historical_runtime')
        row = parameter(run(data), 'moe_kernel')
        assert row['status'] == 'different'
        assert not row['SG']['current_runtime_qualified']
        assert 'not_current_runtime_evidence' in row['SG']['reasons']

    def test_each_correctness_failure_blocks_even_when_all_parameters_match(self):
        for qualification in audit_tool.QUALIFICATIONS:
            with self.subTest(qualification=qualification):
                data = observations()
                data['correctness']['BG'][qualification]['status'] = 'failed'
                result = run(data)
                assert all((row['status'] == 'aligned' for row in result['parameters']))
                assert not result['gate']['formal_test_permitted']

    def test_returned_field_consistency_is_not_an_accepted_qualification(self):
        data = observations()
        data['correctness']['BG'] = {'returned_fields': {'status': 'passed', 'evidence_level': 'runtime'}}
        result = run(data)
        assert all((not row['qualified'] for row in result['correctness'] if row['engine'] == 'BG'))
        assert not result['gate']['formal_test_permitted']

    def test_source_or_input_binding_cannot_be_missing(self):
        for location in ['engine', 'qualification']:
            with self.subTest(location=location):
                data = observations()
                record = data['engines']['BG'] if location == 'engine' else data['correctness']['BG']['numerical']
                del record['binding']
                assert not run(data)['gate']['formal_test_permitted']

    def test_input_manifest_observation_must_match_engine_binding(self):
        data = observations()
        for engine in audit_tool.ENGINES:
            data['engines'][engine]['parameters']['input_manifest_hash']['value'] = 'b' * 64
        assert not run(data)['gate']['formal_test_permitted']

    def test_qualification_cannot_reuse_a_pass_from_a_different_source_or_input(self):
        for key in ('source_sha256', 'input_manifest_sha256'):
            data = observations()
            data['correctness']['SG']['numerical']['binding'][key] = 'f' * 64
            assert not run(data)['gate']['formal_test_permitted']

    def test_separately_frozen_semantic_fixture_requires_exact_binding_and_current_source(self):
        data = observations()
        expected = copy.deepcopy(data['engines']['BG']['binding'])
        expected['input_manifest_sha256'] = 'c' * 64
        data['qualification_expected_bindings'] = {'BG': {'semantic': expected}}
        data['correctness']['BG']['semantic']['binding'] = copy.deepcopy(expected)
        assert run(data)['gate']['formal_test_permitted']
        data['qualification_expected_bindings']['BG']['semantic']['source_sha256'] = 'd' * 64
        data['correctness']['BG']['semantic']['binding']['source_sha256'] = 'd' * 64
        assert not run(data)['gate']['formal_test_permitted']

    def test_required_fields_cannot_be_removed_by_a_caller_supplied_subset(self):
        data = observations()
        data['required_parameters'] = ['kv_dtype']
        del data['engines']['SG']['parameters']['attention_kernel']
        result = run(data)
        assert parameter(result, 'attention_kernel')['status'] == 'unverified'
        assert not result['gate']['formal_test_permitted']

    def test_current_claim_without_evidence_references_is_unverified(self):
        data = observations()
        data['engines']['BG']['parameters']['active_cap']['evidence'] = []
        data['correctness']['SG']['semantic']['evidence'] = []
        assert not run(data)['gate']['formal_test_permitted']

    def test_nonfinite_and_duplicate_json_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = observations()
            data['engines']['BG']['parameters']['prefill_budget']['value'] = {'x': float('nan')}
            with self.assertRaisesRegex(ValueError, 'finite JSON'):
                run(data)
            path = tmp_path / 'bad.json'
            for text in ('{"schema":"a","schema":"b"}', '{"x":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    audit_tool.load_observations(path)

    def test_report_pair_is_exclusive_and_contains_evidence_and_blocked_status(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = observations()
            data['engines']['SG']['parameters']['kv_dtype']['evidence_level'] = 'planned'
            result = run(data)
            json_path, md_path = (tmp_path / 'audit.json', tmp_path / 'audit.md')
            audit_tool.write_reports(result, json_path, md_path)
            assert json.loads(json_path.read_text())['gate']['status'] == 'blocked'
            assert 'planned' in md_path.read_text()
            assert '/evidence/SG/kv_dtype.json' in md_path.read_text()
            before = json_path.read_bytes()
            with self.assertRaises(FileExistsError):
                audit_tool.write_reports(run(observations()), json_path, md_path)
            assert json_path.read_bytes() == before

    def test_existing_markdown_does_not_create_or_overwrite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            md_path = tmp_path / 'audit.md'
            md_path.write_text('preserve')
            json_path = tmp_path / 'audit.json'
            with self.assertRaises(FileExistsError):
                audit_tool.write_reports(run(observations()), json_path, md_path)
            assert md_path.read_text() == 'preserve'
            assert not json_path.exists()

    def test_cli_writes_blocked_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source, output, markdown = (tmp_path / name for name in ('source.json', 'report.json', 'report.md'))
            source.write_text(json.dumps({'schema': audit_tool.INPUT_SCHEMA}))
            code = audit_tool.main(['--observations', str(source), '--json-output', str(output), '--markdown-output', str(markdown), '--as-of-utc', AS_OF])
            assert code == 2
            assert not json.loads(output.read_text())['gate']['formal_test_permitted']
if __name__ == '__main__':
    unittest.main()
