"""CPU contracts for real shared/DP qualification; no distributed execution.

Two source-snapshot checks use optional external fixtures. Set
R1_SGLANG_FROZEN_SOURCES to a captured alignment directory to require them;
missing explicitly requested fixtures and changed hashes always fail.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/qualify_sglang_r1_local_shared.py'
SPEC = importlib.util.spec_from_file_location('sg_distributed_shared_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


CONTRACT = {'reference': 'original_fp8_weights_dequantized_fp64_swiglu',
            'atol': 1e-5, 'rtol': 0.016, 'max_failure_fraction': 0.0001}


def result_fixture(world):
    cases = [{'case': name, 'padding': mode, 'routed_SUM_calls': 1, 'local_shared_calls': 1,
              'shared_added_once_bitwise': True, 'separate_shared_oracle_bitwise': True,
              'native_gather_matches_independent_rows': True, 'reference_failures': 0,
              'checked_FP64_reference_elements': 100}
             for name, _ in M.case_counts(world) for mode in ('SUM_LEN', 'MAX_LEN')]
    return [{'rank': rank, 'world_size': world, 'status': 'passed_real_shared_branch_NOT_full_model_qualified',
             'source_sha256': M.sha(PATH), 'tp_members': list(range(world)),
             'attention_tp_size': 1, 'attention_dp_size': world, 'expert_parallel_size': world,
             'actual_shared_tp_size': 1,
             'cases': copy.deepcopy(cases), 'numerical_contract': dict(CONTRACT),
             'numerical_contract_sha256': 'pinned-numeric',
             'sources': {'native': {'sha256': 'pinned-source'}}, 'weights': {'original': 'pinned-tensors'}}
            for rank in range(world)]


class DistributedSharedContracts(unittest.TestCase):
    def _frozen_sources(self, relative):
        configured = os.environ.get('R1_SGLANG_FROZEN_SOURCES')
        base = (Path(configured) if configured is not None else
                PATH.parents[2] / '.repro-private/r1-alignment-20261026')
        path = base / relative
        if not path.exists():
            if configured is not None:
                self.fail('Explicit frozen SGLang source fixture is missing: ' + str(path))
            self.skipTest('Optional frozen SGLang sources unavailable; set R1_SGLANG_FROZEN_SOURCES')
        return path

    def test_default_plan_imports_no_gpu_library_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            with patch.object(M.importlib, 'import_module') as imported, contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(M.main(['--output-dir', str(output)]), 0)
            imported.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(stream.getvalue())['full_model_qualified'])

    def test_exact_six_original_weight_scale_specs(self):
        values = M.original_tensor_specs(3)
        self.assertEqual(len(values), 6)
        self.assertEqual(values['model.layers.3.mlp.shared_experts.gate_proj.weight']['shape'], [2048, 7168])
        self.assertEqual(values['model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv']['shape'], [56, 16])
        self.assertEqual({v['dtype'] for v in values.values()}, {'torch.float8_e4m3fn', 'torch.float32'})

    def test_padding_cases_include_empty_and_last_owner(self):
        for world in (2, 4, 16):
            counts = dict(M.case_counts(world))['last_owner_only']
            self.assertEqual(M.effective_counts(counts, 'SUM_LEN'), [0] * (world - 1) + [3])
            self.assertEqual(M.effective_counts(counts, 'MAX_LEN'), [3] * world)

    def test_unique_row_signs_preserve_exact_BF16_SUM_for_every_partial_sum(self):
        # CPU simulation of round-to-nearest-even BF16, without importing Torch.
        def bf16(value):
            bits = struct.unpack('>I', struct.pack('>f', value))[0]
            bits = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16
            return struct.unpack('>f', struct.pack('>I', bits))[0]
        codes = {M.fixture_sign_code(owner, token) for owner in range(16) for token in range(4)}
        self.assertEqual(len(codes), 64)
        for world in (2, 4, 16):
            for sign in (-1, 1):
                contribution = sign * 0.0625 / (8 * world)
                value = 0.0
                for k in range(1, world + 1):
                    value = bf16(value + contribution)
                    self.assertEqual(value, contribution * k)
                self.assertEqual(value, sign * 0.0625 / 8)

    def test_current_installed_additional_linear_and_config_sources_match_pins(self):
        root = self._frozen_sources('shared-static-review-actual-sg')
        for name, expected in M.EXTRA_SOURCE_SHA.items():
            path = root / (name.removeprefix('sglang.srt.').replace('.', '/') + '.py')
            self.assertEqual(M.sha(path), expected)

    def test_partial_world_pass_cannot_become_final16(self):
        for world in (2, 4):
            result = M.aggregate_rank_records(result_fixture(world), world)
            self.assertTrue(result['partial_world_precheck_passed'])
            self.assertFalse(result['final16_layout_qualified'])
        result = M.aggregate_rank_records(result_fixture(16), 16)
        self.assertTrue(result['final16_layout_qualified'])
        self.assertFalse(result['full_model_qualified'])
        self.assertFalse(result['formal_test_permitted'])

    def test_composition_only_pass_does_not_claim_independent_MLP_math(self):
        records = result_fixture(16)
        for record in records:
            record['numerical_contract'] = record['numerical_contract_sha256'] = None
            for case in record['cases']:
                case['checked_FP64_reference_elements'] = 0
        result = M.aggregate_rank_records(records, 16)
        self.assertTrue(result['final16_layout_qualified'])
        self.assertFalse(result['independent_shared_MLP_numerical_qualified'])
        records[3]['cases'][2]['separate_shared_oracle_bitwise'] = False
        with self.assertRaisesRegex(ValueError, 'composition/numerical gate'):
            M.aggregate_rank_records(records, 16)

    def test_unknown_generated_source_cannot_be_admitted_by_an_arbitrary_pin(self):
        data = b'class DeepseekV2DecoderLayer: pass\n'
        with self.assertRaisesRegex(ValueError, 'explicitly admitted'):
            M.generated_source_contract(data, M.hashlib.sha256(data).hexdigest())

    def test_actual_frozen_stock_generator_and_exact_combined_bootstrap(self):
        generator = PATH.with_name('prepare_sglang_r1_local_shared.py')
        spec = importlib.util.spec_from_file_location('shared_probe_generator_contract', generator)
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        stock = self._frozen_sources('installed-sg-node1/sglang/srt/models/deepseek_v2.py')
        generated = prepare.patched_source(stock.read_bytes())
        M.generated_source_contract(generated, M.SHARED_MODEL_SHA)
        M.generated_source_contract(generated + M.TRACE_BOOTSTRAP.encode(), M.TRACED_MODEL_SHA)

    def test_missing_rank_case_or_different_weights_fail_closed(self):
        records = result_fixture(16)
        with self.assertRaisesRegex(ValueError, 'Every real rank'):
            M.aggregate_rank_records(records[:-1], 16)
        records[2]['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'complete SUM_LEN'):
            M.aggregate_rank_records(records, 16)
        records = result_fixture(16)
        records[15]['weights'] = {'different': 'values'}
        with self.assertRaisesRegex(ValueError, 'different sources'):
            M.aggregate_rank_records(records, 16)

    def test_numeric_gate_must_be_predeclared_and_cannot_pass_all_failures(self):
        self.assertEqual(M.validate_numeric_contract(CONTRACT), CONTRACT)
        for changed in ({'max_failure_fraction': 1}, {'rtol': float('inf')}, {'atol': -1}, {'reference': 'self-comparison'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                M.validate_numeric_contract(dict(CONTRACT, **changed))
        records = result_fixture(2)
        records[0]['cases'][0]['reference_failures'] = 100
        with self.assertRaisesRegex(ValueError, 'numerical gate'):
            M.aggregate_rank_records(records, 2)

    def test_source_contract_requires_shared_finish_after_scatter(self):
        good = '''class DeepseekV2DecoderLayer:
 def forward(self):
  _r1_capture_local_shared_input()
  self.layer_communicator.postprocess_layer()
  _r1_finish_local_shared()
'''
        M.adapter_source_contract(good)
        wrong = good.replace('  self.layer_communicator.postprocess_layer()\n', '') + '\n'
        with self.assertRaisesRegex(ValueError, 'ordering contract'):
            M.adapter_source_contract(wrong)

    def test_missing_external_guard_fails_before_gpu_import(self):
        with patch.dict(M.os.environ, {}, clear=True), patch.object(M.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(RuntimeError, 'parent-owned distributed guard'):
                M.execute(SimpleNamespace(allowed_hostname=[]))
            imported.assert_not_called()


if __name__ == '__main__':
    unittest.main()
