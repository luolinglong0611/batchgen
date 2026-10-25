import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('phase_weights', Path(__file__).resolve().parents[1] / 'batchgen/models/deepseek/deepseekv3/phase_weights.py')
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)
HAS_TORCH = importlib.util.find_spec('torch') is not None
FLAG = 'BATCHGEN_R1_PHASE_PREPARE'


class PhaseWeightsTest(unittest.TestCase):
    def test_disabled_returns_original_payload(self):
        with patch.dict(os.environ, {FLAG: '0'}):
            source = {'untouched': object()}
            self.assertIs(phase.prepare_worker_skeleton(source, 'other'), source)
            self.assertIsNone(phase.pack_phase_scales(source, 'other', -1, 0, 'cpu'))

    def test_bad_flag_and_model_rejected(self):
        with patch.dict(os.environ, {FLAG: 'yes'}):
            with self.assertRaises(ValueError):
                phase.enabled()
        with patch.dict(os.environ, {FLAG: '1'}):
            with self.assertRaises(ValueError):
                phase.prepare_worker_skeleton({}, 'other')

    @unittest.skipUnless(HAS_TORCH, 'Torch required')
    def test_phase_rank_selection_values_and_storage_lifetime(self):
        import torch
        source = {f'model.layers.3.mlp.experts.{i}.gate_proj.weight_scale_inv': torch.arange(8, dtype=torch.float32).view(2, 4) + i for i in (0, 15, 16, 255)}
        shared = 'model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv'
        attention = 'model.layers.0.self_attn.q_a_proj.weight_scale_inv'
        mtp = 'model.layers.61.mlp.experts.0.gate_proj.weight_scale_inv'
        source.update({shared: torch.ones(2, 4), attention: torch.ones(2, 2), mtp: torch.ones(2, 4)})
        with patch.dict(os.environ, {FLAG: '1'}):
            prefill = phase.pack_phase_scales(source, 'prefill', 0, 16, 'cpu')
            self.assertEqual(set(prefill), set(source) - {mtp})
            for rank, experts in ((0, (0, 15)), (15, (255,))):
                decode = phase.pack_phase_scales(source, 'decode', rank, 16, 'cpu')
                expected = {shared, attention} | {f'model.layers.3.mlp.experts.{i}.gate_proj.weight_scale_inv' for i in experts}
                self.assertEqual(set(decode), expected)
                self.assertTrue(all(torch.equal(v, source[k]) for k, v in decode.items()))
                self.assertEqual(len({v.untyped_storage().data_ptr() for v in decode.values()}), 1)
            retained = prefill[shared].clone()
            source[shared].add_(10)
            self.assertTrue(torch.equal(prefill[shared], retained))
            rebuilt = phase.pack_phase_scales(source, 'prefill', 0, 16, 'cpu')
            self.assertTrue(torch.equal(rebuilt[shared], source[shared]))
            self.assertFalse(torch.equal(rebuilt[shared], prefill[shared]))

    @unittest.skipUnless(HAS_TORCH, 'Torch required')
    def test_bad_layout_dtype_and_missing_dense_scale_rejected(self):
        import torch
        with patch.dict(os.environ, {FLAG: '1'}):
            with self.assertRaises(ValueError):
                phase.pack_phase_scales({}, 'decode', 0, 8, 'cpu')
            with self.assertRaises(ValueError):
                phase.pack_phase_scales({'model.layers.0.self_attn.q_a_proj.weight_scale_inv': torch.ones(2, 2, dtype=torch.bfloat16)}, 'prefill', 0, 16, 'cpu')
            with self.assertRaises(KeyError):
                phase.prepare_worker_skeleton({}, 'deepseek-ai/DeepSeek-R1')


if __name__ == '__main__':
    unittest.main()
