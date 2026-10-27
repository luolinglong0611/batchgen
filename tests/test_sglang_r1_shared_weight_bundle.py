"""CPU-only index/provenance contracts; no checkpoint tensor imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/prepare_sglang_r1_shared_weights.py'
SPEC = importlib.util.spec_from_file_location('shared_weight_bundle_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class OriginalSharedBundleContracts(unittest.TestCase):
    def fixtures(self, root):
        specs = M.probe_module().original_tensor_specs(3)
        (root / 'original.safetensors').write_bytes(b'not read in plan')
        (root / 'config.json').write_text(json.dumps({'hidden_size': 7168,
            'moe_intermediate_size': 2048, 'n_shared_experts': 1, 'first_k_dense_replace': 3,
            'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [128, 128]}}))
        index = {'weight_map': {name: 'original.safetensors' for name in specs}}
        (root / 'model.safetensors.index.json').write_text(json.dumps(index))
        return specs, index

    def test_plan_pins_original_config_index_and_reads_no_tensor_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            args = SimpleNamespace(checkpoint_dir=root, config_sha256=M.sha(root / 'config.json'),
                index_sha256=M.sha(root / 'model.safetensors.index.json'), layer=3,
                shard_sha_manifest=None, execute=False, output_dir=root / 'never-created')
            with patch.dict('sys.modules', {'torch': None, 'safetensors': None}):
                receipt = M.prepare(args)
            self.assertEqual(receipt['status'], 'plan_only_NOT_executed')
            self.assertFalse(receipt['gpu_operations'])
            self.assertFalse(args.output_dir.exists())
            args.config_sha256 = 'wrong'
            with self.assertRaisesRegex(ValueError, 'config/index SHA mismatch'):
                M.prepare(args)

    def test_index_missing_or_traversing_original_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs, index = self.fixtures(root)
            name = next(iter(specs))
            for path in ('../other.safetensors', '/tmp/other.safetensors', 'not-a-shard.bin', None):
                index['weight_map'][name] = path
                with self.subTest(path=path), self.assertRaises(ValueError):
                    M.selected_shards(root, index, specs)

    def test_FP8_scale_contract_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError, 'block128'):
            M.check_config({'hidden_size': 7168, 'moe_intermediate_size': 2048,
                'n_shared_experts': 1, 'first_k_dense_replace': 3,
                'quantization_config': {'quant_method': 'fp8', 'weight_block_size': [64, 128]}})


if __name__ == '__main__':
    unittest.main()
