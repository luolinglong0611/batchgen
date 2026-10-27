import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("checkpoint_verifier", Path(__file__).parents[1] / "scripts/r1_reproduction/prepare_r1_checkpoint.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class CheckpointVerification(unittest.TestCase):
    def test_exact_bytes_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            header = {"weight": {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]},
                      "scale": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
            raw = json.dumps(header).encode()
            source.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x01\x02\x03\x04\xaa\xbb\xcc\xdd")
            # Converter order may differ; compare tensors by their independent ranges.
            (root / "model.bin").write_bytes(b"\xaa\xbb\xcc\xdd\x01\x02\x03\x04")
            metadata = {"total_byte_size": 8, "state_dict": {
                "scale": {"dtype": "float32", "shape": [1], "offset": 0, "byte_size": 4},
                "weight": {"dtype": "float8_e4m3fn", "shape": [2, 2], "offset": 4, "byte_size": 4}}}
            (root / "model.json").write_text(json.dumps(metadata))
            result = verifier.verify_shard(source, root, ["scale", "weight"])
            self.assertEqual(result["bytes"], 8)
            self.assertEqual(result["tensor_count"], 2)
            (root / "model.bin").write_bytes(b"\xaa\xbb\xcc\xdd\x01\x02\x03\x00")
            with self.assertRaisesRegex(ValueError, "Tensor bytes differ"):
                verifier.verify_shard(source, root, ["scale", "weight"])

    def test_reject_missing_and_overlapping_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            header = {"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                      "b": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}}
            raw = json.dumps(header).encode()
            source.write_bytes(struct.pack("<Q", len(raw)) + raw + b"1234")
            metadata = {"total_byte_size": 4, "state_dict": {
                name: {"dtype": "bfloat16", "shape": [1], "offset": 0, "byte_size": 2} for name in header}}
            (root / "model.json").write_text(json.dumps(metadata))
            (root / "model.bin").write_bytes(b"1234")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                verifier.verify_shard(source, root, ["a"])
            with self.assertRaisesRegex(ValueError, "gap or overlap"):
                verifier.verify_shard(source, root, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
