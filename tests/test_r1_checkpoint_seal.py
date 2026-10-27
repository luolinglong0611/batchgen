"""Tiny, independent source/metadata/record fixtures; production stays strict."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_checkpoint_seal_tests", ROOT / "scripts/r1_reproduction/seal_r1_checkpoint.py")
seal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seal)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root, index_total_size=8):
    source, converted, job = (root / name for name in ("original", "converted", "job"))
    for path in (source, converted, job): path.mkdir()
    index = {"metadata": {"total_size": index_total_size}, "weight_map": {"z.weight": "s0.safetensors", "a.scale": "s1.safetensors"}}
    save(source / "model.safetensors.index.json", index)
    for name, tensor, dtype, shape, raw, original_dtype in (
        ("s0", "z.weight", "float8_e4m3fn", [2, 2], b"1234", "F8_E4M3"),
        ("s1", "a.scale", "float32", [1], b"abcd", "F32")):
        header = json.dumps({tensor: {"dtype": original_dtype, "shape": shape, "data_offsets": [0, 4]}}).encode()
        (source / (name + ".safetensors")).write_bytes(struct.pack("<Q", len(header)) + header + raw)
        (converted / (name + ".bin")).write_bytes(raw)
        metadata = {"file_name": name + ".bin", "state_dict": {tensor: {
            "dtype": dtype, "shape": shape, "offset": 0, "byte_size": 4}}, "total_byte_size": 4}
        save(converted / (name + ".json"), metadata)
        verified = {"shard": name + ".safetensors", "tensor_count": 1, "bytes": 4,
            "metadata_sha256": seal.small_sha(converted / (name + ".json")), "tensors": [{
                "name": tensor, "bytes": 4, "original_and_converted_sha256": hashlib.sha256(raw).hexdigest()}]}
        save(job / (name + ".verified.json"), verified)
    manifest = root / "manifest.json"
    entries = [{"Name": path.name, "Path": path.name, "Size": path.stat().st_size, "Sha256": seal.small_sha(path)}
               for path in sorted(source.iterdir())]
    save(manifest, entries)
    total = sum(item["Size"] for item in entries if item["Path"].endswith(".safetensors"))
    inventory = seal.inventory(source, manifest, manifest_sha=seal.small_sha(manifest),
        index_sha=seal.small_sha(source / "model.safetensors.index.json"), expected_shards=2,
        expected_tensors=2, expected_file_bytes=total)
    status = {"schema": "r1-main-checkpoint-byte-verification-v1", "status": seal.SUCCESS,
        "index_sha256": inventory["index_sha256"], "source_bytes": total, "shards": 2,
        "tensor_count": 2, "marlin_repack": False, "converter_sha256": "d" * 64,
        "completed_shards": 2, "verified_bytes": 8, "cuda_initialized": False}
    save(job / "status.json", status)
    args = SimpleNamespace(input=source, converted=converted, conversion_report_dir=job, manifest=manifest,
        report_dir=root / "seal", expected_hostname=socket.gethostname(), workers=4, timeout_seconds=5,
        poll_seconds=0.001)
    return args, inventory, status


class CheckpointSealContracts(unittest.TestCase):
    def test_stale_index_bf16_size_does_not_replace_independent_original_fp8_header_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, status = fixture(Path(folder), index_total_size=16)
            self.assertEqual(inventory["index_declared_total_size_NOT_authoritative"], 16)
            self.assertEqual(inventory["tensor_bytes"], 8)
            self.assertEqual(sum(header["tensor_bytes"] for header in inventory["source_header_shards"].values()), 8)
            self.assertEqual(seal.conversion_status(args.conversion_report_dir, inventory, require_complete=True), status)
            with patch.object(seal, "inventory", return_value=inventory), \
                 patch.dict(os.environ, {"R1_CHECKPOINT_SEAL_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, clear=True):
                self.assertEqual(seal.execute(args), 0)
            report = seal.load_json(args.report_dir / "status.json")
            self.assertEqual(report["identity"]["tensor_bytes"], 8)
            self.assertEqual(report["index_declared_total_size_NOT_authoritative"], 16)
            self.assertEqual(report["status"], seal.SEALED)

    def test_original_headers_independently_reject_dtype_shape_offsets_inventory_and_payload_disagreement(self):
        cases = [
            {"a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 4]}},
            {"a": {"dtype": "F8_E4M3", "shape": [True, 4], "data_offsets": [0, 4]}},
            {"a": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [1, 5]}},
            {"a": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}},
            {"other": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]}},
            {"a": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]},
             "b": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [3, 4]}},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.safetensors"
            for index, header in enumerate(cases):
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"1234")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    seal.source_header(path, ["a", "b"] if index == 5 else ["a"])

    def test_hash_bound_converted_metadata_must_also_match_independent_original_header(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            meta_path, record_path = args.converted / "s0.json", args.conversion_report_dir / "s0.verified.json"
            meta = seal.load_json(meta_path)
            meta["state_dict"]["z.weight"].update(dtype="float32", shape=[1])
            save(meta_path, meta)
            record = seal.load_json(record_path)
            record["metadata_sha256"] = seal.small_sha(meta_path)
            save(record_path, record)
            with self.assertRaisesRegex(ValueError, "independent original safetensors header"):
                seal.verified_tensor_records(inventory, args.converted, args.conversion_report_dir)

    def test_independent_header_is_bound_to_the_same_stream_as_full_frozen_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            name = "s0.safetensors"
            header = inventory["source_header_shards"][name]
            expected = dict(inventory["files"][name], header_bytes=header["header_bytes"], header_sha256=header["header_sha256"])
            result = seal.original_file_sha(args.input / name, expected, time.monotonic() + 1, threading.Event())
            self.assertEqual(result["source_header_sha256"], header["header_sha256"])
            expected["header_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "parsed header differs"):
                seal.original_file_sha(args.input / name, expected, time.monotonic() + 1, threading.Event())

    def test_production_manifest_index_counts_and_byte_totals_are_pinned(self):
        self.assertEqual((seal.EXPECTED_SHARDS, seal.EXPECTED_TENSORS, seal.EXPECTED_FILE_BYTES), (163, 91991, 688586727753))
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                seal.inventory(args.input, args.manifest)
            with patch.object(seal, "inventory", return_value=inventory), \
                 patch.object(seal, "execute", side_effect=AssertionError("Default must not execute")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    seal.main(["--input", str(args.input), "--converted", str(args.converted),
                        "--conversion-report-dir", str(args.conversion_report_dir), "--manifest", str(args.manifest),
                        "--report-dir", str(args.report_dir), "--expected-hostname", args.expected_hostname])
                self.assertEqual(json.loads(output.getvalue())["status"], "plan_only_NOT_hashed_or_sealed")
                self.assertFalse(args.report_dir.exists())

    def test_metadata_filename_is_not_covered_by_prior_verify_but_seal_rejects_it(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            meta = args.converted / "s0.json"
            value = seal.load_json(meta)
            value["file_name"] = "../other.bin"
            save(meta, value)
            record = args.conversion_report_dir / "s0.verified.json"
            value = seal.load_json(record)
            value["metadata_sha256"] = seal.small_sha(meta)
            save(record, value)
            with self.assertRaisesRegex(ValueError, "file_name"):
                seal.verified_tensor_records(inventory, args.converted, args.conversion_report_dir)

    def test_metadata_hash_must_still_match_the_verified_record(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            path = args.converted / "s0.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "metadata SHA256 changed"):
                seal.verified_tensor_records(inventory, args.converted, args.conversion_report_dir)

    def test_tensor_inventory_duplicates_wrong_shards_bytes_shapes_and_extra_receipts_fail(self):
        for bad in ("duplicate", "wrong_name", "bytes", "dtype", "shape", "extra", "bin_size"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                args, inventory, _ = fixture(Path(folder))
                path = args.conversion_report_dir / "s0.verified.json"
                record = seal.load_json(path)
                if bad == "duplicate": record["tensors"].append(copy.deepcopy(record["tensors"][0]))
                elif bad == "wrong_name": record["tensors"][0]["name"] = "a.scale"
                elif bad == "bytes": record["bytes"] = 3
                elif bad == "dtype": record["tensors"][0]["dtype"] = "bfloat16"
                elif bad == "shape": record["tensors"][0]["shape"] = [4]
                elif bad == "extra": save(args.conversion_report_dir / "extra.verified.json", record)
                else: (args.converted / "s0.bin").write_bytes(b"12345")
                save(path, record)
                with self.assertRaises(ValueError):
                    seal.verified_tensor_records(inventory, args.converted, args.conversion_report_dir)

    def test_legacy_records_obtain_dtype_shape_from_bound_metadata_and_hash_is_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            result = seal.verified_tensor_records(inventory, args.converted, args.conversion_report_dir)
            self.assertEqual([item["name"] for item in result["records"]], ["a.scale", "z.weight"])
            self.assertEqual(result["records"][0]["dtype"], "float32")
            self.assertEqual(result["records"][1]["shape"], [2, 2])
            self.assertEqual(result["weight_tensor_hash"], hashlib.sha256(seal.canonical(result["records"])).hexdigest())
            self.assertEqual((result["tensor_count"], result["tensor_bytes"]), (2, 8))

    def test_original_full_file_hash_rejects_same_size_modification_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            path = args.input / "s0.safetensors"
            expected = inventory["files"][path.name]
            result = seal.original_file_sha(path, expected, time.monotonic() + 1, threading.Event())
            self.assertEqual(result["sha256"], expected["Sha256"])
            with self.assertRaises(TimeoutError):
                seal.original_file_sha(path, expected, time.monotonic() - 1, threading.Event())
            original = path.read_bytes()
            path.write_bytes(original[:-1] + b"X")
            with self.assertRaisesRegex(ValueError, "full SHA256"):
                seal.original_file_sha(path, expected, time.monotonic() + 1, threading.Event())

    def test_conversion_success_must_have_complete_counts_bytes_and_no_cuda(self):
        for bad in ("status", "count", "bytes", "cuda", "repack", "index"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                args, inventory, status = fixture(Path(folder))
                if bad == "status": status["status"] = "running"
                elif bad == "count": status["completed_shards"] = 1
                elif bad == "bytes": status["verified_bytes"] = 7
                elif bad == "cuda": status["cuda_initialized"] = True
                elif bad == "repack": status["marlin_repack"] = True
                else: status["index_sha256"] = "0" * 64
                save(args.conversion_report_dir / "status.json", status)
                with self.assertRaises(ValueError):
                    seal.conversion_status(args.conversion_report_dir, inventory, require_complete=True)

    def test_full_tiny_seal_uses_existing_evidence_preserves_data_and_never_qualifies_model(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, _ = fixture(Path(folder))
            before = {str(path): seal.small_sha(path) for directory in (args.input, args.converted, args.conversion_report_dir)
                      for path in directory.iterdir()}
            env = {"R1_CHECKPOINT_SEAL_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}
            with patch.object(seal, "inventory", return_value=inventory), patch.dict(os.environ, env, clear=True):
                self.assertEqual(seal.execute(args), 0)
                with self.assertRaises(FileExistsError): seal.execute(args)
            report = seal.load_json(args.report_dir / "status.json")
            self.assertEqual(report["status"], seal.SEALED)
            self.assertEqual(report["completed_original_files"], 2)
            self.assertTrue(report["original_file_identity_verified"])
            self.assertEqual(report["weights_tensor_hash"], report["weight_tensor_hash"])
            self.assertFalse(report["full_model_qualified"])
            self.assertFalse(report["formal_test_permitted"])
            self.assertEqual(before, {str(path): seal.small_sha(path) for directory in (args.input, args.converted, args.conversion_report_dir)
                          for path in directory.iterdir()})

    def test_wait_for_existing_job_times_out_without_rerunning_or_changing_it(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, status = fixture(Path(folder))
            status["status"] = "running"
            save(args.conversion_report_dir / "status.json", status)
            before = seal.small_sha(args.conversion_report_dir / "status.json")
            args.timeout_seconds = 0.03
            with patch.object(seal, "inventory", return_value=inventory), \
                 patch.dict(os.environ, {"R1_CHECKPOINT_SEAL_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, clear=True):
                self.assertEqual(seal.execute(args), 1)
            result = seal.load_json(args.report_dir / "status.json")
            self.assertEqual(result["error_type"], "TimeoutError")
            self.assertEqual(seal.small_sha(args.conversion_report_dir / "status.json"), before)

    def test_duplicate_json_keys_and_conversion_failure_do_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            args, inventory, status = fixture(Path(folder))
            bad = Path(folder) / "bad.json"
            bad.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                seal.load_json(bad)
            status["status"] = "failed"
            save(args.conversion_report_dir / "status.json", status)
            with self.assertRaisesRegex(ValueError, "will not be rerun"):
                seal.conversion_status(args.conversion_report_dir, inventory)


if __name__ == "__main__":
    unittest.main()
