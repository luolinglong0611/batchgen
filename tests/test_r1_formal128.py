"""CPU-only provenance, unchanged-prefix, tokenizer and closed-gate contracts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_formal128", ROOT / "scripts/r1_reproduction/prepare_r1_formal128.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def sample(ordinal, ids=None):
    ids = [0, 7, 8] if ids is None else ids
    request = {"custom_id": f"r1-{ordinal:06d}", "method": "POST", "url": "/v1/completions",
        "body": {"model": "deepseek-ai/DeepSeek-R1", "prompt": "original 完整正文 " + str(ordinal),
                 "max_tokens": 2, "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}}
    # Repeated source samples are intentional in the original with-replacement selection.
    provenance = {"custom_id": request["custom_id"], "dataset": "real-task", "original_id": "repeated-source-id",
        "source_row_sha256": "a" * 64, "input_tokens": len(ids), "input_token_ids": ids,
        "input_token_ids_sha256": tool.value_sha(ids), "max_output_tokens": 2, "context_modified": True}
    return request, provenance


class Tokenizer:
    def __init__(self, mapping, *, extra_bos=False, bad_batch=False):
        self.mapping, self.extra_bos, self.bad_batch = mapping, extra_bos, bad_batch

    def encode(self, text, add_special_tokens, truncation=False):
        assert add_special_tokens is True and truncation is False
        return ([0] if self.extra_bos else []) + list(self.mapping[text])

    def __call__(self, texts, **kwargs):
        assert kwargs == {"return_tensors": None, "padding": False, "truncation": False, "return_attention_mask": False}
        return {"input_ids": [self.mapping[text][:-1] if self.bad_batch else self.encode(text, True) for text in texts]}


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [sample(i) for i in range(3)]
        self.requests, self.provenance = self.root / "requests.jsonl", self.root / "provenance.jsonl"
        tool.write_rows(self.requests, [row[0] for row in self.rows])
        tool.write_rows(self.provenance, [row[1] for row in self.rows])
        self.artifact = {"input_tokens": 3, "max_output_tokens": 2, "requests": self.requests.name,
            "provenance": self.provenance.name, "requests_sha256": tool.sha(self.requests),
            "provenance_sha256": tool.sha(self.provenance)}

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return tool.load_prefix(self.root, self.artifact, source_count=3, count=2)

    def test_prefix_retains_original_text_order_and_duplicate_source_samples(self):
        rows = self.load()
        self.assertEqual(rows, self.rows[:2])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row[1]["original_id"] for row in rows}), 1)
        self.assertEqual(rows[0][0]["body"]["prompt"], self.rows[0][0]["body"]["prompt"])

    def test_hashes_cover_unselected_suffix_not_only_first128(self):
        self.requests.write_text(self.requests.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA/count/stat"):
            self.load()

    def test_out_of_order_or_duplicate_custom_ids_fail(self):
        changed = copy.deepcopy(self.rows)
        changed[1][0]["custom_id"] = changed[0][0]["custom_id"]
        tool.write_rows(self.root / "bad.jsonl", [row[0] for row in changed])
        self.artifact.update(requests="bad.jsonl", requests_sha256=tool.sha(self.root / "bad.jsonl"))
        with self.assertRaisesRegex(ValueError, "custom-ID"):
            self.load()

    def test_forged_original_native_ids_or_hash_fail(self):
        self.rows[0][1]["input_token_ids"] = [0, True, 8]
        bad = self.root / "bad.jsonl";tool.write_rows(bad, [row[1] for row in self.rows])
        self.artifact.update(provenance=bad.name, provenance_sha256=tool.sha(bad))
        with self.assertRaisesRegex(ValueError, "native token IDs"):
            self.load()

    def test_missing_row_or_hidden_stop_field_fails(self):
        self.rows[0][0]["body"]["stop"] = ["hidden stop"]
        bad = self.root / "bad.jsonl";tool.write_rows(bad, [row[0] for row in self.rows])
        self.artifact.update(requests=bad.name, requests_sha256=tool.sha(bad))
        with self.assertRaisesRegex(ValueError, "stop/transport"):
            self.load()

    def test_manifest_pins_both_artifact_hashes_and_source_count(self):
        source = {"count_per_profile": 3, "same_sample_order_across_profiles": True, "artifacts": [self.artifact]}
        path = self.root / "manifest.json";tool.write_json(path, source)
        profiles = {"input3-output2": (3, 2, self.artifact["requests_sha256"], self.artifact["provenance_sha256"], "unused")}
        actual, _ = tool.inspect_manifest(path, expected_sha=tool.sha(path), source_count=3, profiles=profiles)
        self.assertEqual(actual, source)
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            tool.inspect_manifest(path, expected_sha="0" * 64, source_count=3, profiles=profiles)
        profiles["input3-output2"] = (3, 2, "0" * 64, self.artifact["provenance_sha256"], "unused")
        with self.assertRaisesRegex(ValueError, "artifact hashes"):
            tool.inspect_manifest(path, expected_sha=tool.sha(path), source_count=3, profiles=profiles)

    def test_relative_path_escape_and_source_output_mutation_are_rejected(self):
        with self.assertRaises(ValueError): tool.safe_child(self.root, "../elsewhere")
        with self.assertRaises(ValueError): tool.guard_output(self.root / "new-output", [self.root])
        with self.assertRaises(FileExistsError): tool.guard_output(self.requests, [])


class TokenContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = [sample(i) for i in range(2)]
        self.mapping = {request["body"]["prompt"]: record["input_token_ids"] for request, record in self.rows}

    def test_native_ids_match_current_HF_BG_single_and_real_batch_call_shape(self):
        tokenizer = Tokenizer(self.mapping)
        results, failures = tool.validate_tokens(self.rows, tokenizer, tokenizer, 3)
        self.assertEqual(failures, [])
        self.assertTrue(all(row["original_native_ids_reused"] and row["HF_BG_worker_identical"] for row in results))

    def test_new_BOS_plus_one_is_a_failure_and_never_trimmed(self):
        before = copy.deepcopy(self.rows)
        results, failures = tool.validate_tokens(self.rows, Tokenizer(self.mapping, extra_bos=True), Tokenizer(self.mapping, extra_bos=True), 3)
        self.assertEqual(len(failures), 2)
        self.assertEqual(results[0]["HF_length"], 4)
        self.assertEqual(self.rows, before)

    def test_matching_single_encode_does_not_mask_worker_batch_mismatch(self):
        _, failures = tool.validate_tokens(self.rows, Tokenizer(self.mapping), Tokenizer(self.mapping, bad_batch=True), 3)
        self.assertTrue(all(not row["HF_BG_worker_identical"] for row in failures))

    def test_same_length_different_original_ids_is_explicit_failure(self):
        current = {key: [0, 8, 7] for key in self.mapping}
        _, failures = tool.validate_tokens(self.rows, Tokenizer(current), Tokenizer(current), 3)
        self.assertTrue(all(row["HF_BG_worker_identical"] and not row["original_native_ids_reused"] for row in failures))

    def test_generation_overrides_are_identical_without_prompt_or_source_mutation(self):
        before = copy.deepcopy(self.rows[0][0]);new = tool.policy_envelope(before, 8192)
        self.assertEqual(before, self.rows[0][0])
        self.assertEqual(new["body"]["prompt"], before["body"]["prompt"])
        for key, value in {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "ignore_eos": True, "max_tokens": 8192}.items():
            self.assertEqual(new["body"][key], value)


class PrepareFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
        source = self.root / "source";source.mkdir();self.manifest = source / "manifest.json";self.manifest.write_text("{}")
        self.rows = [sample(i) for i in range(2)]
        self.mapping = {r["body"]["prompt"]: p["input_token_ids"] for r, p in self.rows}
        self.bundle = ROOT / "batchgen/models/deepseek/deepseekv3"
        self.args = SimpleNamespace(source_manifest=self.manifest, repo=ROOT, tokenizer_dir=self.bundle,
            bg_tokenizer_source_sha256=tool.sha(self.bundle / "tokenizer.py"), output=self.root / "output",
            expected_hostname=socket.gethostname(), execute=True)
        self.metadata = {"count_per_profile": 6000, "sampling": "with-replacement", "seed": 20261023}
        self.artifact = {"input_tokens": 3, "max_output_tokens": 2}

    def tearDown(self): self.temp.cleanup()

    def run_flow(self, *, extra_bos=False):
        hf = Tokenizer(self.mapping, extra_bos=extra_bos)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: hf))
        key = "input3-output2"
        with patch.object(tool, "inspect_manifest", return_value=(self.metadata, {key: self.artifact})), \
             patch.object(tool, "load_prefix", return_value=self.rows), \
             patch.object(tool, "load_service_tokenizer", return_value=(hf, {
                 "batchgen/models/deepseek/deepseekv3/tokenizer.py": self.args.bg_tokenizer_source_sha256})), \
             patch.object(tool, "MANIFEST_SHA", tool.sha(self.manifest)), \
             patch.object(tool, "PROFILES", {key: (3, 2, "unused", "unused", tool.effective_sha(self.rows))}), \
             patch.object(tool.importlib.metadata, "version", return_value="CPU-test"), \
             patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), \
             patch.dict(os.environ, {"R1_FORMAL128_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, clear=True):
            return tool.prepare(self.args)

    def test_success_writes_byte_identical_engine_payloads_but_never_releases_formal(self):
        report = self.run_flow()
        self.assertEqual(report["status"], "prepared_unchanged_first128_candidate_NOT_formal_released")
        self.assertTrue(report["current_tokenizer_contract_verified"])
        self.assertFalse(report["formal_test_permitted"])
        bg, sg = self.args.output / "input3-output2.bg.requests.jsonl", self.args.output / "input3-output2.sg.requests.jsonl"
        self.assertEqual(bg.read_bytes(), sg.read_bytes())
        self.assertEqual(json.loads(bg.read_text().splitlines()[0])["body"]["prompt"], self.rows[0][0]["body"]["prompt"])

    def test_BOS_failure_outputs_diagnostics_and_options_without_request_payloads(self):
        report = self.run_flow(extra_bos=True)
        self.assertEqual(report["status"], "blocked_current_tokenizer_contract_NOT_formal_released")
        self.assertFalse(report["current_tokenizer_contract_verified"])
        self.assertIn("remediation_options_NOT_automatically_applied", report)
        self.assertEqual([p.name for p in self.args.output.iterdir()], ["manifest.json"])

    def test_default_plan_never_loads_tokenizers_or_writes_output(self):
        self.args.execute = False
        with patch.object(tool, "inspect_manifest", return_value=(self.metadata, {})), \
             patch.object(tool, "load_service_tokenizer", side_effect=AssertionError("No tokenizer imports in plan")):
            report = tool.prepare(self.args)
        self.assertEqual(report["status"], "plan_only_NOT_executed")
        self.assertEqual(set(report["profiles"]), set(tool.PROFILES))
        self.assertTrue(all(row["selected_count"] == 128 and row["current_tokenization"] == "NOT_executed"
                            for row in report["profiles"].values()))
        self.assertFalse(self.args.output.exists())

    def test_wrong_host_or_visible_GPU_environment_stops_before_import_or_output(self):
        self.args.expected_hostname = "wrong-host"
        with patch.object(tool, "inspect_manifest", return_value=(self.metadata, {})), \
             patch.dict(os.environ, {"R1_FORMAL128_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, clear=True), \
             self.assertRaisesRegex(ValueError, "CPU-only"):
            tool.prepare(self.args)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
