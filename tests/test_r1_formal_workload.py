"""Explicit-N ordered prefixes, frozen helper identity, and closed release gates."""
import hashlib
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
SPEC = importlib.util.spec_from_file_location("r1_prefixN", ROOT / "scripts/r1_reproduction/prepare_r1_formal_workload.py")
tool = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(tool)


class FakeTokenizer:
    def __init__(self, mapping, bad_worker_prompt=None):
        self.mapping, self.bad_worker_prompt = mapping, bad_worker_prompt

    def encode(self, prompt, add_special_tokens, truncation=False):
        assert add_special_tokens is True and truncation is False
        return list(self.mapping[prompt])

    def __call__(self, prompts, **kwargs):
        assert kwargs == {"return_tensors": None, "padding": False, "truncation": False, "return_attention_mask": False}
        return {"input_ids": [self.encode(p, True)[:-1] if p == self.bad_worker_prompt else self.encode(p, True) for p in prompts]}


class FormalWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory();cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source";cls.source.mkdir()
        cls.real_base = tool.pinned_helper()
        cls.helper_sha_before = hashlib.sha256(tool.HELPER_PATH.read_bytes()).hexdigest()
        cls.profiles, cls.artifacts, cls.mapping, cls.rows = {}, {}, {}, {}
        for length, output in ((3, 2), (2, 3)):
            key = f"input{length}-output{output}";rows = []
            for i in range(6000):
                ids = [0] + [7] * (length - 1)
                prompt = f"original-{length}-正文-{i}"
                request = {"custom_id": f"r1-{i:06d}", "method": "POST", "url": "/v1/completions",
                    "body": {"model": "deepseek-ai/DeepSeek-R1", "prompt": prompt, "max_tokens": output,
                             "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}}
                record = {"custom_id": request["custom_id"], "dataset": "original-task", "original_id": "sample-" + str(i // 2),
                    "source_row_sha256": "a" * 64, "input_tokens": length, "input_token_ids": ids,
                    "input_token_ids_sha256": cls.real_base.value_sha(ids), "max_output_tokens": output}
                rows.append((request, record));cls.mapping[prompt] = ids
            cls.rows[key] = rows
            requests, provenance = cls.source / (key + ".requests.jsonl"), cls.source / (key + ".provenance.jsonl")
            cls.real_base.write_rows(requests, [row[0] for row in rows]);cls.real_base.write_rows(provenance, [row[1] for row in rows])
            artifact = {"input_tokens": length, "max_output_tokens": output, "requests": requests.name,
                "provenance": provenance.name, "requests_sha256": cls.real_base.sha(requests), "provenance_sha256": cls.real_base.sha(provenance)}
            cls.artifacts[key] = artifact
            cls.profiles[key] = (length, output, artifact["requests_sha256"], artifact["provenance_sha256"], cls.real_base.effective_sha(rows[:128]))
        cls.manifest = cls.source / "manifest.json"
        cls.real_base.write_json(cls.manifest, {"count_per_profile": 6000, "same_sample_order_across_profiles": True,
            "sampling": "with-replacement", "seed": 20261023, "artifacts": list(cls.artifacts.values())})
        cls.manifest_sha = cls.real_base.sha(cls.manifest)

    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256(tool.HELPER_PATH.read_bytes()).hexdigest() == cls.helper_sha_before
        cls.temp.cleanup()

    def args(self, count, name, execute=True):
        bundled = ROOT / "batchgen/models/deepseek/deepseekv3"
        return SimpleNamespace(count=count, repo=ROOT, source_manifest=self.manifest, tokenizer_dir=bundled,
            bg_tokenizer_source_sha256=self.real_base.sha(bundled / "tokenizer.py"),
            output=self.root / name, execute=execute, expected_hostname=socket.gethostname())

    def run_case(self, args, *, bad_worker_prompt=None, mutate_source=False):
        base = tool.pinned_helper()
        original_inspect = base.inspect_manifest
        base.MANIFEST_SHA, base.PROFILES = self.manifest_sha, self.profiles
        base.inspect_manifest = lambda path: original_inspect(path, expected_sha=self.manifest_sha, profiles=self.profiles)
        hf = FakeTokenizer(self.mapping);service = FakeTokenizer(self.mapping, bad_worker_prompt)
        base.load_service_tokenizer = lambda repo: (service, {"batchgen/models/deepseek/deepseekv3/tokenizer.py": args.bg_tokenizer_source_sha256})
        old_validate = base.validate_tokens
        changed_file = self.source / next(iter(self.artifacts.values()))["requests"]
        original_bytes = changed_file.read_bytes()
        def validate(*values):
            result = old_validate(*values)
            if mutate_source and changed_file.read_bytes() == original_bytes:
                changed_file.write_bytes(original_bytes + b"\n")
            return result
        base.validate_tokens = validate
        try:
            with patch.object(tool, "pinned_helper", return_value=base), \
                 patch.object(tool.importlib.metadata, "version", return_value="CPU-test"), \
                 patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
                    "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: hf))}), \
                 patch.dict(os.environ, {"R1_FORMAL_WORKLOAD_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, clear=True):
                return tool.prepare(args)
        finally:
            if changed_file.read_bytes() != original_bytes: changed_file.write_bytes(original_bytes)

    def test_count_must_be_explicit_integral_multiple16_not_more_than6000(self):
        for count in (16, 176, 2800, 5600, 6000): self.assertEqual(tool.checked_count(count), count)
        for count in (0, -16, 17, 6016, True, 128.0, None):
            with self.subTest(count=count), self.assertRaises(ValueError): tool.checked_count(count)

    def test_frozen128_helper_is_loaded_by_exact_bytes_and_not_modified(self):
        self.assertEqual(self.helper_sha_before, tool.HELPER_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "helper.py";wrong.write_text("raise AssertionError('must not execute')")
            with patch.object(tool, "HELPER_PATH", wrong), self.assertRaisesRegex(ValueError, "helper SHA256"):
                tool.pinned_helper()

    def test_N176_uses_all_original_rows_and_records_new_N_body_hashes(self):
        args = self.args(176, "success176");report = self.run_case(args)
        self.assertEqual(report["status"], "prepared_unchanged_ordered_prefixN_candidate_NOT_formal_released")
        self.assertEqual(report["count_per_profile"], 176)
        self.assertTrue(report["source_artifacts_rehashed_after_derivation"])
        self.assertFalse(report["formal_test_permitted"])
        for key, profile in report["profiles"].items():
            self.assertEqual(len(profile["tokenization_checks"]), 176)
            self.assertEqual(profile["selected_custom_ids"][-1], "r1-000175")
            self.assertEqual(profile["unique_source_samples"], 88)  # Repeats retained.
            self.assertEqual(profile["original_prefixN_effective_payload_sha256"], self.real_base.effective_sha(self.rows[key][:176]))
            self.assertNotEqual(profile["original_prefixN_effective_payload_sha256"], self.profiles[key][4])
            generated = [(self.real_base.policy_envelope(request, self.artifacts[key]["max_output_tokens"]), record)
                         for request, record in self.rows[key][:176]]
            self.assertEqual(profile["new_policy_prefixN_effective_payload_sha256"], self.real_base.effective_sha(generated))
            bg, sg = profile["prepared_artifacts"]["BG"], profile["prepared_artifacts"]["SG"]
            self.assertEqual((args.output / bg["name"]).read_bytes(), (args.output / sg["name"]).read_bytes())
            last = json.loads((args.output / bg["name"]).read_text().splitlines()[-1])
            self.assertEqual(last["body"]["prompt"], self.rows[key][175][0]["body"]["prompt"])

    def test_failure_after_first128_is_checked_and_never_hidden_by_prior128_pass(self):
        key = next(iter(self.rows));bad_prompt = self.rows[key][175][0]["body"]["prompt"]
        args = self.args(176, "blocked176");report = self.run_case(args, bad_worker_prompt=bad_prompt)
        self.assertEqual(report["status"], "blocked_current_allN_tokenizer_contract_NOT_formal_released")
        self.assertEqual(report["profiles"][key]["tokenization_mismatches"][0]["custom_id"], "r1-000175")
        self.assertEqual([path.name for path in args.output.iterdir()], ["manifest.json"])

    def test_source_change_during_tokenization_fails_before_payload_emission(self):
        args = self.args(32, "changed-source");report = self.run_case(args, mutate_source=True)
        self.assertEqual(report["status"], "failed_derivation_NOT_formal_released")
        self.assertFalse(report["prepared_payloads_usable"])
        self.assertIn("changed after complete source read", report["error"]["message"])
        self.assertEqual([path.name for path in args.output.iterdir()], ["manifest.json"])

    def test_plan_has_explicit_N_and_never_imports_tokenizers_or_writes(self):
        args = self.args(32, "plan-only", execute=False)
        base = tool.pinned_helper();old_inspect = base.inspect_manifest
        base.MANIFEST_SHA, base.PROFILES = self.manifest_sha, self.profiles
        base.inspect_manifest = lambda path: old_inspect(path, expected_sha=self.manifest_sha, profiles=self.profiles)
        with patch.object(tool, "pinned_helper", return_value=base), \
             patch.dict(sys.modules, {"torch": None, "transformers": None}):
            report = tool.prepare(args)
        self.assertEqual(report["status"], "plan_only_NOT_executed")
        self.assertEqual(report["count_per_profile"], 32)
        self.assertTrue(all(row["selected_count"] == 32 for row in report["profiles"].values()))
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
