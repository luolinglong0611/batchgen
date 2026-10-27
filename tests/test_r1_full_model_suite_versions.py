"""Versioned semantic budgets: old failures stay failures; gates stay fixed."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r1_reproduction/qualify_r1_full_model.py"
SPEC = importlib.util.spec_from_file_location("r1_suite_versions", SCRIPT)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class TinyTokenizer:
    def encode(self, text, add_special_tokens):
        if not add_special_tokens:
            return SimpleNamespace(ids=[100, 101, 102, 103])
        index = next(i for i, row in enumerate(tool.QA) if row[1] in text)
        return SimpleNamespace(ids=[0, 10 + index])


def prepare_fixture(root, version):
    tokenizer_file = root / "tokenizer.json"
    if not tokenizer_file.exists(): tokenizer_file.write_text("{}")
    output = root / (version + ".json")
    args = SimpleNamespace(suite_version=version, tokenizer_json=tokenizer_file,
                           tokenizer_sha256=tool.sha(tokenizer_file), execute=True, output=output)
    def boundary(tokenizer, length):
        return "original fixed boundary", [0] + [2] * (length - 1)
    with patch.object(tool, "reference_tokenizer", return_value=TinyTokenizer()), \
         patch.object(tool, "exact_length_prompt", side_effect=boundary), \
         patch.object(tool, "exact_text", side_effect=lambda tokenizer, ids: "native-prefix:" + tool.value_sha(ids)), \
         contextlib.redirect_stdout(io.StringIO()):
        if tool.prepare(args) != 0: raise AssertionError("Actual prepare failed")
    return json.loads(output.read_text())


def diagnostic_run(root, suite, *, include_schema=True):
    (root / "trace").mkdir()
    tool.write_new(root / "trace/events.json", [])
    trace = ROOT / "scripts/r1_reproduction/r1_full_model_trace.py"
    deployment = {"schema": "r1-full-model-deployment-v1", "backend": "sglang", "world_size": 16,
        "num_hidden_layers": 61, "vocab_size": tool.VOCAB, "run_id": "diagnostic-run", "artifact_sha256": "b" * 64,
        "tokenizer_json_sha256": suite["tokenizer_json_sha256"], "graphs": False, "prefix_cache": False,
        "speculative_decoding": False, "checkpoint_identity": "original-r1", "model_id": "deepseek-ai/DeepSeek-R1",
        "endpoint": "http://already-running.invalid:30000",
        "source_files": {"full_model_trace": {"path": str(trace), "sha256": tool.sha(trace)}}}
    records = []
    for job in suite["jobs"]:
        if job["stage"] == "long": continue
        truncated = job["id"] == "gen-arithmetic-r0"
        count = 384 if truncated else job["max_new_tokens"] if job["ignore_eos"] else 1
        records.append({"id": job["id"], "native_output_ids": [42] * count,
            "prompt_tokens": len(job["input_ids"]), "completion_tokens": count,
            "text": "<think>17*19=323, still checking" if truncated else "FINAL: " + job["answers"][0],
            "finish_reason": "length" if truncated or job["ignore_eos"] else "stop"})
    result = {"schema": tool.RUN_SCHEMA, "scope": "full_model", "status": "evidence_collected_NOT_pair_qualified",
        "suite_sha256": "c" * 64, "deployment": deployment, "backend": "sglang", "long_correctness": False,
        "records": records, "trace": {"events_file": "trace/events.json", "events_sha256": tool.sha(root / "trace/events.json")}}
    if include_schema: result["suite_schema"] = suite["schema"]
    tool.write_new(root / "run.json", result)
    return root / "run.json", result


class SuiteVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
        self.v1 = prepare_fixture(self.root, "v1")
        self.v2 = prepare_fixture(self.root, "v2")

    def tearDown(self): self.temp.cleanup()

    def test_actual_prepare_changes_exactly12_budgets_and_no_other_job_field(self):
        self.assertEqual(self.v1["schema"], tool.SCHEMA)
        self.assertEqual(self.v2["schema"], tool.SCHEMA_V2)
        self.assertEqual(len(self.v1["jobs"]), 84)
        self.assertEqual(len(self.v2["jobs"]), 84)
        self.assertNotIn("budget_protocol", self.v1)
        self.assertEqual(self.v2["budget_protocol"], tool.V2_BUDGET_PROTOCOL)
        changed, unchanged = 0, 0
        for previous, current in zip(self.v1["jobs"], self.v2["jobs"]):
            if current["kind"] == "generation" and current["stage"] == "short" and current["semantic"]:
                self.assertEqual(previous["max_new_tokens"], 384)
                self.assertEqual(current["max_new_tokens"], 8192)
                self.assertFalse(current["ignore_eos"])
                restored = dict(current, max_new_tokens=384)
                self.assertEqual(previous, restored)
                changed += 1
            else:
                self.assertEqual(previous, current)
                unchanged += 1
        self.assertEqual((changed, unchanged), (12, 72))
        self.assertEqual(self.v1["numeric_contract"], self.v2["numeric_contract"])
        self.assertEqual(self.v1["sampling"], self.v2["sampling"])
        self.assertEqual(self.v1["limitations"], self.v2["limitations"])

    def test_numeric_thresholds_and_original_answers_are_not_relaxed(self):
        expected = {"logprob_sup_abs_max": .5, "logprob_rms_max": .1, "Jensen_Shannon_max_nats": .001,
            "teacher_logprob_abs_max": .1, "teacher_logprob_mean_abs_max": .03,
            "top1_disagreement_near_tie_margin_max": .2, "near_tie_disagreement_fraction_max": .05,
            "nonfinite_allowed": 0, "shape_or_token_bounds_errors_allowed": 0,
            "semantic_answers_required_fraction": 1.0}
        for key, value in expected.items(): self.assertEqual(tool.NUMERIC[key], value)
        aliases = {"arithmetic": ["323"], "algebra": ["5", "x=5"], "percent": ["60%", "0.6"],
                   "recall": ["silver maple 47"], "code": ["14"], "logic": ["yes"]}
        for job in self.v2["jobs"]:
            if job["semantic"]: self.assertEqual(job["answers"], aliases[job["case_id"]])

    def test_each_version_locks_its_own_budget_and_requires_explicit_metadata(self):
        self.assertIs(tool.validate_suite(self.v1), self.v1)
        self.assertIs(tool.validate_suite(self.v2), self.v2)
        bad_cases = []
        v = copy.deepcopy(self.v1);v["jobs"][0]["max_new_tokens"] = 8192;bad_cases.append(v)
        for changes in ({"max_new_tokens": 384}, {"max_new_tokens": 1024}, {"max_new_tokens": 8193}, {"ignore_eos": True}, {"answers": ["anything"]}):
            v = copy.deepcopy(self.v2);v["jobs"][0].update(changes);bad_cases.append(v)
        v = copy.deepcopy(self.v2);v.pop("budget_protocol");bad_cases.append(v)
        v = copy.deepcopy(self.v2);v["budget_protocol"]["max_new_tokens"] = 16384;bad_cases.append(v)
        v = copy.deepcopy(self.v2);v["numeric_contract"]["teacher_logprob_abs_max"] = 1.;bad_cases.append(v)
        v = copy.deepcopy(self.v2);v["jobs"] = v["jobs"][:-1];bad_cases.append(v)
        v = copy.deepcopy(self.v1);v["budget_protocol"] = dict(tool.V2_BUDGET_PROTOCOL);bad_cases.append(v)
        for bad in bad_cases:
            with self.subTest(bad_schema=bad["schema"]), self.assertRaises(ValueError): tool.validate_suite(bad)

    def test_v2_cannot_replace_reviewed_teacher_with_unknown_case_to_keep84(self):
        bad = copy.deepcopy(self.v2)
        teacher = next(job for job in bad["jobs"] if job["id"] == "tf-arithmetic-p3")
        teacher.update(id="tf-unreviewed-p0", case_id="unreviewed", semantic=False,
                       answers=["silver maple 47"], teacher_position=0)
        self.assertEqual(len(bad["jobs"]), 84)
        with self.assertRaisesRegex(ValueError, "job IDs/teacher cases/roles"):
            tool.validate_suite(bad)
        for target in ("coverage-0", "long-length8192"):
            bad = copy.deepcopy(self.v2)
            next(job for job in bad["jobs"] if job["id"] == target)["case_id"] = "length2048"
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "job IDs/teacher cases/roles"):
                tool.validate_suite(bad)

    def test_old384_truncation_stays_hard_failure_with_or_without_legacy_schema_field(self):
        for include in (False, True):
            directory = self.root / ("old-explicit" if include else "old-legacy");directory.mkdir()
            path, _ = diagnostic_run(directory, self.v1, include_schema=include)
            result = tool.validate_evidence(path, self.v1, "c" * 64, False)
            self.assertIn("gen-arithmetic-r0: semantic generation did not finish naturally", result["hard_failures"])
            self.assertEqual(tool.normalized_answer(result["records"]["gen-arithmetic-r0"]["text"]), "")
            self.assertTrue(result["missing_evidence"])  # Never a model qualification.

    def test_relabeling384_output_as_v2_never_turns_length_into_natural_EOS(self):
        directory = self.root / "new-truncated";directory.mkdir()
        path, _ = diagnostic_run(directory, self.v2)
        result = tool.validate_evidence(path, self.v2, "c" * 64, False)
        self.assertIn("gen-arithmetic-r0: semantic generation did not finish naturally", result["hard_failures"])

    def test_v1_v2_captures_cannot_be_mixed_by_schema_or_suite_SHA(self):
        directory = self.root / "mixed";directory.mkdir()
        path, data = diagnostic_run(directory, self.v1)
        with self.assertRaisesRegex(ValueError, "schema"):
            tool.validate_evidence(path, self.v2, "c" * 64, False)
        data["suite_schema"] = tool.SCHEMA_V2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "capture"):
            tool.validate_evidence(path, self.v2, "d" * 64, False)
        data.pop("suite_schema");path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "binding is missing"):
            tool.validate_evidence(path, self.v2, "c" * 64, False)


if __name__ == "__main__":
    unittest.main()
