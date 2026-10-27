"""CPU-only full-model evidence contracts; no service or GPU is contacted."""
import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r1_reproduction/qualify_r1_full_model.py"
TRACE = ROOT / "scripts/r1_reproduction/r1_full_model_trace.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


tool = load("r1_full_model_contract_tool", SCRIPT)
trace = load("r1_full_model_contract_trace", TRACE)
semantic = load("r1_independent_semantic_contract", SCRIPT.with_name("r1_independent_semantic.py"))


def fixed_suite():
    jobs = []
    bases = {}
    for number, (name, question, answer, aliases) in enumerate(tool.QA):
        bases[name] = ([0, 10 + number], True, list(aliases))
    for length in (63, 64, 65, 129, 2048, 8192):
        bases["length" + str(length)] = ([0] + [2] * (length - 1), False, ["silver maple 47"])

    def add(name, job_id, kind, stage, ids, maximum, ignore, **extra):
        prompt = "synthetic reversible prompt: " + job_id
        jobs.append({"id": job_id, "case_id": name, "kind": kind, "stage": stage,
                     "input_ids": list(ids), "input_ids_sha256": tool.value_sha(ids),
                     "prompt": prompt, "prompt_sha256": tool.value_sha(prompt),
                     "max_new_tokens": maximum, "ignore_eos": ignore,
                     "semantic": bases[name][1], "answers": bases[name][2], **extra})
    for name, (ids, semantic, _) in bases.items():
        for repeat in range(2 if semantic else 1):
            add(name, "gen-" + name + "-r" + str(repeat), "generation", "short", ids, 384 if semantic else 129, not semantic)
        for position in range(2):
            add(name, "tf-" + name + "-p" + str(position), "teacher", "short", ids + [100] * position,
                1, True, teacher_token_id=100, teacher_position=position,
                teacher_method="independent_prefix_prefill_NOT_incremental_decode")
        if name in ("length2048", "length8192"):
            add(name, "long-" + name, "generation", "long", ids, 2048 if name == "length8192" else 8192, True)
    for i in range(16):
        add("length63", "coverage-" + str(i), "coverage", "short", bases["length63"][0], 3, True)
    return {"schema": tool.SCHEMA, "scope": "full_model", "numeric_contract": copy.deepcopy(tool.NUMERIC),
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0},
            "world_size": 16, "num_hidden_layers": 61, "vocab_size": tool.VOCAB,
            "tokenizer_json_sha256": "a" * 64, "jobs": jobs}


class FullModelContract(unittest.TestCase):
    def test_imports_and_prepare_plan_never_import_accelerators_or_tokenizers(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "sitecustomize.py").write_text(
                "import sys\nclass Reject:\n"
                " def find_spec(self,fullname,path=None,target=None):\n"
                "  if fullname.split('.')[0] in ('torch','triton','tokenizers','transformers','batchgen'):\n"
                "   raise RuntimeError('unexpected accelerator/tokenizer import')\n"
                "sys.meta_path.insert(0,Reject())\n")
            tokenizer = directory / "tokenizer.json"; tokenizer.write_text("{}")
            out = directory / "not-created"
            result = subprocess.run([sys.executable, str(SCRIPT), "prepare", "--tokenizer-json", str(tokenizer),
                                     "--tokenizer-sha256", tool.sha(tokenizer), "--output", str(out)],
                                    env={**os.environ, "PYTHONPATH": str(directory)}, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "plan_only_NOT_executed")
            self.assertEqual(report["scope"], "full_model")
            self.assertFalse(out.exists())

    def test_fixed_suite_requires_boundaries_both_long_profiles_semantics_and_teacher_positions(self):
        good = fixed_suite()
        self.assertIs(tool.validate_suite(good), good)
        mutations = []
        v = copy.deepcopy(good); v["jobs"] = [j for j in v["jobs"] if j["id"] != "long-length2048"]; mutations.append(v)
        v = copy.deepcopy(good); v["jobs"] = [j for j in v["jobs"] if j["id"] != "coverage-15"]; mutations.append(v)
        v = copy.deepcopy(good); v["jobs"] = [j for j in v["jobs"] if j["id"] != "gen-arithmetic-r1"]; mutations.append(v)
        v = copy.deepcopy(good); v["jobs"] = [j for j in v["jobs"] if j["id"] != "tf-algebra-p0"]; mutations.append(v)
        v = copy.deepcopy(good); v["numeric_contract"]["teacher_logprob_abs_max"] = 100; mutations.append(v)
        v = copy.deepcopy(good); next(j for j in v["jobs"] if j["stage"] == "long")["ignore_eos"] = False; mutations.append(v)
        v = copy.deepcopy(good); v["jobs"][0]["answers"] = ["anything"]; mutations.append(v)
        v = copy.deepcopy(good); v["jobs"][0]["input_ids"][0] = tool.VOCAB; mutations.append(v)
        for bad in mutations:
            with self.assertRaises(ValueError):
                tool.validate_suite(bad)

    def test_trace_off_is_noop_without_inspecting_model_or_worker(self):
        class Reject:
            def __getattribute__(self, name):
                raise AssertionError("disabled trace touched actual model/worker")
        with patch.dict(os.environ, {"R1_FULL_MODEL_TRACE": "0"}):
            self.assertEqual(trace.register_model_hooks(Reject(), 0, 16, "prefill"), [])
            trace.trace_bg_logits(Reject(), Reject(), Reject())
            trace.trace_bg_completion(Reject(), Reject(), Reject())

    def test_native_ids_are_not_inferred_from_text_and_reject_out_of_bounds(self):
        self.assertEqual(trace.ids_list([0, 1, 129279]), [0, 1, 129279])
        for ids in ([129280], [-1], [1.0], [True], [[1]]):
            with self.assertRaises(ValueError):
                trace.ids_list(ids)

    def test_logprob_metrics_accept_additive_logit_shift_and_reject_real_change(self):
        a = [1.0, 0.5, -2.0, -10.0]
        same = tool.logits_metrics(a, [x + 32 for x in a], 2)
        self.assertTrue(same["passed"])
        self.assertLess(same["Jensen_Shannon_nats"], 1e-12)
        changed = tool.logits_metrics([4, 0, -2], [0, 4, -2], 0)
        self.assertFalse(changed["passed"])
        self.assertTrue(changed["top1_disagreement"])
        self.assertFalse(changed["ambiguous_disagreement"])

    def test_only_near_tie_top1_mismatch_is_allowed_and_nonfinite_always_rejected(self):
        near = tool.logits_metrics([0.001, 0, -4], [0, 0.001, -4], 2)
        self.assertTrue(near["passed"])
        self.assertTrue(near["ambiguous_disagreement"])
        # Test stable JS calculation with underflowing tail probabilities.
        extreme = tool.logits_metrics([0, -744, -1000], [0, -745, -1001], 0)
        self.assertTrue(math.isfinite(extreme["Jensen_Shannon_nats"]))
        for bad in ([0, float("nan")], [0, float("inf")], [0, -float("inf")]):
            with self.assertRaises(ValueError):
                tool.logits_metrics([0, 1], bad, 0)

    def test_semantic_scoring_uses_final_suffix_and_rejects_unclosed_thinking(self):
        self.assertEqual(tool.normalized_answer("<think>guess 99</think>\nFINAL: 323."), "323")
        self.assertEqual(tool.normalized_answer("<think>FINAL: 323"), "")
        self.assertEqual(tool.normalized_answer("323"), "")
        self.assertEqual(tool.normalized_answer("</think>FINAL: \\boxed{323}"), "323")
        self.assertEqual(tool.normalized_answer("</think><think>FINAL:323"), "")

    def test_sg_teacher_request_uses_native_prefix_and_requested_teacher_probability(self):
        job = next(j for j in fixed_suite()["jobs"] if j["kind"] == "teacher")
        request = tool.sg_request(job)
        self.assertIs(request["input_ids"], job["input_ids"])
        self.assertEqual(request["token_ids_logprob"], [job["teacher_token_id"]])
        self.assertEqual(request["sampling_params"]["max_new_tokens"], 1)
        self.assertTrue(request["sampling_params"]["ignore_eos"])
        self.assertFalse(request["stream"])

    def test_declared_61_layers_and_16_ranks_without_actual_events_cannot_pass(self):
        suite = fixed_suite()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "trace").mkdir()
            tool.write_new(root / "trace/events.json", [])
            deployment = {"schema": "r1-full-model-deployment-v1", "backend": "sglang", "world_size": 16,
                          "num_hidden_layers": 61, "vocab_size": tool.VOCAB, "run_id": "test-run", "artifact_sha256": "b" * 64,
                          "tokenizer_json_sha256": suite["tokenizer_json_sha256"], "graphs": False, "prefix_cache": False,
                          "speculative_decoding": False, "checkpoint_identity": "original-r1",
                          "model_id": "deepseek-ai/DeepSeek-R1", "endpoint": "http://already-running:30000",
                          "source_files": {"full_model_trace": {"path": str(TRACE), "sha256": tool.sha(TRACE)}}}
            rows = []
            for job in suite["jobs"]:
                if job["stage"] == "long":
                    continue
                count = job["max_new_tokens"] if job["ignore_eos"] else 1
                rows.append({"id": job["id"], "native_output_ids": [42] * count,
                             "prompt_tokens": len(job["input_ids"]), "completion_tokens": count,
                             "text": "FINAL: " + job["answers"][0], "finish_reason": "length" if job["ignore_eos"] else "stop"})
            run = {"schema": tool.RUN_SCHEMA, "scope": "full_model", "status": "evidence_collected_NOT_pair_qualified",
                   "suite_sha256": "c" * 64, "deployment": deployment, "backend": "sglang", "long_correctness": False,
                   "records": rows, "trace": {"events_file": "trace/events.json", "events_sha256": tool.sha(root / "trace/events.json")}}
            tool.write_new(root / "run.json", run)
            result = tool.validate_evidence(root / "run.json", suite, "c" * 64, False)
            self.assertEqual(result["actual_coverage_count"], 0)
            self.assertEqual(result["required_coverage_count"], 16 * 2 * 61 * 2)
            self.assertTrue(result["missing_evidence"])


class ActualHookContract(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.r = types.SimpleNamespace(pending={}, active_phase=None,
                                       emit=lambda event, **fields: self.events.append({"event": event, **fields}))
        trace._HANDLES.clear()

    def model(self):
        class Module:
            def __init__(self): self.hooks = []
            def register_forward_hook(self, fn, with_kwargs):
                self.hooks.append(fn); return fn
        layers = []
        for _ in range(61):
            layer = Module(); layer.self_attn = Module(); layer.mlp = Module(); layers.append(layer)
        class Model: pass
        result = Model()
        result.model = types.SimpleNamespace(layers=layers)
        result.config = types.SimpleNamespace(hidden_size=7168, num_hidden_layers=61, vocab_size=129280, n_routed_experts=256)
        return result

    def test_registering_does_not_fabricate_forward_coverage(self):
        model = self.model()
        with patch.object(trace, "recorder", return_value=self.r):
            handles = trace.register_model_hooks(model, 0, 16, "prefill")
        self.assertEqual(len(handles), 183)
        self.assertEqual([e["event"] for e in self.events], ["model_hooks_registered"])
        self.assertFalse(self.r.pending)

    def test_single_model_cannot_be_registered_with_two_false_phase_labels(self):
        model = self.model()
        with patch.object(trace, "recorder", return_value=self.r):
            trace.register_model_hooks(model, 0, 16, "prefill")
            with self.assertRaisesRegex(ValueError, "two fixed phase"):
                trace.register_model_hooks(model, 0, 16, "decode")

    def test_dynamic_phase_requires_observed_mode_and_actual_hook_execution(self):
        model = self.model()
        summary = {"observed": True, "numel": 7, "checked_elements": 7, "nonfinite_elements": 0}
        with patch.object(trace, "recorder", return_value=self.r), patch.object(trace, "tensor_summary", return_value=summary):
            trace.register_model_hooks(model, 0, 16, "dynamic")
            module = model.model.layers[0].self_attn
            module.hooks[0](module, (), {}, None)
            self.assertEqual(self.events[-1]["event"], "trace_error")
            trace.set_execution_phase(0, 16, "decode")
            module.hooks[0](module, (), {}, None)
            trace.flush_layer_stats(0, 16)
            self.assertEqual(self.events[-1]["event"], "layer_summary")
            self.assertEqual(self.events[-1]["phase"], "decode")
            self.assertEqual(self.r.pending, {"decode": {(0, "attention")}})

    def test_hidden_input_skips_integer_sg_positions(self):
        class Tensor:
            def __init__(self, floating): self.floating = floating
            def detach(self): return self
            def numel(self): return 3
            def is_floating_point(self): return self.floating
        positions, hidden, residual = Tensor(False), Tensor(True), Tensor(True)
        self.assertIs(trace.hidden_input((positions, hidden, residual), {}), hidden)
        self.assertIs(trace.hidden_input((positions,), {"hidden_states": residual}), residual)

    def test_model_lifecycle_uses_weak_identity_and_never_retains_old_gpu_model(self):
        model = self.model()
        previous = self.model(); old = weakref.ref(previous)
        trace._MODEL_REFS[id(model)] = old
        trace._HANDLES[(id(model), 0, "prefill")] = ["stale handles"]
        del previous; gc.collect()
        self.assertIsNone(old())
        with patch.object(trace, "recorder", return_value=self.r):
            handles = trace.register_model_hooks(model, 0, 16, "prefill")
        self.assertEqual(len(handles), 183)
        actual = weakref.ref(model)
        del model; gc.collect()
        self.assertIsNone(actual(), "trace hook registry must not keep a model or its GPU weights alive")


class IndependentSemanticContract(unittest.TestCase):
    def test_original_payload_pins_and_paired_rule_are_frozen_before_new_runs(self):
        self.assertEqual(set(semantic.PINS), {"questions.json", "references.json", "selection.json", "bg-requests.jsonl", "sg-requests.jsonl"})
        self.assertTrue(all(len(digest) == 64 for digest in semantic.PINS.values()))
        self.assertIn("question", semantic.RULE["paired_unit"])
        self.assertIsNone(semantic.RULE["absolute_paper_accuracy_requirement"])
        self.assertIn("no statistical", semantic.RULE["inference_limit"])

    def test_em_f1_aliases_and_unextractable_final_answers_keep_denominator(self):
        self.assertEqual(semantic.scores("The cat!", ["cat", "feline"]), {"EM": 1.0, "F1": 1.0})
        self.assertEqual(semantic.scores("cat dog", ["cat"])["F1"], 2 / 3)
        self.assertEqual(semantic.scores("", ["cat"]), {"EM": 0.0, "F1": 0.0})
        self.assertEqual(semantic.final_suffix("<think>unfinished"), ("", False))
        self.assertEqual(semantic.final_suffix("<think>why</think> cat"), ("cat", True))
        self.assertEqual(semantic.final_suffix("cat"), ("cat", True))

    def test_obvious_paired_regression_fails_without_inventing_paper_absolute_accuracy(self):
        same = [{"delta_EM": 0.0, "delta_F1": 0.0} for _ in range(48)]
        self.assertFalse(semantic.paired_rule(same)["gross_regression"])
        f1_loss = [{"delta_EM": 0.0, "delta_F1": -0.04} for _ in range(48)]
        result = semantic.paired_rule(f1_loss)
        self.assertTrue(result["gross_regression"])
        self.assertEqual(result["F1_question_losses"], 48)
        self.assertLess(result["F1_one_sided_exact_sign_p"], 0.01)
        em_loss = [{"delta_EM": -0.1, "delta_F1": 0.0} for _ in range(48)]
        self.assertTrue(semantic.paired_rule(em_loss)["gross_regression"])
        balanced = [{"delta_EM": 0.0, "delta_F1": (-0.1 if i % 2 else 0.1)} for i in range(48)]
        self.assertFalse(semantic.paired_rule(balanced)["gross_regression"])


if __name__ == "__main__":
    unittest.main()
