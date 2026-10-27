"""CPU-only native termination/retired-row evidence; never runs a model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/r1_reproduction" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load("r1_stop_tool", "qualify_r1_full_model.py")
trace = load("r1_stop_trace", "r1_full_model_trace.py")
bridge = load("r1_stop_bridge", "sglang_r1_full_model_trace.py")


def fixture(backend="sglang", ignore=False, ids=None):
    ids = [32, 7] if ids is None else ids
    job = {"id": "job", "kind": "generation", "stage": "short", "semantic": True,
           "input_ids": [0, 21], "input_ids_sha256": tool.value_sha([0, 21]), "max_new_tokens": len(ids) if ignore else 8192,
           "ignore_eos": ignore, "answers": ["323"]}
    raw = {"type": "length", "length": len(ids)} if ignore else {"type": "stop", "matched": ids[-1]}
    reason = "length" if ignore else "stop"
    record = {"id": "job", "native_output_ids": list(ids), "finish_reason": reason,
              "raw_finish_reason": raw if backend == "sglang" else reason,
              "prompt_tokens": 2, "completion_tokens": len(ids), "text": "FINAL: 323"}
    record["response"] = ({"output_ids": list(ids), "meta_info": {"finish_reason": copy.deepcopy(raw)}}
                          if backend == "sglang" else {"response": {"body": {"choices": [{"finish_reason": reason}]}}})
    policy = {"schema": "r1-native-stop-policy-v1", "status": "observed", "evidence": ["actual object fields"],
              "eos_token_ids": [7], "ignore_eos": ignore, "max_new_tokens": job["max_new_tokens"]}
    completion = {"native_generated_ids": list(ids), "last_native_token": ids[-1],
                  "generated_count": len(ids), "decoded_length": len(ids), "original_input_length": 2,
                  "original_input_ids_sha256": job["input_ids_sha256"], "stop_policy": policy}
    if backend == "sglang":
        completion.update(raw_native_finish_reason=copy.deepcopy(raw), native_finish_is_error=False,
                          native_finish_reason_class="FINISH_LENGTH" if ignore else "FINISH_MATCHED_TOKEN")
        policy.update(additional_stop_token_ids=[], stop_token_ids=[], stop_strs=[], stop_regex_strs=[], grammar_active=False)
    else:
        completion["finish_reason"] = reason
        policy.update(repetition_breaker_enabled=False, repetition_detected=False)
    return job, record, completion


class NativeStopEvidence(unittest.TestCase):
    def test_sg_raw_matched_reason_is_retained(self):
        job, record, _ = fixture()
        response = {"output_ids": record["native_output_ids"], "text": "FINAL: 323", "meta_info": {
            "prompt_tokens": 2, "completion_tokens": 2, "finish_reason": {"type": "stop", "matched": "NaN happened"}}}
        transport = SimpleNamespace(call=lambda *args: response)
        actual = tool.collect_sg(transport, job)
        self.assertEqual(actual["finish_reason"], "stop")
        self.assertEqual(actual["raw_finish_reason"], response["meta_info"]["finish_reason"])

    def test_native_eos_requires_actual_policy_and_matching_native_token(self):
        for backend in ("batchgen", "sglang"):
            job, record, native = fixture(backend)
            result = tool.validate_termination(backend, job, record, native)
            self.assertEqual(result["evidence"]["status"], "verified")
            self.assertFalse(result["hard_failures"])
            self.assertFalse(result["missing_evidence"])
            self.assertEqual(result["evidence"]["last_native_token"], 7)  # Not inferred as R1 fixture token1.
            del native["stop_policy"]
            result = tool.validate_termination(backend, job, record, native)
            self.assertTrue(result["missing_evidence"])
            self.assertNotEqual(result["evidence"]["status"], "verified")

    def test_string_regex_grammar_error_or_non_eos_stop_cannot_be_natural(self):
        for mode in ("string", "regex", "grammar", "error", "extra_token", "wrong_eos", "wrong_last", "raw_mismatch", "unknown_class"):
            job, record, native = fixture()
            if mode in ("string", "regex"):
                native["raw_native_finish_reason"] = record["raw_finish_reason"] = {"type": "stop", "matched": "END"}
                native["native_finish_reason_class"] = "FINISH_MATCHED_STR" if mode == "string" else "FINISHED_MATCHED_REGEX"
            elif mode == "grammar": native["stop_policy"]["grammar_active"] = True
            elif mode == "error": native["native_finish_is_error"] = True
            elif mode == "extra_token": native["stop_policy"]["stop_token_ids"] = [7]
            elif mode == "wrong_eos": native["stop_policy"]["eos_token_ids"] = [8]
            elif mode == "wrong_last": native["last_native_token"] = 8
            elif mode == "raw_mismatch": record["raw_finish_reason"] = {"type": "stop", "matched": 8}
            else: native["native_finish_reason_class"] = "SomethingElse"
            with self.subTest(mode=mode):
                result = tool.validate_termination("sglang", job, record, native)
                self.assertTrue(result["hard_failures"])
                self.assertNotEqual(result["evidence"]["status"], "verified")

    def test_bg_native_repetition_or_mismatched_finish_cannot_be_http_stop(self):
        for mode in ("native_reason", "native_flag", "breaker_enabled"):
            job, record, native = fixture("batchgen")
            if mode == "native_reason": native["finish_reason"] = "repetition"
            elif mode == "native_flag": native["stop_policy"]["repetition_detected"] = True
            else: native["stop_policy"]["repetition_breaker_enabled"] = True
            with self.subTest(mode=mode):
                self.assertTrue(tool.validate_termination("batchgen", job, record, native)["hard_failures"])
        job, record, _ = fixture("batchgen")
        record["finish_reason"] = "repetition"
        result = tool.validate_termination("batchgen", job, record, None)
        self.assertIn("semantic generation did not finish naturally", result["hard_failures"])

    def test_repetition_threshold_is_symmetric_and_fixed_length_is_diagnostic_only(self):
        for backend in ("batchgen", "sglang"):
            for ignore in (False, True):
                job, record, native = fixture(backend, ignore, [9] * 32 + [7])
                result = tool.validate_termination(backend, job, record, native)
                self.assertTrue(result["evidence"]["repetition_diagnostic"]["detected"])
                self.assertEqual(bool(result["hard_failures"]), not ignore)
        self.assertFalse(tool.repetition_diagnostic([9] * 31)["detected"])
        self.assertFalse(tool.repetition_diagnostic([9, 10] * 31)["detected"])
        result = tool.repetition_diagnostic([9, 10] * 32)
        self.assertEqual(result["matches"][0]["first_prefix_length"], 64)
        self.assertEqual(result["matches"][0]["pattern_tokens"], 2)
        self.assertFalse(tool.repetition_diagnostic(list(range(1000, 1101)) * 32)["detected"])

    def test_first_eos_and_raw_length_are_checked(self):
        job, record, native = fixture("sglang", False, [7, 32, 7])
        self.assertTrue(tool.validate_termination("sglang", job, record, native)["hard_failures"])

    def test_native_bool_float_token_and_count_lookalikes_cannot_pass(self):
        for field, value in (("native_generated_ids", [True, 7]), ("native_generated_ids", [1.0, 7]),
                             ("generated_count", 2.0), ("decoded_length", 2.0), ("original_input_length", 2.0)):
            job, record, native = fixture("sglang", False, [1, 7])
            native[field] = value
            with self.subTest(field=field, value=value):
                self.assertTrue(tool.validate_termination("sglang", job, record, native)["hard_failures"])
        job, record, native = fixture("sglang", True, [7])
        native["raw_native_finish_reason"]["length"] = True
        self.assertTrue(tool.validate_termination("sglang", job, record, native)["hard_failures"])
        job, record, native = fixture("sglang", True)
        native["raw_native_finish_reason"]["length"] += 1
        self.assertTrue(tool.validate_termination("sglang", job, record, native)["hard_failures"])
        for raw_ids in ([True, 7], [1.0, 7]):
            job, record, native = fixture("sglang", False, [1, 7])
            record["response"]["output_ids"] = raw_ids
            self.assertTrue(tool.validate_termination("sglang", job, record, native)["hard_failures"])

    def test_paired_actual_eos_difference_is_not_hidden_by_same_stop_string(self):
        job, record, native = fixture("batchgen")
        a = {"records": {"job": record}, "termination_evidence": {
            "job": tool.validate_termination("batchgen", job, record, native)["evidence"]}}
        b = copy.deepcopy(a)
        self.assertEqual(tool.require_same_termination_policy(a, b)["status"], "verified_equal")
        b["termination_evidence"]["job"]["stop_policy"]["eos_token_ids"] = [8]
        with self.assertRaisesRegex(ValueError, "EOS/stop policies differ"):
            tool.require_same_termination_policy(a, b)
        b["termination_evidence"] = {}
        self.assertEqual(tool.require_same_termination_policy(a, b)["status"], "unverified")


class NativeTraceObservation(unittest.TestCase):
    def test_bg_completion_reads_loaded_worker_policy_and_native_eos(self):
        events = []
        r = SimpleNamespace(jobs={"job": {}}, layer_totals={}, emit=lambda event, **fields: events.append(dict(event=event, **fields)))
        def actual_finish_reason(seq):
            return "stop"
        seq = SimpleNamespace(uuid="job@batch", assigned_rank=3, decoded_length=2, original_prompt_length=2,
                              max_decode_length=8192, ignore_eos=False, _rep_detected=False)
        worker = SimpleNamespace(rank=3, world_size=16, eos_token_ids={7}, _get_finish_reason=actual_finish_reason,
                                 _trajectory_tokens=lambda *args, **kwargs: [0, 21])
        with patch.dict(os.environ, {"R1_FULL_MODEL_TRACE": "1"}), patch.object(trace, "recorder", return_value=r), \
                patch.dict(actual_finish_reason.__globals__, {"REP_DETECTION": False}):
            trace.trace_bg_completion(worker, seq, [32, 7])
        native = events[-1]
        self.assertEqual(native["stop_policy"]["eos_token_ids"], [7])
        self.assertFalse(native["stop_policy"]["repetition_breaker_enabled"])
        self.assertEqual(native["last_native_token"], 7)
        job, record, _ = fixture("batchgen")
        self.assertEqual(tool.validate_termination("batchgen", job, record, native)["evidence"]["status"], "verified")
        del worker.eos_token_ids
        self.assertEqual(trace.bg_stop_policy(worker, seq)["status"], "missing")

    def test_actual_sg_req_policy_and_completion_are_read_without_fixture_eos(self):
        events = []
        r = SimpleNamespace(jobs={"job": {}}, layer_totals={}, emit=lambda event, **fields: events.append(dict(event=event, **fields)))
        reason = type("FINISH_MATCHED_TOKEN", (), {"is_error": False, "to_json": lambda self: {"type": "stop", "matched": 7}})()
        req = SimpleNamespace(rid="job", origin_input_ids=[0, 21], output_ids=[32, 7], finished_len=2,
            finished_reason=reason, eos_token_ids={7}, tokenizer=SimpleNamespace(eos_token_id=7, additional_stop_token_ids=None),
            sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=8192, stop_token_ids=None, stop_strs=[], stop_regex_strs=[]),
            grammar=None)
        with patch.dict(os.environ, {"R1_FULL_MODEL_TRACE": "1"}), patch.object(trace, "recorder", return_value=r):
            trace.trace_sg_completion(req, 3, 16)
        native = events[-1]
        self.assertEqual(native["event"], "completion")
        self.assertEqual(native["stop_policy"]["eos_token_ids"], [7])
        self.assertEqual(native["last_native_token"], 7)
        job, record, _ = fixture()
        self.assertEqual(tool.validate_termination("sglang", job, record, native)["evidence"]["status"], "verified")
        del req.eos_token_ids
        self.assertEqual(trace.sg_stop_policy(req)["status"], "missing")

    def test_sg_wrapper_calls_stock_once_and_only_observes_after_finish(self):
        calls, observations = [], []
        class Req:
            finished_reason = None
            def check_finished(self, new_accepted_len=1):
                calls.append(new_accepted_len)
                self.finished_reason = "native-done"
                return "unchanged-return"
        observer = SimpleNamespace(trace_sg_completion=lambda *args: observations.append(args))
        dist = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 5, get_world_size=lambda: 16)
        bridge.install_completion_wrapper(Req, observer)
        req = Req()
        with patch.dict(sys.modules, {"torch.distributed": dist}):
            self.assertEqual(req.check_finished(1), "unchanged-return")
            self.assertEqual(req.check_finished(2), "unchanged-return")
        self.assertEqual(calls, [1, 2])
        self.assertEqual(observations, [(req, 5, 16)])
        with self.assertRaisesRegex(RuntimeError, "installation mismatch"):
            bridge.install_completion_wrapper(Req, observer)

    def test_bg_retired_row_is_logged_as_inactive_without_deduplicating_active_steps(self):
        events, captured = [], []
        retired = SimpleNamespace(uuid="job@batch", decoded_length=262, original_prompt_length=2,
            current_context_length=264, max_decode_length=8192, eos_reached=True, _rep_detected=False,
            status=SimpleNamespace(name="IN_DECODE"))
        active = SimpleNamespace(**{**retired.__dict__, "uuid": "live@batch", "decoded_length": 1,
            "current_context_length": 3, "eos_reached": False})
        r = SimpleNamespace(emit=lambda event, **fields: events.append(dict(event=event, **fields)))
        class Logits:
            shape = (2, 129280)
            def __getitem__(self, rows):
                self.selected = rows
                return "selected-native-rows"
        logits = Logits()
        worker = SimpleNamespace(rank=0, world_size=16, model_context_length=32768,
            model=SimpleNamespace(config=SimpleNamespace(phase="decode")),
            _is_sequence_completed=lambda seq: seq.eos_reached,
            _trajectory_model_input=lambda seq: [0, 21, 32])
        with patch.dict(os.environ, {"R1_FULL_MODEL_TRACE": "1"}), patch.object(trace, "recorder", return_value=r), \
                patch.object(trace, "capture_logits", side_effect=lambda *args: captured.append(args)):
            for _ in range(2):
                trace.trace_bg_logits(worker, logits, [retired, active])
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e["event"] == "inactive_forward" and e["eligible"] is False for e in events))
        self.assertTrue(all(e["generated_before_step"] == 262 and e["sequence_state"]["eos_reached"] for e in events))
        self.assertEqual(logits.selected, [1])
        self.assertEqual([item[0] for item in captured], [["live@batch"], ["live@batch"]])
        self.assertEqual(active.decoded_length, 1)  # Trace never advances or deduplicates either row.


class StrictStepIntegration(unittest.TestCase):
    def evidence(self, directory, mode):
        job, record, native = fixture("batchgen")
        suite = {"schema": tool.SCHEMA, "tokenizer_json_sha256": "a" * 64, "jobs": [job]}
        deployment = {"schema": "r1-full-model-deployment-v1", "backend": "batchgen", "world_size": 16,
            "num_hidden_layers": 61, "vocab_size": tool.VOCAB, "run_id": "test", "artifact_sha256": "b" * 64,
            "tokenizer_json_sha256": "a" * 64, "graphs": False, "prefix_cache": False,
            "speculative_decoding": False, "checkpoint_identity": "r1", "model_id": "r1", "endpoint": "http://unused",
            "source_files": {"trace": {"path": "/unused/source", "sha256": "d" * 64}}}
        envelope = {"schema": "r1-full-model-trace-v1", "run_id": "test", "suite_sha256": "c" * 64,
            "deployment_sha256": "b" * 64, "world_size": 16, "rank": 0, "request_id": "job"}
        events = [dict(envelope, event="input_tokens", input_ids_sha256=job["input_ids_sha256"],
                       input_count=2, input_ids_equal_fixture=True), dict(envelope, event="completion", **native)]
        for step, token in enumerate(record["native_output_ids"]):
            prefix = job["input_ids"] + record["native_output_ids"][:step]
            events.append(dict(envelope, event="logits", generated_before_step=step, input_ids_sha256=tool.value_sha(prefix),
                context_length=len(prefix), vocab_size=tool.VOCAB, nonfinite_elements=0, greedy_token_id=token,
                source_files_at_last_step={"trace": "d" * 64}))
        if mode == "duplicate": events.append(dict(events[-1], generated_before_step=2))
        if mode == "missing": events.pop()
        if mode in ("inactive", "false_inactive"):
            events.append(dict(envelope, event="inactive_forward", eligible=False, completed=True, phase="decode",
                eligibility_source="actual worker._is_sequence_completed(seq)", generated_before_step=2,
                model_context_length=32768, sequence_state={"decoded_length": 2 if mode == "inactive" else 1,
                    "original_prompt_length": 2, "current_context_length": 4, "max_decode_length": 8192,
                    "eos_reached": True, "_rep_detected": False, "status": "IN_DECODE"}))
        (directory / "trace").mkdir()
        tool.write_new(directory / "trace/events.json", events)
        run = {"schema": tool.RUN_SCHEMA, "scope": "full_model", "status": "evidence_collected_NOT_pair_qualified",
            "backend": "batchgen", "suite_sha256": "c" * 64, "deployment": deployment,
            "long_correctness": False, "records": [record], "trace": {"events_file": "trace/events.json",
                "events_sha256": tool.sha(directory / "trace/events.json")}}
        tool.write_new(directory / "run.json", run)
        return tool.validate_evidence(directory / "run.json", suite, "c" * 64, False)

    def test_active_duplicate_missing_and_false_inactive_evidence_fail_closed(self):
        for mode in ("valid", "duplicate", "missing", "inactive", "false_inactive"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                result = self.evidence(Path(folder), mode)
            if mode in ("duplicate", "missing"):
                self.assertTrue(any("missing/duplicate native logits steps" in item for item in result["missing_evidence"]))
            elif mode == "false_inactive":
                self.assertTrue(any("inactive forward" in item for item in result["hard_failures"]))
            else:
                self.assertFalse(result["hard_failures"])
                self.assertFalse(any("native logits steps" in item for item in result["missing_evidence"]))
            self.assertTrue(result["missing_evidence"])  # Synthetic fixture never proves full rank/layer coverage.


if __name__ == "__main__":
    unittest.main()
