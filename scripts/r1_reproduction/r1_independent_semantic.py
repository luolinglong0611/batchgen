#!/usr/bin/env python3
"""Import/collect/score the frozen independent full-context 48x2 QA suite.

Every subcommand defaults to plan-only. The new derived protocol is explicitly
greedy; original 0.6/.95 sampling remains in provenance. No input is clipped.
This is a fixed paired regression gate, not paper-quality equivalence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import string
import time

_spec = importlib.util.spec_from_file_location("_r1_full_model_semantic_common", Path(__file__).with_name("qualify_r1_full_model.py"))
core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(core)
PINS = {
    "questions.json": "cf95e858647603f8ccd40d70419c2384e45aeccc0bbe5611146c1695e37a81ca",
    "references.json": "f88d82c0fc021c32b996ef163fda95f1521d2d2d61411284c19fe26cafa588fd",
    "selection.json": "584eeffc748b40b83b5d080f9b1c5879224e353a1639ab7a8eb598e317312744",
    "bg-requests.jsonl": "df8f5d8f3b230fdd3ef0c36470308dbf602dc6185d6173a488cf9106397c73cb",
    "sg-requests.jsonl": "27436b8613fe2496e965586245014ef7269aa06bda954629819b10bb84b09687",
}
RULE = {"version": "r1-independent-48x2-gross-regression-v1",
        "paired_unit": "question, average two repeats before paired differences",
        "primary": "best-alias normalized EM and token-F1 on observable final suffix; no-think direct answer allowed",
        "missing_or_failed": "score zero in all-96 denominator and fail/incomplete evidence gate",
        "unextractable_final_answer": "score zero; retain extractability rate, no invented absolute paper-accuracy cutoff",
        "gross_regression": "BG-minus-SG mean EM <= -0.05 OR mean F1 <= -0.05 OR (F1 one-sided exact sign p<0.01 AND mean F1<-0.02)",
        "comparison_floor": "if SG mean F1==0, inconclusive; no equivalence/pass claim",
        "absolute_paper_accuracy_requirement": None,
        "inference_limit": "no statistical non-inferiority or paper-quality equivalence inferred from not triggering gross-regression rule"}
SCHEMA = "r1-independent-fullcontext-48x2-greedy-v1"
PASS = "passed_full_model_fixed_correctness_and_paired_semantic_NOT_paper_quality_equivalent"


def import_payload(args):
    cohort_size = getattr(args, "cohort_size", 1)
    protocol = core.collection_protocol(cohort_size, getattr(args, "time_basis", "legacy_wall_v1"))
    sources = {}
    for name, expected in PINS.items():
        path = args.payload / name
        if core.sha(path) != expected:
            raise ValueError("Frozen full-context source mismatch: " + name)
        sources[name] = {"path": str(path.resolve()), "sha256": expected}
    if not args.execute:
        print(json.dumps({"status": "plan_only_NOT_executed", "scope": "full_model", "source_files": sources,
                          "original_sampling": {"temperature": 0.6, "top_p": 0.95, "SG_seeds": [20261024, 20261025], "BG_request_seed": "absent"},
                          "derived_sampling": "greedy temperature0/top_p1/top_k1, natural EOS, max8192",
                          "collection_protocol": protocol,
                          "paired_rule": RULE, "long_context_maximum": 17636, "requests": 96}, indent=2)); return 0
    if core.sha(args.tokenizer_json) != args.tokenizer_sha256:
        raise ValueError("Explicit import tokenizer SHA mismatch")
    tok = core.reference_tokenizer(args.tokenizer_json)
    questions = core.checked_json(args.payload / "questions.json")
    refs = core.checked_json(args.payload / "references.json")
    selections = core.checked_json(args.payload / "selection.json")
    bg = [json.loads(line) for line in (args.payload / "bg-requests.jsonl").read_text().splitlines()]
    sg = [json.loads(line) for line in (args.payload / "sg-requests.jsonl").read_text().splitlines()]
    if (len(questions), len(refs), len(selections), len(bg), len(sg)) != (48, 48, 48, 96, 96):
        raise ValueError("Expected exactly 48 unique questions x two frozen repeats")
    refs_by_key = {(r["dataset"], r["original_id"]): r["answers"] for r in refs}
    selection_by_key = {(s["dataset"], s["original_id"]): s for s in selections}
    if len(refs_by_key) != 48 or len(selection_by_key) != 48:
        raise ValueError("Duplicate source question/reference identity")
    jobs = []
    for rb, rs in zip(bg, sg):
        qi, repeat = rb["question_index"], rb["repeat"]
        q = questions[qi]
        key = (q["dataset"], q["original_id"])
        ids, prompt = q["input_ids"], q["completion_prompt"]
        selected = selection_by_key[key]
        if (rs["question_index"] != qi or rs["repeat"] != repeat or rb["custom_id"] != rs["custom_id"]
                or rb["body"]["prompt"] != prompt or rs["body"]["input_ids"] != ids
                or core.value_sha(ids) != q["input_ids_sha256"] or rb["input_ids_sha256"] != q["input_ids_sha256"]
                or selected["input_ids_sha256"] != q["input_ids_sha256"]
                or len(ids) != q["input_tokens"] or len(ids) + 8192 > 32768
                or tok.encode(prompt, add_special_tokens=True).ids != ids):
            raise ValueError("Full original input/context or tokenizer identity changed")
        answers = refs_by_key[key]
        if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("Missing independent answer aliases")
        jobs.append({"id": rb["custom_id"], "case_id": "independent-q" + str(qi), "kind": "generation", "stage": "short",
                     "question_index": qi, "repeat": repeat, "dataset": q["dataset"], "original_id": q["original_id"],
                     "length_bin": q["length_bin"], "prompt": prompt, "prompt_sha256": core.value_sha(prompt),
                     "input_ids": ids, "input_ids_sha256": q["input_ids_sha256"], "max_new_tokens": 8192,
                     "ignore_eos": False, "semantic": True, "answers": answers,
                     "source_row_sha256": q["source_row_sha256"], "context_sha256": q["context_sha256"]})
    if {(j["question_index"], j["repeat"]) for j in jobs} != {(q, r) for q in range(48) for r in (0, 1)}:
        raise ValueError("Frozen paired-repeat coverage differs")
    suite = {"schema": SCHEMA, "scope": "full_model", "jobs": jobs, "paired_rule": RULE,
             "source_files": sources, "tokenizer_json_sha256": args.tokenizer_sha256,
             "world_size": 16, "num_hidden_layers": 61, "vocab_size": 129280,
             "derived_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "ignore_eos": False, "max_new_tokens": 8192},
             "original_sampling": {"temperature": 0.6, "top_p": 0.95, "SG_seeds": [20261024, 20261025], "BG_request_seed": "absent"},
             "tool_sha256": core.sha(__file__), "full_context_preserved": True,
             "prior_model_results_imported": False, "full_model_qualified": False}
    if cohort_size != 1 or protocol.get("time_basis") == "node_monotonic_causal_v2":
        suite["collection_protocol"] = protocol
        suite["collection_source_sha256"] = core.sha(core.__file__)
    for name, expected in PINS.items():
        if core.sha(args.payload / name) != expected:
            raise ValueError("Source changed during semantic import")
    core.write_new(args.output, suite)
    print(core.canonical({"status": "independent_semantic_fixture_imported_NOT_run", "scope": "full_model", "suite_sha256": core.sha(args.output)}))
    return 0


def validate_suite(suite):
    if (core.suite_cohort_size(suite) == 16 or core.is_causal_suite(suite)) and suite.get("tool_sha256") != core.sha(__file__):
        raise ValueError("Cohort16 independent suite collector source differs")
    if (suite.get("schema") != SCHEMA or suite.get("paired_rule") != RULE or suite.get("full_context_preserved") is not True
            or suite.get("prior_model_results_imported") is not False or len(suite.get("jobs", [])) != 96
            or {name: d["sha256"] for name, d in suite.get("source_files", {}).items()} != PINS):
        raise ValueError("Independent full-context suite/source/scoring contract differs")
    jobs = suite["jobs"]
    if len({j["id"] for j in jobs}) != 96 or {(j["question_index"], j["repeat"]) for j in jobs} != {(q, r) for q in range(48) for r in (0, 1)}:
        raise ValueError("Missing/duplicate paired quality item")
    for j in jobs:
        if (core.value_sha(j["input_ids"]) != j["input_ids_sha256"] or core.value_sha(j["prompt"]) != j["prompt_sha256"]
                or not 1 <= len(j["input_ids"]) <= 24576 or len(j["input_ids"]) + 8192 > 32768
                or any(type(v) is not int or not 0 <= v < 129280 for v in j["input_ids"])
                or j["ignore_eos"] is not False or j["max_new_tokens"] != 8192 or not j["answers"]):
            raise ValueError("Invalid independent original context/output contract")
    return suite


def collect(args):
    suite = validate_suite(core.checked_json(args.suite, args.suite_sha256))
    schedule = core.requested_collection_schedule(args, suite, False)
    d = core.validate_deployment(core.checked_json(args.deployment, args.deployment_sha256), args.backend, suite)
    if args.endpoint.rstrip("/") != d["endpoint"].rstrip("/"):
        raise ValueError("Independent target endpoint differs from frozen deployment")
    causal_capture = core.prepare_causal_capture(args, suite, d)
    if not args.execute:
        print(json.dumps({"status": "plan_only_NOT_executed", "scope": "full_model", "operation": "collect_independent_semantic",
                          "requests": 96, "already_running_service_only": True, "derived_sampling": suite["derived_sampling"],
                          "collection_schedule": schedule, "collection_schedule_sha256": core.value_sha(schedule),
                          "performance_test": False}, indent=2)); return 0
    if not args.run_48x2 or not args.trace_directory.is_dir():
        raise ValueError("Explicit --run-48x2 and existing trace directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    d["artifact_sha256"] = args.deployment_sha256
    report = {"schema": core.RUN_SCHEMA, "scope": "full_model", "status": core.INCOMPLETE, "backend": args.backend,
              "suite_sha256": args.suite_sha256, "tool_sha256": core.sha(__file__), "deployment": d,
              "long_correctness": False, "independent_semantic": True, "records": [],
              "collection_schedule": schedule, "collection_schedule_sha256": core.value_sha(schedule),
              "collection_receipts": [], "collection_source_sha256": core.sha(core.__file__),
              "started_unix_ns": time.time_ns(), "formal_performance_permitted": False, **causal_capture}
    if causal_capture:
        report.update(started_monotonic_ns=time.monotonic_ns(), causal_collector_identity=core.causal_module().identity())
    code = 2
    try:
        with (args.output / "http.jsonl").open("x") as log:
            transport = core.Transport(args.endpoint, args.timeout, log, os.environ.get(args.api_key_env) if args.api_key_env else None)
            core.collect_cohorts(transport, suite, schedule, report, d["model_id"], args.timeout,
                                 args.output / "progress.json", 32768)
        report["finished_unix_ns"] = time.time_ns()
        if causal_capture:
            report["finished_monotonic_ns"] = time.monotonic_ns()
        report["trace"] = core.snapshot_trace(args.trace_directory, args.output / "trace", d, args.suite_sha256, report["started_unix_ns"], report["finished_unix_ns"], causal=core.is_causal_suite(suite))
        if core.sha(args.suite) != args.suite_sha256 or core.sha(args.deployment) != args.deployment_sha256:
            raise ValueError("Frozen semantic suite/deployment changed")
        report["status"] = "evidence_collected_NOT_pair_qualified"; code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["service_request_may_still_be_active"] = True
    report.setdefault("finished_unix_ns", time.time_ns())
    if causal_capture:
        report.setdefault("finished_monotonic_ns", time.monotonic_ns())
    core.write_new(args.output / "run.json", report)
    print(core.canonical({"status": report["status"], "scope": "full_model", "report": str(args.output / "run.json")}))
    return code


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def scores(text, answers):
    a = normalize(text)
    predicted = a.split(); em = f1 = 0.0
    for reference in answers:
        target = normalize(reference).split()
        em = max(em, float(a == " ".join(target)))
        overlap = sum((Counter(predicted) & Counter(target)).values())
        if overlap:
            f1 = max(f1, 2 * overlap / (len(predicted) + len(target)))
    return {"EM": em, "F1": f1}


def final_suffix(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return "", False
    if "<think>" in text:
        return "", False
    return text.strip(), bool(text.strip())


def paired_rule(rows):
    em = math.fsum(r["delta_EM"] for r in rows) / len(rows)
    f1 = math.fsum(r["delta_F1"] for r in rows) / len(rows)
    losses = sum(r["delta_F1"] < 0 for r in rows); wins = sum(r["delta_F1"] > 0 for r in rows)
    n = losses + wins
    p = sum(math.comb(n, k) for k in range(losses, n + 1)) / 2**n if n else 1.0
    fail = em <= -0.05 or f1 <= -0.05 or (p < 0.01 and f1 < -0.02)
    return {"mean_delta_EM_BG_minus_SG": em, "mean_delta_F1_BG_minus_SG": f1,
            "F1_question_losses": losses, "F1_question_wins": wins,
            "F1_one_sided_exact_sign_p": p, "gross_regression": fail}


def score(args):
    suite = validate_suite(core.checked_json(args.suite, args.suite_sha256))
    if not args.execute:
        print(json.dumps({"status": "plan_only_NOT_executed", "scope": "full_model", "paired_rule": RULE,
                          "requires": "new native-ID/source-bound actual full-model runs plus successful separate long-smoke report"}, indent=2)); return 0
    report = {"schema": "r1-independent-semantic-paired-result-v1", "scope": "full_model", "status": core.INCOMPLETE,
              "suite_sha256": args.suite_sha256, "tool_sha256": core.sha(__file__), "paired_rule": RULE,
              "full_model_qualified": False, "formal_performance_permitted": False, "per_item": [], "per_question": []}
    code = 2
    try:
        if core.sha(args.batchgen_run) != args.batchgen_run_sha256 or core.sha(args.sglang_run) != args.sglang_run_sha256:
            raise ValueError("Explicit semantic run SHA mismatch")
        smoke = core.checked_json(args.long_smoke_report, args.long_smoke_report_sha256)
        coverage_count = 16 * 2 * 61 * 2
        if (smoke.get("status") != core.PASS or smoke.get("scope") != "full_model" or smoke.get("long_correctness") is not True
                or smoke.get("numeric_contract") != core.NUMERIC
                or any(smoke.get("observed_execution", {}).get(engine, {}).get("actual_coverage_count") != coverage_count
                       for engine in ("batchgen", "sglang"))):
            raise ValueError("Successful separate short+long+teacher smoke report required")
        if core.sha(args.tokenizer_json) != suite["tokenizer_json_sha256"]:
            raise ValueError("Independent native-ID decoding tokenizer differs")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(args.tokenizer_json)); tok.no_padding(); tok.no_truncation()
        a = core.validate_evidence(args.batchgen_run, suite, args.suite_sha256, False, semantic_smoke=False, context_limit=32768)
        b = core.validate_evidence(args.sglang_run, suite, args.suite_sha256, False, semantic_smoke=False, context_limit=32768)
        core.require_same_collection_schedule(a, b)
        report["termination_policy"] = core.require_same_termination_policy(a, b)
        report["collection"] = {"batchgen": a["collection"], "sglang": b["collection"]}
        if a["run"]["backend"] != "batchgen" or b["run"]["backend"] != "sglang":
            raise ValueError("Independent semantic engine identities reversed")
        if a["run"]["deployment"]["checkpoint_identity"] != b["run"]["deployment"]["checkpoint_identity"]:
            raise ValueError("Paired semantic checkpoints differ")
        for engine, evidence in (("batchgen", a), ("sglang", b)):
            if core.deployment_model_identity(evidence["run"]["deployment"]) != smoke.get("deployment_model_identity", {}).get(engine):
                raise ValueError("Semantic model/source/execution identity differs from the qualified long-smoke run")
        hard, missing = a["hard_failures"] + b["hard_failures"], a["missing_evidence"] + b["missing_evidence"]
        for job in suite["jobs"]:
            record = {"id": job["id"], "question_index": job["question_index"], "repeat": job["repeat"], "dataset": job["dataset"], "length_bin": job["length_bin"], "input_ids_sha256": job["input_ids_sha256"]}
            for name, evidence in (("batchgen", a), ("sglang", b)):
                response = evidence["records"][job["id"]]
                ids = response.get("native_output_ids")
                if not isinstance(ids, list):
                    record[name] = {"EM": 0.0, "F1": 0.0, "extractable": False}; continue
                text = tok.decode(ids, skip_special_tokens=True)
                suffix, extractable = final_suffix(text)
                api_suffix, _ = final_suffix(response.get("text", ""))
                if normalize(suffix) != normalize(api_suffix):
                    hard.append(name + ": native-ID/API semantic text differs: " + job["id"])
                primary = scores(suffix, job["answers"]) if extractable else {"EM": 0.0, "F1": 0.0}
                record[name] = {**primary, "extractable": extractable, "raw_completion": scores(text, job["answers"]),
                                "native_output_count": len(ids), "finish_reason": response["finish_reason"],
                                "final_suffix": suffix, "answers": job["answers"]}
            report["per_item"].append(record)
        for question in range(48):
            group = [r for r in report["per_item"] if r["question_index"] == question]
            averaged = {engine: {metric: math.fsum(r[engine][metric] for r in group) / 2 for metric in ("EM", "F1")} for engine in ("batchgen", "sglang")}
            report["per_question"].append({"question_index": question, "dataset": group[0]["dataset"], "length_bin": group[0]["length_bin"], **averaged,
                "delta_EM": averaged["batchgen"]["EM"] - averaged["sglang"]["EM"], "delta_F1": averaged["batchgen"]["F1"] - averaged["sglang"]["F1"]})
        report["paired_result"] = paired_rule(report["per_question"])
        report["aggregate"] = {engine: {metric: math.fsum(r[engine][metric] for r in report["per_item"]) / 96 for metric in ("EM", "F1")} for engine in ("batchgen", "sglang")}
        report["source_run_sha256"] = {"batchgen": a["run_sha256"], "sglang": b["run_sha256"], "long_smoke": args.long_smoke_report_sha256}
        report["observed_execution"] = {name: {key: evidence[key] for key in ("actual_coverage_count", "required_coverage_count")} for name, evidence in (("batchgen", a), ("sglang", b))}
        report["hard_failures"], report["missing_evidence"] = hard, missing
        if (core.sha(args.batchgen_run) != args.batchgen_run_sha256 or core.sha(args.sglang_run) != args.sglang_run_sha256
                or core.sha(args.long_smoke_report) != args.long_smoke_report_sha256
                or core.sha(args.suite) != args.suite_sha256 or core.sha(args.tokenizer_json) != suite["tokenizer_json_sha256"]):
            raise ValueError("Semantic comparison sources changed during evaluation")
        if hard or report["paired_result"]["gross_regression"]:
            report["status"] = "failed_independent_semantic_or_full_model_evidence"; code = 1
        elif missing or report["aggregate"]["sglang"]["F1"] == 0:
            report["status"] = "inconclusive_independent_semantic_evidence"
        else:
            report["status"] = PASS; report["full_model_qualified"] = True; code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, ValueError):
            report["status"] = "failed_independent_semantic_or_full_model_evidence"; code = 1
    core.write_new(args.output, report)
    print(core.canonical({"status": report["status"], "scope": "full_model", "report": str(args.output)}))
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("import")
    q.add_argument("--cohort-size", type=int, choices=(1, 16), default=1)
    q.add_argument("--time-basis", choices=("legacy_wall_v1", "node_monotonic_causal_v2"), default="legacy_wall_v1")
    q.add_argument("--payload", type=Path, required=True); q.add_argument("--tokenizer-json", type=Path, required=True)
    q.add_argument("--tokenizer-sha256", required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--execute", action="store_true")
    q = sub.add_parser("collect")
    q.add_argument("--causal-clock-config", type=Path); q.add_argument("--causal-clock-config-sha256")
    q.add_argument("--cohort-size", type=int, choices=(1, 16), default=1)
    q.add_argument("--time-basis", choices=("legacy_wall_v1", "node_monotonic_causal_v2"), default="legacy_wall_v1")
    q.add_argument("--suite", type=Path, required=True); q.add_argument("--suite-sha256", required=True)
    q.add_argument("--deployment", type=Path, required=True); q.add_argument("--deployment-sha256", required=True)
    q.add_argument("--backend", choices=("batchgen", "sglang"), required=True); q.add_argument("--endpoint", required=True)
    q.add_argument("--trace-directory", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    q.add_argument("--timeout", type=float, default=7200); q.add_argument("--api-key-env"); q.add_argument("--run-48x2", action="store_true"); q.add_argument("--execute", action="store_true")
    q = sub.add_parser("score")
    q.add_argument("--suite", type=Path, required=True); q.add_argument("--suite-sha256", required=True)
    q.add_argument("--batchgen-run", type=Path, required=True); q.add_argument("--sglang-run", type=Path, required=True)
    q.add_argument("--batchgen-run-sha256", required=True); q.add_argument("--sglang-run-sha256", required=True)
    q.add_argument("--long-smoke-report", type=Path, required=True); q.add_argument("--long-smoke-report-sha256", required=True)
    q.add_argument("--tokenizer-json", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if hasattr(args, "timeout") and (not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400):
        p.error("Bounded overall semantic timeout must be in (0,86400]")
    return {"import": import_payload, "collect": collect, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
