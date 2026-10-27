#!/usr/bin/env python3
"""Fixed full-R1 correctness evidence, never an automatic performance run.

All commands default to plan-only. prepare is local tokenizer-only; collect
uses already-running services; compare is offline. Missing native IDs, full
teacher logits or observed layer/rank coverage yields INCOMPLETE, never pass.
"""
from __future__ import annotations
import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

SCHEMA = "r1-full-model-fixed-suite-v1"
SCHEMA_V2 = "r1-full-model-fixed-suite-v2"
V2_BUDGET_PROTOCOL = {
    "version": "r1-short-semantic-generation-budget-v2",
    "semantic_generation_count": 12,
    "previous_max_new_tokens": 384,
    "max_new_tokens": 8192,
    "ignore_eos": False,
    "basis": "same8192 cap predeclared for the independent96 suite; unchanged answer, naturalEOS and numeric gates; no adaptive budget escalation",
    "previous_384_results": "retained_as_observed_NOT_upgraded_or_reused_as_v2_pass",
}
RUN_SCHEMA = "r1-full-model-evidence-v1"
CONTRACT_VERSION = "r1-full-model-predeclared-contract-v1"
VOCAB = 129280
LAYERS = 61
WORLD = 16
QA = (
    ("arithmetic", "What is 17 times 19?", "323", ("323",)),
    ("algebra", "Solve 3*x + 7 = 22 for x.", "5", ("5", "x=5")),
    ("percent", "There are 12 red balls and 8 blue balls. What percentage are red?", "60%", ("60%", "0.6")),
    ("recall", "The fictional access phrase is silver maple 47. Repeat that phrase.", "silver maple 47", ("silver maple 47",)),
    ("code", "What is sum(i*i for i in range(4)) in Python?", "14", ("14",)),
    ("logic", "All glims are norks. All norks are tevs. Are all glims tevs? Answer yes or no.", "Yes", ("yes",)),
)
NUMERIC = {"version": CONTRACT_VERSION, "logprob_sup_abs_max": 0.5,
           "logprob_rms_max": 0.1, "Jensen_Shannon_max_nats": 0.001,
           "teacher_logprob_abs_max": 0.1, "teacher_logprob_mean_abs_max": 0.03,
           "top1_disagreement_near_tie_margin_max": 0.2,
           "near_tie_disagreement_fraction_max": 0.05,
           "nonfinite_allowed": 0, "shape_or_token_bounds_errors_allowed": 0,
           "semantic_answers_required_fraction": 1.0,
           "basis": "predeclared engineering acceptance for this fixed smoke suite; not a paper threshold or statistical non-inferiority budget"}
PASS = "passed_full_model_long_smoke_INDEPENDENT_SEMANTIC_PENDING"
SHORT = "passed_full_model_short_stage_LONG_CORRECTNESS_PENDING"
FAIL = "failed_full_model_correctness"
INCOMPLETE = "incomplete_full_model_evidence"


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checked_json(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError("Explicit artifact SHA mismatch: " + str(path))
    return json.loads(Path(path).read_text())


def render(question):
    return "<｜User｜>" + question + "\nGive a concise final answer as FINAL: <answer>.<｜Assistant｜>"


def reference_tokenizer(path):
    """Match released R1 tokenizer_config boundaries, not bare JSON defaults."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing
    path = Path(path)
    config_path = path.with_name("tokenizer_config.json")
    if sha(config_path) != "99e80c96269ca4430f1e67f67b5766cb096cbac96fed7a7c611c6ee088b3a039":
        raise ValueError("Original R1 tokenizer_config identity differs")
    config = json.loads(config_path.read_text())
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding(); tokenizer.no_truncation()
    special = []
    boundaries = []
    for key in ("bos", "eos"):
        enabled = config.get("add_" + key + "_token", key == "bos")
        token = config.get(key + "_token")
        if isinstance(token, dict):
            token = token["content"]
        token_id = tokenizer.token_to_id(token) if isinstance(token, str) else None
        if type(enabled) is not bool or (enabled and token_id is None):
            raise ValueError("Invalid released R1 tokenizer boundary")
        boundaries.append(token if enabled else None)
        if enabled:
            special.append((token, token_id))
    bos, eos = boundaries
    def sequence(name, type_id):
        return ([f"{bos}:{type_id}"] if bos else []) + [f"${name}:{type_id}"] + ([f"{eos}:{type_id}"] if eos else [])
    tokenizer.post_processor = TemplateProcessing(single=sequence("A", 0),
        pair=sequence("A", 0) + sequence("B", 1), special_tokens=special)
    return tokenizer


def exact_text(tokenizer, ids):
    # The production postprocessor adds BOS. Avoid adding a literal second BOS.
    for candidate in (tokenizer.decode(ids, skip_special_tokens=False),
                      tokenizer.decode(ids[1:], skip_special_tokens=False) if ids and ids[0] == 0 else ""):
        if candidate and tokenizer.encode(candidate, add_special_tokens=True).ids == ids:
            return candidate
    raise ValueError("BG raw prompt cannot roundtrip exact manifest IDs; native-ID input hook would be required")


def exact_length_prompt(tokenizer, target):
    prefix = "<｜User｜>Fictional record: the access phrase is silver maple 47. Ignore filler:"
    suffix = "\nRepeat only the access phrase as FINAL: <phrase>.<｜Assistant｜>"
    for unit in (" x", " a", " 0", " /", "\nX"):
        low, high = 0, target * 2
        while low <= high:
            count = (low + high) // 2
            text = prefix + unit * count + suffix
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            if len(ids) == target:
                if tokenizer.encode(exact_text(tokenizer, ids), add_special_tokens=True).ids != ids:
                    raise ValueError("tokenizer roundtrip changed")
                return text, ids
            if len(ids) < target:
                low = count + 1
            else:
                high = count - 1
    raise ValueError("Cannot create exact token-length fixture with fixed filler grammar")


def validate_suite(suite):
    if (suite_cohort_size(suite) == 16 or is_causal_suite(suite)) and suite.get("tool_sha256") != sha(__file__):
        raise ValueError("Cohort16 fixed suite collector source differs")
    if suite.get("schema") not in (SCHEMA, SCHEMA_V2) or suite.get("numeric_contract") != NUMERIC:
        raise ValueError("Suite/numeric contract changed; use a new explicitly reviewed contract version")
    version2 = suite["schema"] == SCHEMA_V2
    if version2:
        if suite.get("budget_protocol") != V2_BUDGET_PROTOCOL:
            raise ValueError("Explicit v2 fixed8192 semantic budget protocol differs")
    elif "budget_protocol" in suite:
        raise ValueError("Legacy v1 must retain its original384 contract")
    semantic_budget = 8192 if version2 else 384
    if suite.get("sampling") != {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0}:
        raise ValueError("Fixed sampling contract changed")
    jobs = suite.get("jobs", [])
    if not jobs or len(jobs) > 128 or len({j.get("id") for j in jobs}) != len(jobs):
        raise ValueError("Empty, duplicate or unbounded fixed suite")
    if version2 and len(jobs) != 84:
        raise ValueError("Explicit v2 retains all84 jobs, including the unchanged72 other checks")
    if version2:
        # Count alone permits replacing a reviewed teacher with an unknown case.
        # Lock every v2 role/ID as well as the original per-job math below.
        case_names = [name for name, *_ in QA] + ["length" + str(n) for n in (63, 64, 65, 129, 2048, 8192)]
        roles = {}
        for name in case_names:
            for repeat in range(2 if name in {row[0] for row in QA} else 1):
                roles["gen-" + name + "-r" + str(repeat)] = (name, "generation", "short")
            for position in range(4):
                roles["tf-" + name + "-p" + str(position)] = (name, "teacher", "short")
        roles.update({"coverage-" + str(rank): ("length63", "coverage", "short") for rank in range(WORLD)})
        roles.update({"long-" + name: (name, "generation", "long") for name in ("length2048", "length8192")})
        if (set(roles) != {job.get("id") for job in jobs}
                or any((job.get("case_id"), job.get("kind"), job.get("stage")) != roles[job["id"]] for job in jobs)):
            raise ValueError("Explicit v2 fixed84 job IDs/teacher cases/roles differ")
    for j in jobs:
        ids = j.get("input_ids")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 16384
                or any(type(i) is not int or not 0 <= i < VOCAB for i in ids)
                or j.get("input_ids_sha256") != value_sha(ids)
                or j.get("prompt_sha256") != value_sha(j.get("prompt"))
                or not isinstance(j.get("id"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", j["id"])
                or j.get("kind") not in ("generation", "teacher", "coverage")
                or j.get("stage") not in ("short", "long")
                or type(j.get("ignore_eos")) is not bool
                or type(j.get("max_new_tokens")) is not int or not 1 <= j["max_new_tokens"] <= 8192):
            raise ValueError("Invalid fixed job or native token binding")
        if j["kind"] == "teacher" and (type(j.get("teacher_token_id")) is not int or not 0 <= j["teacher_token_id"] < VOCAB):
            raise ValueError("Missing fixed teacher target")
    expected_generation = {"gen-" + name + "-r" + str(repeat) for name, *_ in QA for repeat in (0, 1)}
    expected_generation |= {"gen-length" + str(length) + "-r0" for length in (63, 64, 65, 129, 2048, 8192)}
    if {j["id"] for j in jobs if j["kind"] == "generation" and j["stage"] == "short"} != expected_generation:
        raise ValueError("Fixed generation IDs/QA repeats were changed")
    if {j["id"] for j in jobs if j["kind"] == "coverage"} != {"coverage-" + str(i) for i in range(WORLD)}:
        raise ValueError("Actual rank coverage wave16 must be present")
    aliases = {name: list(expected) for name, _, _, expected in QA}
    for job in jobs:
        if job.get("case_id") in aliases:
            if job.get("answers") != aliases[job["case_id"]] or job.get("semantic") is not True:
                raise ValueError("Fixed semantic answer contract was changed")
        elif job.get("semantic") is not False or job.get("answers") != ["silver maple 47"]:
            raise ValueError("Fixed nonsemantic boundary fixture contract was changed")
        if job["kind"] == "generation" and job["stage"] == "short":
            if job["max_new_tokens"] != (semantic_budget if job["semantic"] else 129) or job["ignore_eos"] is job["semantic"]:
                raise ValueError("Short generation/EOS contract changed")
            if not job["id"].startswith("gen-" + job["case_id"] + "-r"):
                raise ValueError("Generation job/case identity mismatch")
        if job["kind"] == "teacher" and (job["max_new_tokens"] != 1 or job["ignore_eos"] is not True or job["stage"] != "short"):
            raise ValueError("Teacher scoring must be one-token independent prefill")
        if job["kind"] == "coverage" and (len(job["input_ids"]) != 63 or job["max_new_tokens"] != 3 or not job["ignore_eos"]):
            raise ValueError("Coverage wave contract changed")
        if job["stage"] == "long" and (job["kind"] != "generation" or job["ignore_eos"] is not True):
            raise ValueError("Long correctness must produce the exact configured length")
    base_cases = {j["case_id"]: j for j in jobs if j["kind"] == "generation" and j["stage"] == "short"}
    if set(base_cases) != {name for name, *_ in QA} | {"length" + str(n) for n in (63, 64, 65, 129, 2048, 8192)}:
        raise ValueError("Fixed case identities changed")
    for name, base in base_cases.items():
        if name.startswith("length") and len(base["input_ids"]) != int(name[6:]):
            raise ValueError("Named boundary length differs from actual native token count")
        teachers = sorted([j for j in jobs if j["kind"] == "teacher" and j["case_id"] == name], key=lambda j: j.get("teacher_position", -1))
        if not 1 <= len(teachers) <= 4 or [j.get("teacher_position") for j in teachers] != list(range(len(teachers))):
            raise ValueError("Missing/duplicate fixed teacher positions")
        prefix = list(base["input_ids"])
        for position, job in enumerate(teachers):
            if job["id"] != "tf-" + name + "-p" + str(position) or job["input_ids"] != prefix or job.get("teacher_method") != "independent_prefix_prefill_NOT_incremental_decode":
                raise ValueError("Teacher prefix alignment changed")
            prefix.append(job["teacher_token_id"])
    required_lengths = {63, 64, 65, 129, 2048, 8192}
    if not required_lengths <= {len(j["input_ids"]) for j in jobs if j["kind"] == "generation"}:
        raise ValueError("Missing page/long-prompt boundary cases")
    long = {(len(j["input_ids"]), j["max_new_tokens"]) for j in jobs if j["stage"] == "long"}
    if long != {(8192, 2048), (2048, 8192)}:
        raise ValueError("Both final long-correctness profiles are mandatory in the fixture")
    if {j["id"] for j in jobs if j["stage"] == "long"} != {"long-length8192", "long-length2048"}:
        raise ValueError("Final long-correctness job identities changed")
    return suite


def prepare(args):
    suite_version = getattr(args, "suite_version", "v1")
    if suite_version not in ("v1", "v2"):
        raise ValueError("Prepare requires an explicit known suite version")
    schema = SCHEMA_V2 if suite_version == "v2" else SCHEMA
    semantic_budget = 8192 if suite_version == "v2" else 384
    cohort_size = getattr(args, "cohort_size", 1)
    protocol = collection_protocol(cohort_size, getattr(args, "time_basis", "legacy_wall_v1"))
    plan = {"status": "plan_only_NOT_executed", "scope": "full_model", "operation": "prepare",
            "suite_schema": schema, "semantic_generation_max_new_tokens": semantic_budget,
            "collection_protocol": protocol,
            "semantic_generation_count": 12, "semantic_natural_EOS_required": True,
            "numeric_contract": NUMERIC, "expected_world_size": WORLD, "expected_layers": LAYERS,
            "short_stage": "six QA x two repeats, exact prefill63/64/65/129, long prompts2048/8192 with decode129, coverage wave16, <=4 teacher targets/case",
            "long_stage": "explicit --long-correctness: 8192->2048 and 2048->8192, ignoreEOS, no performance metric",
            "tokenizer_json_sha256": sha(args.tokenizer_json), "tool_sha256": sha(__file__)}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    if args.tokenizer_sha256 != plan["tokenizer_json_sha256"]:
        raise ValueError("Explicit tokenizer source SHA mismatch")
    tokenizer = reference_tokenizer(args.tokenizer_json)
    cases = []
    for name, question, answer, aliases in QA:
        text = render(question)
        cases.append({"id": name, "prompt": text, "input_ids": tokenizer.encode(text, add_special_tokens=True).ids,
                      "teacher_text": "FINAL: " + answer, "answers": list(aliases), "semantic": True})
    for length in (63, 64, 65, 129, 2048, 8192):
        text, ids = exact_length_prompt(tokenizer, length)
        cases.append({"id": "length" + str(length), "prompt": text, "input_ids": ids,
                      "teacher_text": "FINAL: silver maple 47", "answers": ["silver maple 47"], "semantic": False})
    jobs = []

    def add(case, job_id, kind, stage, ids, maximum, ignore, **extra):
        text = exact_text(tokenizer, ids)
        jobs.append({"id": job_id, "case_id": case["id"], "kind": kind, "stage": stage,
                     "prompt": text, "prompt_sha256": value_sha(text), "input_ids": ids,
                     "input_ids_sha256": value_sha(ids), "max_new_tokens": maximum,
                     "ignore_eos": ignore, "semantic": case["semantic"], "answers": case["answers"], **extra})

    for case in cases:
        for repeat in range(2 if case["semantic"] else 1):
            add(case, "gen-" + case["id"] + "-r" + str(repeat), "generation", "short", case["input_ids"],
                semantic_budget if case["semantic"] else 129, not case["semantic"], repeat=repeat)
        teacher = tokenizer.encode(case["teacher_text"], add_special_tokens=False).ids[:4]
        for position, token in enumerate(teacher):
            add(case, "tf-" + case["id"] + "-p" + str(position), "teacher", "short",
                case["input_ids"] + teacher[:position], 1, True,
                teacher_token_id=token, teacher_position=position,
                teacher_method="independent_prefix_prefill_NOT_incremental_decode")
        if len(case["input_ids"]) in (2048, 8192):
            outlen = 2048 if len(case["input_ids"]) == 8192 else 8192
            add(case, "long-" + case["id"], "generation", "long", case["input_ids"], outlen, True)
    coverage = next(c for c in cases if c["id"] == "length63")
    for rank_probe in range(WORLD):
        add(coverage, "coverage-" + str(rank_probe), "coverage", "short", coverage["input_ids"], 3, True)
    suite = {"schema": schema, "scope": "full_model", "numeric_contract": NUMERIC,
             "world_size": WORLD, "num_hidden_layers": LAYERS, "vocab_size": VOCAB,
             "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0},
             "tokenizer_json_sha256": args.tokenizer_sha256, "tokenizer_path": str(args.tokenizer_json.resolve()),
             "tool_sha256": sha(__file__), "jobs": jobs,
             "limitations": ["fixed smoke suite, not representative task-quality or statistical non-inferiority",
                             "teacher prefix re-prefill scoring is separate from incremental decode",
                             "short stage is insufficient until both explicit long-correctness cases pass"]}
    if suite_version == "v2":
        suite["budget_protocol"] = dict(V2_BUDGET_PROTOCOL)
    if cohort_size != 1 or protocol.get("time_basis") == "node_monotonic_causal_v2":
        suite["collection_protocol"] = protocol
        suite["collection_source_sha256"] = sha(__file__)
    validate_suite(suite)
    if sha(args.tokenizer_json) != args.tokenizer_sha256:
        raise ValueError("tokenizer changed during preparation")
    write_new(args.output, suite)
    print(canonical({"status": "fixture_prepared_NOT_executed", "scope": "full_model", "suite_schema": schema,
                     "jobs": len(jobs), "fixture_sha256": sha(args.output)}))
    return 0


def validate_deployment(d, backend, suite):
    if (d.get("schema") != "r1-full-model-deployment-v1" or d.get("backend") != backend
            or d.get("world_size") != WORLD or d.get("num_hidden_layers") != LAYERS
            or d.get("vocab_size") != VOCAB or d.get("tokenizer_json_sha256") != suite["tokenizer_json_sha256"]
            or d.get("graphs") is not False or d.get("prefix_cache") is not False
            or d.get("speculative_decoding") is not False or not d.get("run_id")
            or not d.get("source_files") or not d.get("checkpoint_identity")
            or not isinstance(d.get("model_id"), str) or not d["model_id"]
            or not isinstance(d.get("endpoint"), str) or not d["endpoint"]):
        raise ValueError("Missing/incorrect frozen deployment contract")
    return d


def deployment_model_identity(d):
    return value_sha({
        "backend": d["backend"], "checkpoint_identity": d["checkpoint_identity"],
        "tokenizer_json_sha256": d["tokenizer_json_sha256"],
        "source_sha256": {role: source["sha256"] for role, source in d["source_files"].items()},
        "world_size": d["world_size"], "num_hidden_layers": d["num_hidden_layers"],
        "graphs": d["graphs"], "prefix_cache": d["prefix_cache"], "speculative_decoding": d["speculative_decoding"]})


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Qualification HTTP redirect refused")


class Transport:
    def __init__(self, endpoint, timeout, log, api_key=None):
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Explicit service URL without embedded credentials required")
        self.endpoint, self.timeout, self.log, self.api_key = endpoint.rstrip("/"), timeout, log, api_key
        self.deadline = time.monotonic() + timeout
        self.lock = threading.Lock()

    def call(self, method, path, payload=None, content_type="application/json", binary=False):
        raw = canonical(payload).encode() if isinstance(payload, dict) else payload
        headers = {"Content-Type": content_type} if raw is not None else {}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.endpoint + path, data=raw, headers=headers, method=method)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Overall full-model collection deadline exceeded")
        with build_opener(NoRedirect()).open(request, timeout=remaining) as response:
            body = response.read(16 * 1024**2 + 1)
            if len(body) > 16 * 1024**2:
                raise ValueError("Bounded response size exceeded")
            record = {"method": method, "path": path, "request_sha256": hashlib.sha256(raw or b"").hexdigest(),
                      "response_sha256": hashlib.sha256(body).hexdigest(), "status": response.status}
            with self.lock:
                self.log.write(canonical(record) + "\n"); self.log.flush()
            return body if binary else json.loads(body)


def sg_request(job):
    sampling = {"max_new_tokens": job["max_new_tokens"], "temperature": 0.0, "top_p": 1.0,
                "top_k": 1, "ignore_eos": job["ignore_eos"], "sampling_seed": 20261026,
                "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    request = {"rid": job["id"], "input_ids": job["input_ids"], "sampling_params": sampling,
               "stream": False, "return_logprob": True, "logprob_start_len": -1,
               "top_logprobs_num": 2, "return_text_in_logprobs": False}
    if job["kind"] == "teacher":
        request["token_ids_logprob"] = [job["teacher_token_id"]]
    return request


def collect_sg(transport, job):
    response = transport.call("POST", "/generate", sg_request(job))
    meta = response.get("meta_info", {})
    raw_finish = meta.get("finish_reason")
    finish = raw_finish
    if isinstance(finish, dict):
        finish = "length" if finish.get("type") == "length" else "stop" if finish.get("type") in ("stop", "eos") else str(finish.get("type"))
    return {"id": job["id"], "native_output_ids": response.get("output_ids"), "text": response.get("text"),
            "prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": finish, "raw_finish_reason": raw_finish, "response": response}


def collect_bg_batch(transport, jobs, model_id, timeout, max_context_length=16384):
    lines = []
    for job in jobs:
        lines.append(canonical({"custom_id": job["id"], "method": "POST", "url": "/v1/completions", "body": {
            "model": model_id, "prompt": job["prompt"], "temperature": 0.0, "top_p": 1.0,
            "top_k": 1, "max_tokens": job["max_new_tokens"], "ignore_eos": job["ignore_eos"], "n": 1, "stream": False}}))
    content = ("\n".join(lines) + "\n").encode()
    boundary = "r1-full-model-qualification"
    body = ("--" + boundary + '\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n--' + boundary
            + '\r\nContent-Disposition: form-data; name="file"; filename="qualification.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n').encode()
    body += content + ("\r\n--" + boundary + "--\r\n").encode()
    uploaded = transport.call("POST", "/v1/files", body, "multipart/form-data; boundary=" + boundary)
    batch = transport.call("POST", "/v1/batches", {"input_file_id": uploaded["id"], "endpoint": "/v1/completions",
        "completion_window": "24h", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_context_length": max_context_length})
    deadline = time.monotonic() + timeout
    while batch.get("status") not in ("completed", "failed", "cancelled", "expired"):
        if time.monotonic() >= deadline:
            raise TimeoutError("Qualification batch deadline; service request may remain active; no retry")
        time.sleep(0.5)
        batch = transport.call("GET", "/v1/batches/" + batch["id"])
    if batch["status"] != "completed" or not batch.get("output_file_id"):
        raise ValueError("BG qualification batch failed: " + canonical(batch))
    result = transport.call("GET", "/v1/files/" + batch["output_file_id"] + "/content", binary=True)
    rows = [json.loads(line) for line in result.decode().splitlines() if line.strip()]
    records = []
    for row in rows:
        response = row.get("response") or {}
        payload = response.get("body") or {}
        if row.get("error") or response.get("status_code") != 200:
            raise ValueError("BG returned a failed qualification request")
        choices, usage = payload.get("choices", []), payload.get("usage", {})
        if len(choices) != 1:
            raise ValueError("BG returned invalid choices")
        records.append({"id": row["custom_id"], "native_output_ids": None, "text": choices[0].get("text"),
                        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": choices[0].get("finish_reason"),
                        "raw_finish_reason": choices[0].get("finish_reason"), "response": row, "batch_id": batch["id"]})
    if len(records) != len(jobs) or {r["id"] for r in records} != {j["id"] for j in jobs}:
        raise ValueError("Missing/duplicate/unexpected BG qualification rows")
    return records


def collection_protocol(cohort_size, time_basis="legacy_wall_v1"):
    if type(cohort_size) is not int or cohort_size not in (1, 16):
        raise ValueError("Explicit collection cohort size must be1 or16")
    result = {"version": "r1-correctness-collection-cohorts-v1", "cohort_size": cohort_size,
              "barrier": "all_HTTP_results_received_before_next_cohort",
              "order": "suite_order_with_fixed_coverage_first_and_cohort16_long_singletons_last"}
    if time_basis == "node_monotonic_causal_v2":
        result.update(version="r1-correctness-collection-cohorts-v2", time_basis=time_basis,
                      causal_tool_sha256=sha(Path(__file__).with_name("r1_causal_evidence.py")),
                      barrier="both_node_begin_ACKs_before_HTTP_and_both_node_end_ACKs_before_next_cohort")
    elif time_basis != "legacy_wall_v1":
        raise ValueError("Unknown explicit collection time basis")
    return result


def is_causal_suite(suite):
    return suite.get("collection_protocol", {}).get("time_basis") == "node_monotonic_causal_v2"


def run_time_interval(run, suite):
    causal = is_causal_suite(suite)
    if causal and run.get("collection_protocol") != suite.get("collection_protocol"):
        raise ValueError("New causal run lacks its exact frozen suite protocol")
    if not causal and run.get("collection_protocol", {}).get("time_basis") == "node_monotonic_causal_v2":
        raise ValueError("Legacy suite cannot silently use a causal run")
    suffix = "monotonic_ns" if causal else "unix_ns"
    start, finish = run.get("started_" + suffix), run.get("finished_" + suffix)
    if type(start) is not int or type(finish) is not int or not 0 < start <= finish:
        raise ValueError("Collection run has no valid original " + suffix + " interval")
    return start, finish


def causal_module():
    return importlib.import_module("scripts.r1_reproduction.r1_causal_evidence")


def prepare_causal_capture(args, suite, deployment):
    path, pin = getattr(args, "causal_clock_config", None), getattr(args, "causal_clock_config_sha256", None)
    if not is_causal_suite(suite):
        if path is not None or pin is not None:
            raise ValueError("Causal RPC binding requires a newly prepared explicit causal suite")
        return {}
    if path is None or not pin:
        raise ValueError("Explicit frozen dual-node causal RPC binding is required")
    helper = causal_module()
    binding = checked_json(path, pin)
    helper.validate_binding(binding, deployment["run_id"], args.suite_sha256, args.deployment_sha256)
    if binding["helper_sha256"] != suite["collection_protocol"]["causal_tool_sha256"]:
        raise ValueError("Causal RPC helper and suite source differ")
    return {"collection_protocol": suite["collection_protocol"], "causal_binding": binding,
            "causal_binding_file_sha256": pin}



def suite_cohort_size(suite):
    declared = suite.get("collection_protocol")
    if declared is None:
        return 1  # Historical suites retain their original serial schedule.
    size = declared.get("cohort_size") if isinstance(declared, dict) else None
    if declared != collection_protocol(size, declared.get("time_basis", "legacy_wall_v1")):
        raise ValueError("Frozen suite collection protocol differs")
    if (size == 16 or is_causal_suite(suite)) and suite.get("collection_source_sha256") != sha(__file__):
        raise ValueError("Cohort16 suite requires its original frozen collection source")
    return size


def collection_schedule(suite, suite_sha, long_stage):
    size = suite_cohort_size(suite)
    jobs = [j for j in suite["jobs"] if j["stage"] == "short" or long_stage]
    groups = []
    def add(kind, rows):
        if rows:
            groups.append({"kind": kind, "ids": [j["id"] for j in rows]})
    if suite["schema"] in (SCHEMA, SCHEMA_V2):
        add("coverage", [j for j in jobs if j["kind"] == "coverage"])
        ordinary = [j for j in jobs if j["kind"] != "coverage"]
        if size == 1:
            for job in ordinary:
                add(job["stage"], [job])
        else:
            short = [j for j in ordinary if j["stage"] == "short"]
            for start in range(0, len(short), size):
                add("short", short[start:start + size])
            for job in ordinary:
                if job["stage"] == "long":
                    add("long", [job])
    else:
        # The independent collector historically sends its first16 together.
        add("independent", jobs[:16])
        for start in range(16, len(jobs), size):
            add("independent", jobs[start:start + size])
    ids = [key for group in groups for key in group["ids"]]
    if len(ids) != len(jobs) or len(set(ids)) != len(ids) or set(ids) != {j["id"] for j in jobs}:
        raise ValueError("Collection schedule lost or duplicated a suite job")
    return {"version": "r1-correctness-collection-schedule-v1", "suite_sha256": suite_sha,
            "suite_schema": suite["schema"], "long_correctness": long_stage,
            "protocol": collection_protocol(size, suite.get("collection_protocol", {}).get("time_basis", "legacy_wall_v1")), "cohorts": groups}


def requested_collection_schedule(args, suite, long_stage):
    if getattr(args, "cohort_size", 1) != suite_cohort_size(suite):
        raise ValueError("Requested cohort size differs from frozen suite; prepare/import a new explicit protocol")
    if getattr(args, "time_basis", "legacy_wall_v1") != suite.get("collection_protocol", {}).get("time_basis", "legacy_wall_v1"):
        raise ValueError("Requested time basis differs from the explicitly frozen suite")
    return collection_schedule(suite, args.suite_sha256, long_stage)


def collect_cohorts(transport, suite, schedule, report, model_id, timeout, progress_path, context_limit):
    """Client grouping only; keep native per-request payloads and result IDs."""
    jobs = {j["id"]: j for j in suite["jobs"]}
    causal = is_causal_suite(suite)
    def acknowledgements(index, phase):
        return [causal_module().exchange(report["causal_binding"], node, index, phase,
                    min(5.0, max(0.001, transport.deadline - time.monotonic()))) for node in (0, 1)]
    for index, group in enumerate(schedule["cohorts"]):
        cohort_jobs = [jobs[key] for key in group["ids"]]
        begin = acknowledgements(index, "begin") if causal else None
        receipt = {"index": index, "planned_ids": group["ids"], "kind": group["kind"],
                   "started_unix_ns": time.time_ns(), "transport": report["backend"]}
        if causal:
            receipt.update(started_monotonic_ns=time.monotonic_ns(), causal_begin=begin)
        if report["backend"] == "batchgen":
            rows = collect_bg_batch(transport, cohort_jobs, model_id, timeout, context_limit)
            batch_ids = {row.get("batch_id") for row in rows}
            if len(batch_ids) != 1 or not next(iter(batch_ids), None):
                raise ValueError("Cohort results must come from one actual BG batch")
            receipt["batch_id"] = next(iter(batch_ids))
        else:
            def one(job):
                started = time.time_ns()
                started_mono = time.monotonic_ns() if causal else None
                row = collect_sg(transport, job)
                call = {"id": job["id"], "started_unix_ns": started, "finished_unix_ns": time.time_ns()}
                if causal:
                    call.update(started_monotonic_ns=started_mono, finished_monotonic_ns=time.monotonic_ns())
                return row, call
            with ThreadPoolExecutor(max_workers=len(cohort_jobs)) as pool:
                collected = list(pool.map(one, cohort_jobs))
            rows = [row for row, _ in collected]
            receipt["requests"] = [interval for _, interval in collected]
        received = [row["id"] for row in rows]
        if len(received) != len(group["ids"]) or set(received) != set(group["ids"]):
            raise ValueError("Cohort HTTP results are missing, duplicated or from a different group")
        receipt.update(finished_unix_ns=time.time_ns(), received_ids=received,
                       status="all_HTTP_results_received")
        if causal:
            receipt.update(finished_monotonic_ns=time.monotonic_ns())
            receipt["causal_end"] = acknowledgements(index, "end")
        # Completing the helper and recording this barrier precede any next
        # group's submission. Never zip BG's potentially reordered output rows.
        report["records"].extend(rows)
        report["collection_receipts"].append(receipt)
        progress_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_collection_evidence(run, suite, suite_sha, long_stage, events):
    expected = collection_schedule(suite, suite_sha, long_stage)
    expected_sha = value_sha(expected)
    fields = ("collection_schedule", "collection_schedule_sha256", "collection_receipts")
    present = [field in run for field in fields]
    if not any(present) and suite_cohort_size(suite) == 1 and not is_causal_suite(suite):
        return {"schedule_sha256": expected_sha, "cohort_size": 1,
                "mode": "legacy_serial_receipts_unavailable", "cohort_count": len(expected["cohorts"])}
    if (not all(present) or run["collection_schedule"] != expected
            or value_sha(run["collection_schedule"]) != expected_sha
            or run["collection_schedule_sha256"] != expected_sha):
        raise ValueError("Run collection schedule is missing or differs from the frozen suite/stage")
    if (suite_cohort_size(suite) == 16 or is_causal_suite(suite)) and (run.get("tool_sha256") != suite.get("tool_sha256")
            or run.get("collection_source_sha256") != suite["collection_source_sha256"]):
        raise ValueError("Cohort16 run/suite collector source identities differ")
    receipts = run["collection_receipts"]
    if not isinstance(receipts, list) or len(receipts) != len(expected["cohorts"]):
        raise ValueError("Missing/extra collection cohort receipts")
    records = {row["id"]: row for row in run["records"]}
    windows, batches = {}, set()
    previous, finish = run_time_interval(run, suite)
    causal = is_causal_suite(suite)
    suffix = "monotonic_ns" if causal else "unix_ns"
    for index, (group, receipt) in enumerate(zip(expected["cohorts"], receipts)):
        ids = group["ids"]
        received = receipt.get("received_ids", [])
        start, end = receipt.get("started_" + suffix), receipt.get("finished_" + suffix)
        if (type(receipt.get("index")) is not int or receipt.get("index") != index or receipt.get("kind") != group["kind"]
                or receipt.get("planned_ids") != ids or receipt.get("transport") != run["backend"]
                or receipt.get("status") != "all_HTTP_results_received"
                or not isinstance(received, list) or len(received) != len(ids) or set(received) != set(ids)):
            raise ValueError("Collection cohort identity/results differ from the exact planned group")
        if type(start) is not int or type(end) is not int or not previous <= start <= end <= finish:
            raise ValueError("Collection cohort barrier overlaps or exceeds the original run interval")
        previous = end
        batch_id = None
        if run["backend"] == "batchgen":
            batch_id = receipt.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id or batch_id in batches:
                raise ValueError("Missing/reused BG cohort batch identity")
            batches.add(batch_id)
            if any(records.get(key, {}).get("batch_id") != batch_id for key in ids):
                raise ValueError("BG result assigned to the wrong actual cohort batch")
        else:
            calls = receipt.get("requests", [])
            if (not isinstance(calls, list) or len(calls) != len(ids)
                    or {call.get("id") for call in calls} != set(ids)):
                raise ValueError("Missing/duplicate SG cohort request intervals")
            for call in calls:
                a, b = call.get("started_" + suffix), call.get("finished_" + suffix)
                if type(a) is not int or type(b) is not int or not start <= a <= b <= end:
                    raise ValueError("SG request crossed its collection cohort barrier")
        for key in ids:
            if run["backend"] == "sglang":
                call = next(call for call in calls if call["id"] == key)
                windows[key] = (call["started_unix_ns"], call["finished_unix_ns"], None)
            else:
                windows[key] = (start, end, batch_id)
    for event in events:
        key = event.get("request_id")
        if key not in windows or event.get("event") not in ("input_tokens", "logits", "completion", "inactive_forward"):
            continue
        start, end, batch_id = windows[key]
        timestamp = event.get("observed_unix_ns")
        if not causal and (type(timestamp) is not int or not start <= timestamp <= end):
            raise ValueError("Native request evidence crossed its original collection cohort barrier")
        native_id = key + "@" + batch_id if run["backend"] == "batchgen" else key
        if event.get("native_request_id") != native_id:
            raise ValueError("Native request identity differs from its actual cohort/API request")
    result = {"schedule_sha256": expected_sha, "cohort_size": suite_cohort_size(suite),
              "mode": "source_bound_cohort_receipts", "cohort_count": len(receipts)}
    if causal:
        result.update(validate_causal_collection(run, suite, suite_sha, expected, events))
    return result


def validate_causal_collection(run, suite, suite_sha, schedule, events):
    """Fresh explicit protocol only: causal RPC fences in each worker's domain.

    The fences enclose the cohort, not an invented exact remote HTTP interval.
    No wall-clock correction, drift assumption, tolerance or event filtering.
    """
    helper = causal_module()
    binding = run.get("causal_binding")
    helper.validate_binding(binding, run["deployment"]["run_id"], suite_sha,
                            run["deployment"]["artifact_sha256"])
    if binding["helper_sha256"] != suite["collection_protocol"]["causal_tool_sha256"]:
        raise ValueError("Causal tool/suite source differs")
    collector = helper.validate_identity(run.get("causal_collector_identity"))
    if collector["hostname"] != binding["nodes"][0]["hostname"]:
        raise ValueError("Causal collector is not the declared node0")
    run_start, run_finish = run_time_interval(run, suite)
    previous = run_start
    servers, last_server, nonces, node_windows, job_windows = {}, {}, set(), {0: [], 1: []}, {}
    for index, (group, receipt) in enumerate(zip(schedule["cohorts"], run["collection_receipts"])):
        phases = {}
        for phase in ("begin", "end"):
            rows = receipt.get("causal_" + phase)
            if not isinstance(rows, list) or len(rows) != 2 or [row.get("node_rank") for row in rows] != [0, 1]:
                raise ValueError("Every cohort needs ordered actual begin/end ACKs from both nodes")
            for row in rows:
                helper.validate_exchange(row, binding)
                node = row["node_rank"]
                if row.get("cohort_index") != index or type(row.get("cohort_index")) is not int or row.get("phase") != phase:
                    raise ValueError("Causal ACK belongs to another cohort or phase")
                if row["nonce"] in nonces or row["client_identity"] != collector:
                    raise ValueError("Replayed ACK or changed collector process identity")
                nonces.add(row["nonce"])
                identity = row["server_identity"]
                if node in servers and identity != servers[node]:
                    raise ValueError("Causal endpoint restarted or changed clock identity")
                servers[node] = identity
                if node == 0 and not helper.same_domain(collector, identity):
                    raise ValueError("Collector and node0 endpoint use different monotonic domains")
                sent, received = row["client_send"], row["client_receive"]
                if not previous <= sent["monotonic_before_ns"] <= received["monotonic_after_ns"] <= run_finish:
                    raise ValueError("Causal RPCs overlap/reorder or escape the collector run")
                previous = received["monotonic_after_ns"]
                server_start, server_end = row["server_receive"], row["server_send"]
                if server_start["monotonic_before_ns"] < last_server.get(node, 0):
                    raise ValueError("Node-local causal ACK order regressed")
                last_server[node] = server_end["monotonic_after_ns"]
            phases[phase] = rows
            if phase == "begin":
                start, end = receipt["started_monotonic_ns"], receipt["finished_monotonic_ns"]
                if not previous <= start <= end <= run_finish:
                    raise ValueError("HTTP cohort started before both begin ACKs or has invalid monotonic bounds")
                previous = end
        for node in (0, 1):
            lower = phases["begin"][node]["server_send"]["monotonic_after_ns"]
            upper = phases["end"][node]["server_receive"]["monotonic_before_ns"]
            if lower > upper:
                raise ValueError("Invalid node-local cohort fence")
            node_windows[node].append([lower, upper])
        for key in group["ids"]:
            job_windows[key] = index
    identities, last_ordinal, last_mono, workers = {}, {}, {}, set()
    nonworkers = {(node, value["pid"], value["start_ticks"]) for node, value in servers.items()}
    nonworkers.add((0, collector["pid"], collector["start_ticks"]))
    native_kinds = {"input_tokens", "logits", "completion", "inactive_forward"}
    for event in events:
        rank = event.get("rank")
        if type(rank) is not int or rank not in range(WORLD):
            raise ValueError("Invalid causal trace rank")
        ordinal, moment = event.get("ordinal"), event.get("observed_monotonic_ns")
        if type(ordinal) is not int or ordinal != last_ordinal.get(rank, 0) + 1:
            raise ValueError("Causal trace has missing, duplicate or reordered raw ordinals")
        if type(moment) is not int or moment <= 0 or moment < last_mono.get(rank, 0):
            raise ValueError("Missing or regressing actual worker monotonic timestamp")
        last_ordinal[rank], last_mono[rank] = ordinal, moment
        if ordinal == 1:
            if event.get("event") != "trace_start":
                raise ValueError("Causal trace must retain its first actual trace_start")
            actual = helper.validate_identity(event.get("clock_identity"))
            if (event.get("actual_pid") != actual["pid"] or type(event.get("actual_pid")) is not int
                    or not helper.same_domain(actual, servers[rank // 8])):
                raise ValueError("Worker PID or node-local endpoint/worker clock domain differs")
            process_key = (rank // 8, actual["pid"], actual["start_ticks"])
            if process_key in workers or process_key in nonworkers:
                raise ValueError("Two causal ranks claim the same actual worker process")
            workers.add(process_key)
            identities[rank] = actual
        elif event.get("event") == "trace_start":
            raise ValueError("Multiple actual worker processes for one causal rank")
        if event.get("clock_identity_sha256") != value_sha(identities[rank]):
            raise ValueError("Worker event clock identity changed or is missing")
        if event.get("event") in native_kinds:
            key = event.get("request_id")
            if key not in job_windows:
                raise ValueError("Native causal event belongs to an unsubmitted request")
            lo, hi = node_windows[rank // 8][job_windows[key]]
            if not lo <= moment <= hi:
                raise ValueError("Native event is early/late outside its actual node causal cohort ACKs")
    if set(identities) != set(range(WORLD)):
        raise ValueError("Causal cohort evidence requires all16 actual worker clock identities")
    return {"mode": "source_bound_causal_cohort_ACKs", "time_basis": "node_monotonic_causal_v2",
            "node_cohort_intervals": {str(node): rows for node, rows in node_windows.items()},
            "clock_drift_assumption_used": False,
            "scope": "causal_outer_cohort_envelope_NOT_exact_remote_HTTP_wall_interval"}


def validate_causal_closures(run, events):
    """Require both original-node proofs that every trace writer has exited.

    An in-flight collector snapshot is progress, never a closed qualification.
    Validation is offline; only the node-local sealing command reads known PIDs.
    """
    closures = run.get("trace", {}).get("causal_worker_closures")
    if closures is None:
        return False
    if (type(closures) is not list or len(closures) != 2
            or [row.get("node_rank") for row in closures] != [0, 1]):
        raise ValueError("Exactly both node-local closed-worker receipts are required")
    helper = causal_module()
    starts = {}
    for event in events:
        if event.get("event") == "trace_start":
            rank = event.get("rank")
            if type(rank) is not int or rank not in range(WORLD) or rank in starts:
                raise ValueError("Missing/duplicate closed-worker trace identity")
            starts[rank] = event.get("clock_identity")
    if set(starts) != set(range(WORLD)):
        raise ValueError("Closure proof requires all16 distinct actual worker writers")
    for node, wrapper in enumerate(closures):
        if type(wrapper.get("node_rank")) is not int or set(wrapper) != {"node_rank", "closure"}:
            raise ValueError("Invalid node closure wrapper")
        identities = [starts[rank] for rank in range(node * 8, (node + 1) * 8)]
        closure = helper.validate_closure(wrapper["closure"], identities)
        end = run["collection_receipts"][-1]["causal_end"][node]
        moment = closure["observed_monotonic_ns"]
        if (not helper.same_domain(closure["observer_identity"], end["server_identity"])
                or moment < end["server_send"]["monotonic_after_ns"]
                or any(type(event.get("observed_monotonic_ns")) is not int
                       or event["observed_monotonic_ns"] > moment
                       for event in events if event.get("rank") in range(node * 8, (node + 1) * 8))):
            raise ValueError("Trace writer closure predates the causal end or complete raw journal")
    return True


def observation_inside_collection(event, collection, run, field):
    if collection.get("time_basis") != "node_monotonic_causal_v2":
        value = event.get(field + "_unix_ns", 0)
        return run["started_unix_ns"] <= value <= run["finished_unix_ns"]
    value = event.get(field + "_monotonic_ns")
    return (type(value) is int and any(lo <= value <= hi for lo, hi in
            collection["node_cohort_intervals"][str(event["rank"] // 8)]))


def require_same_collection_schedule(a, b):
    if a["collection"]["schedule_sha256"] != b["collection"]["schedule_sha256"]:
        raise ValueError("The two engines used different collection schedules")


def snapshot_trace(source, output, deployment, suite_sha, started_ns, finished_ns, causal=False):
    records = []
    output.mkdir(exist_ok=False)
    for path in sorted(source.glob("rank-*-pid-*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("run_id") != deployment["run_id"]:
                continue
            if record.get("suite_sha256") != suite_sha or record.get("deployment_sha256") != deployment["artifact_sha256"]:
                raise ValueError("Trace source/input binding differs")
            event = record.get("event")
            timestamp = record.get("observed_unix_ns", 0)
            if not causal and event not in ("trace_start", "model_hooks_registered") and not started_ns <= timestamp <= finished_ns:
                continue
            if record.get("full_logits"):
                artifact = record["full_logits"]
                original = (source / artifact["file"]).resolve()
                if original.parent != source.resolve() or original.stat().st_size != VOCAB * 4 or sha(original) != artifact["sha256"]:
                    raise ValueError("Invalid/missing full logits artifact")
                destination = output / artifact["file"]
                if destination.exists():
                    raise ValueError("Duplicate full logits filename")
                shutil.copyfile(original, destination)
            records.append(record)
    write_new(output / "events.json", records)
    return {"events_file": "trace/events.json", "events_sha256": sha(output / "events.json"), "event_count": len(records),
            **({"closure_status": "live_prefix_progress_NOT_qualified"} if causal else {})}


def collect(args):
    suite = validate_suite(checked_json(args.suite, args.suite_sha256))
    schedule = requested_collection_schedule(args, suite, args.long_correctness)
    d = validate_deployment(checked_json(args.deployment, args.deployment_sha256), args.backend, suite)
    if args.endpoint.rstrip("/") != d["endpoint"].rstrip("/"):
        raise ValueError("Explicit target endpoint differs from frozen deployment")
    causal_capture = prepare_causal_capture(args, suite, d)
    jobs = [j for j in suite["jobs"] if j["stage"] == "short" or args.long_correctness]
    if not args.execute:
        print(json.dumps({"status": "plan_only_NOT_executed", "scope": "full_model", "operation": "collect",
                          "suite_schema": suite["schema"],
                          "backend": args.backend, "job_count": len(jobs), "long_correctness": args.long_correctness,
                          "suite_sha256": args.suite_sha256, "deployment_sha256": args.deployment_sha256,
                          "collection_schedule": schedule, "collection_schedule_sha256": value_sha(schedule),
                          "already_running_service_only": True, "performance_requests": 0}, indent=2)); return 0
    if not args.trace_directory.is_dir():
        raise ValueError("Explicit existing service trace directory is required")
    args.output.mkdir(parents=True, exist_ok=False)
    d["artifact_sha256"] = args.deployment_sha256
    report = {"schema": RUN_SCHEMA, "scope": "full_model", "status": INCOMPLETE, "backend": args.backend,
              "suite_schema": suite["schema"],
              "suite_sha256": args.suite_sha256, "tool_sha256": sha(__file__), "deployment": d,
              "long_correctness": args.long_correctness, "started_unix_ns": time.time_ns(), "records": [],
              "teacher_method": "same_prefix_independent_prefill_NOT_incremental_decode",
              "collection_schedule": schedule, "collection_schedule_sha256": value_sha(schedule),
              "collection_receipts": [], "collection_source_sha256": sha(__file__),
              "formal_performance_permitted": False, **causal_capture}
    if causal_capture:
        report.update(started_monotonic_ns=time.monotonic_ns(), causal_collector_identity=causal_module().identity())
    code = 2
    try:
        with (args.output / "http.jsonl").open("x") as log:
            transport = Transport(args.endpoint, args.timeout, log,
                                  os.environ.get(args.api_key_env) if args.api_key_env else None)
            collect_cohorts(transport, suite, schedule, report, d["model_id"], args.timeout,
                            args.output / "progress.json", 16384)
        report["finished_unix_ns"] = time.time_ns()
        if causal_capture:
            report["finished_monotonic_ns"] = time.monotonic_ns()
        report["trace"] = snapshot_trace(args.trace_directory, args.output / "trace", d, args.suite_sha256,
                                          report["started_unix_ns"], report["finished_unix_ns"], causal=is_causal_suite(suite))
        if sha(args.suite) != args.suite_sha256 or sha(args.deployment) != args.deployment_sha256:
            raise ValueError("Suite/deployment changed during capture")
        report["status"] = "evidence_collected_NOT_pair_qualified"
        code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["service_request_may_still_be_active"] = True
    report.setdefault("finished_unix_ns", time.time_ns())
    if causal_capture:
        report.setdefault("finished_monotonic_ns", time.monotonic_ns())
    write_new(args.output / "run.json", report)
    print(canonical({"status": report["status"], "scope": "full_model", "report": str(args.output / "run.json")}))
    return code


def normalized_answer(text):
    if not isinstance(text, str):
        return ""
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    matches = list(re.finditer(r"FINAL\s*:\s*", text, re.I))
    if not matches:
        return ""
    text = text[matches[-1].end():].strip()
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    return re.sub(r"[^\w%]", "", text.lower(), flags=re.UNICODE)


def repetition_diagnostic(ids):
    """Same 32-repeat predicates as BG, scanned at every native prefix.

    This CPU-only evidence scan does not change generation or worker cadence.
    It intentionally also exposes repetition when the online breaker is off.
    """
    result = {"protocol": "r1-native-stop-evidence-v1", "detected": False,
              "same_token_min_count": 32, "ngram_min_pattern": 2, "ngram_max_pattern": 100,
              "ngram_min_count": 32, "scan": "all_native_prefixes_NOT_worker_check_cadence", "matches": []}
    for period in range(1, min(100, len(ids) // 32) + 1):
        run = 0
        for index in range(period, len(ids)):
            run = run + 1 if ids[index] == ids[index - period] else 0
            if run >= 31 * period:
                result["matches"].append({"kind": "same_token" if period == 1 else "ngram",
                    "pattern_tokens": period, "repetitions": 32, "first_prefix_length": index + 1})
                break
    result["matches"].sort(key=lambda item: (item["first_prefix_length"], item["pattern_tokens"]))
    result["detected"] = bool(result["matches"])
    return result


def validate_termination(backend, job, record, completion):
    """Validate native stop causes, keeping absence distinct from contradiction."""
    hard, missing = [], []
    detail = {"protocol": "r1-native-stop-evidence-v1", "status": "unverified",
              "raw_http_finish_reason": record.get("raw_finish_reason")}
    ids = record.get("native_output_ids")
    valid_ids = (type(ids) is list and bool(ids)
                 and all(type(token) is int and 0 <= token < VOCAB for token in ids))
    if valid_ids:
        detail["last_native_token"] = ids[-1]
        detail["repetition_diagnostic"] = repetition_diagnostic(ids)
        if not job["ignore_eos"] and detail["repetition_diagnostic"]["detected"]:
            hard.append("semantic generation repeats under the unchanged 32-repeat predicate")
    response = record.get("response")
    if type(response) is not dict:
        missing.append("raw native API response unavailable")
    else:
        if backend == "sglang":
            raw_source = response.get("meta_info", {}).get("finish_reason")
            if canonical(response.get("output_ids")) != canonical(ids):
                hard.append("raw SG HTTP/native output token IDs differ")
        else:
            choices = response.get("response", {}).get("body", {}).get("choices", [])
            raw_source = choices[0].get("finish_reason") if len(choices) == 1 else None
        if raw_source is None:
            missing.append("raw API finish reason unavailable")
        elif canonical(raw_source) != canonical(record.get("raw_finish_reason", record.get("finish_reason"))):
            hard.append("raw API response/retained finish reasons differ")
    if not job["ignore_eos"] and record.get("finish_reason") != "stop":
        hard.append("semantic generation did not finish naturally")
    if completion is None:
        missing.append("native termination observation unavailable")
        return {"hard_failures": hard, "missing_evidence": missing, "evidence": detail}
    detail["native_completion"] = completion
    native_ids = completion.get("native_generated_ids")
    valid_native_ids = (type(native_ids) is list and bool(native_ids)
                        and all(type(token) is int and 0 <= token < VOCAB for token in native_ids))
    if not valid_ids or not valid_native_ids or native_ids != ids:
        hard.append("native completion/API output token IDs differ")
    if (not valid_ids or completion.get("last_native_token") != ids[-1]
            or type(completion.get("last_native_token")) is not int):
        if "last_native_token" not in completion:
            missing.append("native last-token observation unavailable")
        else:
            hard.append("native last-token observation differs from complete output")
    if (any(type(completion.get(field)) is not int for field in ("original_input_length", "generated_count", "decoded_length"))
            or completion.get("original_input_ids_sha256") != job["input_ids_sha256"]
            or completion.get("original_input_length") != len(job["input_ids"])
            or completion.get("generated_count") != (len(ids) if type(ids) is list else None)
            or completion.get("decoded_length") != (len(ids) if type(ids) is list else None)):
        hard.append("native completion input/output length binding mismatch")
    policy = completion.get("stop_policy")
    detail["stop_policy"] = policy
    if (type(policy) is not dict or policy.get("schema") != "r1-native-stop-policy-v1"
            or policy.get("status") != "observed" or not policy.get("evidence")):
        missing.append("actual native stop policy/EOS set unavailable")
        policy = None
    if policy is not None:
        eos = policy.get("eos_token_ids")
        if (type(eos) is not list or not eos or any(type(token) is not int or not 0 <= token < VOCAB for token in eos)
                or len(set(eos)) != len(eos)):
            hard.append("invalid actual EOS token set")
            eos = []
        if (policy.get("ignore_eos") is not job["ignore_eos"]
                or type(policy.get("max_new_tokens")) is not int
                or policy.get("max_new_tokens") != job["max_new_tokens"]):
            hard.append("native stop/length policy differs from request")
        if not job["ignore_eos"] and valid_ids:
            if ids[-1] not in eos or any(token in eos for token in ids[:-1]):
                hard.append("natural completion does not end at the first actual EOS token")
        if backend == "batchgen":
            if (type(policy.get("repetition_breaker_enabled")) is not bool
                    or type(policy.get("repetition_detected")) is not bool):
                missing.append("actual BG repetition policy/flag unavailable")
            elif policy["repetition_breaker_enabled"] or policy["repetition_detected"]:
                hard.append("native BG repetition stopping differs from the shared stop contract")
        else:
            fields = ("additional_stop_token_ids", "stop_token_ids", "stop_strs", "stop_regex_strs")
            if any(field not in policy for field in fields) or type(policy.get("grammar_active")) is not bool:
                missing.append("actual SG additional-stop/grammar policy unavailable")
            elif any(policy[field] != [] for field in fields) or policy["grammar_active"]:
                hard.append("native SG extra stop/regex/grammar differs from the shared stop contract")
    if backend == "batchgen":
        native_reason = completion.get("finish_reason")
        detail["raw_native_finish_reason"] = native_reason
        if type(native_reason) is not str:
            missing.append("native BG finish reason unavailable")
        elif native_reason != record.get("finish_reason"):
            hard.append("native BG/HTTP finish reasons differ")
        raw_http = record.get("raw_finish_reason", record.get("finish_reason"))
        detail["raw_http_finish_reason"] = raw_http
        if raw_http != record.get("finish_reason"):
            hard.append("raw BG/normalized HTTP finish reasons differ")
    else:
        raw_http = record.get("raw_finish_reason")
        raw_native = completion.get("raw_native_finish_reason")
        detail["raw_native_finish_reason"] = raw_native
        if type(raw_http) is not dict or type(raw_native) is not dict:
            missing.append("raw SG HTTP/native finish reason unavailable")
        else:
            if canonical(raw_http) != canonical(raw_native):
                hard.append("native SG/raw HTTP finish reasons differ")
            if type(completion.get("native_finish_is_error")) is not bool:
                missing.append("native SG finish error state unavailable")
            elif completion["native_finish_is_error"]:
                hard.append("native SG completion reports an error")
            if job["ignore_eos"]:
                if (raw_native.get("type") != "length" or type(raw_native.get("length")) is not int
                        or raw_native.get("length") != job["max_new_tokens"]
                        or completion.get("native_finish_reason_class") != "FINISH_LENGTH"):
                    hard.append("native SG fixed-length finish reason differs")
            elif (raw_native.get("type") != "stop" or type(raw_native.get("matched")) is not int
                    or not valid_ids or raw_native.get("matched") != ids[-1]
                    or completion.get("native_finish_reason_class") != "FINISH_MATCHED_TOKEN"):
                hard.append("native SG semantic completion was not an EOS token match")
    if not hard and not missing:
        detail["status"] = "verified"
    return {"hard_failures": hard, "missing_evidence": missing, "evidence": detail}


def require_same_termination_policy(a, b):
    """Compare only actually verified stop policies for the paired requests."""
    left, right = a.get("termination_evidence", {}), b.get("termination_evidence", {})
    expected = set(a.get("records", {})) | set(b.get("records", {}))
    missing, compared = [], 0
    for key in sorted(expected):
        x, y = left.get(key, {}), right.get(key, {})
        if x.get("status") != "verified" or y.get("status") != "verified":
            missing.append(key)
            continue
        fields = ("eos_token_ids", "ignore_eos", "max_new_tokens")
        if any(canonical(x["stop_policy"].get(field)) != canonical(y["stop_policy"].get(field)) for field in fields):
            raise ValueError("Paired actual EOS/stop policies differ: " + key)
        compared += 1
    return {"protocol": "r1-native-stop-evidence-v1", "status": "verified_equal" if compared and not missing else "unverified",
            "compared_requests": compared, "unverified_requests": missing,
            "batchgen": left, "sglang": right}


def read_logits(path, artifact):
    if artifact.get("dtype") != "float32_le" or artifact.get("elements") != VOCAB or artifact.get("bytes") != VOCAB * 4:
        raise ValueError("Explicit full-logits dtype/shape metadata required")
    resolved = (path / "trace" / artifact["file"]).resolve()
    if resolved.parent != (path / "trace").resolve() or resolved.stat().st_size != VOCAB * 4 or sha(resolved) != artifact["sha256"]:
        raise ValueError("Logits file identity/shape mismatch")
    result = array("f"); result.frombytes(resolved.read_bytes())
    if sys.byteorder != "little":
        result.byteswap()
    if len(result) != VOCAB or not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite or wrong-vocabulary full logits")
    return result


def logits_metrics(a, b, target):
    if len(a) != len(b) or not a or not all(math.isfinite(v) for v in (*a, *b)):
        raise ValueError("Finite equal full-vocabulary vectors required")
    def normalized(v):
        maximum = max(v)
        logz = maximum + math.log(math.fsum(math.exp(x - maximum) for x in v))
        return [x - logz for x in v]
    la, lb = normalized(a), normalized(b)
    errors = [abs(x - y) for x, y in zip(la, lb)]
    pa, pb = [math.exp(x) for x in la], [math.exp(x) for x in lb]
    js = 0.0
    for p, q, x, y in zip(pa, pb, la, lb):
        logmean = max(x, y) + math.log1p(math.exp(-abs(x - y))) - math.log(2.0)
        if p:
            js += 0.5 * p * (x - logmean)
        if q:
            js += 0.5 * q * (y - logmean)
    ia, ib = max(range(len(a)), key=a.__getitem__), max(range(len(b)), key=b.__getitem__)
    va, vb = sorted(a, reverse=True)[:2], sorted(b, reverse=True)[:2]
    disagreement = ia != ib
    margin_a, margin_b = va[0] - va[1], vb[0] - vb[1]
    result = {"logprob_sup_abs": max(errors), "logprob_rms": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
              "Jensen_Shannon_nats": max(0.0, js), "teacher_logprob_abs": errors[target],
              "top1_a": ia, "top1_b": ib, "top1_disagreement": disagreement,
              "top2_margin_a": margin_a, "top2_margin_b": margin_b,
              "ambiguous_disagreement": disagreement and max(margin_a, margin_b) <= NUMERIC["top1_disagreement_near_tie_margin_max"]}
    result["passed"] = (result["logprob_sup_abs"] <= NUMERIC["logprob_sup_abs_max"]
                        and result["logprob_rms"] <= NUMERIC["logprob_rms_max"]
                        and result["Jensen_Shannon_nats"] <= NUMERIC["Jensen_Shannon_max_nats"]
                        and result["teacher_logprob_abs"] <= NUMERIC["teacher_logprob_abs_max"]
                        and (not disagreement or result["ambiguous_disagreement"]))
    return result


def validate_evidence(run_path, suite, suite_sha, long_stage, semantic_smoke=True, context_limit=16384):
    run = checked_json(run_path)
    # Legacy v1 and the independent96 collector predate this metadata field.
    # New fixed v2 must state it; a present conflicting schema always fails.
    if (run.get("suite_schema", suite["schema"]) != suite["schema"]
            or (suite["schema"] == SCHEMA_V2 and run.get("suite_schema") != SCHEMA_V2)):
        raise ValueError("Run/suite schema differs or new fixed v2 binding is missing")
    if (run.get("schema") != RUN_SCHEMA or run.get("scope") != "full_model"
            or run.get("status") != "evidence_collected_NOT_pair_qualified" or run.get("suite_sha256") != suite_sha
            or run.get("long_correctness") is not long_stage):
        raise ValueError("Incomplete/wrong-stage full-model capture")
    deployment = validate_deployment(run["deployment"], run["backend"], suite)
    jobs = {j["id"]: j for j in suite["jobs"] if j["stage"] == "short" or long_stage}
    records = {r["id"]: r for r in run["records"]}
    if len(records) != len(run["records"]) or set(records) != set(jobs):
        raise ValueError("Missing/duplicate/unknown HTTP evidence jobs")
    event_path = run_path.parent / run["trace"]["events_file"]
    events = checked_json(event_path, run["trace"]["events_sha256"])
    collection = validate_collection_evidence(run, suite, suite_sha, long_stage, events)
    logits, completions, inputs, inactive = {}, {}, {}, {}
    coverage = set(); hard = []; missing = []; termination = {}
    if is_causal_suite(suite) and not validate_causal_closures(run, events):
        missing.append("causal trace is an unclosed prefix; both original-node worker closure seals required")
    trace_started_ranks = set()
    registered = set()
    for event in events:
        if (event.get("schema") != "r1-full-model-trace-v1" or event.get("suite_sha256") != suite_sha
                or event.get("run_id") != deployment["run_id"] or event.get("world_size") != WORLD
                or event.get("deployment_sha256") != deployment["artifact_sha256"]
                or not 0 <= event.get("rank", -1) < WORLD):
            raise ValueError("Trace envelope source/input/rank binding differs")
        kind = event.get("event")
        if kind == "trace_start":
            expected_sources = {role: data["sha256"] for role, data in deployment["source_files"].items()}
            if event.get("source_files") != expected_sources or event.get("helper_sha256") not in expected_sources.values():
                hard.append("actual worker source/helper identity differs from deployment")
            else:
                trace_started_ranks.add(event["rank"])
        if kind == "model_hooks_registered":
            expected_config = {"hidden_size": 7168, "num_hidden_layers": 61, "vocab_size": VOCAB, "n_routed_experts": 256}
            if (event.get("actual_layer_count") != LAYERS or event.get("actual_hook_count") != LAYERS * 3
                    or event.get("actual_config") != expected_config):
                hard.append("actual loaded model config/layers differ from complete R1")
            else:
                registered.add(event["rank"])
        if kind == "trace_error":
            hard.append("runtime trace_error: " + str(event.get("location")))
        if kind in ("layer_summary", "layer_failure"):
            if event.get("nonfinite_elements") != 0 or event.get("unobserved_activations") != 0:
                hard.append("nonfinite/unobserved actual layer activation")
            if (event.get("component") in ("attention", "mlp") and event.get("phase") in ("prefill", "decode")
                    and 0 <= event.get("layer_index", -1) < LAYERS
                    and observation_inside_collection(event, collection, run, "last_nonempty_observed")
                    and event.get("forward_calls", 0) > 0 and event.get("output_checked_elements", 0) > 0):
                coverage.add((event["rank"], event["phase"], event["layer_index"], event["component"]))
        if kind == "layer_forward":
            for key in ("activation_input", "activation_output"):
                summary = event.get(key, {})
                if (summary.get("observed") is not True or summary.get("checked_elements") != summary.get("numel")
                        or summary.get("nonfinite_elements") != 0):
                    hard.append("missing/nonfinite/partial actual layer activation")
            if (event.get("component") in ("attention", "mlp") and event.get("phase") in ("prefill", "decode")
                    and 0 <= event.get("layer_index", -1) < LAYERS
                    and event.get("activation_output", {}).get("numel", 0) > 0
                    and (not is_causal_suite(suite) or observation_inside_collection(event, collection, run, "observed"))):
                coverage.add((event["rank"], event["phase"], event["layer_index"], event["component"]))
        key = event.get("request_id")
        if key not in jobs:
            continue
        if kind == "logits":
            logits.setdefault(key, []).append(event)
        elif kind == "completion":
            completions.setdefault(key, []).append(event)
        elif kind == "input_tokens":
            inputs.setdefault(key, []).append(event)
        elif kind == "inactive_forward":
            inactive.setdefault(key, []).append(event)
    required = {(r, p, layer, component) for r in range(WORLD) for p in ("prefill", "decode")
                for layer in range(LAYERS) for component in ("attention", "mlp")}
    if trace_started_ranks != set(range(WORLD)) or registered != set(range(WORLD)):
        missing.append("not every actual rank observed the pinned sources and full model registration")
    if coverage != required:
        missing.append("actual rank/phase/layer/attention+MLP coverage missing: " + str(len(required - coverage)))
    for key, job in jobs.items():
        record = records[key]
        if not job["ignore_eos"] and record.get("finish_reason") != "stop":
            hard.append(key + ": semantic generation did not finish naturally")
        admitted = inputs.get(key, [])
        if not admitted:
            missing.append(key + ": native input observation unavailable")
        elif any(e.get("input_ids_sha256") != job["input_ids_sha256"] or e.get("input_count") != len(job["input_ids"])
                 or e.get("input_ids_equal_fixture") is not True for e in admitted):
            hard.append(key + ": native intake token IDs differ from fixture")
        if record.get("prompt_tokens") != len(job["input_ids"]) or not isinstance(record.get("text"), str):
            hard.append(key + ": HTTP prompt length/text differs")
        cs = completions.get(key, [])
        completion = cs[0] if len(cs) == 1 else None
        if run["backend"] == "batchgen":
            if len(cs) != 1:
                missing.append(key + ": missing/duplicate BG native completion")
                result = validate_termination(run["backend"], job, record, None)
                termination[key] = result["evidence"]
                hard.extend(key + ": " + reason for reason in result["hard_failures"])
                missing.extend(key + ": " + reason for reason in result["missing_evidence"])
                continue
            c = cs[0]
            record["native_output_ids"] = c.get("native_generated_ids")
            if (c.get("original_input_ids_sha256") != job["input_ids_sha256"]
                    or c.get("original_input_length") != len(job["input_ids"])
                    or c.get("generated_count") != c.get("decoded_length")):
                hard.append(key + ": BG native input/output length binding mismatch")
        ids = record.get("native_output_ids")
        if not isinstance(ids, list):
            missing.append(key + ": native IDs unavailable; retokenized text is not accepted")
            continue
        if any(type(i) is not int or not 0 <= i < VOCAB for i in ids):
            hard.append(key + ": output token outside vocabulary")
        if record.get("completion_tokens") != len(ids) or not 1 <= len(ids) <= job["max_new_tokens"]:
            hard.append(key + ": native/API output length differs")
        if job["ignore_eos"] and (len(ids) != job["max_new_tokens"] or record.get("finish_reason") != "length"):
            hard.append(key + ": fixed length not respected")
        if not job["ignore_eos"] and record.get("finish_reason") != "stop":
            hard.append(key + ": semantic generation did not finish naturally")
        if run["backend"] == "sglang" and len(cs) != 1:
            missing.append(key + ": missing/duplicate SG native completion")
        result = validate_termination(run["backend"], job, record, completion)
        termination[key] = result["evidence"]
        hard.extend(key + ": " + reason for reason in result["hard_failures"])
        missing.extend(key + ": " + reason for reason in result["missing_evidence"])
        for event in inactive.get(key, []):
            state = event.get("sequence_state", {})
            fields = ("decoded_length", "original_prompt_length", "current_context_length", "max_decode_length")
            valid = (run["backend"] == "batchgen" and event.get("eligible") is False
                     and event.get("completed") is True and event.get("phase") in ("prefill", "decode")
                     and event.get("eligibility_source") == "actual worker._is_sequence_completed(seq)"
                     and type(state) is dict and all(type(state.get(field)) is int for field in fields)
                     and type(state.get("eos_reached")) is bool and type(state.get("_rep_detected")) is bool
                     and type(event.get("model_context_length")) is int
                     and state.get("decoded_length") == len(ids)
                     and event.get("generated_before_step") == len(ids)
                     and state.get("original_prompt_length") == len(job["input_ids"])
                     and state.get("current_context_length") == len(job["input_ids"]) + len(ids)
                     and state.get("max_decode_length") == job["max_new_tokens"])
            if valid:
                valid = (state["decoded_length"] >= state["max_decode_length"]
                         or state["current_context_length"] >= event["model_context_length"]
                         or state["eos_reached"] or state["_rep_detected"])
            if not valid:
                hard.append(key + ": inactive forward lacks actual completed-row/final-ledger evidence")
        ls = sorted(logits.get(key, []), key=lambda e: e.get("generated_before_step", -1))
        if [e.get("generated_before_step") for e in ls] != list(range(len(ids))):
            missing.append(key + ": missing/duplicate native logits steps")
            continue
        for step, event in enumerate(ls):
            prefix = job["input_ids"] + ids[:step]
            if (event.get("input_ids_sha256") != value_sha(prefix) or event.get("context_length") != len(prefix)
                    or event.get("vocab_size") != VOCAB or event.get("nonfinite_elements") != 0
                    or event.get("greedy_token_id") != ids[step] or len(prefix) + 1 > context_limit):
                hard.append(key + ": nonfinite/shape/context/native sampling binding failure")
        expected_sources = {role: data["sha256"] for role, data in deployment["source_files"].items()}
        if ls and ls[-1].get("source_files_at_last_step") is None:
            missing.append(key + ": end-of-request source identities missing")
        elif ls and ls[-1].get("source_files_at_last_step") != expected_sources:
            hard.append(key + ": source identities changed during the actual request")
        if job["kind"] == "teacher" and (len(ls) != 1 or not ls[0].get("full_logits")):
            missing.append(key + ": full teacher vocabulary logits unavailable")
        if job["kind"] == "teacher" and ls and ls[0].get("phase") != "prefill":
            hard.append(key + ": teacher prefix scoring did not use the declared prefill route")
        if semantic_smoke and job["semantic"] and job["kind"] == "generation":
            if normalized_answer(record["text"]) not in {re.sub(r"[^\w%]", "", x.lower()) for x in job["answers"]}:
                hard.append(key + ": semantic final answer mismatch")
        first = ls[0] if ls else {}
        if first.get("input_ids_sha256") != job["input_ids_sha256"]:
            hard.append(key + ": native admission/prefill input binding mismatch")
    return {"run": run, "records": records, "logits": logits, "hard_failures": sorted(set(hard)),
            "collection": collection,
            "termination_evidence": termination,
            "inactive_forward_evidence": inactive,
            "missing_evidence": sorted(set(missing)), "actual_coverage_count": len(coverage),
            "required_coverage_count": len(required), "run_sha256": sha(run_path)}


def compare(args):
    suite = validate_suite(checked_json(args.suite, args.suite_sha256))
    if not args.execute:
        print(json.dumps({"status": "plan_only_NOT_executed", "scope": "full_model", "operation": "compare",
                          "suite_schema": suite["schema"],
                          "numeric_contract": NUMERIC, "long_correctness": args.long_correctness,
                          "suite_sha256": args.suite_sha256}, indent=2)); return 0
    report = {"schema": "r1-full-model-comparison-v1", "scope": "full_model", "status": INCOMPLETE,
              "suite_schema": suite["schema"],
              "suite_sha256": args.suite_sha256, "tool_sha256": sha(__file__), "numeric_contract": NUMERIC,
              "long_correctness": args.long_correctness, "full_model_qualified": False,
              "formal_performance_permitted": False, "teacher_comparisons": []}
    code = 2
    try:
        if sha(args.batchgen_run) != args.batchgen_run_sha256 or sha(args.sglang_run) != args.sglang_run_sha256:
            raise ValueError("Explicit input run SHA mismatch")
        if sha(args.tokenizer_json) != suite["tokenizer_json_sha256"]:
            raise ValueError("Native completion decoding tokenizer differs from frozen suite")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        tokenizer.no_padding(); tokenizer.no_truncation()
        a = validate_evidence(args.batchgen_run, suite, args.suite_sha256, args.long_correctness)
        b = validate_evidence(args.sglang_run, suite, args.suite_sha256, args.long_correctness)
        require_same_collection_schedule(a, b)
        report["collection"] = {"batchgen": a["collection"], "sglang": b["collection"]}
        report["termination_policy"] = require_same_termination_policy(a, b)
        if a["run"]["backend"] != "batchgen" or b["run"]["backend"] != "sglang":
            raise ValueError("Engine identities reversed")
        if a["run"]["deployment"]["checkpoint_identity"] != b["run"]["deployment"]["checkpoint_identity"]:
            raise ValueError("The two model/checkpoint identities differ")
        report["input_run_sha256"] = {"batchgen": a["run_sha256"], "sglang": b["run_sha256"]}
        report["deployment_model_identity"] = {"batchgen": deployment_model_identity(a["run"]["deployment"]),
                                               "sglang": deployment_model_identity(b["run"]["deployment"])}
        report["observed_execution"] = {name: {key: evidence[key] for key in ("actual_coverage_count", "required_coverage_count")}
                                          for name, evidence in (("batchgen", a), ("sglang", b))}
        hard = a["hard_failures"] + b["hard_failures"]
        missing = a["missing_evidence"] + b["missing_evidence"]
        for name, evidence in (("batchgen", a), ("sglang", b)):
            for job in suite["jobs"]:
                if job["kind"] != "generation" or not job["semantic"]:
                    continue
                record = evidence["records"][job["id"]]
                native = record.get("native_output_ids")
                if not isinstance(native, list):
                    continue
                decoded = tokenizer.decode(native, skip_special_tokens=True)
                actual = normalized_answer(decoded)
                expected = {re.sub(r"[^\w%]", "", x.lower()) for x in job["answers"]}
                if actual not in expected or actual != normalized_answer(record["text"]):
                    hard.append(name + ": semantic native-ID decoding/API final answer mismatch: " + job["id"])
        if not hard and not missing:
            for job in suite["jobs"]:
                if job["kind"] != "teacher":
                    continue
                ea, eb = a["logits"][job["id"]][0], b["logits"][job["id"]][0]
                va = read_logits(args.batchgen_run.parent, ea["full_logits"])
                vb = read_logits(args.sglang_run.parent, eb["full_logits"])
                metrics = logits_metrics(va, vb, job["teacher_token_id"])
                if (metrics["top1_a"] != a["records"][job["id"]]["native_output_ids"][0]
                        or metrics["top1_b"] != b["records"][job["id"]]["native_output_ids"][0]):
                    hard.append("saved teacher logits do not reproduce actual native greedy token")
                report["teacher_comparisons"].append({"id": job["id"], "teacher_method": job["teacher_method"],
                    "input_ids_sha256": job["input_ids_sha256"], "teacher_token_id": job["teacher_token_id"], **metrics})
            rows = report["teacher_comparisons"]
            if not rows or not all(row["passed"] for row in rows):
                hard.append("teacher full-vocabulary numeric contract failed")
            if rows:
                mean = math.fsum(row["teacher_logprob_abs"] for row in rows) / len(rows)
                fraction = sum(row["top1_disagreement"] for row in rows) / len(rows)
                report["teacher_mean_logprob_abs"] = mean
                report["teacher_near_tie_disagreement_fraction"] = fraction
                if mean > NUMERIC["teacher_logprob_mean_abs_max"] or fraction > NUMERIC["near_tie_disagreement_fraction_max"]:
                    hard.append("aggregate teacher logprob/top1 contract failed")
        report["hard_failures"] = hard
        report["missing_evidence"] = missing
        if (sha(args.batchgen_run) != args.batchgen_run_sha256 or sha(args.sglang_run) != args.sglang_run_sha256
                or sha(args.suite) != args.suite_sha256 or sha(args.tokenizer_json) != suite["tokenizer_json_sha256"]):
            raise ValueError("Comparison sources/inputs changed while being read")
        if hard:
            report["status"] = FAIL; code = 1
        elif missing:
            report["status"] = INCOMPLETE
        else:
            report["status"] = PASS if args.long_correctness else SHORT
            report["full_model_qualified"] = False
            report["independent_semantic_qualification"] = "pending_frozen_full_context_48x2_paired_suite"
            code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, ValueError):
            report["status"] = FAIL
            code = 1
    write_new(args.output, report)
    print(canonical({"status": report["status"], "scope": "full_model", "report": str(args.output)}))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--time-basis", choices=("legacy_wall_v1", "node_monotonic_causal_v2"), default="legacy_wall_v1")
    p.add_argument("--suite-version", choices=("v1", "v2"), default="v1",
                   help="v1 retains384; explicitly choose v2 for the fixed8192 budget on12 semantic generations")
    p.add_argument("--cohort-size", type=int, choices=(1, 16), default=1)
    p.add_argument("--tokenizer-json", type=Path, required=True); p.add_argument("--tokenizer-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--execute", action="store_true")
    p = sub.add_parser("collect")
    p.add_argument("--time-basis", choices=("legacy_wall_v1", "node_monotonic_causal_v2"), default="legacy_wall_v1")
    p.add_argument("--causal-clock-config", type=Path); p.add_argument("--causal-clock-config-sha256")
    p.add_argument("--cohort-size", type=int, choices=(1, 16), default=1)
    p.add_argument("--suite", type=Path, required=True); p.add_argument("--suite-sha256", required=True)
    p.add_argument("--deployment", type=Path, required=True); p.add_argument("--deployment-sha256", required=True)
    p.add_argument("--backend", choices=("batchgen", "sglang"), required=True)
    p.add_argument("--endpoint", required=True); p.add_argument("--trace-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--api-key-env")
    p.add_argument("--timeout", type=float, default=1800); p.add_argument("--long-correctness", action="store_true")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--suite", type=Path, required=True); p.add_argument("--suite-sha256", required=True)
    p.add_argument("--batchgen-run", type=Path, required=True); p.add_argument("--sglang-run", type=Path, required=True)
    p.add_argument("--batchgen-run-sha256", required=True); p.add_argument("--sglang-run-sha256", required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--long-correctness", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if hasattr(args, "timeout") and (not math.isfinite(args.timeout) or not 0 < args.timeout <= 7200):
        parser.error("Explicit bounded timeout must be in (0,7200] seconds")
    return {"prepare": prepare, "collect": collect, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
