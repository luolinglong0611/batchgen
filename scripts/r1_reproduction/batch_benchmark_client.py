#!/usr/bin/env python3
"""Audit whole-batch completion time through BatchGen's files/batches API.

Consumes the SAME prepare_workload manifest/body order as benchmark_client.py.
No request retries, redirects, implicit re-submission, server restart or cleanup.
One --endpoint is the two-node BatchGen deployment's coordinator, not a replica.

  python batch_benchmark_client.py --manifest /data/workload/manifest.json \
    --profile input8192-output2048 --endpoint http://node0:8000 \
    --server-revision <deployed-commit> --warmup-count 16 --output /data/run1

Primary wall_seconds includes POST /v1/files upload, POST /v1/batches creation,
polling to terminal state and complete output/error-file retrieval. Also records
create_to_results_seconds separately, excluding upload, for an explicit batch-
creation timing boundary. Neither includes offline payload preparation or warmup.
Polling contributes observation delay; no server completion precision is invented.
Per-request server-created timestamps are retained, not treated as client latency.

The current repository explicitly supports per-request ignore_eos in
CompletionRequest, scheduler intake, SequenceEntry, EOS checks and the repetition
breaker. Context limits still apply. --ignore-eos is opt-in and each actual output
must equal max_tokens; otherwise the run fails. Without it, EOS is allowed and
actual lengths/quality must accompany comparisons. The deployed commit is recorded
but must be verified operationally. Incremental output saving must be enabled:
a terminal batch with missing/empty outputs is NOT a successful measurement.

This whole-batch API metric is NOT interchangeable with synchronous closed-loop
concurrency. Compare matched manifests, effective-payload hashes, tokenizer/model
revisions, EOS policy, context capacity, prefix-cache policy and admission method.
This is not a recovered paper artifact or a throughput result until actually run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, build_opener
import uuid


HELPER_SOURCE = Path(__file__).resolve().with_name("benchmark_client.py")
_spec = importlib.util.spec_from_file_location("r1_sync_workload_helpers", HELPER_SOURCE)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
dumps, sha256 = _helper.dumps, _helper.sha256
TERMINAL = {"completed", "failed", "cancelled", "expired"}
ACTIVE = {"validating", "in_progress", "finalizing", "cancelling"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def base_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must be an http(s) origin/base path without credentials/query/fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def remaining(deadline):
    value = deadline - time.perf_counter()
    if value <= 0:
        raise TimeoutError("Batch phase deadline exceeded; server job is not automatically cancelled")
    return value


class Transport:
    """One HTTP attempt per call; polling only repeats successful status reads."""

    def __init__(self, endpoint, api_key, timeout, deadline, started, log):
        self.endpoint, self.api_key, self.timeout = endpoint, api_key, timeout
        self.deadline, self.started, self.log = deadline, started, log

    def call(self, method, path, payload=None, content_type=None, destination=None):
        started = time.perf_counter()
        record = {"method": method, "path": path, "attempts": 1,
                  "started_seconds": started - self.started,
                  "request_bytes": len(payload) if payload is not None else 0}
        if payload is not None:
            record["request_sha256"] = hashlib.sha256(payload).hexdigest()
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint + path, data=payload, headers=headers, method=method)
        try:
            try:
                response = build_opener(_helper.NoRedirect()).open(
                    request, timeout=min(self.timeout, remaining(self.deadline)))
            except HTTPError as error:
                response = error
            with response:
                record["http_status"] = response.status
                if not 200 <= response.status < 300:
                    record["error_response"] = response.read(1024 * 1024).decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {response.status} for {method} {path}; not retried")
                if destination is not None:
                    total = 0
                    with destination.open("xb") as output:
                        while True:
                            remaining(self.deadline)
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            total += len(block)
                    record["download_bytes"] = total
                    result = None
                else:
                    content = response.read(8 * 1024 * 1024 + 1)
                    if len(content) > 8 * 1024 * 1024:
                        raise ValueError("Control response exceeds 8 MiB")
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        raise ValueError("Control response must be a JSON object")
                    record["response"] = result
            remaining(self.deadline)
            return result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["completed_seconds"] = time.perf_counter() - self.started
            self.log.write(dumps(record) + "\n")
            self.log.flush()


def prepare_payload(directory, phase, requests):
    path = directory / f"{phase}.submitted.jsonl"
    with path.open("x", encoding="utf-8") as output:
        for item in requests:
            output.write(dumps({"custom_id": item["custom_id"], "method": "POST",
                                "url": "/v1/completions", "body": item["body"]}) + "\n")
    data = path.read_bytes()
    boundary = "r1-batch-" + uuid.uuid4().hex
    multipart = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return path, multipart, "multipart/form-data; boundary=" + boundary


def validate_results(paths, requests, output_path, ignore_eos, received_seconds):
    expected = {item["custom_id"]: item for item in requests}
    found, errors, counts = {}, [], {"successful": 0, "failed": 0, "prompt_tokens": 0, "output_tokens": 0}
    with output_path.open("x", encoding="utf-8") as output:
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        errors.append(f"{path.name}:{line_number}: invalid JSON: {error}")
                        continue
                    if not isinstance(row, dict):
                        errors.append(f"{path.name}:{line_number}: result is not an object")
                        continue
                    custom_id = row.get("custom_id")
                    if not isinstance(custom_id, str) or custom_id not in expected:
                        errors.append(f"Unknown custom_id in {path.name}:{line_number}: {custom_id!r}")
                        continue
                    if custom_id in found:
                        errors.append(f"Duplicate result custom_id: {custom_id}")
                        continue
                    found[custom_id] = True
                    item = expected[custom_id]
                    problems = []
                    if row.get("error"):
                        problems.append("Batch result contains an error")
                    response = row.get("response")
                    body = response.get("body") if isinstance(response, dict) else None
                    if not isinstance(response, dict) or response.get("status_code") != 200 or not isinstance(body, dict):
                        problems.append("Missing successful HTTP completion response")
                        body = body if isinstance(body, dict) else {}
                    if body.get("error"):
                        problems.append("Completion body contains an error")
                    usage = body.get("usage")
                    if not isinstance(usage, dict) or not all(_helper.positive_integer(usage.get(key)) for key in ("prompt_tokens", "completion_tokens")):
                        problems.append("Missing/invalid actual usage")
                    else:
                        if usage["prompt_tokens"] != item["expected_input_tokens"]:
                            problems.append("Actual prompt_tokens differs from manifest")
                        if usage["completion_tokens"] > item["max_output_tokens"]:
                            problems.append("Actual output exceeds requested max_tokens")
                        if ignore_eos and usage["completion_tokens"] != item["max_output_tokens"]:
                            problems.append("Fixed output length not enforced")
                    choices = body.get("choices")
                    text, finish_reason = None, None
                    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or not isinstance(choices[0].get("text"), str):
                        problems.append("Expected one completion choice with text")
                    else:
                        text, finish_reason = choices[0]["text"], choices[0].get("finish_reason")
                        if finish_reason not in {"stop", "length"}:
                            problems.append("Missing/non-successful finish_reason (including repetition termination)")
                    success = not problems
                    record = {"custom_id": custom_id, "original_id": item["original_id"], "dataset": item["dataset"],
                              "success": success, "usage": usage, "output_text": text, "finish_reason": finish_reason,
                              "errors": problems, "batch_result": row, "source_result_file": path.name,
                              "expected_input_tokens": item["expected_input_tokens"], "max_output_tokens": item["max_output_tokens"],
                              "server_created_unix": body.get("created"),
                              "batch_results_received_seconds": received_seconds,
                              "per_request_client_completion_time_available": False}
                    output.write(dumps(record) + "\n")
                    if success:
                        counts["successful"] += 1
                        counts["prompt_tokens"] += usage["prompt_tokens"]
                        counts["output_tokens"] += usage["completion_tokens"]
                    else:
                        counts["failed"] += 1
        missing = sorted(set(expected) - set(found))
        for custom_id in missing:
            output.write(dumps({"custom_id": custom_id, "original_id": expected[custom_id]["original_id"],
                                "dataset": expected[custom_id]["dataset"], "success": False,
                                "errors": ["No result returned for submitted request"]}) + "\n")
        counts["failed"] += len(missing)
    return {**counts, "expected": len(requests), "found": len(found), "missing_custom_ids": missing,
            "structural_errors": errors, "all_valid": not errors and counts["failed"] == 0 and len(found) == len(expected)}


def run_phase(args, phase, requests, directory, endpoint, api_key):
    # Build/hash the request file and multipart body before timing.
    submitted, payload, content_type = prepare_payload(directory, phase, requests)
    result = {"phase": phase, "status": "failed", "submitted_file": submitted.name,
              "submitted_sha256": sha256(submitted), "requested_count": len(requests),
              "started_utc": utc_now(), "batch_id": None, "output_files": [], "all_requests_valid": False}
    started = time.perf_counter()
    deadline = started + args.timeout_seconds
    create_started, terminal_observed, received = None, None, None
    batch = None
    try:
        with (directory / f"{phase}.http.jsonl").open("x", encoding="utf-8") as log:
            transport = Transport(endpoint, api_key, args.http_timeout_seconds, deadline, started, log)
            uploaded = transport.call("POST", "/v1/files", payload, content_type)
            file_id = uploaded.get("id")
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("Upload did not return a file id")
            result["input_file_id"] = file_id
            if uploaded.get("checksum") and uploaded["checksum"] != result["submitted_sha256"]:
                raise ValueError("Server upload checksum differs from submitted JSONL")
            creation = {"input_file_id": file_id, "endpoint": "/v1/completions", "completion_window": "24h"}
            if args.max_context_length is not None:
                creation["max_context_length"] = args.max_context_length
            result["create_parameters"] = creation
            create_started = time.perf_counter()
            batch = transport.call("POST", "/v1/batches", dumps(creation).encode(), "application/json")
            batch_id = batch.get("id")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError("Batch creation did not return a batch id")
            result["batch_id"] = batch_id
            while batch.get("status") not in TERMINAL:
                if batch.get("status") not in ACTIVE:
                    raise ValueError(f"Unknown batch status: {batch.get('status')!r}")
                result["last_nonterminal_observed_seconds"] = time.perf_counter() - started
                time.sleep(min(args.poll_seconds, remaining(deadline)))
                batch = transport.call("GET", "/v1/batches/" + quote(batch_id, safe=""))
                if batch.get("id") != batch_id:
                    raise ValueError("Retrieved batch id changed")
            terminal_observed = time.perf_counter()
            result["terminal_batch"] = batch
            # Current BatchGen exposes output_file_id. Download error_file_id if
            # an API-compatible server actually returns it; do not request fields.
            file_ids = set()
            for key in ("output_file_id", "error_file_id"):
                output_id = batch.get(key)
                if output_id is None:
                    continue
                if not isinstance(output_id, str) or not output_id or output_id in file_ids:
                    raise ValueError("Invalid/duplicate output or error file id")
                file_ids.add(output_id)
                path = directory / f"{phase}.{key}.jsonl"
                transport.call("GET", "/v1/files/" + quote(output_id, safe="") + "/content", destination=path)
                result["output_files"].append(path.name)
            # The end boundary is complete retrieval, BEFORE offline validation
            # and checksum work. This includes HTTP framing/file-write overhead.
            received = time.perf_counter()
        paths = [directory / name for name in result["output_files"]]
        validation = validate_results(paths, requests, directory / f"{phase}.requests.jsonl",
                                      args.ignore_eos, received - started)
        result["validation"] = validation
        result["all_requests_valid"] = batch.get("status") == "completed" and validation["all_valid"]
        result["status"] = "completed" if result["all_requests_valid"] else "invalid_batch_or_results"
        result["result_sha256"] = {path.name: sha256(path) for path in paths}
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if batch is not None:
            result["last_batch_response"] = batch
        # A lost/failed create response does not prove that no job was created.
        result["server_job_may_still_be_active"] = create_started is not None and (batch or {}).get("status") not in TERMINAL
    finally:
        result["finished_utc"] = utc_now()
        result["elapsed_until_return_seconds"] = time.perf_counter() - started
        result["wall_seconds"] = received - started if received is not None else None
        result["create_to_results_seconds"] = received - create_started if received is not None and create_started is not None else None
        result["terminal_observed_seconds"] = terminal_observed - started if terminal_observed is not None else None
        result["upload_and_create_preparation_seconds"] = create_started - started if create_started is not None else None
        valid = result["all_requests_valid"]
        result["requests_per_second"] = len(requests) / result["wall_seconds"] if valid else None
        result["output_tokens_per_second"] = result["validation"]["output_tokens"] / result["wall_seconds"] if valid else None
        (directory / f"{phase}.summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", required=True, choices=("input8192-output2048", "input2048-output8192"))
    parser.add_argument("--endpoint", required=True, help="One BatchGen coordinator URL; whole-batch submission")
    parser.add_argument("--server-revision", required=True, help="Actual deployed server commit, recorded without claiming remote verification")
    parser.add_argument("--count", type=int, help="Prefix count for smoke/tuning; default entire workload")
    parser.add_argument("--warmup-count", type=int, default=1, help="Requests in a separate warmup batch; 0 disables warmup explicitly")
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=86400, help="Per-phase deadline including upload, creation, polling and retrieval")
    parser.add_argument("--http-timeout-seconds", type=float, default=300, help="Per-socket timeout; all HTTP failures abort without retry")
    parser.add_argument("--max-context-length", type=int, help="Optional supported batch-level limit; must cover input+output and match baseline capacity")
    parser.add_argument("--ignore-eos", action="store_true", help="Supported per-request extension; require exact actual max_tokens output")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, required=True, help="New directory; no overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    if args.warmup_count < 0 or (args.count is not None and args.count <= 0):
        parser.error("warmup-count must be nonnegative; count positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.poll_seconds, args.timeout_seconds, args.http_timeout_seconds)) or args.poll_seconds > 60:
        parser.error("Timeouts must be finite/positive; poll-seconds must be in (0,60]")
    endpoint = base_url(args.endpoint)
    manifest, artifact, requests, _, effective_hash = _helper.load_workload(args.manifest, args.profile, args.count, args.ignore_eos)
    if args.warmup_count > len(requests):
        parser.error("warmup-count exceeds selected workload count")
    if args.max_context_length is not None and args.max_context_length < artifact["input_tokens"] + artifact["max_output_tokens"]:
        parser.error("max-context-length would truncate the requested input+output")
    # No extra per-request fields are invented. The source JSONL is audited by
    # load_workload; only its already-supported ignore_eos/temperature are fixed.
    supported = {"model", "prompt", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "n", "stream",
                 "logprobs", "echo", "presence_penalty", "frequency_penalty", "user", "ignore_eos"}
    for item in requests:
        unknown = set(item["body"]) - supported
        if unknown:
            raise ValueError(f"Unsupported CompletionRequest keys: {sorted(unknown)}")
    api_key = os.environ.get(args.api_key_env)
    source_root = Path(__file__).resolve().parents[2]
    evidence_paths = ["batchgen/server/io_struct.py", "batchgen/server/http_server.py", "batchgen/server/batch_scheduler.py",
                      "batchgen/worker/completion.py", "batchgen/batchgen_worker.py"]
    configuration = {
        "schema_version": 1, "created_utc": utc_now(), "client_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(HELPER_SOURCE), "python_version": platform.python_version(),
        "manifest_sha256": sha256(args.manifest), "source_requests_sha256": artifact["requests_sha256"],
        "source_provenance_sha256": artifact["provenance_sha256"], "effective_payloads_sha256": effective_hash,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "endpoint": endpoint, "server_revision_operator_supplied": args.server_revision,
        "api_key_present": bool(api_key), "request_count": len(requests), "source_request_count": manifest["count_per_profile"],
        "temperature": 0.0, "ignore_eos": args.ignore_eos, "source_ignore_eos": manifest.get("ignore_eos"),
        "local_api_source_sha256": {name: sha256(source_root / name) for name in evidence_paths if (source_root / name).is_file()},
        "submission_mode": "one whole asynchronous batch; no client per-request concurrency limit",
        "primary_timing": "start upload POST through terminal state and complete result/error-file retrieval",
        "secondary_timing": "start batch-create POST through complete result/error-file retrieval",
        "polling_observation_delay_included": True, "retries": 0, "follow_redirects": False,
        "paper_exact_reproduction": False,
        "limitations": ["Do not equate whole-batch BCT with synchronous closed-loop admission.",
                        "Verify deployed commit and tokenizer/model assets externally; local source hashes are evidence, not remote attestation.",
                        "Enable incremental output saving; empty or missing result files invalidate the run.",
                        "Fixed-length runs still require sufficient model context and per-request actual-length verification.",
                        "Match baseline EOS, context, prefix-cache and workload settings; warmup must cover relevant compilation shapes.",
                        "Per-request response timestamps are server-created seconds, not precise client completion times.",
                        "Timeout or interruption does not automatically cancel a server-side batch."],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"config_sha256": sha256(config_path), "status": "failed", "warmup": None, "measurement": None,
               "all_requests_valid": False}
    try:
        if args.warmup_count:
            summary["warmup"] = run_phase(args, "warmup", requests[:args.warmup_count], args.output, endpoint, api_key)
            if not summary["warmup"]["all_requests_valid"]:
                summary["status"] = "warmup_failed_measurement_not_started"
                return 1
        summary["measurement"] = run_phase(args, "measurement", requests, args.output, endpoint, api_key)
        summary["all_requests_valid"] = summary["measurement"]["all_requests_valid"]
        summary["status"] = "completed" if summary["all_requests_valid"] else "measurement_failed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted_server_batch_not_automatically_cancelled"
        return 130
    finally:
        summary["finished_utc"] = utc_now()
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "status": summary["status"],
                          "all_requests_valid": summary["all_requests_valid"],
                          "measurement": summary["measurement"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_requests_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
