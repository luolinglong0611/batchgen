#!/usr/bin/env python3
"""Measure audited workloads against synchronous OpenAI completion endpoints.

Standard library only. Example, two eight-GPU replicas:
  python benchmark_client.py --manifest /data/workload/manifest.json \
    --profile input8192-output2048 \
    --endpoint http://node0:30000/v1 --endpoint http://node1:30000/v1 \
    --concurrency 32 --warmup-per-endpoint 1 --output /data/results/run1

Requests are assigned round-robin in manifest order, with globally bounded
in-flight concurrency and NO retries (including HTTP redirects). Warmup completes
before timing starts. All measured requests are available at phase start; the
reported wall time includes client admission, HTTP submission, server queueing,
inference, and complete response receipt/validation, ending at the last request.
Model startup/compilation must be completed by warmup before a formal run.

This is a synchronous, closed-loop HTTP measurement, NOT the original paper
artifact and NOT BatchGen's asynchronous whole-batch API. Do not compare those
submission modes without recording and accounting for the different admission
policies. This client makes a fresh connection per request. It does not discover
server parallelism, disable server prefix reuse, or prove tokenizer identity.
Keep tokenizer/model revisions and prefix-cache settings equal on both engines;
actual per-request usage.prompt_tokens is checked against the workload.

Output defaults to EOS-allowed caps even if the input workload requests fixed
output. --ignore-eos explicitly enables fixed-length enforcement and records this
effective override. A run is invalid if any request fails, token counts disagree,
or fixed output is not actually enforced. Warmup failures abort before timing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoints must be absolute http(s) URLs")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Do not put credentials, query strings or fragments in endpoint URLs")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/completions"):
        path += "/completions" if path.endswith("/v1") else "/v1/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def safe_artifact(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest artifact escapes its directory")
    return path


def load_workload(manifest_path: Path, profile: str, count: int | None, ignore_eos: bool):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in manifest["artifacts"]
               if f"input{item['input_tokens']}-output{item['max_output_tokens']}" == profile]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest profile {profile!r}; found {len(matches)}")
    artifact = matches[0]
    requests_path = safe_artifact(manifest_path.parent, artifact["requests"])
    provenance_path = safe_artifact(manifest_path.parent, artifact["provenance"])
    if sha256(requests_path) != artifact["requests_sha256"]:
        raise ValueError("Request file SHA256 does not match manifest")
    if sha256(provenance_path) != artifact["provenance_sha256"]:
        raise ValueError("Provenance file SHA256 does not match manifest")
    provenance = {}
    with provenance_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["custom_id"] in provenance:
                raise ValueError("Duplicate custom_id in provenance")
            # Keep the hash-verified token IDs/answers on disk; retaining tens of
            # millions of Python integers is unnecessary for the HTTP client.
            provenance[row["custom_id"]] = {key: row[key] for key in (
                "input_tokens", "max_output_tokens", "original_id", "dataset")}
    requests, seen = [], set()
    with requests_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            custom_id = row["custom_id"]
            if custom_id in seen:
                raise ValueError("Duplicate custom_id in requests")
            seen.add(custom_id)
            if row.get("method") != "POST" or row.get("url") != "/v1/completions":
                raise ValueError("Only /v1/completions POST envelopes are supported")
            body = dict(row["body"])
            if not isinstance(body.get("prompt"), str) or not isinstance(body.get("model"), str):
                raise ValueError("Each request must contain one string prompt and a model alias")
            if body.get("stream", False) or body.get("n", 1) != 1:
                raise ValueError("Only non-streaming n=1 requests are supported")
            if body.get("temperature") != 0:
                raise ValueError("Workload temperature must be explicitly 0")
            metadata = provenance[custom_id]
            if metadata["input_tokens"] != artifact["input_tokens"] or metadata["max_output_tokens"] != artifact["max_output_tokens"]:
                raise ValueError("Provenance lengths disagree with selected profile")
            if body.get("max_tokens") != metadata["max_output_tokens"]:
                raise ValueError("Request max_tokens disagrees with provenance")
            # No implicit activation through a source file; operator opts in.
            body["ignore_eos"] = ignore_eos
            body["temperature"] = 0.0
            requests.append({"custom_id": custom_id, "body": body,
                             "expected_input_tokens": metadata["input_tokens"],
                             "max_output_tokens": metadata["max_output_tokens"],
                             "original_id": metadata["original_id"], "dataset": metadata["dataset"]})
    if seen != set(provenance):
        raise ValueError("Request/provenance custom_id sets differ")
    if len(requests) != manifest["count_per_profile"]:
        raise ValueError("Request count disagrees with manifest")
    if count is not None:
        if count > len(requests):
            raise ValueError("--count exceeds workload size")
        requests = requests[:count]
    if not requests:
        raise ValueError("Empty workload")
    payloads = [dumps(item["body"]).encode("utf-8") for item in requests]
    effective_hash = hashlib.sha256()
    for item, payload in zip(requests, payloads):
        effective_hash.update(item["custom_id"].encode("utf-8") + b"\0" + payload + b"\n")
    return manifest, artifact, requests, payloads, effective_hash.hexdigest()


def request_once(item: dict, payload: bytes, endpoint: str, endpoint_index: int,
                 ordinal: int, phase: str, start_ns: int, timeout: float,
                 max_response_bytes: int, api_key: str | None, ignore_eos: bool) -> dict:
    started_ns = time.perf_counter_ns()
    record = {"phase": phase, "ordinal": ordinal, "custom_id": item["custom_id"],
              "original_id": item["original_id"], "dataset": item["dataset"],
              "endpoint_index": endpoint_index, "endpoint": endpoint, "attempts": 1,
              "expected_input_tokens": item["expected_input_tokens"],
              "max_output_tokens": item["max_output_tokens"], "started_utc": utc_now(),
              "started_seconds": (started_ns - start_ns) / 1e9,
              "success": False, "http_status": None, "usage": None,
              "output_text": None, "finish_reason": None, "errors": []}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response_bytes = b""
    try:
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        # A new opener/connection per call; no automatic redirect or retry.
        opener = build_opener(NoRedirect())
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            record["http_status"] = response.status
            response_bytes = response.read(max_response_bytes + 1)
        record["response_received_seconds"] = (time.perf_counter_ns() - start_ns) / 1e9
        if len(response_bytes) > max_response_bytes:
            response_bytes = response_bytes[:max_response_bytes]
            raise ValueError(f"Response exceeded {max_response_bytes} bytes; raw response truncated")
        if record["http_status"] != 200:
            record["errors"].append(f"HTTP {record['http_status']}")
        result = json.loads(response_bytes)
        record["response"] = result
        if not isinstance(result, dict):
            raise ValueError("Response must be a JSON object")
        usage = result.get("usage")
        record["usage"] = usage
        if not isinstance(usage, dict) or not all(positive_integer(usage.get(key)) for key in ("prompt_tokens", "completion_tokens")):
            record["errors"].append("Missing or invalid actual token usage")
        else:
            if usage["prompt_tokens"] != item["expected_input_tokens"]:
                record["errors"].append("Actual prompt_tokens disagrees with workload")
            if usage["completion_tokens"] > item["max_output_tokens"]:
                record["errors"].append("Actual completion_tokens exceeds requested cap")
            if ignore_eos and usage["completion_tokens"] != item["max_output_tokens"]:
                record["errors"].append("ignore_eos fixed output length was not enforced")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or not isinstance(choices[0].get("text"), str):
            record["errors"].append("Expected exactly one completion choice with text")
        else:
            record["output_text"] = choices[0]["text"]
            record["finish_reason"] = choices[0].get("finish_reason")
            if record["finish_reason"] not in {"stop", "length"}:
                record["errors"].append("Missing or non-successful finish_reason")
        if result.get("error"):
            record["errors"].append("Server returned an error object")
        record["success"] = not record["errors"]
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    if "response" not in record:
        record["raw_response"] = response_bytes.decode("utf-8", errors="replace")
    completed_ns = time.perf_counter_ns()
    record.update({"completed_utc": utc_now(), "completed_seconds": (completed_ns - start_ns) / 1e9,
                   "request_latency_seconds": (completed_ns - started_ns) / 1e9})
    return record


def run_phase(items: list[dict], payloads: list[bytes], assignments: list[int],
              endpoints: list[str], phase: str, log_path: Path, concurrency: int,
              timeout: float, max_response_bytes: int, api_key: str | None,
              ignore_eos: bool) -> dict:
    records, pending, next_index = [], {}, 0
    phase_started_utc = utc_now()
    # Construct executor before the timed boundary, but create no network traffic.
    with log_path.open("w", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=concurrency) as executor:
        start_ns = time.perf_counter_ns()
        while next_index < len(items) or pending:
            while next_index < len(items) and len(pending) < concurrency:
                index = next_index
                endpoint_index = assignments[index]
                future = executor.submit(request_once, items[index], payloads[index],
                                         endpoints[endpoint_index], endpoint_index, index, phase,
                                         start_ns, timeout, max_response_bytes, api_key, ignore_eos)
                pending[future] = index
                next_index += 1
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Sorting simultaneous completions makes logs easier to audit. Actual
            # end time is recorded inside each request, not when log I/O finishes.
            for future in sorted(completed, key=lambda future: pending[future]):
                record = future.result()
                log.write(dumps(record) + "\n")
                log.flush()
                records.append(record)
                del pending[future]
    wall = max((record["completed_seconds"] for record in records), default=0.0)
    failures = [record for record in records if not record["success"]]
    successful = [record for record in records if record["success"]]
    output_tokens = sum(record["usage"]["completion_tokens"] for record in successful)
    all_valid = len(records) == len(items) and not failures
    return {"phase": phase, "started_utc": phase_started_utc,
            "scheduled_requests": len(items), "completed_requests": len(records),
            "successful_requests": len(successful), "failed_requests": len(failures),
            "failed_custom_ids": [record["custom_id"] for record in failures],
            "requests_per_endpoint": dict(Counter(str(record["endpoint_index"]) for record in records)),
            "wall_seconds": wall, "all_requests_valid": all_valid,
            "successful_prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in successful),
            "successful_output_tokens": output_tokens,
            "requests_per_second": len(records) / wall if all_valid and wall else None,
            "output_tokens_per_second": output_tokens / wall if all_valid and wall else None,
            "results_file": log_path.name, "results_sha256": sha256(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("input8192-output2048", "input2048-output8192"))
    parser.add_argument("--endpoint", required=True, action="append", help="Repeat for replicas; each URL receives ordinal modulo endpoint count")
    parser.add_argument("--concurrency", type=int, default=16, help="Global max in-flight requests, not per replica")
    parser.add_argument("--warmup-per-endpoint", type=int, default=1, help="Full workload requests per endpoint, excluded from timing; 0 must be justified separately")
    parser.add_argument("--count", type=int, help="Use this prefix of the workload, e.g. smoke/tuning; default all")
    parser.add_argument("--timeout-seconds", type=float, default=3600, help="Per-socket timeout; not a hard overall request or experiment deadline")
    parser.add_argument("--max-response-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name; secret value is never stored in config")
    parser.add_argument("--ignore-eos", action="store_true", help="Explicit fixed-output experiment; any shorter output invalidates the run")
    parser.add_argument("--output", required=True, type=Path, help="New result directory; refuses overwrite")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.warmup_per_endpoint < 0 or (args.count is not None and args.count <= 0):
        parser.error("concurrency/count must be positive and warmup nonnegative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.max_response_bytes <= 0:
        parser.error("timeout and response size limit must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new run directory")
    endpoints = [endpoint_url(value) for value in args.endpoint]
    if len(set(endpoints)) != len(endpoints):
        parser.error("Duplicate replica endpoints are not allowed")
    manifest, artifact, requests, payloads, effective_hash = load_workload(args.manifest, args.profile, args.count, args.ignore_eos)
    api_key = os.environ.get(args.api_key_env)
    configuration = {
        "schema_version": 1, "created_utc": utc_now(), "client_sha256": sha256(Path(__file__)),
        "python_version": platform.python_version(), "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest), "profile": args.profile,
        "request_file_sha256": artifact["requests_sha256"], "provenance_file_sha256": artifact["provenance_sha256"],
        "effective_payloads_sha256": effective_hash, "request_count": len(requests),
        "source_request_count": manifest["count_per_profile"], "endpoints": endpoints,
        "concurrency_global": args.concurrency, "warmup_per_endpoint": args.warmup_per_endpoint,
        "timeout_seconds_per_socket": args.timeout_seconds, "max_response_bytes": args.max_response_bytes,
        "temperature": 0.0, "ignore_eos": args.ignore_eos, "source_ignore_eos": manifest.get("ignore_eos"),
        "api_key_environment_variable": args.api_key_env, "api_key_present": bool(api_key),
        "models": sorted({item["body"]["model"] for item in requests}),
        "tokenizer_revision": manifest.get("tokenizer_revision"), "tokenizer_files": manifest.get("tokenizer_files"),
        "assignment": "request ordinal modulo endpoint count; bounded global concurrency",
        "timing": "first measured task submission through last complete response validation; warmup excluded",
        "retry_count": 0, "follow_redirects": False, "stream": False,
        "submission_mode": "synchronous completions, closed-loop admission, fresh connection per request",
        "paper_exact_reproduction": False,
        "limitations": ["Not BatchGen asynchronous full-batch API timing.",
                        "Runtime revisions, parallel topology and disabled prefix cache must be recorded externally.",
                        "A failed/invalid request invalidates throughput; no retries or successful-only speedup claims.",
                        "Response usage is checked, but matching counts alone does not prove matching token IDs.",
                        "Zero warmup cannot establish that model compilation/loading is excluded.",
                        "Socket timeout is not a hard deadline and interrupted clients cannot guarantee server-side request cancellation."],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "started", "config_sha256": sha256(config_path), "warmup": None,
               "measurement": None, "valid_for_comparison": False}
    exit_code = 1
    try:
        warmup_assignments = list(range(len(endpoints))) * args.warmup_per_endpoint
        if warmup_assignments:
            indexes = [index % len(requests) for index in range(len(warmup_assignments))]
            summary["warmup"] = run_phase([requests[index] for index in indexes], [payloads[index] for index in indexes],
                                          warmup_assignments, endpoints, "warmup", args.output / "warmup.jsonl",
                                          args.concurrency, args.timeout_seconds, args.max_response_bytes, api_key, args.ignore_eos)
            if not summary["warmup"]["all_requests_valid"]:
                summary["status"] = "warmup_failed_measurement_not_started"
                return 1
        assignments = [index % len(endpoints) for index in range(len(requests))]
        summary["measurement"] = run_phase(requests, payloads, assignments, endpoints, "measurement",
                                           args.output / "requests.jsonl", args.concurrency, args.timeout_seconds,
                                           args.max_response_bytes, api_key, args.ignore_eos)
        valid = summary["measurement"]["all_requests_valid"]
        summary["valid_for_comparison"] = valid
        summary["status"] = "completed" if valid else "completed_with_invalid_requests"
        exit_code = 0 if valid else 1
    except KeyboardInterrupt:
        summary["status"] = "interrupted_incomplete"
        exit_code = 130
    except Exception as error:
        summary["status"] = "client_error_incomplete"
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        summary["finished_utc"] = utc_now()
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"summary": str(summary_path.resolve()), "status": summary["status"],
                          "valid_for_comparison": summary["valid_for_comparison"],
                          "measurement": summary["measurement"]}, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
