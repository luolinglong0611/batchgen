#!/usr/bin/env python3
"""Measure audited workloads against non-streaming OpenAI completion endpoints.

Example, two eight-GPU replicas (async-burst requires aiohttp >= 3.10):
  python benchmark_client.py --manifest /data/workload/manifest.json \
    --profile input8192-output2048 \
    --endpoint http://node0:30000/v1 --endpoint http://node1:30000/v1 \
    --submission-mode async-burst --warmup-per-endpoint 1 --output /data/results/run1

Requests are assigned round-robin in manifest order, with NO retries or redirects.
closed-loop (default) uses a bounded thread pool; async-burst creates all request
tasks before running them, with no completion-dependent admission or connector
limit below the phase size. It checks the process FD budget before network I/O.
Warmup completes before measurement. Main wall time runs from the first network
attempt until every response body is received; JSON parsing, validation and result
serialization happen after all network work. Client admission, DNS/connection
setup, submission, server queueing, inference and body transfer are included.
Model startup/compilation must be completed by warmup before a formal run.

Neither mode recovers the paper's original artifact or BatchGen's batch API.
Record the admission mode and HTTP overhead when comparing end-to-end makespans.
Each request uses a fresh connection. The client does not discover
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
import asyncio
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import threading
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


class PhaseClock:
    """One origin at the first HTTP attempt, shared by threads or async tasks."""

    def __init__(self):
        self.origin_ns = None
        self.started_utc = None
        self.lock = threading.Lock()

    def begin(self):
        with self.lock:
            now = time.perf_counter_ns()
            if self.origin_ns is None:
                self.origin_ns, self.started_utc = now, utc_now()
            return (now - self.origin_ns) / 1e9

    def elapsed(self):
        return (time.perf_counter_ns() - self.origin_ns) / 1e9


def request_record(item, endpoint, endpoint_index, ordinal, phase):
    return {"phase": phase, "ordinal": ordinal, "custom_id": item["custom_id"],
              "original_id": item["original_id"], "dataset": item["dataset"],
              "endpoint_index": endpoint_index, "endpoint": endpoint, "attempts": 1,
              "expected_input_tokens": item["expected_input_tokens"],
              "max_output_tokens": item["max_output_tokens"],
              "success": False, "http_status": None, "usage": None,
              "output_text": None, "finish_reason": None, "errors": [],
              "response_body_complete": False, "_body": b""}


def request_headers(api_key):
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "Accept-Encoding": "identity"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def network_finished(record, clock):
    record["completed_seconds"] = clock.elapsed()
    record["completed_utc"] = utc_now()
    record["request_latency_seconds"] = record["completed_seconds"] - record["started_seconds"]


def request_once(item, payload, endpoint, endpoint_index, ordinal, phase,
                 clock, timeout, max_response_bytes, api_key):
    """Transport only. Response JSON is parsed after the entire phase drains."""
    record = request_record(item, endpoint, endpoint_index, ordinal, phase)
    request = Request(endpoint, data=payload, headers=request_headers(api_key), method="POST")
    opener = build_opener(NoRedirect())
    record["started_seconds"] = clock.begin()
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            record["http_status"] = response.status
            record["response_headers_received_seconds"] = clock.elapsed()
            data = response.read(max_response_bytes + 1)
            record["_body"] = data[:max_response_bytes]
            if len(data) > max_response_bytes:
                raise ValueError(f"Response exceeded {max_response_bytes} bytes; body incomplete")
            # urllib's bounded read need not raise on a short Content-Length body.
            length = response.headers.get("Content-Length")
            if length is not None and len(data) != int(length):
                raise ValueError("Incomplete response body (Content-Length mismatch)")
            record["response_body_complete"] = True
            record["response_received_seconds"] = clock.elapsed()
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    network_finished(record, clock)
    return record


def validate_response(record, ignore_eos):
    """Offline validation shared by both admission modes, never in network timing."""
    response_bytes = record.pop("_body")
    record["response_body_bytes"] = len(response_bytes)
    started = time.perf_counter()
    try:
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
            if usage["prompt_tokens"] != record["expected_input_tokens"]:
                record["errors"].append("Actual prompt_tokens disagrees with workload")
            if usage["completion_tokens"] > record["max_output_tokens"]:
                record["errors"].append("Actual completion_tokens exceeds requested cap")
            if ignore_eos and usage["completion_tokens"] != record["max_output_tokens"]:
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
    except Exception as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    if "response" not in record:
        record["raw_response"] = response_bytes.decode("utf-8", errors="replace")
    record["success"] = record["response_body_complete"] and not record["errors"]
    record["offline_validation_seconds"] = time.perf_counter() - started
    return record


def async_capacity(request_count, connection_limit=None, fd_reserve=64):
    """Refuse hidden connector throttling or insufficient local descriptor budget.

    This checks the process, not server capacity, ephemeral ports or network health.
    No process limits are changed. DNS/TLS/control descriptors get reserve space.
    """
    if request_count <= 0 or fd_reserve < 16:
        raise ValueError("Request count must be positive and FD reserve at least 16")
    limit = request_count if connection_limit is None else connection_limit
    if limit < request_count:
        raise ValueError("async-burst connection limit must cover every phase request")
    fd_directory = next((path for path in (Path("/proc/self/fd"), Path("/dev/fd")) if path.is_dir()), None)
    if fd_directory is None:
        raise RuntimeError("Cannot inventory open file descriptors for async-burst")
    open_fds = len(list(fd_directory.iterdir()))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = open_fds + request_count + fd_reserve
    if soft != resource.RLIM_INFINITY and required > soft:
        raise ValueError(f"Insufficient RLIMIT_NOFILE: soft={soft}, need at least {required} "
                         f"({request_count} sockets + {open_fds} open + {fd_reserve} reserve)")
    return {"request_count": request_count, "connector_limit": limit,
            "connector_limit_per_host": 0, "happy_eyeballs_delay": None,
            "open_fds": open_fds,
            "fd_soft_limit": soft, "fd_hard_limit": hard,
            "fd_reserve": fd_reserve, "required_fd_budget": required,
            "fd_inventory_path": str(fd_directory), "limits_changed": False}


async def async_transport(items, payloads, assignments, endpoints, phase, clock,
                          timeout, max_response_bytes, api_key, connection_limit, fd_reserve):
    import aiohttp

    capacity = async_capacity(len(items), connection_limit, fd_reserve)
    # No semaphore; a connector slot is available for every request. Force fresh
    # connections to avoid persistent-connection retry behavior and warmup reuse.
    # Disable parallel address-family connection races: one outstanding socket
    # per request keeps the FD budget meaningful even for dual-stack hostnames.
    connector = aiohttp.TCPConnector(limit=capacity["connector_limit"], limit_per_host=0,
                                    force_close=True, happy_eyeballs_delay=None)
    trace = aiohttp.TraceConfig()

    async def payload_dispatch(session, context, params):
        context.trace_request_ctx["payload_dispatch_seconds"] = clock.elapsed()

    trace.on_request_chunk_sent.append(payload_dispatch)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout,
                                     trust_env=False, auto_decompress=False,
                                     trace_configs=[trace]) as session:
        async def one(index):
            endpoint_index = assignments[index]
            record = request_record(items[index], endpoints[endpoint_index], endpoint_index, index, phase)
            body = bytearray()
            record["started_seconds"] = clock.begin()
            try:
                # POST is never retried; redirects are returned as failed results.
                async with session.post(endpoints[endpoint_index], data=payloads[index],
                                        headers=request_headers(api_key), allow_redirects=False,
                                        trace_request_ctx=record) as response:
                    record["http_status"] = response.status
                    record["response_headers_received_seconds"] = clock.elapsed()
                    async for block in response.content.iter_chunked(64 * 1024):
                        remaining = max_response_bytes - len(body)
                        body.extend(block[:remaining])
                        if len(block) > remaining:
                            raise ValueError(f"Response exceeded {max_response_bytes} bytes; body incomplete")
                    record["response_body_complete"] = True
                    record["response_received_seconds"] = clock.elapsed()
            except Exception as error:
                record["errors"].append(f"{type(error).__name__}: {error}")
            record["_body"] = bytes(body)
            network_finished(record, clock)
            return record

        # No await in this loop: all requests exist before any starts. Transport
        # scheduling is concurrent, independent of preceding response completion.
        tasks = [asyncio.create_task(one(index)) for index in range(len(items))]
        records = await asyncio.gather(*tasks)
    capacity["aiohttp_version"] = aiohttp.__version__
    return records, capacity


def run_phase(items: list[dict], payloads: list[bytes], assignments: list[int],
              endpoints: list[str], phase: str, log_path: Path, concurrency: int,
              timeout: float, max_response_bytes: int, api_key: str | None,
              ignore_eos: bool, submission_mode="closed-loop", connection_limit=None,
              fd_reserve=64) -> dict:
    if not items or not (len(items) == len(payloads) == len(assignments)):
        raise ValueError("Nonempty items/payloads/assignments must have equal lengths")
    clock, capacity = PhaseClock(), None
    records, pending, next_index = [], {}, 0
    if submission_mode == "async-burst":
        records, capacity = asyncio.run(async_transport(
            items, payloads, assignments, endpoints, phase, clock, timeout,
            max_response_bytes, api_key, connection_limit, fd_reserve))
    elif submission_mode == "closed-loop":
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while next_index < len(items) or pending:
                while next_index < len(items) and len(pending) < concurrency:
                    index = next_index
                    endpoint_index = assignments[index]
                    future = executor.submit(request_once, items[index], payloads[index],
                                             endpoints[endpoint_index], endpoint_index, index, phase,
                                             clock, timeout, max_response_bytes, api_key)
                    pending[future] = index
                    next_index += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda future: pending[future]):
                    records.append(future.result())
                    del pending[future]
    else:
        raise ValueError(f"Unknown submission mode: {submission_mode}")
    all_bodies_received = all(record["response_body_complete"] for record in records)
    wall = max(record["response_received_seconds"] for record in records) if all_bodies_received else None
    transport_finished = max(record["completed_seconds"] for record in records)
    starts = [record["started_seconds"] for record in records]
    dispatches = [record["payload_dispatch_seconds"] for record in records if "payload_dispatch_seconds" in record]
    received = [record["response_received_seconds"] for record in records if record["response_body_complete"]]
    # No parsing, validation, per-response JSON serialization, or logging delays
    # can affect admission or any other request's timed network progress.
    with log_path.open("x", encoding="utf-8") as log:
        for record in records:
            validate_response(record, ignore_eos)
            log.write(dumps(record) + "\n")
    failures = [record for record in records if not record["success"]]
    successful = [record for record in records if record["success"]]
    output_tokens = sum(record["usage"]["completion_tokens"] for record in successful)
    all_valid = len(records) == len(items) and not failures
    return {"phase": phase, "started_utc": clock.started_utc,
            "scheduled_requests": len(items), "completed_requests": len(records),
            "successful_requests": len(successful), "failed_requests": len(failures),
            "failed_custom_ids": [record["custom_id"] for record in failures],
            "requests_per_endpoint": dict(Counter(str(record["endpoint_index"]) for record in records)),
            "wall_seconds": wall, "all_requests_valid": all_valid,
            "submission_mode": submission_mode, "capacity_check": capacity,
            "all_response_bodies_received": all_bodies_received,
            "transport_finished_seconds": transport_finished,
            "submission_attempt_span_seconds": max(starts) - min(starts),
            "last_submission_attempt_seconds": max(starts),
            "payload_dispatch_span_seconds": max(dispatches) - min(dispatches) if dispatches else None,
            "payload_dispatch_observed_requests": len(dispatches),
            "last_payload_dispatch_seconds": max(dispatches) if dispatches else None,
            "all_attempts_started_before_first_body_complete": max(starts) <= min(received) if received else None,
            "payload_dispatch_semantics": "aiohttp client write trace, not server acceptance; unavailable for urllib",
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
    parser.add_argument("--submission-mode", choices=("closed-loop", "async-burst"), default="closed-loop",
                        help="closed-loop waits for free request slots; async-burst submits all without completion-dependent admission")
    parser.add_argument("--concurrency", type=int, default=16, help="Closed-loop only: global max in-flight requests, not per replica; ignored in async-burst")
    parser.add_argument("--connection-limit", type=int, help="Async-burst connector capacity; default phase size; must cover every phase request")
    parser.add_argument("--fd-reserve", type=int, default=64, help="Async-burst extra process FD budget, minimum 16; no ulimit changes")
    parser.add_argument("--warmup-per-endpoint", type=int, default=1, help="Full workload requests per endpoint, excluded from timing; 0 must be justified separately")
    parser.add_argument("--count", type=int, help="Use this prefix of the workload, e.g. smoke/tuning; default all")
    parser.add_argument("--timeout-seconds", type=float, default=3600, help="Async-burst: total deadline per request. Closed-loop: per-socket timeout, not overall deadline")
    parser.add_argument("--max-response-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name; secret value is never stored in config")
    parser.add_argument("--ignore-eos", action="store_true", help="Explicit fixed-output experiment; any shorter output invalidates the run")
    parser.add_argument("--output", required=True, type=Path, help="New result directory; refuses overwrite")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.warmup_per_endpoint < 0 or (args.count is not None and args.count <= 0):
        parser.error("concurrency/count must be positive and warmup nonnegative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.max_response_bytes <= 0:
        parser.error("timeout and response size limit must be positive")
    if args.fd_reserve < 16 or (args.connection_limit is not None and args.connection_limit <= 0):
        parser.error("fd-reserve must be >=16 and connection-limit positive")
    if args.submission_mode == "closed-loop" and args.connection_limit is not None:
        parser.error("--connection-limit applies only to async-burst")
    if args.output.exists():
        parser.error("--output already exists; choose a new run directory")
    endpoints = [endpoint_url(value) for value in args.endpoint]
    if len(set(endpoints)) != len(endpoints):
        parser.error("Duplicate replica endpoints are not allowed")
    manifest, artifact, requests, payloads, effective_hash = load_workload(args.manifest, args.profile, args.count, args.ignore_eos)
    capacity_preflight, aiohttp_version = None, None
    if args.submission_mode == "async-burst":
        import aiohttp
        aiohttp_version = aiohttp.__version__
        capacity_preflight = async_capacity(max(len(requests), args.warmup_per_endpoint * len(endpoints)),
                                           args.connection_limit, args.fd_reserve)
    api_key = os.environ.get(args.api_key_env)
    configuration = {
        "schema_version": 2, "created_utc": utc_now(), "client_sha256": sha256(Path(__file__)),
        "python_version": platform.python_version(), "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest), "profile": args.profile,
        "request_file_sha256": artifact["requests_sha256"], "provenance_file_sha256": artifact["provenance_sha256"],
        "effective_payloads_sha256": effective_hash, "request_count": len(requests),
        "source_request_count": manifest["count_per_profile"], "endpoints": endpoints,
        "concurrency_global": args.concurrency if args.submission_mode == "closed-loop" else None,
        "warmup_per_endpoint": args.warmup_per_endpoint,
        "connection_limit": args.connection_limit, "capacity_preflight": capacity_preflight,
        "aiohttp_version": aiohttp_version, "fd_reserve": args.fd_reserve,
        "timeout_seconds": args.timeout_seconds,
        "timeout_semantics": "per request total" if args.submission_mode == "async-burst" else "per socket operation",
        "max_response_bytes": args.max_response_bytes,
        "temperature": 0.0, "ignore_eos": args.ignore_eos, "source_ignore_eos": manifest.get("ignore_eos"),
        "api_key_environment_variable": args.api_key_env, "api_key_present": bool(api_key),
        "models": sorted({item["body"]["model"] for item in requests}),
        "tokenizer_revision": manifest.get("tokenizer_revision"), "tokenizer_files": manifest.get("tokenizer_files"),
        "assignment": "request ordinal modulo endpoint count",
        "timing": "first network attempt through final complete response body; all JSON parsing/validation/serialization after transport; warmup excluded",
        "retry_count": 0, "follow_redirects": False, "stream": False,
        "submission_mode": args.submission_mode,
        "admission": "all tasks ready before network; no response-dependent admission" if args.submission_mode == "async-burst" else "bounded completion-dependent admission",
        "connections": "fresh connection per request; no redirects or request retries",
        "paper_exact_reproduction": False,
        "limitations": ["OpenAI completion transport differs from BatchGen file/batch API; compare declared submission and retrieval boundaries.",
                        "Runtime revisions, parallel topology and disabled prefix cache must be recorded externally.",
                        "A failed/invalid request invalidates throughput; no retries or successful-only speedup claims.",
                        "Response usage is checked, but matching counts alone does not prove matching token IDs.",
                        "Zero warmup cannot establish that model compilation/loading is excluded.",
                        "Closed-loop socket timeout is not a hard deadline; interrupted clients cannot guarantee server-side request cancellation.",
                        "FD checks do not establish server connection capacity, ephemeral-port availability, or client/network saturation.",
                        "Raw response bodies are held in client memory until all network requests finish; a process interruption may leave an incomplete run without per-request artifacts."],
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
                                          args.concurrency, args.timeout_seconds, args.max_response_bytes, api_key, args.ignore_eos,
                                          args.submission_mode, args.connection_limit, args.fd_reserve)
            if not summary["warmup"]["all_requests_valid"]:
                summary["status"] = "warmup_failed_measurement_not_started"
                return 1
        assignments = [index % len(endpoints) for index in range(len(requests))]
        summary["measurement"] = run_phase(requests, payloads, assignments, endpoints, "measurement",
                                           args.output / "requests.jsonl", args.concurrency, args.timeout_seconds,
                                           args.max_response_bytes, api_key, args.ignore_eos,
                                           args.submission_mode, args.connection_limit, args.fd_reserve)
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
