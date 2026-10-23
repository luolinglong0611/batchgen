"""Local HTTP contract tests for audited R1 client admission and BCT boundaries."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import sys

import pytest

pytest.importorskip("aiohttp")
SOURCE = Path(__file__).resolve().parents[1] / "scripts/r1_reproduction/benchmark_client.py"
spec = importlib.util.spec_from_file_location("r1_benchmark_test", SOURCE)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class State:
    def __init__(self, expected=1, barrier=False, status=200, incomplete=False, delay=0.04, prompt_tokens=8):
        self.expected, self.barrier, self.status = expected, barrier, status
        self.incomplete, self.delay = incomplete, delay
        self.prompt_tokens = prompt_tokens
        self.requests, self.finished = [], 0
        self.lock, self.all_arrived = threading.Lock(), threading.Event()


@pytest.fixture
def servers():
    running = []

    def make(state):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with state.lock:
                    state.requests.append((self.path, body, self.server.server_port))
                    if len(state.requests) == state.expected:
                        state.all_arrived.set()
                if state.barrier and not state.all_arrived.wait(2):
                    self.send_error(503, "Client waited for completion instead of submitting all requests")
                    return
                result = {"usage": {"prompt_tokens": state.prompt_tokens, "completion_tokens": 2},
                          "choices": [{"text": "result-" + body["prompt"], "finish_reason": "length"}]}
                raw = json.dumps(result).encode()
                self.send_response(state.status)
                if state.status == 307:
                    self.send_header("Location", "/redirected")
                if state.incomplete:
                    self.send_header("Content-Length", str(len(raw) + 10))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(raw)
                    self.close_connection = True
                    return
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # Headers and the first half arrive early; final body is delayed.
                first, second = raw[:len(raw)//2], raw[len(raw)//2:]
                try:
                    self.wfile.write(f"{len(first):x}\r\n".encode() + first + b"\r\n")
                    self.wfile.flush()
                    time.sleep(state.delay)
                    with state.lock:
                        state.finished += 1
                    self.wfile.write(f"{len(second):x}\r\n".encode() + second + b"\r\n0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/v1/completions"

    yield make
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join()


def workload(count):
    items = [{"custom_id": str(i), "original_id": str(i), "dataset": "fixture",
              "expected_input_tokens": 8, "max_output_tokens": 2,
              "body": {"prompt": str(i), "model": "fixture", "max_tokens": 2,
                       "temperature": 0.0, "ignore_eos": True}} for i in range(count)]
    return items, [json.dumps(item["body"]).encode() for item in items]


def run(tmp_path, endpoints, count=1, mode="async-burst", max_bytes=10000):
    items, payloads = workload(count)
    return client.run_phase(items, payloads, [i % len(endpoints) for i in range(count)],
                            endpoints, "test", tmp_path / "responses.jsonl", 1, 5,
                            max_bytes, None, True, mode)


def test_async_burst_admits_all_before_completion_and_keeps_round_robin(servers, tmp_path):
    state = State(expected=6, barrier=True)
    endpoints = [servers(state), servers(state)]
    summary = run(tmp_path, endpoints, count=6)
    assert summary["all_requests_valid"]
    assert state.all_arrived.is_set() and len(state.requests) == 6
    assert summary["requests_per_endpoint"] == {"0": 3, "1": 3}
    assert summary["all_attempts_started_before_first_body_complete"]
    assert summary["payload_dispatch_observed_requests"] == 6
    assert summary["submission_attempt_span_seconds"] >= 0
    assert summary["capacity_check"]["connector_limit"] == 6
    records = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    for row in records:
        assert row["endpoint_index"] == row["ordinal"] % 2
        assert row["output_text"] == "result-" + str(row["ordinal"])
        assert row["response_received_seconds"] - row["response_headers_received_seconds"] >= state.delay * 0.7
        assert row["attempts"] == 1 and row["response_body_complete"]
    assert summary["wall_seconds"] == max(row["response_received_seconds"] for row in records)


@pytest.mark.parametrize("mode", ["closed-loop", "async-burst"])
def test_validation_is_after_all_network_and_excluded_from_wall(servers, tmp_path, monkeypatch, mode):
    state = State(expected=3)
    original = client.validate_response

    def slow_validation(record, ignore_eos):
        assert state.finished == state.expected
        time.sleep(0.04)
        return original(record, ignore_eos)

    monkeypatch.setattr(client, "validate_response", slow_validation)
    before = time.perf_counter()
    summary = run(tmp_path, [servers(state)], count=3, mode=mode)
    elapsed = time.perf_counter() - before
    assert summary["all_requests_valid"]
    assert elapsed - summary["wall_seconds"] >= 0.1
    records = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    if mode == "closed-loop":
        ordered = sorted(records, key=lambda row: row["ordinal"])
        assert ordered[1]["started_seconds"] >= ordered[0]["response_received_seconds"]
        assert summary["capacity_check"] is None


@pytest.mark.parametrize("status", [503, 307])
@pytest.mark.parametrize("mode", ["closed-loop", "async-burst"])
def test_failures_and_redirects_are_not_retried(servers, tmp_path, status, mode):
    state = State(status=status, delay=0)
    summary = run(tmp_path, [servers(state)], mode=mode)
    assert len(state.requests) == 1
    assert state.requests[0][0] == "/v1/completions"
    assert summary["failed_requests"] == 1
    assert not summary["all_requests_valid"]
    assert summary["requests_per_second"] is None
    assert summary["output_tokens_per_second"] is None


@pytest.mark.parametrize("mode", ["closed-loop", "async-burst"])
def test_short_body_is_not_success_even_if_json_is_complete(servers, tmp_path, mode):
    state = State(incomplete=True)
    summary = run(tmp_path, [servers(state)], mode=mode)
    assert not summary["all_response_bodies_received"]
    assert summary["wall_seconds"] is None
    assert summary["failed_requests"] == 1
    assert len(state.requests) == 1


@pytest.mark.parametrize("mode", ["closed-loop", "async-burst"])
def test_response_size_limit_invalidates_run(servers, tmp_path, mode):
    summary = run(tmp_path, [servers(State(delay=0))], mode=mode, max_bytes=10)
    assert not summary["all_response_bodies_received"]
    assert summary["wall_seconds"] is None
    row = json.loads((tmp_path / "responses.jsonl").read_text())
    assert row["response_body_bytes"] <= 10
    assert any("exceeded" in error for error in row["errors"])


def test_connection_limit_cannot_silently_create_closed_loop():
    with pytest.raises(ValueError, match="cover every phase request"):
        client.async_capacity(6, connection_limit=5)


def test_insufficient_fd_capacity_fails_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(client.resource, "getrlimit", lambda key: (10, 10))
    with pytest.raises(ValueError, match="Insufficient RLIMIT_NOFILE"):
        run(tmp_path, ["http://127.0.0.1:1/v1/completions"])
    assert not (tmp_path / "responses.jsonl").exists()


def test_usage_and_fixed_length_validation_remain_strict():
    item = workload(1)[0][0]
    for usage in ({"prompt_tokens": 9, "completion_tokens": 2},
                  {"prompt_tokens": 8, "completion_tokens": 1}):
        record = client.request_record(item, "http://unused", 0, 0, "test")
        record.update(http_status=200, response_body_complete=True,
                      _body=json.dumps({"usage": usage, "choices": [{"text": "x", "finish_reason": "length"}]}).encode())
        client.validate_response(record, True)
        assert not record["success"] and record["errors"]


def write_cli_manifest(directory):
    requests, provenance = [], []
    for item in workload(4)[0]:
        item["body"]["max_tokens"] = 2048
        requests.append({"custom_id": item["custom_id"], "method": "POST", "url": "/v1/completions", "body": item["body"]})
        provenance.append({"custom_id": item["custom_id"], "original_id": item["original_id"], "dataset": "fixture",
                           "input_tokens": 8192, "max_output_tokens": 2048})
    for filename, rows in (("requests.jsonl", requests), ("provenance.jsonl", provenance)):
        (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"count_per_profile": 4, "artifacts": [{"input_tokens": 8192, "max_output_tokens": 2048,
                "requests": "requests.jsonl", "requests_sha256": client.sha256(directory / "requests.jsonl"),
                "provenance": "provenance.jsonl", "provenance_sha256": client.sha256(directory / "provenance.jsonl")}]}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("mode", ["closed-loop", "async-burst"])
def test_cli_reuses_manifest_keeps_warmup_separate_and_records_mode(servers, tmp_path, monkeypatch, mode):
    state = State(expected=6, prompt_tokens=8192, delay=0)
    endpoints = [servers(state), servers(state)]
    manifest, output = write_cli_manifest(tmp_path), tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--manifest", str(manifest),
        "--profile", "input8192-output2048", "--endpoint", endpoints[0], "--endpoint", endpoints[1],
        "--submission-mode", mode, "--concurrency", "1", "--output", str(output)])
    assert client.main() == 0
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert config["submission_mode"] == mode
    assert config["manifest_sha256"] == client.sha256(manifest)
    assert config["concurrency_global"] == (None if mode == "async-burst" else 1)
    assert summary["warmup"]["completed_requests"] == 2
    assert summary["measurement"]["completed_requests"] == 4
    assert summary["measurement"]["requests_per_endpoint"] == {"0": 2, "1": 2}
    assert len(state.requests) == 6
    assert all(body["ignore_eos"] is False for _, body, _ in state.requests)
    assert summary["valid_for_comparison"]


def test_failed_async_warmup_never_starts_measurement(servers, tmp_path, monkeypatch):
    state = State(status=503, delay=0, prompt_tokens=8192)
    manifest, output = write_cli_manifest(tmp_path), tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--manifest", str(manifest),
        "--profile", "input8192-output2048", "--endpoint", servers(state),
        "--submission-mode", "async-burst", "--output", str(output)])
    assert client.main() == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "warmup_failed_measurement_not_started"
    assert summary["measurement"] is None and len(state.requests) == 1
