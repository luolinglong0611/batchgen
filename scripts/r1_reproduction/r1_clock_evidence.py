#!/usr/bin/env python3
"""Independent clock evidence; never rewrites or qualifies model evidence.

The network commands are opt-in. Tests exercise the arithmetic without sockets.
Relative drift is an explicit assumption, not a bound measured by this tool.
"""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
import time

SCHEMA = "r1-cross-node-clock-exchange-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {"unix_ns": wall, "monotonic_before_ns": before, "monotonic_after_ns": after}


def identity(address):
    # Deliberately require Linux boot identity for the two deployment hosts.
    return {"address": address, "hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "source_sha256": digest(__file__)}


def check_stamp(value):
    fields = ("unix_ns", "monotonic_before_ns", "monotonic_after_ns")
    if (not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] <= 0 for k in fields)
            or value["monotonic_before_ns"] > value["monotonic_after_ns"]):
        raise ValueError("Invalid actual clock observation")


def check_elapsed(a, b, tolerance_ns):
    check_stamp(a)
    check_stamp(b)
    low = b["monotonic_before_ns"] - a["monotonic_after_ns"]
    high = b["monotonic_after_ns"] - a["monotonic_before_ns"]
    wall = b["unix_ns"] - a["unix_ns"]
    if low < 0 or not low - tolerance_ns <= wall <= high + tolerance_ns:
        raise ValueError("Clock step or inconsistent monotonic/wall elapsed time")


def exchange_bounds(row, tolerance_ns):
    """server_clock - client_clock; include unknown one-way network asymmetry."""
    a, b, c, d = (row[k] for k in ("client_send", "server_receive", "server_send", "client_receive"))
    check_elapsed(a, d, tolerance_ns)
    check_elapsed(b, c, tolerance_ns)
    low = c["unix_ns"] - d["unix_ns"]
    high = b["unix_ns"] - a["unix_ns"]
    if low > high:
        raise ValueError("Inconsistent four-timestamp exchange")
    return low, high


def validate_samples(rows, tolerance_ns):
    if type(tolerance_ns) is not int or tolerance_ns < 0 or len(rows) < 2:
        raise ValueError("Need at least two complete exchanges and an explicit step tolerance")
    first = rows[0]
    for i, row in enumerate(rows):
        if (row.get("schema") != SCHEMA or type(row.get("ordinal")) is not int or row["ordinal"] != i
                or row.get("session_id") != first.get("session_id")
                or row.get("client") != first.get("client") or row.get("server") != first.get("server")):
            raise ValueError("Mixed/reordered clock samples or changed boot/source/host identity")
        for peer in (row["client"], row["server"]):
            if (not all(isinstance(peer.get(k), str) and peer[k] for k in ("address", "hostname", "boot_id"))
                    or peer.get("source_sha256") != digest(__file__)):
                raise ValueError("Clock source/host identity is missing or changed")
        exchange_bounds(row, tolerance_ns)
        if i:
            check_elapsed(rows[i - 1]["client_receive"], row["client_send"], tolerance_ns)
            check_elapsed(rows[i - 1]["server_send"], row["server_receive"], tolerance_ns)
    return rows


def event_client_interval(event_ns, rows, relative_drift_ppb, max_probe_gap_ns, tolerance_ns):
    """Conservative event interval, conditional on a continuous relative drift bound.

    For offset O, server timestamp e = client timestamp x + O(x). At exchange
    time a in [c1,c4], O(a) is in [L,U]. With |O(x)-O(a)| <= rho*|x-a|,
    expanding [e-U,e-L] by rho*D/(1-rho) encloses x. Intersect every exchange;
    retain asymmetry rather than assuming half-RTT is the offset error.
    """
    if (type(event_ns) is not int or event_ns <= 0 or type(relative_drift_ppb) is not int
            or not 0 <= relative_drift_ppb < 1_000_000_000
            or type(max_probe_gap_ns) is not int or max_probe_gap_ns <= 0):
        raise ValueError("Invalid timestamp, drift assumption or probe-gap policy")
    low, high = None, None
    for row in rows:
        offset_low, offset_high = exchange_bounds(row, tolerance_ns)
        a = row["client_send"]["unix_ns"]
        b = row["client_receive"]["unix_ns"]
        # Drift during the exchange must also be covered; NTP's fixed-offset
        # inequalities alone are not a proof of constant clock offset.
        exchange_drift = (relative_drift_ppb * (b - a) + 999_999_999) // 1_000_000_000
        offset_low -= exchange_drift
        offset_high += exchange_drift
        x_low, x_high = event_ns - offset_high, event_ns - offset_low
        distance = max(abs(x_low - a), abs(x_low - b), abs(x_high - a), abs(x_high - b))
        denominator = 1_000_000_000 - relative_drift_ppb
        margin = (relative_drift_ppb * distance + denominator - 1) // denominator
        candidate = (x_low - margin, x_high + margin)
        low = candidate[0] if low is None else max(low, candidate[0])
        high = candidate[1] if high is None else min(high, candidate[1])
    if low is None or low > high:
        raise ValueError("Clock observations conflict with the explicit drift assumption")
    before = [r["client_receive"]["unix_ns"] for r in rows if r["client_receive"]["unix_ns"] <= low]
    after = [r["client_send"]["unix_ns"] for r in rows if r["client_send"]["unix_ns"] >= high]
    if not before or not after or min(after) - max(before) > max_probe_gap_ns:
        raise ValueError("Event lacks close pre/post clock observations; no extrapolation")
    return low, high


def classify(interval, start, finish):
    low, high = interval
    if not start <= finish or not low <= high:
        raise ValueError("Invalid original time window")
    if start <= low <= high <= finish:
        return "supported_under_declared_clock_assumption"
    if high < start or low > finish:
        return "outside_original_window"
    return "uncertain_at_original_window_boundary"


class ClockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        received = stamp()
        if (self.client_address[0] != self.server.client_ip or self.path != "/clock/" + self.server.session_id
                or self.headers.get("Content-Length") not in (None, "0") or self.headers.get("Transfer-Encoding")):
            self.send_error(403)
            return
        nonce = self.headers.get("X-R1-Clock-Probe", "")
        if not re.fullmatch("[0-9a-f]{32}", nonce):
            self.send_error(400)
            return
        response = {"schema": SCHEMA, "session_id": self.server.session_id, "nonce": nonce,
                    "server": self.server.identity, "server_receive": received, "server_send": stamp()}
        raw = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ClockServer(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(3)
        return connection, address


def serve(args):
    with ClockServer((args.server_ip, args.port), ClockHandler) as server:
        server.client_ip, server.session_id = args.client_ip, args.session_id
        server.identity = identity(args.server_ip)
        server.timeout = 1
        end = time.monotonic() + args.duration_seconds
        while time.monotonic() < end:
            server.handle_request()
    return {"status": "clock_endpoint_closed", "session_id": args.session_id}


def probe(args):
    own = identity(args.client_ip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        for ordinal in range(args.count):
            connection = http.client.HTTPConnection(args.server_ip, args.port, timeout=args.timeout_seconds,
                                                    source_address=(args.client_ip, 0))
            nonce = secrets.token_hex(16)
            try:
                sent = stamp()
                connection.request("GET", "/clock/" + args.session_id, headers={"X-R1-Clock-Probe": nonce})
                response = connection.getresponse()
                raw = response.read(4097)
                received = stamp()
                if response.status != 200 or len(raw) > 4096:
                    raise ValueError("Invalid clock endpoint response")
                value = json.loads(raw)
                if (value.get("schema") != SCHEMA or value.get("session_id") != args.session_id
                        or value.get("nonce") != nonce or value.get("server", {}).get("address") != args.server_ip
                        or value["server"].get("source_sha256") != own["source_sha256"]):
                    raise ValueError("Clock endpoint/session/source identity mismatch")
                value.update(ordinal=ordinal, client=own, client_send=sent, client_receive=received)
                del value["nonce"]
                output.write(json.dumps(value, separators=(",", ":")) + "\n")
                output.flush()
            finally:
                connection.close()
            if ordinal + 1 < args.count:
                time.sleep(args.interval_seconds)
        os.fsync(output.fileno())
    return {"status": "clock_observations_NOT_model_qualification", "output": str(args.output),
            "sha256": digest(args.output), "samples": args.count}


def audit(args):
    for path, expected in ((args.samples, args.samples_sha256), (args.run, args.run_sha256), (args.events, args.events_sha256)):
        if digest(path) != expected:
            raise ValueError("Pinned original input bytes changed")
    if not args.samples.read_bytes().endswith(b"\n"):
        raise ValueError("Clock journal has an unclosed final observation")
    rows = validate_samples([json.loads(line) for line in args.samples.read_text().splitlines()], args.step_tolerance_ns)
    run = json.loads(args.run.read_text())
    events = json.loads(args.events.read_text())
    if rows[0]["session_id"] != run["deployment"]["run_id"]:
        raise ValueError("Clock observations belong to a different run session")
    # This independent check only adds clock evidence. The frozen qualifier must
    # still validate schedule, all receipts, identities, layers and full logits.
    windows = {}
    for group in run["collection_receipts"]:
        if run["backend"] == "sglang":
            calls = group["requests"]
        else:
            calls = [{"id": key, "started_unix_ns": group["started_unix_ns"],
                      "finished_unix_ns": group["finished_unix_ns"]} for key in group["planned_ids"]]
        for call in calls:
            if call["id"] in windows:
                raise ValueError("Duplicate request in original cohort receipts")
            windows[call["id"]] = (call["started_unix_ns"], call["finished_unix_ns"])
    results = []
    for event in events:
        if (event.get("run_id") != run["deployment"]["run_id"]
                or event.get("suite_sha256") != run["suite_sha256"]
                or event.get("deployment_sha256") != run["deployment"]["artifact_sha256"]):
            raise ValueError("Raw events belong to different original run/suite/deployment")
        if event.get("rank") not in range(8, 16) or event.get("event") not in ("input_tokens", "logits", "completion"):
            continue
        if event.get("request_id") not in windows:
            continue
        start, finish = windows[event["request_id"]]
        row = {"rank": event["rank"], "ordinal": event["ordinal"], "request_id": event["request_id"],
               "observed_unix_ns": event["observed_unix_ns"], "original_window": [start, finish]}
        try:
            interval = event_client_interval(event["observed_unix_ns"], rows, args.relative_drift_ppb,
                                             args.max_probe_gap_ns, args.step_tolerance_ns)
            row.update(client_time_interval=list(interval), status=classify(interval, start, finish))
        except ValueError as error:
            row.update(status="clock_evidence_incomplete", reason=str(error))
        row["original_unadjusted_gate"] = start <= event["observed_unix_ns"] <= finish
        results.append(row)
    supported = (bool(results) and all(r["status"] == "supported_under_declared_clock_assumption"
                                       and r["original_unadjusted_gate"] for r in results))
    report = {"schema": "r1-independent-clock-window-audit-v1",
              "status": "supplementary_windows_supported_NOT_qualified" if supported else "clock_window_evidence_NOT_supported",
              "tool_sha256": digest(__file__), "run_sha256": args.run_sha256, "events_sha256": args.events_sha256,
              "samples_sha256": args.samples_sha256, "client": rows[0]["client"], "server": rows[0]["server"],
              "session_id": rows[0]["session_id"], "assumed_max_relative_drift_ppb": args.relative_drift_ppb,
              "max_probe_gap_ns": args.max_probe_gap_ns, "step_tolerance_ns": args.step_tolerance_ns,
              "original_evidence_modified": False, "formal_performance_permitted": False, "events": results}
    report["limitations"] = ["continuous_relative_drift_bound_is_declared_not_measured",
        "caller_must_bind_collector_host_and_boot_to_client_identity",
        "caller_must_supply_full_raw_event_union_before_any_time_filter",
        "original_qualifier_and_all_other_gates_still_required"]
    offsets = [exchange_bounds(row, args.step_tolerance_ns) for row in rows]
    report["offset_observations"] = {"sign": "server_unix_minus_client_unix",
        "first_interval_ns": list(offsets[0]), "last_interval_ns": list(offsets[-1]),
        "minimum_exchange_uncertainty_width_ns": min(hi - lo for lo, hi in offsets),
        "exchanges": len(rows)}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    return {"status": report["status"], "output": str(args.output), "sha256": digest(args.output), "events": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "probe"):
        sub = commands.add_parser(name)
        sub.add_argument("--server-ip", required=True)
        sub.add_argument("--client-ip", required=True)
        sub.add_argument("--port", type=int, required=True)
        sub.add_argument("--session-id", required=True)
        sub.add_argument("--execute", action="store_true")
        if name == "serve":
            sub.add_argument("--duration-seconds", type=float, required=True)
        else:
            sub.add_argument("--count", type=int, required=True)
            sub.add_argument("--interval-seconds", type=float, default=1)
            sub.add_argument("--timeout-seconds", type=float, default=3)
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("audit")
    for name in ("samples", "run", "events"):
        sub.add_argument("--" + name, type=Path, required=True)
        sub.add_argument("--" + name + "-sha256", required=True)
    sub.add_argument("--relative-drift-ppb", type=int, required=True)
    sub.add_argument("--max-probe-gap-ns", type=int, required=True)
    sub.add_argument("--step-tolerance-ns", type=int, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command in ("serve", "probe"):
        for address in (args.server_ip, args.client_ip):
            if str(ipaddress.IPv4Address(address)) != address or address == "0.0.0.0":
                raise ValueError("Use the explicit literal deployment IPv4 addresses")
        if (args.client_ip == args.server_ip or not 1 <= args.port <= 65535
                or not re.fullmatch("[A-Za-z0-9_-]{1,100}", args.session_id)):
            raise ValueError("Invalid clock endpoint identity")
        if args.command == "probe" and (args.count < 2 or not 0 < args.interval_seconds <= 60 or not 0 < args.timeout_seconds <= 60):
            raise ValueError("Invalid finite probe count/interval/timeout")
        if args.command == "serve" and not 0 < args.duration_seconds <= 86400:
            raise ValueError("Invalid finite endpoint lifetime")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "command": args.command, "source_sha256": digest(__file__)}))
        return
    print(json.dumps({"serve": serve, "probe": probe, "audit": audit}[args.command](args)))


if __name__ == "__main__":
    main()
