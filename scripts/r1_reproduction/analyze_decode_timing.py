#!/usr/bin/env python3
"""Attribute an archived BG measurement using its original rank-zero log.

GPU-event durations include stream idle time between CPU launches. They are
not a measure of occupied GPU SM time. Rank times must never be added together.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


STAMP = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - \[BatchGenWorker-0\]")
TIMING = re.compile(r"step_ms=([\d.]+).*fwd_launch ([\d.]+).*gpu_fwd_ms=([\d.]+).*\(n=(\d+)\)")


def analyze(log, measurement, timezone="Asia/Shanghai"):
    start = dt.datetime.fromisoformat(measurement["started_utc"]).timestamp()
    end = start + measurement["wall_seconds"]
    rows = []
    for line in log.replace("\r", "\n").splitlines():
        match = STAMP.search(line)
        if match:
            timestamp = dt.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo(timezone)).timestamp()
            rows.append((timestamp, line))
    rows.sort(key=lambda row: row[0])
    current = "other"
    cursor = start
    seconds = {"decode_window": 0., "other": 0.}
    samples = []
    waves = []
    frees = []
    for timestamp, line in rows:
        state = None
        if "stage=deep_free_enter" in line or "[DECODE] Breaking for prefill" in line:
            state = "other"
        if "[DECODE] Prepared batch:" in line:
            state = "decode_window"
        if timestamp < start:
            if state: current = state
            continue
        if timestamp > end: break
        if state:
            seconds[current] += timestamp - cursor
            cursor, current = timestamp, state
        timing = TIMING.search(line)
        if timing:
            step, launch, event, count = timing.groups()
            samples.append({"step_ms": float(step), "forward_wall_ms": float(launch), "forward_cuda_event_ms": float(event), "steps": int(count)})
        selected = re.search(r"\[PREFILL\] Selected (\d+) sequences \((\d+) recompute", line)
        if selected and int(selected[1]): waves.append({"selected": int(selected[1]), "recomputed": int(selected[2])})
        free = re.search(r"deep_free_model_memory sync=([\d.]+)s del=([\d.]+)s release_streamed=([\d.]+)s empty_cache=([\d.]+)s gc=([\d.]+)s", line)
        if free: frees.append(sum(map(float, free.groups())))
    seconds[current] += end - cursor
    count = sum(row["steps"] for row in samples)
    return {
        "measurement_seconds": measurement["wall_seconds"],
        "upload_create_seconds": measurement.get("upload_and_create_preparation_seconds"),
        "rank0_wall_windows_seconds": seconds,
        "logged_decode_steps": count,
        "weighted_decode_timing_ms": {key: sum(row[key]*row["steps"] for row in samples)/count if count else None for key in ("step_ms", "forward_wall_ms", "forward_cuda_event_ms")},
        "prefill_waves": waves,
        "recomputed_admissions": sum(w["recomputed"] for w in waves),
        "rank0_model_free_calls": len(frees),
        "rank0_model_free_wall_seconds": sum(frees),
        "scope": "Original measurement bounds and rank0 timestamps. Decode windows include boundary/control costs. CUDA events include CPU launch gaps. Other includes prefill, phase rebuilds and all unclassified time. Recomputed admissions may count a request more than once.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-log", type=Path, required=True)
    parser.add_argument("--client-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = args.model_log.read_bytes()
    client = args.client_log.read_bytes()
    result = analyze(log.decode(), json.loads(client)["measurement"])
    result["source_sha256"] = {"model_log": hashlib.sha256(log).hexdigest(), "client_log": hashlib.sha256(client).hexdigest()}
    with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
