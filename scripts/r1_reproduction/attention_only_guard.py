"""Bounded supervisor for attention_only_qualification; imported only on --execute.

This supervisor never inspects or signals a model service. It requires an empty
assigned GPU, then owns one new session. WNOWAIT pins its session leader until
all final signals/audits finish. Different compiler PGIDs in that same session
are included; descendants that create a new SID are outside the audit's scope.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback


def process_identity(pid):
    try:
        path = Path(f"/proc/{pid}")
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "rss_bytes": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_identity(current, saved):
    return current is not None and all(
        current[k] == saved[k] for k in ("pid", "pgid", "sid", "start_ticks")
    )


def session_members(saved):
    if not same_identity(process_identity(saved["pid"]), saved):
        raise RuntimeError("Unreaped session leader no longer matches")
    return [
        row
        for p in Path("/proc").iterdir()
        if p.name.isdecimal()
        and (row := process_identity(int(p.name))) is not None
        and row["sid"] == saved["sid"]
    ]


def child_exited(child):
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def signal_session(saved, sig):
    rows = session_members(saved)
    if saved["pid"] != saved["pgid"] or saved["pid"] != saved["sid"]:
        raise RuntimeError("Child was not started as a fresh session leader")
    groups = {r["pgid"] for r in rows if r["state"] not in ("Z", "X")}
    signalled = []
    for group in sorted(groups):
        # The unreaped leader pins this SID. A group still inside it can only
        # belong to descendants of the new session, including compiler groups.
        current = session_members(saved)
        if not any(r["pgid"] == group and r["state"] not in ("Z", "X") for r in current):
            continue
        try:
            os.killpg(group, sig)
            signalled.append(group)
        except ProcessLookupError:
            pass
    return signalled


def cleanup(child, saved):
    record = {"signals": []}
    live = lambda: [r for r in session_members(saved) if r["state"] not in ("Z", "X")]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not live():
            break
        record["signals"].append({"signal": int(sig), "pgids": signal_session(saved, sig)})
        deadline = time.monotonic() + 10
        while live() and time.monotonic() < deadline:
            time.sleep(0.1)
    # Audit BEFORE wait/reap. Never signal a numeric PID/PGID afterward.
    record["remaining_live_session_members"] = live()
    record["leader_exited_before_reap"] = child_exited(child)
    if record["leader_exited_before_reap"]:
        record["exit_code"] = child.wait(timeout=1)
    record["cleanup_complete"] = (
        not record["remaining_live_session_members"] and record["leader_exited_before_reap"]
    )
    return record


def gpu_apps(gpu):
    response = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    result = []
    for line in response.stdout.splitlines():
        if line.strip():
            pid, mib = line.split(",")
            result.append({"pid": int(pid), "bytes": int(mib) * 1024**2})
    return result


def memory_available():
    return next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def violation(sample, limits):
    if sample["elapsed_seconds"] >= limits["seconds"]:
        return "wall_time_limit"
    if sample["session_RSS_bytes"] >= limits["session_RSS_bytes"]:
        return "session_RSS_limit"
    if sample["GPU_bytes"] >= limits["target_GPU_bytes"]:
        return "assigned_GPU_memory_limit"
    if sample["MemAvailable_bytes"] < limits["MemAvailable_min_bytes"]:
        return "host_memory_floor"
    if sample["unexpected_GPU_pids"]:
        return "unexpected_process_on_assigned_GPU"
    return None


def supervise(args, script):
    from attention_only_qualification import (
        LIMITS,
        REPO,
        SOURCE_FILES,
        plan,
        sha256_file,
    )

    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").exists():
        raise RuntimeError("Wrong allocated container")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("SIGCHLD default/WNOWAIT ownership required")
    # All checks precede launch; a busy GPU is never reclaimed by this tool.
    initial = gpu_apps(args.gpu)
    if initial:
        raise RuntimeError(
            "Assigned GPU has compute processes; resource owner must release it first"
        )
    if memory_available() < LIMITS["MemAvailable_min_bytes"]:
        raise RuntimeError("Insufficient host memory")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    record = plan(args)
    record.update(
        status="started_NOT_qualified",
        samples=[],
        initial_GPU_apps=initial,
        guard_scope="same UID, same new SID, visible container /proc; new-SID descendants excluded",
    )
    for name in SOURCE_FILES:
        destination = out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
        if sha256_file(destination) != record["source_sha256"][name]:
            raise RuntimeError("Source snapshot mismatch")
    for name, path in (
        ("config.json", args.config),
        ("weights-manifest.json", args.weights_manifest),
    ):
        shutil.copyfile(path, out / "sources" / name)
    argv = [arg for arg in sys.argv[1:] if arg != "--execute"] + ["--_worker"]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "MAX_JOBS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "BATCHGEN_ATTENTION_ONLY_GUARDED": "1",
        "TRITON_CACHE_DIR": str(out / "triton-cache"),
    }
    child = None
    saved = None
    start = time.monotonic()

    def save():
        temporary = out / "guard.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "guard.json")

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (out / "child.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, str(script), *argv],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            saved = process_identity(child.pid)
            if not saved:
                raise RuntimeError("Could not capture child identity")
            record["child_identity"] = saved
            while not child_exited(child):
                rows = session_members(saved)
                apps = gpu_apps(args.gpu)
                pids = {r["pid"] for r in rows}
                sample = {
                    "elapsed_seconds": time.monotonic() - start,
                    "session_RSS_bytes": sum(r["rss_bytes"] for r in rows),
                    "members": rows,
                    "GPU_bytes": sum(r["bytes"] for r in apps),
                    "MemAvailable_bytes": memory_available(),
                    "unexpected_GPU_pids": [r["pid"] for r in apps if r["pid"] not in pids],
                }
                record["samples"].append(sample)
                save()
                reason = violation(sample, LIMITS)
                if reason:
                    record["stopped_reason"] = reason
                    break
                time.sleep(0.5)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        if child is not None and saved is not None:
            try:
                record["cleanup"] = cleanup(child, saved)
            except BaseException:
                record["cleanup_error"] = traceback.format_exc()
        record["source_sha256_after"] = {name: sha256_file(REPO / name) for name in SOURCE_FILES}
        record["sources_unchanged"] = record["source_sha256_after"] == record["source_sha256"]
        result = out / "report.json"
        if result.exists():
            record["report_sha256"] = sha256_file(result)
        clean = record.get("cleanup", {}).get("cleanup_complete", False)
        code = record.get("cleanup", {}).get("exit_code")
        if (
            clean
            and code in (0, 2)
            and not record.get("stopped_reason")
            and not record.get("error")
            and record["sources_unchanged"]
        ):
            record["status"] = "diagnostics_collected_NOT_model_or_performance_qualified"
        else:
            record["status"] = "failed_or_stopped_NOT_qualified"
        record["elapsed_seconds_NOT_benchmark"] = time.monotonic() - start
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if record["status"].startswith("diagnostics_collected"):
        return code
    return 1
