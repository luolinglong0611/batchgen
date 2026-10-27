#!/usr/bin/env python3
"""Bound one isolated MLA tail diagnostic using the existing owned-process guard.

Default is plan-only. This never launches a model server or a performance job.
The parent must supply an empty assigned GPU and the exact isolated environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

from attention_only_guard import (
    child_exited, cleanup, cleanup_signal_shield, gpu_apps, memory_available,
    process_identity, session_members, violation,
)
from r1_launch_safety import prepare_launch_safety

LIMITS = {"seconds": 900, "session_RSS_bytes": 8 * 1024**3,
          "target_GPU_bytes": 8 * 1024**3, "MemAvailable_min_bytes": 64 * 1024**3}
SOURCES = ("guard_mla_tail_qualification.py", "mla_tail_safety.py",
           "qualify_r1_mla_tail_safety.py", "attention_only_guard.py", "r1_launch_safety.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--worker-kind", choices=("mla-tail", "w8a8-index", "fp8-compat", "mla-compat", "sg-page64", "mgn"), default="mla-tail")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--flashinfer-version", choices=("0.5.2", "0.6.13"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--compat-source", type=Path)
    parser.add_argument("--compat-source-sha256")
    parser.add_argument("--quant-source", type=Path)
    parser.add_argument("--quant-source-sha256")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--backend-source", type=Path)
    parser.add_argument("--backend-source-sha256")
    parser.add_argument("--metadata-source", type=Path)
    parser.add_argument("--metadata-source-sha256")
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--native-extension", type=Path)
    parser.add_argument("--native-extension-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captured-inputs", type=Path)
    parser.add_argument("--captured-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_kind == "mla-tail" and (not args.flashinfer_version or not args.cuda_home):
        parser.error("MLA tail diagnostics require --flashinfer-version and --cuda-home")
    if args.worker_kind == "w8a8-index" and (not args.source or not args.source_sha256):
        parser.error("W8A8 diagnostics require --source and --source-sha256")
    bound_sources = {}
    if args.worker_kind in ("w8a8-index", "fp8-compat", "sg-page64", "mgn"):
        if not args.source or not args.source_sha256:
            parser.error("GEMM diagnostics require --source and --source-sha256")
        bound_sources[str(args.source)] = args.source_sha256
    if args.worker_kind == "fp8-compat":
        for name in ("compat_source", "quant_source"):
            path, digest = getattr(args, name), getattr(args, name + "_sha256")
            if not path or not digest:
                parser.error("FP8 compatibility requires bound compatibility and quantization sources")
            bound_sources[str(path)] = digest
        if not args.checkpoint_root:
            parser.error("FP8 compatibility requires --checkpoint-root")
    if args.worker_kind == "mla-compat":
        if not all((args.backend_source, args.backend_source_sha256,
                    args.metadata_source, args.metadata_source_sha256)):
            parser.error("MLA compatibility requires bound backend and metadata sources")
        bound_sources[str(args.backend_source)] = args.backend_source_sha256
        bound_sources[str(args.metadata_source)] = args.metadata_source_sha256
    if args.worker_kind == "mgn":
        if not args.native_extension or not args.native_extension_sha256:
            parser.error("MGN requires the bound AOT binary")
        bound_sources[str(args.native_extension)] = args.native_extension_sha256
    if args.worker_kind in ("fp8-compat", "mla-compat", "sg-page64") and not args.cuda_home:
        parser.error("Compatibility qualifications require an explicit --cuda-home")
    tools = Path(__file__).resolve().parent
    workers = {"mla-tail": "qualify_r1_mla_tail_safety.py",
               "w8a8-index": "qualify_r1_w8a8_index.py",
               "fp8-compat": "qualify_r1_sm103_fp8_compat.py",
               "mla-compat": "qualify_r1_common_mla.py",
               "sg-page64": "qualify_sglang_r1_page64_worker.py",
               "mgn": "qualify_r1_mgn.py"}
    sources = SOURCES if args.worker_kind == "mla-tail" else (
        "guard_mla_tail_qualification.py", workers[args.worker_kind],
        "attention_only_guard.py", "r1_launch_safety.py")
    limits = dict(LIMITS)
    if args.worker_kind == "sg-page64":
        limits["target_GPU_bytes"] = 2 * 1024**3
    record = {"schema": "r1-mla-tail-owned-guard-v1", "status": "plan_only_NOT_executed",
              "scope": "operator-only; not full-model correctness or performance", "limits": limits,
              "physical_gpu": args.gpu, "hostname": args.expected_hostname,
              "worker_kind": args.worker_kind,
              "flashinfer_version": args.flashinfer_version,
              "source_sha256": {name: sha(tools / name) for name in sources},
              "runtime_source_sha256": bound_sources}
    if not args.execute:
        print(json.dumps(record, indent=2))
        return 0
    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").exists():
        raise RuntimeError("Unexpected container identity")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("Default SIGCHLD handling is required to retain child identity")
    if args.gpu < 0 or gpu_apps(args.gpu):
        raise RuntimeError("Assigned GPU is invalid or occupied")
    if memory_available() < limits["MemAvailable_min_bytes"]:
        raise RuntimeError("Host memory is below the fixed diagnostic floor")
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Unexpected inherited ELF preload")
    if not args.python.is_file():
        raise RuntimeError("Pinned Python is missing")
    if args.cuda_home and not (args.cuda_home / "bin/nvcc").is_file():
        raise RuntimeError("Private CUDA compiler is missing")
    if args.sanitizer and (args.worker_kind != "mgn" or not args.sanitizer.is_file()):
        raise RuntimeError("The explicit sanitizer is supported only for the native MGN probe")
    for path, digest in bound_sources.items():
        if sha(path) != digest:
            raise RuntimeError("Qualification runtime source does not match its bound SHA256: " + path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Fresh guard output directory required")
    record["launch_safety"] = prepare_launch_safety(args.output.parent)
    args.output.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MAX_JOBS="1", NCCL_RAS_ENABLE="0",
                       TRITON_CACHE_DIR=str(args.output / "triton-cache"))
    if args.cuda_home:
        environment.update(CUDA_HOME=str(args.cuda_home),
                           PATH=str(args.cuda_home / "bin") + ":" + environment.get("PATH", ""),
                           LD_LIBRARY_PATH=str(args.cuda_home / "lib"),
                           TRITON_PTXAS_PATH=str(args.cuda_home / "bin/ptxas"))
    worker = workers[args.worker_kind]
    command = [str(args.python), str(tools / worker),
               "--expected-hostname", args.expected_hostname,
               "--output", str(args.output / "diagnostic"), "--execute"]
    if args.worker_kind == "mla-tail":
        environment["R1_MLA_TAIL_GUARDED"] = "1"
        command += ["--flashinfer-version", args.flashinfer_version]
    elif args.worker_kind == "w8a8-index":
        environment["R1_MAIN_GEMM_GUARDED"] = "1"
        environment["DISABLE_MMA_V5"] = "1"
        command += ["--source", str(args.source), "--source-sha256", args.source_sha256]
    elif args.worker_kind == "fp8-compat":
        environment.update(R1_MAIN_FP8_COMPAT_GUARDED="1", BATCHGEN_R1_SM103_FP8_COMPAT="1",
                           DISABLE_MMA_V5="1")
        command += ["--source", str(args.source), "--source-sha256", args.source_sha256,
                    "--compat-source", str(args.compat_source),
                    "--compat-source-sha256", args.compat_source_sha256,
                    "--quant-source", str(args.quant_source),
                    "--quant-source-sha256", args.quant_source_sha256,
                    "--checkpoint-root", str(args.checkpoint_root)]
    elif args.worker_kind == "mla-compat":
        environment.update(R1_MAIN_MLA_COMPAT_GUARDED="1")
        command += ["--backend-source", str(args.backend_source),
                    "--backend-source-sha256", args.backend_source_sha256,
                    "--metadata-source", str(args.metadata_source),
                    "--metadata-source-sha256", args.metadata_source_sha256]
    elif args.worker_kind == "sg-page64":
        environment.update(R1_SG_ALLOCATOR_GUARDED="1", DISABLE_MMA_V5="1")
        command += ["--source", str(args.source), "--source-sha256", args.source_sha256]
    else:
        environment.update(R1_MAIN_MGN_GUARDED="1", PYTORCH_NO_CUDA_MEMORY_CACHING="1")
        command += ["--source", str(args.source), "--source-sha256", args.source_sha256,
                    "--native-extension", str(args.native_extension),
                    "--native-extension-sha256", args.native_extension_sha256]
        if args.sanitizer:
            record["sanitizer"] = {"path": str(args.sanitizer), "sha256": sha(args.sanitizer)}
            command = [str(args.sanitizer), "--tool", "memcheck", "--error-exitcode", "99",
                       "--target-processes", "all"] + command
    if args.captured_inputs and args.worker_kind == "mla-tail":
        if not args.captured_sha256:
            raise ValueError("Captured input SHA256 is required")
        command += ["--captured-inputs", str(args.captured_inputs),
                    "--captured-sha256", args.captured_sha256]
    record["runtime_environment"] = {
        key: environment[key] for key in
        ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "TRITON_PTXAS_PATH", "TRITON_CACHE_DIR",
         "DISABLE_MMA_V5", "BATCHGEN_R1_SM103_FP8_COMPAT", "OMP_NUM_THREADS", "MAX_JOBS", "LD_LIBRARY_PATH")
        if key in environment
    }
    if args.cuda_home:
        record["assembler"] = {"path": str(args.cuda_home / "bin/ptxas"),
                               "sha256": sha(args.cuda_home / "bin/ptxas")}
    record.update(status="running_NOT_qualified", command=command, samples=[], started_unix=time.time())
    child = saved = None
    started = time.monotonic()

    def save():
        temporary = args.output / "guard.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "guard.json")

    def interrupt(signum, frame):
        raise InterruptedError(f"Diagnostic supervisor received signal {signum}")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / "child.log").open("x") as log:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            saved = process_identity(child.pid)
            if not saved:
                raise RuntimeError("Unable to retain child identity")
            record["child_identity"] = saved
            while not child_exited(child):
                rows, apps = session_members(saved), gpu_apps(args.gpu)
                pids = {r["pid"] for r in rows}
                sample = {"elapsed_seconds": time.monotonic() - started,
                          "session_RSS_bytes": sum(r["rss_bytes"] for r in rows),
                          "GPU_bytes": sum(r["bytes"] for r in apps),
                          "MemAvailable_bytes": memory_available(),
                          "unexpected_GPU_pids": [r["pid"] for r in apps if r["pid"] not in pids]}
                record["samples"].append(sample)
                save()
                reason = violation(sample, limits)
                if reason:
                    record["stopped_reason"] = reason
                    break
                time.sleep(1)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        with cleanup_signal_shield():
            if child is not None and saved is not None:
                try:
                    record["cleanup"] = cleanup(child, saved)
                except BaseException:
                    record["cleanup_error"] = traceback.format_exc()
            record["sources_unchanged"] = all(sha(tools / name) == digest
                                                for name, digest in record["source_sha256"].items())
            record["runtime_sources_unchanged"] = all(
                sha(path) == digest for path, digest in bound_sources.items())
            result = args.output / "diagnostic/report.json"
            if result.exists():
                record["report_sha256"] = sha(result)
                try:
                    report = json.loads(result.read_text())
                    record["reported_qualification_status"] = report.get("status")
                except (ValueError, OSError):
                    record["report_parse_failed"] = True
            clean = record.get("cleanup", {}).get("cleanup_complete", False)
            code = record.get("cleanup", {}).get("exit_code")
            success = (clean and code in (0, 2) and record["sources_unchanged"]
                       and record["runtime_sources_unchanged"]
                       and result.exists() and not record.get("report_parse_failed")
                       and not record.get("stopped_reason") and not record.get("error"))
            record["status"] = "closed_operator_diagnostic" if success else "failed_or_stopped"
            expected_pass = {
                "mla-tail": "passed_isolated_tail_contract_NOT_model_or_performance_qualified",
                "w8a8-index": "passed_isolated_w8a8_index_contract_NOT_model_or_performance_qualified",
                "fp8-compat": "passed_isolated_sm103_fp8_compat_contract_NOT_model_or_performance_qualified",
                "mla-compat": "passed_isolated_sm103_mla_compat_contract_NOT_model_or_performance_qualified",
                "sg-page64": "passed_native_page64_token_address_contract_NOT_model_qualified",
                "mgn": "passed_isolated_native_mgn_dispatch_contract_NOT_model_qualified",
            }[args.worker_kind]
            record["qualification_passed"] = (
                success and code == 0 and record.get("reported_qualification_status")
                == expected_pass
            )
            record["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
            save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
