#!/usr/bin/env python3
"""Qualify an explicitly supplied Triton MLA candidate with the unchanged R1 gate.

This does not modify or import the BatchGen package. --kernel-source points to
the user's existing mla_triton.py or owned triton_precision.py, loaded by file path. Its SHA256 and the
reference script SHA256 are recorded. No candidate source is copied or edited.

Run inside an allocated SM103 container, for example:
  python triton_attention_qualification.py \
    --kernel-source /work/batchgen/models/moonshotai/kimi_linear/mla_triton.py \
    --expected-hostname HOST --device 0 --report triton-attention.json

Uses attention_qualification.py's FP32 reference and unchanged BF16 error gate:
abs(error) <= 1e-5 + 1.6e-2 * abs(BF16-rounded FP32 reference), with fewer than
1e-4 failures. Inputs match that script's small profile: causal prefill lengths
1/63/65, 128 heads, QK192/V128; three paged-decode steps starting at 1/63/65,
QK576/latent512, page64, shuffled physical pages and NaN unused cache.

Only layout adapters differ: decode Q [B,1,H,576] -> [B,H,576], cache
[P,64,576] -> [P,64,1,576], and reference [B,1,H,512] -> [B,H,512]. No values
or tolerances change. This is an independent correctness candidate; passing
does not qualify FA4, FlashInfer, full-model inference or throughput.
TRITON_PTXAS_PATH and DISABLE_MMA_V5 are recorded without changing either.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import traceback


FIXED = {"HEADS": 128, "QK_DIM": 192, "LATENT_DIM": 512, "ROPE_DIM": 64,
         "PAGE_SIZE": 64, "ATOL": 1e-5, "RTOL": 1.6e-2,
         "MAX_FAILURE_FRACTION": 1e-4}
LENGTHS = [1, 63, 65]
DECODE_STEPS = 3


def emit(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path, name, expected_hash=None):
    """Use importlib's module context, compiling exact bytes rather than .pyc."""
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and source_hash != expected_hash:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Triton/inspect need the module's source context.
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
        if file_digest(path) != source_hash:
            raise ValueError(f"Source changed during import: {path}")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, source_hash


def fixed_gate_check(reference_module, actual, oracle, record, cases):
    # The shared gate records a failure before raising. Keep running independent
    # cases so a failing prefill/decode result does not conceal the other cases.
    previous = len(cases)
    try:
        reference_module.check(actual, oracle, record, cases)
    except AssertionError:
        if len(cases) == previous:
            raise


def run_prefill(torch, reference, kernel, device, scale, cases):
    boundaries = [0]
    for length in LENGTHS:
        boundaries.append(boundaries[-1] + length)
    total = boundaries[-1]
    q = torch.randn(total, 128, 192, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(total, 128, 128, device=device, dtype=torch.bfloat16)
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    actual = kernel.varlen_prefill(q, k, v, cu, max(LENGTHS), softmax_scale=scale)
    torch.cuda.synchronize(device)
    oracle = reference.prefill_reference(q, k, v, boundaries, scale)
    fixed_gate_check(reference, actual, oracle, {
        "case": "triton_varlen_causal_prefill", "lengths": list(LENGTHS),
        "heads": 128, "qk_dim": 192, "value_dim": 128,
        "expected_output_shape": [total, 128, 128],
    }, cases)


def run_decode(torch, reference, kernel, device, scale, cases):
    lengths = list(LENGTHS)
    batch_size, page_size = len(lengths), 64
    max_pages = (max(lengths) + DECODE_STEPS - 1 + page_size - 1) // page_size
    # Match the earlier FlashInfer small-profile table exactly, including its
    # even table-width padding; this is not needed by the candidate itself.
    max_pages = ((max_pages + 1) // 2) * 2
    active_pages = batch_size * max_pages
    cache = torch.full((active_pages + 7, page_size, 576), float("nan"),
                       device=device, dtype=torch.bfloat16)
    page_order = torch.randperm(cache.shape[0], device=device)[:active_pages]
    logical_pages = page_order.reshape(batch_size, max_pages)
    table = logical_pages.to(torch.int32)
    histories = [torch.randn(length, 576, device=device, dtype=torch.bfloat16) for length in lengths]
    for batch, history in enumerate(histories):
        positions = torch.arange(history.shape[0], device=device)
        pages = logical_pages[batch, positions // page_size]
        cache[pages, positions % page_size] = history
    for step in range(DECODE_STEPS):
        if step:
            for batch, history in enumerate(histories):
                token = torch.randn(1, 576, device=device, dtype=torch.bfloat16)
                position = lengths[batch]
                cache[logical_pages[batch, position // page_size], position % page_size] = token[0]
                histories[batch] = torch.cat((history, token))
                lengths[batch] += 1
        query = torch.randn(batch_size, 1, 128, 576, device=device, dtype=torch.bfloat16)
        seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        actual = kernel.paged_mla_decode(query[:, 0], cache.unsqueeze(2), table, seq_lens,
                                         softmax_scale=scale)
        torch.cuda.synchronize(device)
        oracle = reference.decode_reference(query, histories, scale).squeeze(1)
        fixed_gate_check(reference, actual, oracle, {
            "case": f"triton_dense_paged_decode_step_{step}", "lengths": list(lengths),
            "page_size": page_size, "heads": 128, "qk_dim": 576, "value_dim": 512,
            "expected_output_shape": [batch_size, 128, 512], "block_table_width": max_pages,
            "shuffled_physical_pages": True, "poisoned_unused_cache": True,
            "physical_pages": cache.shape[0],
        }, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--kernel-sha256", help="Optional expected hash of the explicitly selected kernel file")
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--stage", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--softmax-scale", type=float, default=192 ** -0.5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.device < 0 or not math.isfinite(args.softmax_scale) or args.softmax_scale <= 0:
        parser.error("device must be nonnegative and softmax-scale finite and positive")
    if args.kernel_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.kernel_sha256):
        parser.error("kernel-sha256 must be lowercase SHA256")
    if args.report and args.report.exists():
        parser.error("Report already exists; use a fresh path to retain earlier failures")
    reference_source = Path(__file__).resolve().with_name("attention_qualification.py")
    kernel_source = args.kernel_source.resolve()
    result = {
        "status": "failed", "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "candidate": "explicitly selected Triton MLA low-level primitives",
        "kernel_source": str(kernel_source), "reference_source": str(reference_source),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {key: os.environ.get(key) for key in ("TRITON_PTXAS_PATH", "DISABLE_MMA_V5")},
        "gate": {"atol": 1e-5, "rtol": 1.6e-2, "max_failure_fraction_exclusive": 1e-4,
                 "oracle": "FP32 IEEE matmul and softmax rounded to BF16", "unchanged_shared_gate": True},
        "fixed_workload": {"prefill_lengths": LENGTHS, "heads": 128, "prefill_qk_dim": 192,
                           "prefill_value_dim": 128, "decode_initial_lengths": LENGTHS,
                           "decode_qk_dim": 576, "decode_value_dim": 512, "page_size": 64, "decode_steps": DECODE_STEPS},
        "scope": "candidate attention primitives only; not full model, graph, projection, transfer or performance qualification",
        "fa4_qualified_by_this_run": False, "flashinfer_qualified_by_this_run": False,
        "throughput_result": False, "cases": [], "stage_errors": [],
    }
    torch, device = None, None
    try:
        if not Path("/.dockerenv").exists():
            raise RuntimeError("Run only inside the allocated container")
        if socket.gethostname() != args.expected_hostname:
            raise RuntimeError("Allocated container hostname does not match --expected-hostname")
        if kernel_source.name not in ("mla_triton.py", "triton_precision.py"):
            raise ValueError("kernel-source must explicitly identify mla_triton.py or triton_precision.py")
        result["kernel_source_sha256"] = file_digest(kernel_source)
        result["reference_source_sha256"] = file_digest(reference_source)
        if args.kernel_sha256 and result["kernel_source_sha256"] != args.kernel_sha256:
            raise ValueError("Kernel SHA256 differs from expected staged source")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback is forbidden")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        capability = torch.cuda.get_device_capability(device)
        result["device"] = {"index": args.device, "name": torch.cuda.get_device_name(device), "capability": list(capability)}
        if capability != (10, 3):
            raise RuntimeError(f"Expected SM103, got {capability}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        reference, _ = load_source(reference_source, "r1_fixed_attention_reference", result["reference_source_sha256"])
        for name, expected in FIXED.items():
            if getattr(reference, name) != expected:
                raise ValueError(f"Shared reference contract changed: {name}")
        kernel, _ = load_source(kernel_source, "r1_user_triton_mla_candidate", result["kernel_source_sha256"])
        result["derived_kernel_source_sha256"] = getattr(kernel, "SOURCE_PROVENANCE_SHA256", None)
        result["versions"] = reference.versions()
        result["kernel_signatures"] = {name: str(inspect.signature(getattr(kernel, name)))
                                       for name in ("varlen_prefill", "paged_mla_decode")}
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        emit({"event": "start", **result})
        with torch.inference_mode():
            stages = [("prefill", run_prefill), ("decode", run_decode)]
            for stage, function in stages:
                if args.stage not in ("all", stage):
                    continue
                try:
                    function(torch, reference, kernel, device, args.softmax_scale, result["cases"])
                except Exception as error:
                    record = {"stage": stage, "type": type(error).__name__, "message": str(error)}
                    result["stage_errors"].append(record)
                    emit({"event": "stage_error", **record})
                    traceback.print_exc(file=sys.stderr)
        expected_cases = (1 if args.stage in ("all", "prefill") else 0) + (3 if args.stage in ("all", "decode") else 0)
        if not result["stage_errors"] and len(result["cases"]) == expected_cases and all(case["status"] == "passed" for case in result["cases"]):
            result["status"] = "passed"
        if file_digest(kernel_source) != result["kernel_source_sha256"] or file_digest(reference_source) != result["reference_source_sha256"]:
            raise RuntimeError("Kernel/reference source changed during qualification")
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc(file=sys.stderr)
    finally:
        if torch is not None and device is not None:
            try:
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            except Exception:
                pass
        emit({"event": "summary", **result})
        if args.report:
            args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
