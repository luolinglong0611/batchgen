#!/usr/bin/env python3
"""Compare R1-shape prefill candidates against the unchanged FP32 attention gate.

The BF16-probability oracle is DIAGNOSTIC ONLY: it approximates one source of
low-precision P@V error, not FA4's exact tiled/online softmax implementation.
A better match does not qualify FA4 against FP32 or prove the whole cause.
The optional Triton candidate loads the existing local primitive read-only;
no Kimi model, dispatch, or source file is changed.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import traceback

import torch

from attention_qualification import check, emit, prefill_reference, versions, ATOL, RTOL, MAX_FAILURE_FRACTION


ROOT = Path(__file__).resolve().parents[2]
TRITON_SOURCE = ROOT / "batchgen/models/moonshotai/kimi_linear/mla_triton.py"


def bf16_probability_reference(q, k, v, boundaries, scale):
    output = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.float32)
    for start, end in zip(boundaries, boundaries[1:]):
        length = end-start
        positions = torch.arange(length, device=q.device)
        for head in range(0, 128, 16):
            keys, values = [tensor[start:end, head:head+16].float().transpose(0, 1) for tensor in (k, v)]
            for offset in range(0, length, 64):
                stop = min(offset+64, length)
                query = q[start+offset:start+stop, head:head+16].float().transpose(0, 1)
                scores = (query @ keys.transpose(-1, -2)) * scale
                mask = positions[None, :] > positions[offset:stop, None]
                probabilities = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
                output[start+offset:start+stop, head:head+16] = (probabilities.bfloat16().float() @ values).transpose(0, 1)
    return output


def compare(actual, reference, case, records):
    try:
        check(actual, reference, {"case": case}, records)
    except AssertionError:
        pass


def run(args, result):
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run only inside the allocated container")
    if args.expected_hostname and socket.gethostname() != args.expected_hostname:
        raise RuntimeError("Unexpected container hostname")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.device) != (10, 3):
        raise RuntimeError("SM103 CUDA device required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    result["device"] = {"name": torch.cuda.get_device_name(device), "capability": [10, 3]}
    lengths = [1, 63, 65] if args.profile == "small" else [1, 63, 64, 65, 257, 2048]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1]+length)
    result["lengths"] = lengths
    q = torch.randn((boundaries[-1], 128, 192), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((boundaries[-1], 128, 128), device=device, dtype=torch.bfloat16)
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    reference = prefill_reference(q, k, v, boundaries, args.softmax_scale)
    probability_reference = bf16_probability_reference(q, k, v, boundaries, args.softmax_scale)
    candidates = ("fa4", "triton") if args.candidate == "both" else (args.candidate,)
    for candidate in candidates:
        entry = {"candidate": candidate, "status": "failed", "primary_gate": [], "diagnostics": []}
        result["candidates"].append(entry)
        try:
            if candidate == "fa4":
                from flash_attn.cute import flash_attn_varlen_func
                returned = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                    softmax_scale=args.softmax_scale, causal=True,
                )
                if not isinstance(returned, tuple) or len(returned) != 2:
                    raise RuntimeError("Expected FA4 b36 (output, lse) tuple")
                actual, _ = returned
            else:
                entry["source_sha256"] = hashlib.sha256(TRITON_SOURCE.read_bytes()).hexdigest()
                spec = importlib.util.spec_from_file_location("r1_readonly_triton_mla_primitive", TRITON_SOURCE)
                primitive = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = primitive
                spec.loader.exec_module(primitive)
                actual = primitive.varlen_prefill(q, k, v, cu, max(lengths), softmax_scale=args.softmax_scale)
            torch.cuda.synchronize(device)
            compare(actual, reference, candidate + "_unchanged_fp32_gate", entry["primary_gate"])
            compare(actual, probability_reference, candidate + "_bf16_probability_diagnostic_only", entry["diagnostics"])
            if bool(torch.isfinite(actual).all()):
                primary_rms = float((actual.float()-reference).square().mean().sqrt())
                diagnostic_rms = float((actual.float()-probability_reference).square().mean().sqrt())
                entry["rms_vs_fp32"] = primary_rms
                entry["rms_vs_bf16_probability"] = diagnostic_rms
                entry["diagnostic_to_primary_rms_ratio"] = diagnostic_rms / max(primary_rms, 1e-30)
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                compare(actual[start:end], reference[start:end], f"{candidate}_sequence_{index}_length_{end-start}", entry["diagnostics"])
            entry["status"] = entry["primary_gate"][0]["status"]
        except Exception as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            traceback.print_exc(file=sys.stderr)
        emit(entry)
    result["status"] = "passed" if all(entry["status"] == "passed" for entry in result["candidates"]) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("fa4", "triton", "both"), default="both")
    parser.add_argument("--profile", choices=("small", "standard"), default="small")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--softmax-scale", type=float, default=192**-0.5)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0 < args.softmax_scale < float("inf"):
        parser.error("softmax scale must be finite and positive")
    result = {"status": "failed", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "versions": versions(), "candidates": [],
              "gate": {"atol": ATOL, "rtol": RTOL, "max_failure_fraction_exclusive": MAX_FAILURE_FRACTION},
              "scope": "FP32 gate unchanged; BF16 probability oracle is diagnostic only, not qualification",
              "compiler_environment": {name: os.environ.get(name) for name in ("TRITON_PTXAS_PATH", "DISABLE_MMA_V5")},
              "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    try:
        with torch.inference_mode():
            run(args, result)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=sys.stderr)
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        emit(result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
