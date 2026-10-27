#!/usr/bin/env python3
"""Bounded operator qualification of the exact common FI attention shim.

Default is plan-only. Execute only under the coordinator's owned single-GPU
guard. The same source file that model import hooks select is loaded by an
explicit path/hash, together with its explicit metadata helper path/hash.
Synthetic post-projection/RoPE BF16 fixtures and independent CPU FP64 formulas
do not qualify R1 projections, KV lifecycle, model quality or paper performance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from types import ModuleType
import traceback


PASS = "passed_isolated_sm103_mla_compat_contract_NOT_model_or_performance_qualified"
FAIL = "failed_isolated_sm103_mla_compat_contract"
LENGTHS = (1, 63, 64, 65, 2047, 2048, 2049, 8191, 8192, 8193)
HEADS = 128
CAP_BYTES = 6 * 1024**3
REFERENCE_CHUNK = 32
MAX_BF16_ULPS = 1
RUNTIME = {"torch": "2.9.0+cu130", "triton": "3.5.0", "flashinfer-python": "0.5.2"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def source_contract(args):
    backend, metadata = args.backend_source.resolve(), args.metadata_source.resolve()
    if backend.parent != metadata.parent or metadata.name != "common_sdpa_backend.py":
        raise ValueError("The explicit shim and its common_sdpa_backend.py helper must be siblings")
    actual = {"backend": {"path": str(backend), "sha256": sha(backend)},
              "metadata": {"path": str(metadata), "sha256": sha(metadata)}}
    if actual["backend"]["sha256"] != args.backend_source_sha256:
        raise ValueError("Common FI backend source hash mismatch")
    if actual["metadata"]["sha256"] != args.metadata_source_sha256:
        raise ValueError("Common metadata source hash mismatch")
    return actual


def plan(args):
    return {"schema": "r1-common-fi-attention-qualification-v1", "status": "plan_only_NOT_executed",
            "scope": "operator", "full_model_qualified": False, "formal_test_permitted": False,
            "expected_hostname": args.expected_hostname, "sources": source_contract(args),
            "worker_sha256": sha(Path(__file__)), "runtime_required": RUNTIME,
            "required_environment": {"R1_MAIN_MLA_COMPAT_GUARDED": "1", "CUDA_VISIBLE_DEVICES": "one assigned GPU"},
            "torch_allocator_cap_bytes": CAP_BYTES,
            "external_limits_required": {"seconds": 900, "RSS_bytes": 8 * 1024**3, "GPU_bytes": 8 * 1024**3},
            "lengths": list(LENGTHS), "heads": HEADS,
            "prefill": "actual shim flash_attn_varlen_func; fresh causal QK192/V128 BF16, every length plus packed63/65",
            "decode": "actual shim get_mla_metadata and flash_mla_with_kvcache; inclusive valid length, page64 allocator/page1 FI indices",
            "reference": {
                "kind": "independent CPU FP64 analytic attention on exact binary-representable BF16 fixtures",
                "chunk_rows": REFERENCE_CHUNK, "max_BF16_ULPs": MAX_BF16_ULPS,
                "envelope": "at most one BF16 representable step from independently FP64-computed, once-rounded BF16 output",
                "FP32_accumulation_note": "The envelope allows isolated optimized-attention accumulation/softmax rounding only. Positive fixture outputs avoid ambiguous zero/subnormal ULP comparisons. It is not a composed FP8/model error budget.",
                "metrics": ["max_absolute_error_vs_FP64", "normalized_RMSE_vs_FP64", "max_BF16_ULPs", "out_of_envelope_elements"],
            },
            "mutation_checks": "All Q/K/V, page table, lengths and full cache byte hashes before/after; includes NaN payloads",
            "invalid_fixture": "A separate valid-KV NaN case must remain unchanged and produce nonfinite output; not folded into normal finite-fixture pass",
            "projections_RoPE_KV_lifecycle_model_quality_and_performance": "not qualified"}


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_backend(args):
    source_contract(args)
    name = "r1_exact_common_mla_qualification_sources"
    if name in sys.modules:
        raise RuntimeError("Qualification source namespace already exists")
    package = ModuleType(name)
    package.__path__ = [str(args.backend_source.resolve().parent)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".common_flashinfer_backend", args.backend_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for entry in ("flash_attn_varlen_func", "get_mla_metadata", "flash_mla_with_kvcache"):
        if Path(inspect.getsourcefile(getattr(module, entry))).resolve() != args.backend_source.resolve():
            raise RuntimeError("Qualified entrypoint does not come from the exact model shim source")
    helper = sys.modules.get(name + ".common_sdpa_backend")
    if helper is None or Path(helper.__file__).resolve() != args.metadata_source.resolve():
        raise RuntimeError("Qualified metadata helper does not come from the explicit source")
    source_contract(args)
    return module


def _ordered_bf16(tensor):
    import torch
    bits = tensor.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


def compare_output(actual, reference_chunks):
    """Compare every finite output against independent CPU FP64 chunks."""
    import torch
    if actual.dtype != torch.bfloat16:
        raise ValueError("The qualified attention output must be BF16")
    metrics = {"compared_elements": 0, "nonfinite_elements": 0,
               "max_BF16_ULPs": 0, "out_of_envelope_elements": 0,
               "max_absolute_error_vs_FP64": 0.0}
    square_error = reference_squares = 0.0
    actual_digest, reference_digest = hashlib.sha256(), hashlib.sha256()
    next_row = 0
    for first, stop, reference in reference_chunks:
        if type(first) is not int or type(stop) is not int or first != next_row or stop <= first or stop > actual.shape[0]:
            raise ValueError("CPU reference chunks must cover output rows exactly once in order")
        next_row = stop
        observed = actual[first:stop].detach().cpu().contiguous()
        if tuple(observed.shape) != tuple(reference.shape) or reference.dtype != torch.float64:
            raise ValueError("CPU reference and actual output shape/dtype disagree")
        finite = torch.isfinite(observed) & torch.isfinite(reference)
        metrics["nonfinite_elements"] += int((~finite).sum())
        rounded = reference.bfloat16()
        ulps = (_ordered_bf16(observed) - _ordered_bf16(rounded)).abs()
        # Preserve a usable JSON failure report even if native code returns
        # NaN/Inf. Nonfinite data never becomes an accepted numerical result.
        if bool(finite.any()):
            metrics["max_BF16_ULPs"] = max(metrics["max_BF16_ULPs"], int(ulps[finite].max()))
        metrics["out_of_envelope_elements"] += int(((ulps > MAX_BF16_ULPS) | ~finite).sum())
        error = observed.double() - reference
        if bool(finite.any()):
            metrics["max_absolute_error_vs_FP64"] = max(metrics["max_absolute_error_vs_FP64"], float(error[finite].abs().max()))
            square_error += float(error[finite].square().sum())
            reference_squares += float(reference[finite].square().sum())
        metrics["compared_elements"] += observed.numel()
        actual_digest.update(observed.view(torch.uint8).numpy().tobytes())
        reference_digest.update(rounded.view(torch.uint8).numpy().tobytes())
    metrics["normalized_RMSE_vs_FP64"] = (
        math.sqrt(square_error / reference_squares)
        if reference_squares and metrics["nonfinite_elements"] == 0 else None
    )
    if metrics["nonfinite_elements"]:
        metrics["max_absolute_error_vs_FP64"] = None
    metrics["actual_output_sha256"] = actual_digest.hexdigest()
    metrics["FP64_rounded_BF16_reference_sha256"] = reference_digest.hexdigest()
    metrics["within_fixed_envelope"] = (
        metrics["compared_elements"] == actual.numel()
        and next_row == actual.shape[0]
        and metrics["nonfinite_elements"] == 0
        and metrics["out_of_envelope_elements"] == 0
    )
    return metrics


def prefill_fixture(lengths):
    """Sparse Q/K and separable V; all stored input values are exact in BF16."""
    import torch
    parts = []
    for length in lengths:
        pos = torch.arange(length, dtype=torch.int64)
        head = torch.arange(HEADS, dtype=torch.int64)
        query = torch.zeros(length, HEADS, 192, dtype=torch.bfloat16)
        key = torch.zeros_like(query)
        query[:, :, 0] = ((pos % 3 + 1)[:, None] * (head % 5 + 1)[None, :]).float() / 8
        query[:, :, 128] = ((head % 3 + 1).float() / 16)[None, :]
        key[:, :, 0] = (((pos % 7) - 3).float() / 8)[:, None]
        key[:, :, 128] = (((pos % 5) - 2).float() / 8)[:, None]
        value = (2 + (pos % 11).double()[:, None, None] / 16
                 + (head % 7).double()[None, :, None] / 32
                 + (torch.arange(128) % 5).double()[None, None, :] / 64).bfloat16()
        parts.append((query, key, value))
    if len(parts) == 1:
        q, k, v = parts[0]
    else:
        q, k, v = (torch.cat([part[index] for part in parts]) for index in range(3))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return q, k, v, torch.tensor(cu, dtype=torch.int32)


def prefill_reference(q, k, v, lengths, scale):
    """Independent FP64 weighted-prefix formula, bounded CPU reference chunks.

    Each query has exactly two nonzero components. For each head/query-class,
    compute scores from CPU Q/K, then independent exp and weighted prefix sums.
    V is a token scalar plus exact head/feature constants; the caller owns the
    synthetic fixture. No optimized output or intermediate is used as oracle.
    """
    import torch
    offset = 0
    for length in lengths:
        means = torch.empty(HEADS, 3, length, dtype=torch.float64)
        for head in range(HEADS):
            for group in range(min(3, length)):
                scores = (q[offset + group, head, 0].double() * k[offset:offset + length, head, 0].double()
                          + q[offset + group, head, 128].double() * k[offset:offset + length, head, 128].double()) * scale
                weights = scores.exp()
                means[head, group] = (weights * v[offset:offset + length, head, 0].double()).cumsum(0) / weights.cumsum(0)
        feature_offsets = v[offset].double() - v[offset, :, 0].double().unsqueeze(-1)
        for first in range(0, length, REFERENCE_CHUNK):
            stop = min(length, first + REFERENCE_CHUNK)
            positions = torch.arange(first, stop)
            mean = means[torch.arange(HEADS).unsqueeze(0), (positions % 3).unsqueeze(1), positions.unsqueeze(1)]
            expected = mean.unsqueeze(-1) + feature_offsets.unsqueeze(0)
            yield offset + first, offset + stop, expected
        offset += length


def decode_fixture(length):
    import torch
    pages = (length + 63) // 64
    physical = torch.randperm(pages + 3, generator=torch.Generator().manual_seed(20261026 + length))[:pages]
    cache = torch.full((pages + 3, 64, 1, 576), float("nan"), dtype=torch.bfloat16)
    position = torch.arange(length, dtype=torch.int64)
    history = (2 + (position % 11).double()[:, None] / 16
               + (torch.arange(512) % 7).double()[None, :] / 64).bfloat16()
    rope = torch.zeros(length, 64, dtype=torch.bfloat16)
    rope[:, 0] = ((position % 5 - 2).float() / 8)
    cache[physical[position // 64], position % 64, 0] = torch.cat((history, rope), dim=-1)
    query = torch.zeros(1, 1, HEADS, 576, dtype=torch.bfloat16)
    head = torch.arange(HEADS)
    query[0, 0, :, 0] = (head % 5 + 1).float() / 8
    query[0, 0, :, 512] = (head % 3 + 1).float() / 16
    table = torch.tensor([physical.tolist() + [-1, -999]], dtype=torch.int32)
    lengths = torch.tensor([length], dtype=torch.int32)
    return query, cache, table, lengths, history, rope


def decode_reference(query, history, rope, scale):
    import torch
    result = torch.empty(1, 1, HEADS, 512, dtype=torch.float64)
    offsets = history[0].double() - history[0, 0].double()
    for head in range(HEADS):
        scores = (query[0, 0, head, 0].double() * history[:, 0].double()
                  + query[0, 0, head, 512].double() * rope[:, 0].double()) * scale
        weights = scores.exp()
        result[0, 0, head] = (weights * history[:, 0].double()).sum() / weights.sum() + offsets
    return [(0, 1, result)]


def execute(args):
    if os.environ.get("R1_MAIN_MLA_COMPAT_GUARDED") != "1":
        raise RuntimeError("Execution requires the parent-owned MLA compatibility guard")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one assigned GPU must be visible")
    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
        raise RuntimeError("Expected allocated container identity")
    versions = {name: importlib.metadata.version(name) for name in RUNTIME}
    if versions != RUNTIME:
        raise RuntimeError(f"Frozen Torch/FI runtime required: {versions}")
    report = plan(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Fresh exclusive output directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(status=FAIL, cases=[], started_unix=time.time(), actual_versions=versions)
    started = time.monotonic()
    worker_sha = report["worker_sha256"]

    def save():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "report.json")

    save()
    exit_code = 1
    try:
        import torch
        torch.set_num_threads(1)
        if torch.version.cuda != "13.0":
            raise RuntimeError("CUDA13.0 Torch runtime required")
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError("Exactly one assigned SM103 GPU required")
        torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)
        module = load_backend(args)
        report["runtime_identity"] = module.runtime_identity()
        report["device"] = {"name": torch.cuda.get_device_name(0), "capability": [10, 3]}
        scale = 0.173
        for lengths in [*[ [length] for length in LENGTHS ], [63, 65]]:
            source_contract(args)
            torch.cuda.reset_peak_memory_stats()
            q, k, v, cu = prefill_fixture(lengths)
            cpu = {"Q": q, "K": k, "V": v, "cu_seqlens": cu}
            inputs = {name: tensor.to("cuda") for name, tensor in cpu.items()}
            before = {name: tensor_sha(tensor) for name, tensor in inputs.items()}
            if before != {name: tensor_sha(tensor) for name, tensor in cpu.items()}:
                raise RuntimeError("Prefill fixture GPU transfer differs from CPU")
            output = module.flash_attn_varlen_func(inputs["Q"], inputs["K"], inputs["V"],
                inputs["cu_seqlens"], inputs["cu_seqlens"], max(lengths), max(lengths),
                softmax_scale=scale, causal=True)
            torch.cuda.synchronize()
            metrics = compare_output(output, prefill_reference(q, k, v, lengths, scale))
            unchanged = before == {name: tensor_sha(tensor) for name, tensor in inputs.items()}
            case = {"phase": "prefill", "lengths": lengths, "input_sha256": before,
                    "inputs_unchanged": unchanged, "comparison": metrics,
                    "peak_torch_bytes": torch.cuda.max_memory_allocated(),
                    "passed": unchanged and metrics["within_fixed_envelope"]}
            report["cases"].append(case)
            save()
            if not case["passed"]:
                raise RuntimeError("Prefill contract/reference failure; original failure preserved")
            del output, inputs, cpu, q, k, v, cu
            gc.collect()
            torch.cuda.empty_cache()

        scale = 0.091
        for length in LENGTHS:
            source_contract(args)
            torch.cuda.reset_peak_memory_stats()
            q, cache, table, lengths, history, rope = decode_fixture(length)
            cpu = {"Q": q, "cache": cache, "table": table, "lengths": lengths}
            inputs = {name: tensor.to("cuda") for name, tensor in cpu.items()}
            before = {name: tensor_sha(tensor) for name, tensor in inputs.items()}
            if before != {name: tensor_sha(tensor) for name, tensor in cpu.items()}:
                raise RuntimeError("Decode fixture GPU transfer differs from CPU")
            metadata, splits = module.get_mla_metadata(inputs["lengths"], HEADS, 1)
            output, lse = module.flash_mla_with_kvcache(inputs["Q"], inputs["cache"], inputs["table"],
                inputs["lengths"], 512, metadata, splits, scale, True)
            torch.cuda.synchronize()
            metrics = compare_output(output, decode_reference(q, history, rope, scale))
            unchanged = before == {name: tensor_sha(tensor) for name, tensor in inputs.items()}
            case = {"phase": "decode", "lengths": [length], "scattered_physical_pages": table[0, :-2].tolist(),
                    "unused_cache_fill": "NaN", "unused_table_tail": [-1, -999],
                    "input_sha256": before, "full_cache_unchanged": unchanged,
                    "comparison": metrics, "LSE_finite": bool(torch.isfinite(lse).all()),
                    "peak_torch_bytes": torch.cuda.max_memory_allocated(),
                    "passed": unchanged and metrics["within_fixed_envelope"] and bool(torch.isfinite(lse).all())}
            report["cases"].append(case)
            save()
            if not case["passed"]:
                raise RuntimeError("Decode contract/reference failure; original failure preserved")
            del output, lse, metadata, inputs, cpu, q, cache, table, lengths, history, rope
            gc.collect()
            torch.cuda.empty_cache()

        # Invalid data is intentionally a separate negative fixture. Its
        # nonfinite output is expected here, never accepted in normal cases.
        q, cache, table, lengths, _, _ = decode_fixture(65)
        cache[table[0, 0].item(), 0, 0, 0] = float("nan")
        q, cache, table, lengths = (tensor.to("cuda") for tensor in (q, cache, table, lengths))
        before = tensor_sha(cache)
        metadata, splits = module.get_mla_metadata(lengths, HEADS, 1)
        output, _ = module.flash_mla_with_kvcache(q, cache, table, lengths, 512, metadata, splits, scale, True)
        torch.cuda.synchronize()
        invalid = {"scope": "invalid_valid_KV_NaN_negative_fixture", "normal_finite_fixture": False,
                   "full_cache_unchanged": tensor_sha(cache) == before,
                   "nonfinite_output_elements": int((~torch.isfinite(output)).sum()),
                   "required_behavior": "do not sanitize valid data; propagate nonfinite invalid fixture"}
        invalid["passed_negative_contract"] = invalid["full_cache_unchanged"] and invalid["nonfinite_output_elements"] > 0
        report["valid_nan_negative_fixture"] = invalid
        if not invalid["passed_negative_contract"]:
            raise RuntimeError("Valid-NaN negative fixture was mutated or did not propagate; investigate")
        source_contract(args)
        if sha(Path(__file__)) != worker_sha:
            raise RuntimeError("Qualification worker source changed")
        report.update(status=PASS, sources_unchanged=True, worker_unchanged=True)
        exit_code = 0
    except BaseException as exc:
        report.update(status=FAIL, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
        save()
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-source", type=Path, required=True)
    parser.add_argument("--backend-source-sha256", required=True)
    parser.add_argument("--metadata-source", type=Path, required=True)
    parser.add_argument("--metadata-source-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        return execute(args)
    print(json.dumps(plan(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
