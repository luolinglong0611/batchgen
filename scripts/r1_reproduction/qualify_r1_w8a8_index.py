#!/usr/bin/env python3
"""Bounded qualification of the current source's repaired W8A8 index path.

Default is CPU-only plan output. Execution requires a parent-owned single-GPU
guard. No unsafe original kernel, model, FlashInfer or benchmark is executed.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

CASES = ((2048, 24576, 128), (98304, 24576, 128), (65537, 32768, 128))
CHUNK_ROWS = 256
CAP_BYTES = 6 * 1024**3
PASS = "passed_isolated_w8a8_index_contract_NOT_model_or_performance_qualified"
FAIL = "failed_index_contract"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract(source, expected_sha):
    actual = sha(source)
    if actual != expected_sha:
        raise ValueError("Explicit W8A8 source SHA256 does not match")
    tree = ast.parse(Path(source).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"w8a8_gemm_kernel_optimized", "w8a8_gemm"} <= functions.keys():
        raise ValueError("Expected actual W8A8 kernel and public default wrapper")
    kernel = functions["w8a8_gemm_kernel_optimized"]
    expected = ("offs_am", "offs_bn", "offs_k", "offs_cm", "offs_cn")
    checks = {}
    for name in expected:
        assignments = [n.value for n in ast.walk(kernel) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        value = assignments[0] if len(assignments) == 1 else None
        checks[name] = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and value.func.attr == "to" and len(value.args) == 1
                        and isinstance(value.args[0], ast.Attribute)
                        and isinstance(value.args[0].value, ast.Name)
                        and value.args[0].value.id == "tl" and value.args[0].attr == "int64")
    if not all(checks.values()):
        raise ValueError("Refuse large execution without all five int64 index widenings")
    wrapper = functions["w8a8_gemm"]
    launches = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "w8a8_gemm_kernel_optimized"]
    if len(launches) != 1:
        raise ValueError("Expected exactly one default W8A8 kernel launch")
    return {"source_sha256": actual, "index_widenings": checks,
            "kernel_AST_sha256": hashlib.sha256(ast.dump(kernel).encode()).hexdigest(),
            "wrapper_AST_sha256": hashlib.sha256(ast.dump(wrapper).encode()).hexdigest(),
            "default_launch_keyword_names": [k.arg for k in launches[0].keywords]}


def first_unsafe_row(n):
    """First nonnegative row whose signed-int32 row*N product overflows."""
    if type(n) is not int or n <= 0:
        raise ValueError("Positive integer output row stride required")
    return (2**31 + n - 1) // n


def scalar_oracle(row, col, k=128):
    """Independent scalar FP64 formula; values are exactly binary-representable."""
    return (float(k) * (1 + row % 7) * (1 + col % 3)
            * ((1 + row % 5) / 8.0) * ((1 + (col // 128) % 5) / 16.0))


def case_passed(record):
    elements = record.get("M", 0) * record.get("N", 0)
    return (elements > 0 and record.get("compared_elements") == elements
            and record.get("compared_bytes") == elements * 2
            and record.get("different_bytes") == 0 and record.get("mismatched_elements") == 0
            and record.get("nonfinite_elements") == 0
            and record.get("inputs_unchanged") is True
            and record.get("FP64_spotchecks_passed") is True
            and record.get("output_contract_passed") is True
            and record.get("actual_output_sha256") is not None
            and record.get("actual_output_sha256") == record.get("analytic_BF16_output_sha256"))


def plan(args):
    source = source_contract(args.source, args.source_sha256)
    return {"schema": "r1-current-main-w8a8-index-v1", "status": "plan_only_NOT_executed",
            "scope": "operator", "qualification_limit": "not model or performance qualification",
            "expected_hostname": args.expected_hostname, "output": str(args.output),
            "source": str(args.source), "source_contract": source,
            "worker_sha256": sha(Path(__file__)),
            "runtime_required": {"torch": "2.9.0+cu130", "triton": "3.5.0", "CUDA": "13.0"},
            "environment_required": {"R1_MAIN_GEMM_GUARDED": "1", "DISABLE_MMA_V5": "1",
                                     "CUDA_VISIBLE_DEVICES": "one assigned device"},
            "torch_allocator_cap_bytes": CAP_BYTES,
            "external_limits_required": {"seconds": 900, "RSS_bytes": 8 * 1024**3,
                                         "GPU_bytes": 8 * 1024**3},
            "comparison_chunk_rows": CHUNK_ROWS,
            "fixture": {"A": "E4M3FN row constant: 1+row%7",
                        "W": "E4M3FN row constant: 1+column%3",
                        "A_scale": "FP32 (1+row%5)/8",
                        "W_scale": "FP32 (1+(column//128)%5)/16",
                        "reference": "independent FP64 analytic formula, one final BF16 cast",
                        "non_power_of_two_scales_included": True},
            "cases": [{"M": m, "N": n, "K": k, "output_bytes": m * n * 2,
                       "first_unsafe_int32_row_product_row": first_unsafe_row(n),
                       "first_flat_offset_2pow31_row_column": list(divmod(2**31, n))}
                      for m, n, k in CASES],
            "launch_policy": "actual source w8a8_gemm(a,a_scale,w,w_scale), no optional/tile overrides",
            "unsafe_original_kernel_executed": False}


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def spot_coordinates(m, n):
    boundary = first_unsafe_row(n)
    flat_row, flat_col = divmod(2**31, n)
    rows = {0, 1, min(127, m - 1), m - 1, boundary - 1, boundary, boundary + 1, flat_row}
    cols = {0, 1, 127, 128, n - 1, flat_col - 1, flat_col, flat_col + 1}
    return [(r, c) for r in sorted(rows) if 0 <= r < m
            for c in sorted(cols) if 0 <= c < n]


def execute(args):
    if os.environ.get("R1_MAIN_GEMM_GUARDED") != "1":
        raise RuntimeError("Execution requires the parent-owned GEMM resource guard")
    if os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("SM103 qualification requires explicit DISABLE_MMA_V5=1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one assigned CUDA device must be visible")
    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
        raise RuntimeError("Expected allocated container identity")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton")}
    if versions != {"torch": "2.9.0+cu130", "triton": "3.5.0"}:
        raise RuntimeError("Frozen Torch2.9cu130/Triton3.5 runtime required")
    report = plan(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Fresh exclusive output directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(status=FAIL, cases=[], actual_versions=versions, started_unix=time.time())
    started = time.monotonic()
    exit_code = 1

    def save():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "report.json")

    save()
    try:
        import torch
        torch.set_num_threads(1)
        if torch.version.cuda != "13.0":
            raise RuntimeError("Frozen CUDA13.0 Torch runtime required")
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError("Exactly one assigned SM103 device required")
        torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        spec = importlib.util.spec_from_file_location("r1_actual_main_w8a8_index_probe", args.source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        source_contract(args.source, args.source_sha256)
        spec.loader.exec_module(module)
        source_contract(args.source, args.source_sha256)
        if Path(inspect.getsourcefile(module.w8a8_gemm)).resolve() != args.source.resolve():
            raise RuntimeError("Public GEMM did not originate from explicit source")
        report["actual_entrypoint"] = {"path": inspect.getsourcefile(module.w8a8_gemm),
                                       "signature": str(inspect.signature(module.w8a8_gemm))}
        report["device"] = {"name": torch.cuda.get_device_name(0), "capability": [10, 3]}

        for m, n, k in CASES:
            source_contract(args.source, args.source_sha256)
            torch.cuda.reset_peak_memory_stats()
            case = {"M": m, "N": n, "K": k, "status": "started",
                    "first_unsafe_int32_row_product_row": first_unsafe_row(n),
                    "compared_elements": 0, "compared_bytes": 0, "different_bytes": 0,
                    "mismatched_elements": 0, "nonfinite_elements": 0}
            report["cases"].append(case)
            save()
            rows = torch.arange(m, dtype=torch.int64)
            columns = torch.arange(n, dtype=torch.int64)
            a_cpu = ((rows % 7 + 1).float()[:, None].expand(m, k)).to(torch.float8_e4m3fn).contiguous()
            w_cpu = ((columns % 3 + 1).float()[:, None].expand(n, k)).to(torch.float8_e4m3fn).contiguous()
            a_scale_cpu = ((rows % 5 + 1).float() / 8).reshape(m, 1).contiguous()
            w_scale_cpu = ((torch.arange(n // 128) % 5 + 1).float() / 16).reshape(n // 128, 1).contiguous()
            cpu_inputs = {"A": a_cpu, "A_scale": a_scale_cpu, "W": w_cpu, "W_scale": w_scale_cpu}
            inputs = {name: value.to("cuda") for name, value in cpu_inputs.items()}
            expected_input_hashes = {name: tensor_sha(value) for name, value in cpu_inputs.items()}
            before_hashes = {name: tensor_sha(value) for name, value in inputs.items()}
            if before_hashes != expected_input_hashes:
                raise RuntimeError("Transferred fixture bytes disagree with CPU source")
            case["inputs"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                     "sha256": before_hashes[name]} for name, value in inputs.items()}
            case["fixture_sha256"] = hashlib.sha256(json.dumps(
                {"shape": [m, n, k], "formula": report["fixture"], "input_sha256": before_hashes},
                sort_keys=True).encode()).hexdigest()
            # Exactly the public wrapper defaults: no tile or launch changes.
            output = module.w8a8_gemm(inputs["A"], inputs["A_scale"], inputs["W"], inputs["W_scale"])
            torch.cuda.synchronize()
            case["output_contract_passed"] = (isinstance(output, torch.Tensor)
                and tuple(output.shape) == (m, n) and output.dtype == torch.bfloat16
                and output.device == inputs["A"].device and output.is_contiguous())
            if not case["output_contract_passed"]:
                raise RuntimeError("Output shape/BF16/device/layout contract failed")
            actual_digest, expected_digest = hashlib.sha256(), hashlib.sha256()
            right = (columns % 3 + 1).double() * ((columns // 128) % 5 + 1).double() / 16
            for first in range(0, m, CHUNK_ROWS):
                stop = min(first + CHUNK_ROWS, m)
                r = rows[first:stop]
                left = k * (r % 7 + 1).double() * (r % 5 + 1).double() / 8
                expected = (left[:, None] * right[None, :]).bfloat16()
                actual = output[first:stop].cpu()
                actual_bytes, expected_bytes = actual.view(torch.uint8), expected.view(torch.uint8)
                case["different_bytes"] += int((actual_bytes != expected_bytes).sum())
                case["mismatched_elements"] += int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
                case["nonfinite_elements"] += int((~torch.isfinite(actual)).sum())
                case["compared_elements"] += actual.numel()
                case["compared_bytes"] += actual.numel() * actual.element_size()
                actual_digest.update(actual_bytes.numpy().tobytes())
                expected_digest.update(expected_bytes.numpy().tobytes())
                del actual, expected, actual_bytes, expected_bytes, left, r
                if first % (CHUNK_ROWS * 16) == 0:
                    save()
            case["actual_output_sha256"] = actual_digest.hexdigest()
            case["analytic_BF16_output_sha256"] = expected_digest.hexdigest()
            spots = []
            for row, column in spot_coordinates(m, n):
                fp64 = scalar_oracle(row, column, k)
                expected = torch.tensor(fp64, dtype=torch.float64).bfloat16()
                actual = output[row, column].cpu()
                same = tensor_sha(actual.reshape(1)) == tensor_sha(expected.reshape(1))
                spots.append({"row": row, "column": column, "CPU_FP64": fp64,
                              "expected_BF16": float(expected), "actual_BF16": float(actual),
                              "same_bytes": same})
            case["FP64_spotchecks"] = spots
            case["FP64_spotchecks_passed"] = bool(spots) and all(s["same_bytes"] for s in spots)
            case["inputs_unchanged"] = before_hashes == {name: tensor_sha(value) for name, value in inputs.items()}
            case["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            case["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            case["status"] = "passed" if case_passed(case) else "failed"
            save()
            del output, inputs, cpu_inputs, a_cpu, w_cpu, a_scale_cpu, w_scale_cpu, rows, columns, right
            gc.collect()
            torch.cuda.empty_cache()
            case["torch_allocated_bytes_after_release"] = torch.cuda.memory_allocated()
            if case["status"] != "passed":
                break
        all_passed = len(report["cases"]) == len(CASES) and all(c.get("status") == "passed" for c in report["cases"])
        report["status"] = PASS if all_passed else FAIL
        exit_code = 0 if all_passed else 2
    except BaseException:
        report.update(status=FAIL, error=traceback.format_exc())
        exit_code = 1
    finally:
        report["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
        try:
            report["source_unchanged"] = sha(args.source) == args.source_sha256
            report["worker_unchanged"] = sha(Path(__file__)) == report["worker_sha256"]
        except OSError as error:
            report.update(source_unchanged=False, worker_unchanged=False, provenance_error=str(error))
        if not report["source_unchanged"] or not report["worker_unchanged"]:
            report["status"] = FAIL
            exit_code = 1
        save()
    return exit_code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--expected-hostname", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--source-sha256", required=True)
    result.add_argument("--execute", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if args.execute:
        return execute(args)
    print(json.dumps(plan(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
