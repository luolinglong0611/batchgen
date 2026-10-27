#!/usr/bin/env python3
"""Qualify exclusive-page tail initialization; never a throughput benchmark.

Default is CPU-only plan output. Execute only inside a parent-owned bounded
single-GPU supervisor with R1_MLA_TAIL_GUARDED=1 and CUDA_VISIBLE_DEVICES set.
This script does not stop services, select another GPU, or change a backend.
The caller must keep the allocator snapshot stable until writes are enqueued.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from mla_tail_safety import initialize_owned_tails_, plan_tail_initialization

LENGTHS = (1, 63, 64, 65, 2047, 2048, 2049, 8191, 8192, 8193)
LIMITS = {"external_wall_seconds": 900, "external_RSS_bytes": 8 * 1024**3,
          "external_GPU_bytes": 8 * 1024**3, "torch_allocator_bytes": 6 * 1024**3}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan(args):
    return {"schema": "r1-mla-tail-qualification-v1", "status": "plan_only_NOT_executed",
            "scope": "isolated attention primitive and tail write-set; not model or performance qualification",
            "expected_hostname": args.expected_hostname,
            "flashinfer_version": args.flashinfer_version, "seed": args.seed,
            "torch_version_required": "2.9.0+cu130",
            "flashinfer_cubin_version_required": args.flashinfer_version,
            "lengths": list(LENGTHS), "modes": ["zero", "finite_poison", "nan_latent"],
            "max_elements_written": len(LENGTHS) * 63 * 576,
            "repeats": 2, "limits": LIMITS,
            "original_gate_changed": False,
            "historical_error_envelope": {"atol": 1e-5, "rtol": 0.016,
                                          "maximum_failure_fraction_exclusive": 1e-4},
            "historical_gate_role": "diagnostic only; helper qualification requires zero-tail bitwise parity",
            "capture_source": str(args.captured_inputs) if args.captured_inputs else "seeded synthetic BF16 logical Q/KV",
            "output": str(args.output),
            "source_sha256": {str(p.name): sha(p) for p in
                              (Path(__file__), Path(__file__).with_name("mla_tail_safety.py"))},
            "requires_external_guard": True, "production_backend_changed": False}


def tensor_sha(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def fixture(torch, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = [(length + 63) // 64 for length in LENGTHS]
    pages = sum(counts) + 7
    permutation = torch.randperm(pages, generator=generator).tolist()
    cache = torch.zeros((pages, 64, 1, 576), dtype=torch.bfloat16)
    query = torch.randn(len(LENGTHS), 128, 576, generator=generator).bfloat16()
    width = max(counts)
    width += width % 2
    table = torch.zeros((len(LENGTHS), width), dtype=torch.int32)
    allocations, cursor = {}, 0
    for row, (length, count) in enumerate(zip(LENGTHS, counts)):
        selected = permutation[cursor:cursor + count]; cursor += count
        allocations[row] = selected
        table[row, :count] = torch.tensor(selected, dtype=torch.int32)
        history = torch.randn(length, 576, generator=generator).bfloat16()
        positions = torch.arange(length)
        cache[table[row, positions // 64].long(), positions % 64, 0] = history
    return {"q": query, "cache": cache, "table": table,
            "lengths": torch.tensor(LENGTHS, dtype=torch.int32)}, allocations


def ownership_for(data):
    lengths = data["lengths"].tolist()
    table = data["table"].tolist()
    allocations = {row: list(pages[:(length + 63) // 64])
                   for row, (pages, length) in enumerate(zip(table, lengths))}
    counts = Counter(page for pages in allocations.values() for page in pages)
    return {"sequence_ids": list(range(len(lengths))), "lengths": lengths,
            "table_rows": table, "all_sequence_pages": allocations,
            "page_refcounts": dict(counts), "page_count": data["cache"].shape[0]}


def cpu_reference(torch, data, scale):
    """Independent FP64 reference reads only logical tokens, in 16-head tiles."""
    results = []
    for row, length in enumerate(data["lengths"].tolist()):
        positions = torch.arange(length)
        history = data["cache"][data["table"][row, positions // 64].long(), positions % 64, 0].double()
        q = data["q"][row].double()
        parts = []
        for first in range(0, 128, 16):
            probabilities = ((q[first:first + 16] @ history.T) * scale).softmax(-1)
            parts.append(probabilities @ history[:, :512])
        results.append(torch.cat(parts))
    return torch.stack(results)


def metrics(torch, actual, reference):
    a = actual.detach().cpu().double()
    finite = bool(torch.isfinite(a).all())
    result = {"nonfinite_elements": int((~torch.isfinite(a)).sum()), "finite": finite,
              "elements": a.numel(), "tensor_sha256": tensor_sha(actual)}
    if finite:
        rounded = reference.bfloat16().double()
        error = a - reference
        failures = int(((a - rounded).abs() > 1e-5 + 0.016 * rounded.abs()).sum())
        result.update(max_abs_to_FP64=float(error.abs().max()),
                      normalized_RMS_to_FP64=float(error.norm() / reference.norm().clamp_min(1e-30)),
                      historical_failure_count=failures,
                      historical_failure_fraction=failures / a.numel())
    return result


def case_contract_passed(record):
    """Fail closed on every independent contract, including write-plan parity."""
    if record.get("mode") not in ("zero", "finite_poison", "nan_latent"):
        return False
    required = ("full_cache_exact_expected_write_set", "write_plan_equal",
                "initialized_repeat_bitwise_equal", "initialized_bitwise_equal_to_zero_tail",
                "kernel_inputs_unchanged", "raw_finite_tail_invariance")
    return (all(record.get(name) is True for name in required)
            and record.get("initialized", {}).get("finite") is True)


def execute(args):
    if os.environ.get("R1_MLA_TAIL_GUARDED") != "1":
        raise RuntimeError("Execution requires a parent-owned bounded single-GPU guard")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Exactly one assigned visible CUDA device is required")
    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
        raise RuntimeError("Expected allocated container hostname/marker")
    if args.output.exists():
        raise RuntimeError("Fresh exclusive output directory required; never overwrite evidence")
    actual_version = importlib.metadata.version("flashinfer-python")
    if actual_version != args.flashinfer_version:
        raise RuntimeError("FlashInfer distribution version mismatch")
    if importlib.metadata.version("flashinfer-cubin") != actual_version:
        raise RuntimeError("FlashInfer Python and cubin versions disagree")
    if importlib.metadata.version("torch") != "2.9.0+cu130":
        raise RuntimeError("This qualification requires the frozen Torch2.9.0+cu130 runtime")
    if args.captured_inputs and (not args.captured_sha256 or sha(args.captured_inputs) != args.captured_sha256):
        raise RuntimeError("Original captured inputs require an exact SHA256")
    args.output.mkdir(parents=True, exist_ok=False)
    report = plan(args)
    report.update(status="started_NOT_qualified", cases=[], started_unix=time.time())
    started = time.monotonic()

    def save():
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "report.json")

    save()
    try:
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError("Exactly one assigned SM103 GPU required")
        torch.cuda.set_per_process_memory_fraction(LIMITS["torch_allocator_bytes"] /
                                                   torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        report["actual_versions"] = {name: importlib.metadata.version(name)
                                     for name in ("torch", "flashinfer-python", "flashinfer-cubin")}
        if args.captured_inputs:
            captured = torch.load(args.captured_inputs, weights_only=True, map_location="cpu", mmap=True)
            data = {name: captured[name] for name in ("q", "cache", "table", "lengths")}
            del captured
        else:
            data, _ = fixture(torch, args.seed)
        if (data["q"].dtype != torch.bfloat16 or tuple(data["q"].shape[1:]) != (128, 576)
                or not data["q"].is_contiguous()):
            raise ValueError("Expected captured BF16 q[batch,128,576]")
        if (data["cache"].dtype != torch.bfloat16 or tuple(data["cache"].shape[1:]) != (64, 1, 576)
                or not data["cache"].is_contiguous()):
            raise ValueError("Expected captured BF16 page64 MLA576 cache")
        if data["table"].dtype != torch.int32 or data["lengths"].dtype != torch.int32:
            raise ValueError("Expected int32 page table and sequence lengths")
        if data["table"].ndim != 2 or data["lengths"].ndim != 1:
            raise ValueError("Expected table[batch,width] and lengths[batch]")
        if len(data["lengths"]) != data["q"].shape[0] or data["table"].shape[0] != data["q"].shape[0]:
            raise ValueError("Captured batch dimensions disagree")
        if sum(t.numel() * t.element_size() for t in data.values()) > 256 * 1024**2:
            raise ValueError("Qualification fixture exceeds 256 MiB")
        ownership = ownership_for(data)
        tail_plan = plan_tail_initialization(**ownership)
        scale = args.softmax_scale
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Positive finite original model softmax scale required")
        input_hashes = {name: tensor_sha(value) for name, value in data.items()}
        reference = cpu_reference(torch, data, scale)
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError("Reference has nonfinite logical input; no kernel has been qualified")
        with (args.output / "fixed-inputs.pt").open("xb") as output:
            torch.save(data, output)
        report.update(input_sha256=input_hashes, actual_lengths=data["lengths"].tolist(),
                      tail_plan=tail_plan, softmax_scale=scale,
                      reference_FP64_sha256=tensor_sha(reference),
                      allocator_snapshot_scope="trusted isolated fixture, not a production manager lifetime")
        if args.flashinfer_version == "0.5.2":
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attend
        else:
            from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as attend
        source = inspect.getsourcefile(attend)
        report["kernel_entry"] = {"path": source, "sha256": sha(source),
                                  "signature": str(inspect.signature(attend)),
                                  "distribution_version": actual_version}
        query = data["q"].unsqueeze(1).to("cuda")
        table = data["table"].to("cuda")
        if table.shape[1] % 2:
            table = torch.cat((table, table.new_zeros((table.shape[0], 1))), 1)
        lengths = data["lengths"].to("cuda")
        workspace = torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda")
        report["kernel_calls"] = []

        def invoke(cache):
            # API layout adaptation is lossless. Arithmetic and lengths stay exact.
            presented = (cache.transpose(1, 2).contiguous() if actual_version == "0.5.2"
                         else cache.squeeze(2))
            restored = (presented.transpose(1, 2) if actual_version == "0.5.2"
                        else presented.unsqueeze(2))
            if tensor_sha(restored) != tensor_sha(cache):
                raise RuntimeError("Cache API layout adaptation changed bytes")
            kwargs = dict(query=query, kv_cache=presented, workspace_buffer=workspace,
                          qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                          block_tables=table, seq_lens=lengths,
                          max_seq_len=max(ownership["lengths"]), bmm1_scale=scale, bmm2_scale=1.0)
            if actual_version != "0.5.2":
                kwargs.update(sparse_mla_top_k=0, backend="trtllm-gen")
            signature = inspect.signature(attend)
            if "enable_pdl" in signature.parameters:
                kwargs["enable_pdl"] = False
            signature.bind(**kwargs)
            inputs = {"query": query, "cache": cache, "API_cache": presented,
                      "table": table, "lengths": lengths}
            before_hashes = {name: tensor_sha(value) for name, value in inputs.items()}
            workspace.zero_()
            result = attend(**kwargs)
            expected_shape = (len(ownership["sequence_ids"]), 1, 128, 512)
            if (not isinstance(result, torch.Tensor) or tuple(result.shape) != expected_shape
                    or result.dtype != torch.bfloat16 or result.device != query.device):
                raise RuntimeError("FlashInfer returned unexpected output shape, BF16 dtype or device")
            # Own the observed bytes before a later invocation can reuse its
            # workspace/output storage; repeated calls must not overwrite raw.
            snapshot = result.squeeze(1).clone()
            torch.cuda.synchronize()
            after_hashes = {name: tensor_sha(value) for name, value in inputs.items()}
            call = {"call": len(report["kernel_calls"]) + 1,
                    "input_hashes": before_hashes,
                    "inputs_unchanged": before_hashes == after_hashes,
                    "API_cache_shape": list(presented.shape),
                    "output_shape": list(snapshot.shape), "output_dtype": str(snapshot.dtype)}
            report["kernel_calls"].append(call)
            if not call["inputs_unchanged"]:
                call["after_input_hashes"] = after_hashes
                raise RuntimeError("FlashInfer mutated an immutable query/cache/table/length input")
            return snapshot

        zero_output = None
        all_contracts = True
        for mode in ("zero", "finite_poison", "nan_latent"):
            # Only this small immutable qualification fixture is copied, never a
            # production KV pool. The helper itself makes no whole-cache copy.
            cache = data["cache"].to("cuda")
            for write in tail_plan["ranges"]:
                tail = cache[write["physical_page"], write["start_token"]:64, 0, :]
                if mode == "finite_poison":
                    tail.copy_(((torch.arange(576, device="cuda") % 2) * 2 - 1).to(torch.bfloat16) * 64)
                elif mode == "nan_latent":
                    tail[:, :512] = float("nan")
                else:
                    tail.zero_()
            raw = invoke(cache)
            raw2 = invoke(cache)
            if mode == "zero":
                zero_output = raw.detach().cpu()
            expected_cache = cache.cpu()
            for write in tail_plan["ranges"]:
                expected_cache[write["physical_page"], write["start_token"]:64, 0, :].zero_()
            actual_plan = initialize_owned_tails_(cache, **ownership)
            torch.cuda.synchronize()
            full_bytes_match = tensor_sha(cache) == tensor_sha(expected_cache)
            safe = invoke(cache)
            safe2 = invoke(cache)
            equal_to_zero = torch.equal(safe.cpu(), zero_output)
            repeated = torch.equal(safe, safe2)
            raw_invariance = (mode == "nan_latent" or
                              (bool(torch.isfinite(raw).all())
                               and torch.equal(raw.cpu(), zero_output)
                               and tensor_sha(raw) == tensor_sha(raw2)))
            record = {"mode": mode, "raw": metrics(torch, raw, reference),
                      "raw_repeat_same_bytes": tensor_sha(raw) == tensor_sha(raw2),
                      "initialized": metrics(torch, safe, reference),
                      "initialized_repeat_bitwise_equal": repeated,
                      "initialized_bitwise_equal_to_zero_tail": equal_to_zero,
                      "full_cache_exact_expected_write_set": full_bytes_match,
                      "write_plan_equal": actual_plan == tail_plan,
                      "kernel_inputs_unchanged": all(c["inputs_unchanged"] for c in report["kernel_calls"]),
                      "raw_finite_tail_invariance": raw_invariance}
            case_pass = case_contract_passed(record)
            record["status"] = "passed" if case_pass else "failed"
            all_contracts &= case_pass
            with (args.output / f"output-{mode}-initialized.pt").open("xb") as output:
                torch.save({"output": safe.cpu()}, output)
            report["cases"].append(record)
            save()
            del cache, raw, raw2, safe, safe2, expected_cache

        # A valid NaN must survive initialization exactly. Do not pass this cache
        # to the independent finite-input oracle or replace its valid values.
        cache = data["cache"].to("cuda")
        first_page = ownership["all_sequence_pages"][0][0]
        cache[first_page, 0, 0, 0] = float("nan")
        before = cache[first_page, 0, 0].clone()
        expected_cache = cache.cpu()
        for write in tail_plan["ranges"]:
            expected_cache[write["physical_page"], write["start_token"]:64, 0, :].zero_()
        initialize_owned_tails_(cache, **ownership)
        preserved = tensor_sha(before) == tensor_sha(cache[first_page, 0, 0])
        report["valid_NaN_preserved_bitwise"] = preserved
        report["valid_NaN_case_exact_cache_write_set"] = tensor_sha(cache) == tensor_sha(expected_cache)
        all_contracts &= preserved and report["valid_NaN_case_exact_cache_write_set"]
        report["input_source_unchanged"] = input_hashes == {name: tensor_sha(value) for name, value in data.items()}
        all_contracts &= report["input_source_unchanged"]
        report["status"] = ("passed_isolated_tail_contract_NOT_model_or_performance_qualified"
                            if all_contracts else "failed_tail_contract")
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        return 0 if all_contracts else 2
    except BaseException:
        report.update(status="failed_NOT_qualified", error=traceback.format_exc())
        return 1
    finally:
        report["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
        report["source_unchanged"] = report["source_sha256"] == plan(args)["source_sha256"]
        if args.captured_inputs:
            report["original_capture_file_unchanged"] = sha(args.captured_inputs) == args.captured_sha256
        if not report["source_unchanged"] or not report.get("original_capture_file_unchanged", True):
            report["status"] = "failed_source_or_capture_mutation"
        save()
        if report["status"] == "failed_source_or_capture_mutation":
            return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--expected-hostname", required=True)
    result.add_argument("--flashinfer-version", choices=("0.5.2", "0.6.13"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20261026)
    result.add_argument("--softmax-scale", type=float,
                        default=(1 + 0.1 * math.log(40))**2 / math.sqrt(192))
    result.add_argument("--captured-inputs", type=Path)
    result.add_argument("--captured-sha256")
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
