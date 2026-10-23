#!/usr/bin/env python3
"""Diagnose FlashInfer 0.6.13 dense MLA padding without changing its strict gate.

Run only in the allocated SM103 container. Each decode step uses identical valid
KV, queries and scrambled page tables across zero, finite-poison and NaN cases.
Failures do not short-circuit the matrix. Exit status remains nonzero if ANY case
fails attention_qualification.check; a completed diagnostic is not a pass.

Examples:
  python decode_padding_diagnostic.py --expected-hostname HOST --report padding.json
  python decode_padding_diagnostic.py --profile standard --report padding-standard.json

The 512 latent channels are both K and V; the last 64 RoPE channels affect K
only. Contrasting them helps identify masked P*NaN V propagation. An FP32
masked-logit/P@V experiment is recorded as explanatory evidence, never an oracle
that replaces the original valid-token-only reference or relaxes its tolerance.
"""

import argparse
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import socket
import sys
import traceback

import torch

from attention_qualification import (
    ATOL, HEADS, LATENT_DIM, MAX_FAILURE_FRACTION, PAGE_SIZE, QK_DIM,
    ROPE_DIM, RTOL, check, decode_reference, emit, versions,
)


REGIONS = ("tail", "table_padding_pages", "unaddressed_pages", "all_unused")
CHANNELS = {"latent_k_and_v": (0, LATENT_DIM),
            "rope_k_only": (LATENT_DIM, LATENT_DIM + ROPE_DIM),
            "all": (0, LATENT_DIM + ROPE_DIM)}


def matrix():
    yield "zero", "all_unused", "all"
    for mode in ("finite_poison", "nan"):
        for region in REGIONS:
            for channels in CHANNELS:
                yield mode, region, channels


def masks_and_cache(histories, logical_pages, physical_pages, device):
    shape = (physical_pages, PAGE_SIZE)
    base = torch.zeros((*shape, LATENT_DIM + ROPE_DIM), device=device, dtype=torch.bfloat16)
    valid = torch.zeros(shape, device=device, dtype=torch.bool)
    tail = torch.zeros_like(valid)
    table_padding = torch.zeros_like(valid)
    addressed = torch.zeros(physical_pages, device=device, dtype=torch.bool)
    addressed[logical_pages.flatten()] = True
    for batch, history in enumerate(histories):
        length = history.shape[0]
        positions = torch.arange(length, device=device)
        pages = logical_pages[batch, positions // PAGE_SIZE]
        slots = positions % PAGE_SIZE
        base[pages, slots] = history
        valid[pages, slots] = True
        used_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        if length % PAGE_SIZE:
            tail[logical_pages[batch, used_pages - 1], length % PAGE_SIZE:] = True
        table_padding[logical_pages[batch, used_pages:]] = True
    unaddressed = (~addressed)[:, None].expand(shape)
    masks = {"tail": tail, "table_padding_pages": table_padding,
             "unaddressed_pages": unaddressed, "all_unused": ~valid}
    assert torch.equal(tail | table_padding | unaddressed, ~valid)
    assert not bool((tail & table_padding).any())
    return base, valid, masks


def poison_cache(base, mask, mode, channel_name, magnitude):
    cache = base.clone()
    lo, hi = CHANNELS[channel_name]
    view = cache[..., lo:hi]
    if mode == "nan":
        view[mask] = float("nan")
    elif mode == "finite_poison":
        signs = torch.arange(lo, hi, device=cache.device).remainder(2).mul(2).sub(1)
        view[mask] = (signs * magnitude).to(cache.dtype)
    return cache


def nonfinite_by_batch(tensor):
    return (~torch.isfinite(tensor)).reshape(tensor.shape[0], -1).sum(-1).tolist()


def masked_pv_diagnostic(query, cache, logical_pages, lengths, scale):
    """Demonstrate the mathematical mechanism, not the fused kernel's internals."""
    records = []
    for batch, length in enumerate(lengths):
        keys = cache[logical_pages[batch]].reshape(-1, LATENT_DIM + ROPE_DIM).float()
        scores = query[batch, 0].float() @ keys.T * scale
        invalid = torch.arange(keys.shape[0], device=keys.device) >= length
        masked_scores = scores.masked_fill(invalid[None], -torch.inf)
        probabilities = masked_scores.softmax(-1)
        values = keys[:, :LATENT_DIM]
        output = probabilities @ values
        sanitized = probabilities @ values.masked_fill(invalid[:, None], 0.)
        records.append({
            "batch": batch, "length": length,
            "raw_score_nonfinite": int((~torch.isfinite(scores)).sum()),
            "probability_nonfinite": int((~torch.isfinite(probabilities)).sum()),
            "masked_probabilities_nonzero": int((probabilities[:, invalid] != 0).sum()),
            "padding_value_nonfinite": int((~torch.isfinite(values[invalid])).sum()),
            "masked_p_times_v_nonfinite": int((~torch.isfinite(output)).sum()),
            "masked_p_times_zeroed_v_nonfinite": int((~torch.isfinite(sanitized)).sum()),
        })
    return records


def run(args, result, device):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as decode

    result["flashinfer_signature"] = str(inspect.signature(decode))
    try:
        result["installed_wrapper_source_sha256"] = hashlib.sha256(
            inspect.getsource(decode).encode()).hexdigest()
    except (OSError, TypeError):
        result["installed_wrapper_source_sha256"] = None
    pattern = [1, 63, 65] if args.profile == "small" else [1, 63, 64, 65, 127, 129, 257, 2048]
    lengths = [min(pattern[i % len(pattern)], args.max_seqlen) for i in range(args.decode_batch)]
    max_pages = (max(lengths) + args.decode_steps - 1 + PAGE_SIZE - 1) // PAGE_SIZE
    # Same table-width alignment as the original gate; no sentinel substitution.
    max_pages = ((max_pages + 1) // 2) * 2
    active_pages = args.decode_batch * max_pages
    physical_pages = active_pages + 7
    logical_pages = torch.randperm(physical_pages, device=device)[:active_pages].reshape(
        args.decode_batch, max_pages)
    table = logical_pages.to(torch.int32)
    result["block_tables"] = table.tolist()
    result["physical_pages"] = physical_pages
    histories = [torch.randn(length, LATENT_DIM + ROPE_DIM, device=device, dtype=torch.bfloat16)
                 for length in lengths]
    workspace = torch.zeros(args.workspace_mib * 1024**2, device=device, dtype=torch.uint8)
    for step in range(args.decode_steps):
        if step:
            for batch in range(args.decode_batch):
                token = torch.randn(1, LATENT_DIM + ROPE_DIM, device=device, dtype=torch.bfloat16)
                histories[batch] = torch.cat((histories[batch], token))
                lengths[batch] += 1
        query = torch.randn(args.decode_batch, 1, HEADS, LATENT_DIM + ROPE_DIM,
                            device=device, dtype=torch.bfloat16)
        reference = decode_reference(query, histories, args.softmax_scale)
        base, valid, masks = masks_and_cache(histories, logical_pages, physical_pages, device)
        seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        zero_output = None
        for mode, region, channels in matrix():
            record = {
                "case": f"step_{step}/{mode}/{region}/{channels}",
                "step": step, "mode": mode, "region": region, "channels": channels,
                "lengths": list(lengths), "page_size": PAGE_SIZE,
                "poisoned_token_slots": int(masks[region].sum()) if mode != "zero" else 0,
                "finite_poison_magnitude": args.poison_magnitude if mode == "finite_poison" else None,
            }
            before = len(result["cases"])
            try:
                cache = poison_cache(base, masks[region], mode, channels, args.poison_magnitude)
                assert torch.equal(cache[valid], base[valid]), "Poison modified valid KV"
                workspace.zero_()
                actual = decode(
                    query=query, kv_cache=cache, workspace_buffer=workspace,
                    qk_nope_head_dim=128, kv_lora_rank=LATENT_DIM, qk_rope_head_dim=ROPE_DIM,
                    block_tables=table, seq_lens=seq_lens, max_seq_len=max(lengths),
                    sparse_mla_top_k=0, bmm1_scale=args.softmax_scale, bmm2_scale=1.0,
                    backend="trtllm-gen",
                )
                torch.cuda.synchronize(device)
                record["output_nonfinite_by_batch"] = nonfinite_by_batch(actual)
                if mode == "zero":
                    zero_output = actual.clone()
                elif zero_output is not None:
                    common = torch.isfinite(actual) & torch.isfinite(zero_output)
                    record["comparison_to_zero_common_finite_elements"] = int(common.sum())
                    record["max_abs_difference_from_zero_on_common_finite"] = (
                        float((actual.float()[common] - zero_output.float()[common]).abs().max())
                        if bool(common.any()) else None
                    )
                if mode == "nan":
                    record["fp32_masked_pv_diagnostic"] = masked_pv_diagnostic(
                        query, cache, logical_pages, lengths, args.softmax_scale)
                # Always use the original valid-token reference and unchanged gate.
                check(actual, reference, record, result["cases"])
            except Exception as exc:
                # check() already records its strict failure. Keep running the
                # other fills instead of treating an expected NaN as success.
                if len(result["cases"]) == before:
                    record.update(status="failed", reason="diagnostic execution error",
                                  error={"type": type(exc).__name__, "message": str(exc)})
                    result["cases"].append(record)
                    emit(record)
                if "illegal memory access" in str(exc).lower() or "device-side assert" in str(exc).lower():
                    raise  # This CUDA context can no longer execute valid comparisons.
    result["diagnostic_completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("small", "standard"), default="small")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--max-seqlen", type=int)
    parser.add_argument("--decode-batch", type=int)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--softmax-scale", type=float, default=QK_DIM ** -0.5)
    parser.add_argument("--poison-magnitude", type=float, default=64.)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.max_seqlen = args.max_seqlen if args.max_seqlen is not None else (257 if args.profile == "small" else 2048)
    args.decode_batch = args.decode_batch if args.decode_batch is not None else (3 if args.profile == "small" else 8)
    for name in ("max_seqlen", "decode_batch", "decode_steps", "workspace_mib"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.softmax_scale < float("inf") or not 1 < args.poison_magnitude <= 1024:
        parser.error("Require finite positive scale and finite poison magnitude in (1, 1024]")
    result = {
        "status": "failed", "diagnostic_completed": False,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "versions": versions(),
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "gate": {"atol": ATOL, "rtol": RTOL,
                 "max_failure_fraction_exclusive": MAX_FAILURE_FRACTION,
                 "oracle": "Original valid-token-only FP32 reference rounded to BF16"},
        "cases": [], "expected_cases": len(list(matrix())) * args.decode_steps,
        "scope": "Padding diagnosis only; no gate waiver, model correctness or performance claim",
        "interpretation": "Finite padding pass plus latent NaN failure and RoPE-only NaN pass is consistent with masked P*NaN V propagation. It does not prove fused-kernel internals or establish a documented NaN-padding guarantee.",
        "upstream_source": "https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/mla/_core.py",
    }
    try:
        if not Path("/.dockerenv").exists():
            raise RuntimeError("Run only inside the allocated container")
        if args.expected_hostname and socket.gethostname() != args.expected_hostname:
            raise RuntimeError("Allocated container hostname mismatch")
        if result["versions"].get("flashinfer-python") != "0.6.13":
            raise RuntimeError("This diagnostic targets flashinfer-python==0.6.13")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        capability = torch.cuda.get_device_capability(device)
        if capability != (10, 3):
            raise RuntimeError(f"Expected SM103, got {capability}")
        result["device"] = {"name": torch.cuda.get_device_name(device), "capability": list(capability)}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        emit({"event": "start", **result})
        with torch.inference_mode():
            run(args, result, device)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["status"] = "passed" if all(case["status"] == "passed" for case in result["cases"]) else "failed"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=sys.stderr)
    finally:
        emit({"event": "summary", **result})
        if args.report:
            args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
