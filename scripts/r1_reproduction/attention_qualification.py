#!/usr/bin/env python3
"""Standalone SM103 R1 attention correctness gate; never a throughput benchmark.

Run inside an allocated container with an isolated CUDA environment, for example:
  python attention_qualification.py --profile small --expected-hostname HOST
  python attention_qualification.py --profile standard --report attention.json

No BatchGen imports, model weights, installation, or runtime modifications occur.
The fixed BF16 envelope is the repository's existing MLA standard gate:
abs(error) <= 1e-5 + 1.6e-2 * abs(BF16-rounded FP32 oracle), with fewer than
1e-4 elements outside the envelope. The script never adjusts this gate on failure.
This qualifies attention primitives only, not projections, KV transport, graphs,
complete-model correctness, or performance. Start with eager execution.
"""

import argparse
import datetime
import importlib.metadata
import inspect
import json
from pathlib import Path
import socket
import sys
import traceback

import torch


HEADS = 128
QK_DIM = 192
LATENT_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
ATOL = 1e-5
RTOL = 1.6e-2
MAX_FAILURE_FRACTION = 1e-4


def emit(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def versions():
    result = {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda}
    for name in (
        "flash-attn-4", "flashinfer-python", "flashinfer-cubin",
        "flashinfer-jit-cache", "nvidia-cutlass-dsl", "triton",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def check(actual, reference, record, reports):
    """Log errors against FP32 and the fixed BF16 oracle, then fail explicitly."""
    record = dict(record)
    record.update(actual_shape=list(actual.shape), reference_shape=list(reference.shape))
    if actual.shape != reference.shape or actual.dtype != torch.bfloat16:
        record.update(status="failed", reason="shape or BF16 output contract mismatch")
        reports.append(record)
        emit(record)
        raise AssertionError(record["reason"])
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        record.update(status="failed", reason="nonfinite output or FP32 reference")
        reports.append(record)
        emit(record)
        raise AssertionError(record["reason"])
    oracle = reference.bfloat16().float()
    error = actual.float() - oracle
    failures = int((error.abs() > ATOL + RTOL * oracle.abs()).sum())
    fraction = failures / max(actual.numel(), 1)
    fp32_error = actual.float() - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    record.update(
        status="passed" if fraction < MAX_FAILURE_FRACTION else "failed",
        elements=actual.numel(), failure_count=failures, failure_fraction=fraction,
        max_abs_error_bf16_oracle=float(error.abs().max()),
        max_abs_error_fp32=float(fp32_error.abs().max()),
        rms_error_fp32=float(fp32_error.square().mean().sqrt()),
        normalized_rms_error_fp32=float(fp32_error.square().mean().sqrt() / reference_rms),
    )
    reports.append(record)
    emit(record)
    if record["status"] != "passed":
        raise AssertionError(f"{record['case']}: fixed BF16 envelope failed ({fraction})")


def prefill_reference(q, k, v, boundaries, scale):
    """FP32 matmuls/softmax, bounded scratch via head and query chunking."""
    output = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.float32)
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        positions = torch.arange(length, device=q.device)
        for head in range(0, HEADS, 16):
            keys = k[start:end, head:head + 16].float().transpose(0, 1)
            values = v[start:end, head:head + 16].float().transpose(0, 1)
            for offset in range(0, length, 64):
                stop = min(offset + 64, length)
                queries = q[start + offset:start + stop, head:head + 16].float().transpose(0, 1)
                scores = (queries @ keys.transpose(-1, -2)) * scale
                mask = positions[None, :] > positions[offset:stop, None]
                probabilities = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
                output[start + offset:start + stop, head:head + 16] = (probabilities @ values).transpose(0, 1)
    return output


def decode_reference(query, histories, scale):
    outputs = []
    for batch, history in enumerate(histories):
        q = query[batch, 0].float()
        keys = history.float()
        probabilities = ((q @ keys.T) * scale).softmax(-1)
        outputs.append(probabilities @ keys[:, :LATENT_DIM])
    return torch.stack(outputs).unsqueeze(1)


def run_prefill(args, device, scale, reports, metadata):
    from flash_attn.cute import flash_attn_varlen_func

    metadata["fa4_signature"] = str(inspect.signature(flash_attn_varlen_func))
    lengths = [1, 63, 65] if args.profile == "small" else [1, 63, 64, 65, 257, 2048]
    lengths = [min(length, args.max_seqlen) for length in lengths]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    total = boundaries[-1]
    q = torch.randn(total, HEADS, QK_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(total, HEADS, 128, device=device, dtype=torch.bfloat16)
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    actual = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
        softmax_scale=scale, causal=True,
    )
    # FA4 b36 returns (output, optional_lse), even with return_lse=False.
    if isinstance(actual, tuple):
        if len(actual) != 2 or not isinstance(actual[0], torch.Tensor):
            raise TypeError('Unexpected FA4 return contract')
        actual = actual[0]
    torch.cuda.synchronize(device)
    reference = prefill_reference(q, k, v, boundaries, scale)
    check(actual, reference, {
        "case": "fa4_varlen_causal_prefill", "lengths": lengths,
        "heads": HEADS, "qk_dim": QK_DIM, "value_dim": 128,
    }, reports)


def run_decode(args, device, scale, reports, metadata):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    metadata["flashinfer_signature"] = str(inspect.signature(trtllm_batch_decode_with_kv_cache_mla))
    pattern = [1, 63, 65] if args.profile == "small" else [1, 63, 64, 65, 127, 129, 257, 2048]
    lengths = [min(pattern[i % len(pattern)], args.max_seqlen) for i in range(args.decode_batch)]
    max_pages = (max(lengths) + args.decode_steps - 1 + PAGE_SIZE - 1) // PAGE_SIZE
    # FlashInfer 0.6.13 TRTLLM requires table width divisible by 128/page_size.
    max_pages = ((max_pages + 1) // 2) * 2
    active_pages = args.decode_batch * max_pages
    # Extra unused pages contain NaNs, so accidental padding reads are exposed.
    cache = torch.full(
        (active_pages + 7, PAGE_SIZE, LATENT_DIM + ROPE_DIM), float("nan"),
        device=device, dtype=torch.bfloat16,
    )
    page_order = torch.randperm(cache.shape[0], device=device)[:active_pages]
    logical_pages = page_order.reshape(args.decode_batch, max_pages)
    table = logical_pages.to(torch.int32)
    histories = [torch.randn(length, LATENT_DIM + ROPE_DIM, device=device, dtype=torch.bfloat16)
                 for length in lengths]
    # Fill each logical sequence through its deliberately scrambled physical map.
    for batch, history in enumerate(histories):
        positions = torch.arange(history.shape[0], device=device)
        pages = logical_pages[batch, positions // PAGE_SIZE]
        cache[pages, positions % PAGE_SIZE] = history
    workspace = torch.zeros(args.workspace_mib * 1024**2, device=device, dtype=torch.uint8)
    for step in range(args.decode_steps):
        if step:
            for batch, history in enumerate(histories):
                token = torch.randn(1, LATENT_DIM + ROPE_DIM, device=device, dtype=torch.bfloat16)
                position = lengths[batch]
                cache[logical_pages[batch, position // PAGE_SIZE], position % PAGE_SIZE] = token[0]
                histories[batch] = torch.cat((history, token))
                lengths[batch] += 1
        query = torch.randn(args.decode_batch, 1, HEADS, LATENT_DIM + ROPE_DIM,
                            device=device, dtype=torch.bfloat16)
        seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        actual = trtllm_batch_decode_with_kv_cache_mla(
            query=query, kv_cache=cache, workspace_buffer=workspace,
            qk_nope_head_dim=128, kv_lora_rank=LATENT_DIM, qk_rope_head_dim=ROPE_DIM,
            block_tables=table, seq_lens=seq_lens, max_seq_len=max(lengths),
            sparse_mla_top_k=0, bmm1_scale=scale, bmm2_scale=1.0,
            backend="trtllm-gen",
        )
        torch.cuda.synchronize(device)
        reference = decode_reference(query, histories, scale)
        check(actual, reference, {
            "case": f"flashinfer_dense_paged_decode_step_{step}",
            "lengths": list(lengths), "page_size": PAGE_SIZE, "heads": HEADS,
            "qk_dim": LATENT_DIM + ROPE_DIM, "value_dim": LATENT_DIM,
            "shuffled_physical_pages": True, "poisoned_unused_cache": True,
        }, reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("small", "standard"), default="small")
    parser.add_argument("--stage", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--max-seqlen", type=int, default=None)
    parser.add_argument("--decode-batch", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--softmax-scale", type=float, default=QK_DIM ** -0.5,
                        help="Use model's effective RoPE/YaRN-adjusted scale for integration checks")
    parser.add_argument("--expected-hostname")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_seqlen is None:
        args.max_seqlen = 257 if args.profile == "small" else 2048
    if args.decode_batch is None:
        args.decode_batch = 3 if args.profile == "small" else 8
    for name in ("max_seqlen", "decode_batch", "decode_steps", "workspace_mib"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.softmax_scale < float("inf"):
        parser.error("--softmax-scale must be finite and positive")
    result = {
        "status": "failed", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "versions": versions(),
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "gate": {"atol": ATOL, "rtol": RTOL, "max_failure_fraction_exclusive": MAX_FAILURE_FRACTION,
                 "oracle": "FP32 IEEE matmul and softmax rounded to BF16"},
        "cases": [], "scope": "attention primitives only; no model or throughput claim",
    }
    try:
        if not Path("/.dockerenv").exists():
            raise RuntimeError("Run only inside the allocated container")
        if args.expected_hostname and socket.gethostname() != args.expected_hostname:
            raise RuntimeError("Allocated container hostname does not match --expected-hostname")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback is forbidden")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        capability = torch.cuda.get_device_capability(device)
        result["device"] = {"index": args.device, "name": torch.cuda.get_device_name(device),
                            "capability": list(capability)}
        if capability != (10, 3):
            raise RuntimeError(f"Expected SM103, got {capability}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        emit({"event": "start", **result})
        with torch.inference_mode():
            if args.stage in ("all", "prefill"):
                run_prefill(args, device, args.softmax_scale, result["cases"], result)
            if args.stage in ("all", "decode"):
                run_decode(args, device, args.softmax_scale, result["cases"], result)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["status"] = "passed"
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
