#!/usr/bin/env python3
"""Plan an isolated, same-input R1 L42 attention diagnostic; never a benchmark.

Default is plan-only. --execute starts the separate bounded supervisor after the
resource owner releases one GPU. No production selector or gate is changed.
Prefill compares precision/FA4/FlashInfer on identical captured Q/K/V. Paged
latent decode compares precision/FlashInfer; FA4 has no equivalent paged-latent
entry point in the existing integration. NaN-tail probes are reported separately
from finite-input execution and from the unchanged historical strict gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
CONFIG_SHA256 = "4422016f41c506339e0510a32812a63da554dc9febd5f1368e7240ded14e4a27"
HISTORICAL_GATE_SHA256 = "a00869226d16695ec7c009942a401eb90914b9b77b00418edb9485b5ba916a0b"
STAGES = {
    "prefill": (63, 64, 65),
    "decode2048": (2047, 2048, 2049),
    "decode8192": (8191, 8192, 8193),
}
LIMITS = {
    "seconds": 900,
    "session_RSS_bytes": 8 * 1024**3,
    "target_GPU_bytes": 8 * 1024**3,
    "MemAvailable_min_bytes": 64 * 1024**3,
}
PINNED_VERSIONS = {
    "torch": "2.9.0+cu130",
    "triton": "3.5.0",
    "flash-attn-4": "4.0.0b36",
    "flashinfer-python": "0.6.13",
    "flashinfer-cubin": "0.6.13",
    "nvidia-cutlass-dsl": "4.8.1",
}
SOURCE_FILES = (
    "scripts/r1_reproduction/attention_only_qualification.py",
    "scripts/r1_reproduction/attention_only_guard.py",
    "scripts/r1_reproduction/attention_qualification.py",
    "scripts/r1_reproduction/attention_layer_qualification.py",
    "scripts/r1_reproduction/convert_verified_shards.py",
    "batchgen/attention/mla/sm103_backend.py",
    "batchgen/attention/mla/triton_precision.py",
    "batchgen/attention/mla/rotary_embedding.py",
    "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py",
    "batchgen/gemm/w8a8.py",
    "batchgen_kernels/triton/fp8_quantize.py",
    "batchgen_kernels/common/mgn.py",
)


class CapturedInputMutation(RuntimeError):
    """A later candidate must never receive a previous candidate's writes."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_lengths(stage):
    return list(STAGES[stage])


def plan(args):
    return {
        "status": "plan_only_NOT_executed",
        "stage": args.stage,
        "lengths_including_current_decode_token": stage_lengths(args.stage),
        "layer_index": 42,
        "seed": args.seed,
        "repeats": args.repeats,
        "physical_GPU": args.gpu,
        "limits": LIMITS,
        "limit_enforcement": "Sampled external session/GPU guard; 6 GiB Torch allocator cap. Not a hard OS RSS limit.",
        "versions_required": PINNED_VERSIONS,
        "config_sha256_required": CONFIG_SHA256,
        "historical_gate_source_sha256_required": HISTORICAL_GATE_SHA256,
        "required_environment": {"DISABLE_MMA_V5": "1", "BATCHGEN_W8A16_DEQUANT": "0"},
        "source_sha256": {name: sha256_file(REPO / name) for name in SOURCE_FILES},
        "config": str(args.config),
        "weights_root": str(args.weights_root),
        "weights_manifest": str(args.weights_manifest),
        "output": str(args.output),
        "prefill_candidates": ["triton_precision", "fa4", "flashinfer_trtllm_ragged"],
        "decode_candidates": ["triton_precision", "flashinfer_trtllm_mla"],
        "FA4_decode": "not_applicable: no equivalent paged-latent entry point in existing integration",
        "capture_contract": "One shared production-projected BF16 boundary; not independent upstream/model validation",
        "capture_scheduling": "History projections use 128-token chunks; not a byte-equivalence test of serving scheduling",
        "page_layout_contract": "Shuffled page64 fixture and real SM103 table adapter; production allocator/offload lifecycle excluded",
        "strict_gate": {
            "atol": 1e-5,
            "rtol": 0.016,
            "max_failure_fraction_exclusive": 1e-4,
        },
        "nan_tail_policy": "separate failed diagnostic; never folded into finite execution status",
        "scope": "attention-only diagnostics; excludes model quality, scheduler/offload and performance qualification",
    }


def tensor_record(tensor):
    import torch

    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def save_tensors(path, tensors):
    """Save exact CPU boundary bytes, with no overwrite and a bounded artifact."""
    import torch

    payload = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
    if sum(t.numel() * t.element_size() for t in payload.values()) > 2 * 1024**3:
        raise RuntimeError("Refusing an input artifact larger than 2 GiB")
    records = {name: tensor_record(tensor) for name, tensor in payload.items()}
    with Path(path).open("xb") as handle:
        torch.save(payload, handle)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(path),
        "tensors": records,
    }


def independent_metrics(actual, reference):
    """No threshold here; keep these observations distinct from the old gate."""
    import torch

    if actual.shape != reference.shape:
        return {"shape_match": False}
    a, b = actual.detach().double(), reference.detach().double().to(actual.device)
    finite = torch.isfinite(a) & torch.isfinite(b)
    result = {
        "shape_match": True,
        "actual_nonfinite": int((~torch.isfinite(a)).sum()),
        "reference_nonfinite": int((~torch.isfinite(b)).sum()),
    }
    if bool(finite.all()):
        error = a - b
        result.update(
            max_abs=float(error.abs().max()),
            relative_l2=float(error.norm() / b.norm().clamp_min(1e-30)),
            rms_error=float(error.square().mean().sqrt()),
        )
    return result


def repeat_metrics(actual, first):
    return {
        "same_raw_bytes": tensor_record(actual) == tensor_record(first),
        **independent_metrics(actual, first),
    }


def assert_inputs_unchanged(captured, fixed, cache=None, cache_before=None):
    current = {key: tensor_record(captured[key]) for key in fixed}
    if current != fixed or (cache is not None and tensor_record(cache) != cache_before):
        raise CapturedInputMutation("Candidate mutated shared captured input bytes")


def historical_gate(actual, reference, label):
    from attention_qualification import check

    records = []
    try:
        check(actual, reference, {"case": label}, records)
    except AssertionError:
        pass
    return records[0]


def build_paged_capture(histories, seed):
    """Real page64/latent576 layout, shuffled pages and manager-style -1 tails.

    The private kernel table replaces unused sentinels with page0 and pads odd
    width, exactly as the existing SM103 adapter. Original table is retained.
    """
    import torch

    if not histories or any(
        h.ndim != 2 or h.shape[1] != 576 or h.shape[0] < 1 or h.dtype != torch.bfloat16
        for h in histories
    ):
        raise ValueError("Nonempty BF16 histories[length,576] required")
    device = histories[0].device
    if any(h.device != device for h in histories):
        raise ValueError("Histories must share one device")
    lengths = torch.tensor([h.shape[0] for h in histories], device=device, dtype=torch.int32)
    width = (int(lengths.max()) + 63) // 64
    width += 1 - width % 2  # Deliberately odd manager table.
    pages = len(histories) * width + 7
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = torch.randperm(pages, generator=generator)[: len(histories) * width]
    assignments = assignments.reshape(len(histories), width).to(device=device)
    used = torch.arange(width, device=device)[None] < ((lengths + 63) // 64)[:, None]
    raw_table = assignments.to(torch.int32).masked_fill(~used, -1)
    table = torch.where(used, raw_table, 0)
    table = torch.cat((table, table.new_zeros((len(histories), 1))), dim=1).contiguous()
    cache = torch.zeros((pages, 64, 1, 576), device=device, dtype=torch.bfloat16)
    tail = torch.zeros((pages, 64), device=device, dtype=torch.bool)
    for row, history in enumerate(histories):
        positions = torch.arange(history.shape[0], device=device)
        cache[assignments[row, positions // 64], positions % 64, 0] = history
        if history.shape[0] % 64:
            tail[assignments[row, (history.shape[0] - 1) // 64], history.shape[0] % 64 :] = True
    return {
        "cache": cache,
        "raw_table": raw_table,
        "table": table,
        "lengths": lengths,
        "tail_mask": tail,
    }


def fill_tail(cache, mask, mode):
    import torch

    if mode not in ("zero", "finite_poison", "nan_latent"):
        raise ValueError("Unknown tail mode")
    result = cache.clone()
    if mode == "finite_poison":
        values = (torch.arange(576, device=cache.device).remainder(2) * 2 - 1) * 64
        result[:, :, 0][mask] = values.to(cache.dtype)
    elif mode == "nan_latent":
        result[:, :, 0, :512][mask] = float("nan")
    return result


def capture_qkv(layer, backend, hidden, positions, cos, sin):
    """Execute upstream operations once; every candidate receives these bytes."""
    import torch
    from batchgen.attention.mla.rotary_embedding import (
        rotary_pos_emb_interleaved_native as rotate,
    )

    quantized = backend._quantize(hidden)
    q = backend._project_decode(layer, "q_a_proj", quantized, layer._captured_scales)
    q = backend._project_decode(
        layer,
        "q_b_proj",
        backend._quantize(layer.q_a_layernorm(q)),
        layer._captured_scales,
    )
    q_nope, q_pe = q.view(-1, 128, 192).split((128, 64), -1)
    kv = backend._project_decode(layer, "kv_a_proj_with_mqa", quantized, layer._captured_scales)
    latent = layer.kv_a_layernorm(kv[:, :512])
    q_pe = rotate(q_pe.unsqueeze(0), cos, sin, positions.unsqueeze(0), 2).squeeze(0)
    k_pe = rotate(kv[:, 512:].view(1, -1, 1, 64), cos, sin, positions.unsqueeze(0), 2).squeeze(0)
    return q_nope, q_pe, latent, k_pe


def capture_inputs(args, layer, backend, device):
    import torch
    from batchgen.attention.mla.rotary_embedding import (
        rotary_pos_emb_interleaved_native as rotate,
    )
    from batchgen_kernels.triton.fp8_quantize import deepseek_v3_dequantization

    lengths = stage_lengths(args.stage)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cos, sin = layer.rotary_emb(
        torch.empty(0, device=device, dtype=torch.bfloat16), seq_len=max(lengths)
    )
    common = {"yarn_cos": cos, "yarn_sin": sin}
    if args.stage == "prefill":
        hidden = torch.randn(sum(lengths), 7168, generator=generator).bfloat16().to(device)
        positions = torch.cat([torch.arange(n, device=device) for n in lengths])
        q_nope, q_pe, latent, k_pe = capture_qkv(layer, backend, hidden, positions, cos, sin)
        expanded = backend._project_prefill(
            layer, "kv_b_proj", latent, layer._captured_scales
        ).view(-1, 128, 256)
        k_nope, values = expanded.split((128, 128), -1)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        common.update(
            hidden=hidden,
            positions=positions,
            cu_seqlens=torch.tensor(boundaries, device=device, dtype=torch.int32),
            lengths=torch.tensor(lengths, device=device, dtype=torch.int32),
            q=torch.cat((q_nope, q_pe), -1).contiguous(),
            k=torch.cat((k_nope, k_pe.expand(-1, 128, -1)), -1).contiguous(),
            v=values.contiguous(),
            latent_kv=torch.cat((latent, k_pe[:, 0]), -1),
        )
        return common, None
    histories, raw_hidden, raw_positions = [], {}, {}
    for row, length in enumerate(lengths):
        history_chunks, hidden_chunks = [], []
        for start in range(0, length - 1, 128):
            stop = min(start + 128, length - 1)
            hidden_cpu = torch.randn(stop - start, 7168, generator=generator).bfloat16()
            hidden_chunks.append(hidden_cpu)
            hidden = hidden_cpu.to(device)
            kv = backend._project_prefill(
                layer, "kv_a_proj_with_mqa", hidden, layer._captured_scales
            )
            latent = layer.kv_a_layernorm(kv[:, :512])
            positions = torch.arange(start, stop, device=device)
            k_pe = rotate(
                kv[:, 512:].view(1, -1, 1, 64), cos, sin, positions.unsqueeze(0), 2
            ).reshape(-1, 64)
            history_chunks.append(torch.cat((latent, k_pe), -1))
        histories.append(torch.cat(history_chunks))
        raw_hidden[f"history_hidden_{row}"] = torch.cat(hidden_chunks)
        raw_positions[f"history_positions_{row}"] = torch.arange(length - 1, dtype=torch.int64)
    hidden_cpu = torch.randn(len(lengths), 7168, generator=generator).bfloat16()
    positions = torch.tensor([n - 1 for n in lengths], device=device, dtype=torch.int64)
    q_nope, q_pe, latent, k_pe = capture_qkv(
        layer, backend, hidden_cpu.to(device), positions, cos, sin
    )
    new_kv = torch.cat((latent, k_pe[:, 0]), -1)
    histories = [
        torch.cat((history, new_kv[row : row + 1])) for row, history in enumerate(histories)
    ]
    absorb = deepseek_v3_dequantization(
        layer.kv_b_proj.weight, layer._captured_scales["kv_b_proj.weight_scale_inv"]
    )
    absorb = absorb.view(128, 256, 512)
    latent_query = torch.einsum("bhd,hdc->bhc", q_nope, absorb[:, :128])
    common.update(
        raw_hidden,
        **raw_positions,
        current_hidden=hidden_cpu,
        current_positions=positions,
        current_latent_kv=new_kv,
        q=torch.cat((latent_query, q_pe), -1).contiguous(),
    )
    paged = build_paged_capture(histories, args.seed + 17)
    actual_table = backend._validated_block_table(
        paged["cache"], paged["raw_table"], paged["lengths"], len(histories), device
    )
    if not torch.equal(paged["table"], actual_table):
        raise RuntimeError("Captured page table differs from the actual SM103 adapter")
    paged["table"] = actual_table
    common.update(paged)
    return common, histories


def function_record(function):
    path = inspect.getsourcefile(function)
    return {
        "signature": str(inspect.signature(function)),
        "source_path": path,
        "source_file_sha256": (sha256_file(path) if path and Path(path).is_file() else None),
    }


def candidates(args, captured, scale, report):
    import torch
    from batchgen.attention.mla.triton_precision import varlen_prefill, paged_mla_decode

    workspace = {}

    def space(name):
        if name not in workspace:
            workspace[name] = torch.zeros(
                128 * 1024**2, device=captured["q"].device, dtype=torch.uint8
            )
        return workspace[name]

    if args.stage == "prefill":
        from flash_attn.cute import flash_attn_varlen_func
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        q, k, v, cu = (captured[n] for n in ("q", "k", "v", "cu_seqlens"))
        lengths = captured["lengths"]

        def fa4(_):
            value = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=65,
                max_seqlen_k=65,
                softmax_scale=scale,
                causal=True,
            )
            if not isinstance(value, tuple) or len(value) != 2:
                raise RuntimeError("Expected pinned FA4 b36 (output,lse) return contract")
            return value[0]

        def flashinfer(_):
            kwargs = dict(
                query=q,
                key=k,
                value=v,
                workspace_buffer=space("flashinfer_prefill"),
                seq_lens=lengths,
                max_q_len=65,
                max_kv_len=65,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=3,
                window_left=-1,
                cum_seq_lens_q=cu,
                cum_seq_lens_kv=cu,
                enable_pdl=False,
                is_causal=True,
                return_lse=False,
            )
            inspect.signature(trtllm_ragged_attention_deepseek).bind(**kwargs)
            return trtllm_ragged_attention_deepseek(**kwargs)

        functions = {
            "triton_precision": varlen_prefill,
            "fa4": flash_attn_varlen_func,
            "flashinfer_trtllm_ragged": trtllm_ragged_attention_deepseek,
        }
        result = [
            (
                "triton_precision",
                lambda _: varlen_prefill(q, k, v, cu, 65, softmax_scale=scale),
            ),
            ("fa4", fa4),
            ("flashinfer_trtllm_ragged", flashinfer),
        ]
    else:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        q, table, lengths = (captured[n] for n in ("q", "table", "lengths"))

        def flashinfer(cache):
            kwargs = dict(
                query=q.unsqueeze(1),
                kv_cache=cache.squeeze(2),
                workspace_buffer=space("flashinfer_decode"),
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=table,
                seq_lens=lengths,
                max_seq_len=int(lengths.max()),
                sparse_mla_top_k=0,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                backend="trtllm-gen",
            )
            inspect.signature(trtllm_batch_decode_with_kv_cache_mla).bind(**kwargs)
            return trtllm_batch_decode_with_kv_cache_mla(**kwargs).squeeze(1)

        functions = {
            "triton_precision": paged_mla_decode,
            "flashinfer_trtllm_mla": trtllm_batch_decode_with_kv_cache_mla,
        }
        result = [
            (
                "triton_precision",
                lambda cache: paged_mla_decode(q, cache, table, lengths, softmax_scale=scale),
            ),
            ("flashinfer_trtllm_mla", flashinfer),
        ]
    report["candidate_functions"] = {
        name: function_record(function) for name, function in functions.items()
    }
    return result


def worker(args):
    if os.environ.get("BATCHGEN_ATTENTION_ONLY_GUARDED") != "1":
        raise RuntimeError("GPU worker must be launched by the bounded supervisor")
    import torch
    import attention_qualification as primitive
    import attention_layer_qualification as layer_test

    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").exists():
        raise RuntimeError("Wrong allocated container")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("Assigned physical GPU must be the only visible CUDA device")
    if os.environ.get("DISABLE_MMA_V5") != "1" or torch.version.cuda != "13.0":
        raise RuntimeError("Requires explicit DISABLE_MMA_V5=1 and Torch cu130")
    if os.environ.get("BATCHGEN_W8A16_DEQUANT", "0") != "0":
        raise RuntimeError("Original FP8 activation GEMM must be retained")
    actual_versions = {name: metadata.version(name) for name in PINNED_VERSIONS}
    if actual_versions != PINNED_VERSIONS:
        raise RuntimeError(f"Unpinned environment: {actual_versions}")
    if (primitive.ATOL, primitive.RTOL, primitive.MAX_FAILURE_FRACTION) != (
        1e-5,
        0.016,
        1e-4,
    ):
        raise RuntimeError("Historical gate changed; refuse to run")
    if sha256_file(Path(primitive.__file__)) != HISTORICAL_GATE_SHA256:
        raise RuntimeError("Historical gate source changed; explicit re-audit required")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("Only allocated SM103 is supported")
    torch.cuda.set_per_process_memory_fraction(
        6 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = json.loads(args.config.read_text())
    if sha256_file(args.config) != CONFIG_SHA256:
        raise RuntimeError("Requires the pinned original R1 config and YaRN parameters")
    layer_test.validate_config(config)
    report = plan(args)
    report.update(
        status="started_NOT_qualified",
        execution_status="started",
        cases=[],
        nan_tail_diagnostic=[],
        versions=actual_versions,
        started_unix=time.time(),
        config_sha256=sha256_file(args.config),
    )
    report["compiler_environment"] = {
        name: os.environ.get(name)
        for name in (
            "DISABLE_MMA_V5",
            "BATCHGEN_W8A16_DEQUANT",
            "TRITON_PTXAS_PATH",
            "CUDA_HOME",
        )
    }
    out = args.output

    def save():
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "report.json")

    started = time.monotonic()
    try:
        weight_args = argparse.Namespace(
            weights_root=args.weights_root,
            weights_manifest=args.weights_manifest,
            weights_shard=None,
            weights_sha256=None,
            layer_index=42,
            seed=args.seed,
        )
        weights, scales, norms = layer_test.load_weights(weight_args, device, report)
        report["weight_tensor_sha256"] = {
            name: tensor_record(tensor)
            for group in (weights, scales, norms)
            for name, tensor in group.items()
        }
        layer = layer_test.build_layer(config, weights, norms, device)
        layer._captured_scales = scales
        report["actual_yarn_softmax_scale"] = layer.softmax_scale
        backend = layer_test.isolated_backend()
        captured, histories = capture_inputs(args, layer, backend, device)
        report["loaded_native_modules"] = {
            name: {"path": str(module.__file__), "sha256": sha256_file(module.__file__)}
            for name, module in tuple(sys.modules.items())
            if ("batchgen" in name or "mgn" in name)
            and getattr(module, "__file__", "")
            and str(module.__file__).endswith(".so")
        }
        report["inputs"] = save_tensors(out / "captured-inputs.pt", captured)
        report["captured_nonfinite_counts"] = {
            name: int((~torch.isfinite(value)).sum())
            for name, value in captured.items()
            if value.is_floating_point()
        }
        if any(report["captured_nonfinite_counts"].values()):
            raise RuntimeError(
                "Upstream capture is nonfinite; do not attribute it to an attention candidate"
            )
        reference = (
            primitive.prefill_reference(
                captured["q"],
                captured["k"],
                captured["v"],
                captured["cu_seqlens"].tolist(),
                layer.softmax_scale,
            )
            if args.stage == "prefill"
            else primitive.decode_reference(
                captured["q"].unsqueeze(1), histories, layer.softmax_scale
            ).squeeze(1)
        )
        report["reference"] = save_tensors(out / "fp32-reference.pt", {"output": reference})
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError(
                "Shared-input FP32 oracle is nonfinite; candidates have not been tested"
            )
        functions = candidates(args, captured, layer.softmax_scale, report)
        fixed = {
            name: tensor_record(captured[name])
            for name in ("q", "k", "v", "cu_seqlens", "table", "raw_table", "lengths")
            if name in captured
        }
        report["fixed_candidate_inputs"] = fixed
        report["cache_artifacts"] = {}
        modes = ["prefill"] if args.stage == "prefill" else ["zero", "finite_poison", "nan_latent"]
        zero_outputs = {}
        precision_outputs = {}
        for mode in modes:
            cache = (
                None
                if mode == "prefill"
                else fill_tail(captured["cache"], captured["tail_mask"], mode)
            )
            cache_before = None if cache is None else tensor_record(cache)
            if cache is not None:
                report["cache_artifacts"][mode] = save_tensors(
                    out / f"cache-{mode}.pt", {"cache": cache}
                )
            for name, function in functions:
                first = None
                for repeat in range(args.repeats):
                    label = f"{name}/{mode}/repeat{repeat}"
                    entry = {
                        "case": label,
                        "candidate": name,
                        "mode": mode,
                        "repeat": repeat,
                        "cache_input": cache_before,
                    }
                    destination = (
                        report["nan_tail_diagnostic"] if mode == "nan_latent" else report["cases"]
                    )
                    try:
                        actual = function(cache)
                        torch.cuda.synchronize(device)
                        if actual.dtype != torch.bfloat16 or actual.shape != reference.shape:
                            raise RuntimeError("Candidate output shape/BF16 contract mismatch")
                        assert_inputs_unchanged(captured, fixed, cache, cache_before)
                        finite = bool(torch.isfinite(actual).all())
                        entry.update(
                            execution_status=(
                                "completed"
                                if finite or mode == "nan_latent"
                                else "nonfinite_on_finite_input"
                            ),
                            historical_strict_gate=historical_gate(actual, reference, label),
                            independent_error=independent_metrics(actual, reference),
                            output=save_tensors(
                                out / f"output-{name}-{mode}-{repeat}.pt",
                                {"output": actual},
                            ),
                        )
                        current_cpu = actual.detach().cpu()
                        if mode == "zero" and repeat == 0:
                            zero_outputs[name] = current_cpu.clone()
                        elif mode != "prefill" and name in zero_outputs:
                            entry["tail_comparison_to_zero"] = repeat_metrics(
                                current_cpu, zero_outputs[name]
                            )
                        if name == "triton_precision" and repeat == 0:
                            precision_outputs[mode] = current_cpu.clone()
                        elif name != "triton_precision" and mode in precision_outputs:
                            entry["same_input_comparison_to_precision_DIAGNOSTIC_ONLY"] = (
                                independent_metrics(current_cpu, precision_outputs[mode])
                            )
                        if first is None:
                            first = current_cpu.clone()
                        entry["repeat_comparison"] = repeat_metrics(current_cpu, first)
                    except Exception as exc:
                        entry.update(
                            execution_status="error",
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(),
                        )
                        if (
                            isinstance(exc, CapturedInputMutation)
                            or "illegal memory access" in str(exc).lower()
                            or "device-side assert" in str(exc).lower()
                        ):
                            destination.append(entry)
                            save()
                            raise
                    destination.append(entry)
                    save()
        report["execution_status"] = (
            "completed"
            if all(c["execution_status"] == "completed" for c in report["cases"])
            else "finite_input_execution_failure"
        )
        report["historical_strict_gate_status"] = (
            "passed"
            if all(
                c.get("historical_strict_gate", {}).get("status") == "passed"
                for c in report["cases"]
            )
            else "failed"
        )
        report["repeat_consistency"] = (
            "byte_identical"
            if all(c.get("repeat_comparison", {}).get("same_raw_bytes") for c in report["cases"])
            else "differences_or_errors"
        )
        report["status"] = "diagnostics_collected_NOT_model_or_performance_qualified"
    except Exception:
        report.update(
            status="failed_NOT_qualified",
            execution_status="error",
            error=traceback.format_exc(),
        )
    finally:
        report["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        save()
    if report["execution_status"] != "completed":
        return 1
    return (
        0
        if report["historical_strict_gate_status"] == "passed"
        and report["repeat_consistency"] == "byte_identical"
        else 2
    )


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=tuple(STAGES), required=True)
    ap.add_argument("--gpu", type=int, choices=range(8), required=True)
    ap.add_argument("--expected-hostname", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights-root", type=Path, required=True)
    ap.add_argument("--weights-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if not 3 <= args.repeats <= 5:
        ap.error("Require 3-5 same-input repetitions")
    if args._worker:
        return worker(args)
    if args.execute:
        from attention_only_guard import supervise

        return supervise(args, Path(__file__).resolve())
    print(json.dumps(plan(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
