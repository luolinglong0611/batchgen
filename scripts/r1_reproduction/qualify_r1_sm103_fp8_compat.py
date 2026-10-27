#!/usr/bin/env python3
"""Bounded, operator-only R1 original-scale SM103 compatibility qualification.

Plan-only by default. Original safetensors are range-read from a mounted local
checkpoint; no model/shard loader, network client, FI, or NVCC is used. Numeric
bounds below are fixed before execution and are never adjusted from results.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time
import traceback
import types

INDEX_SHA = "a2c5d238ab1e8301a117336e9797e0d3551cb8f6dc9dc787318b23d8f54c9248"
PASS = "passed_isolated_sm103_fp8_compat_contract_NOT_model_or_performance_qualified"
FAIL = "failed_sm103_fp8_compat_contract"
CAP_BYTES = 6 * 1024**3
N_CHUNK = 256
SEED = 20261026
U32 = 2.0**-24
UBF16 = 2.0**-8
SCALE_AUDIT = {
    "model.layers.0.self_attn.q_a_proj": ((1536, 7168), "bdd9b6f225c6cf02432e94368636f2d14eb7369194ca814db94dcbf9cad0c80c"),
    "model.layers.0.self_attn.q_b_proj": ((24576, 1536), "9c48c91a70a34b9197b241a162364f425b78816d357f13325c82f6b7ed6aaa19"),
    "model.layers.0.self_attn.kv_a_proj_with_mqa": ((576, 7168), "eb8702ccebabc841d7c6fe3b9f0605fff2c1548da0c9e72d1dab2784a8e84036"),
    "model.layers.0.self_attn.kv_b_proj": ((32768, 512), "749bf1bae1e971f73fe16cc6e9dc0ee86200a035a33e26acc7691b0c1e3ee67b"),
    "model.layers.0.self_attn.o_proj": ((7168, 16384), "2834d5cd5d25aa051a60d4bb034bf222f12a5bbc9b4b7a78de55fc7ed4323b00"),
    "model.layers.3.mlp.experts.0.gate_proj": ((2048, 7168), "3febe52c9cd45fa7429ee3a0579d8d1fee34eed703b9c5ea2cce3f436bdb2d6d"),
    "model.layers.3.mlp.experts.0.up_proj": ((2048, 7168), "4925a99a8f0337d5fda18cfd0a89365a51d7ebdf4d33da0ec6517760427bf5d6"),
    "model.layers.3.mlp.experts.0.down_proj": ((7168, 2048), "00711d12cd7c39b282c37bcc61f282aa8ca72e99028a74c64c499b852d43a72c"),
}


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def gamma32(operations):
    if type(operations) is not int or operations < 0 or operations * U32 >= 1:
        raise ValueError("invalid FP32 rounding-bound operation count")
    return operations * U32 / (1.0 - operations * U32)


def rounding_contract(k):
    blocks = (k + 127) // 128
    return {"contract": "fp32_dot128_scale_products_block_sum_plus_bf16_RNE_v1",
            "FP32_unit_roundoff": U32, "BF16_unit_roundoff": UBF16,
            "dot_terms_per_block_bound": 128, "K_blocks": blocks,
            "FP32_gamma_operations": 128 + 2 + blocks,
            "FP32_gamma": gamma32(128 + 2 + blocks),
            "element_error_bound": "gamma*(1+u_bf16)*sum_abs_dequant_products + u_bf16*abs(FP64_reference) + 2^-134 + (128+2+Kb)*2^-149",
            "allowed_violations": 0,
            "assumption": "Tests the stated FP32-dot128 accumulation contract; a lower-precision hardware path failing it is not automatically accepted",
            "excluded_error": "BF16-to-FP8 quantization is checked separately, then GEMM is compared against the actual validated FP8 operands"}


def source_receipts(args):
    sources = [("w8a8", args.source, args.source_sha256),
               ("compat", args.compat_source, args.compat_source_sha256),
               ("quant", args.quant_source, args.quant_source_sha256)]
    records = {}
    for name, path, expected in sources:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Explicit lowercase SHA256 required")
        actual = sha_file(path)
        if actual != expected:
            raise ValueError(f"{name} source SHA256 mismatch")
        records[name] = {"path": str(path.resolve()), "sha256": actual}
    return records


def read_index(root):
    root = Path(root).resolve()
    path = root / "model.safetensors.index.json"
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("checkpoint index exceeds bounded metadata size")
    raw = path.read_bytes()
    if sha_bytes(raw) != INDEX_SHA:
        raise ValueError("Frozen original checkpoint index mismatch")
    result = json.loads(raw)
    if not isinstance(result.get("weight_map"), dict):
        raise ValueError("invalid safetensors index")
    return result["weight_map"]


def read_tensor_range(root, index, name, expected_dtype, expected_shape, expected_sha=None):
    """Read exactly one header and one bounded tensor range, never a full shard."""
    root = Path(root).resolve()
    filename = index[name]
    if not isinstance(filename, str) or not re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", filename):
        raise ValueError("unsafe/unexpected checkpoint shard filename")
    path = root / filename
    if path.resolve().parent != root:
        raise ValueError("checkpoint shard escapes mounted root")
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise ValueError("short safetensors header prefix")
        header_size = struct.unpack("<Q", raw_size)[0]
        if not 0 < header_size <= 64 * 1024**2:
            raise ValueError("unbounded safetensors header")
        header_raw = f.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError("short safetensors header")
        metadata = json.loads(header_raw)[name]
        dtype = metadata["dtype"]
        if dtype not in expected_dtype or tuple(metadata["shape"]) != tuple(expected_shape):
            raise ValueError(f"unexpected original tensor dtype/shape: {name}")
        lo, hi = metadata["data_offsets"]
        element_size = 4 if dtype == "F32" else 1
        if (type(lo) is not int or type(hi) is not int or not 0 <= lo < hi
                or hi - lo != math.prod(expected_shape) * element_size
                or hi - lo > 256 * 1024**2 or 8 + header_size + hi > path.stat().st_size):
            raise ValueError("invalid/unbounded safetensors tensor range")
        f.seek(8 + header_size + lo)
        raw = f.read(hi - lo)
        if len(raw) != hi - lo:
            raise ValueError("short safetensors tensor range")
    digest = sha_bytes(raw)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Frozen original FP32 scale range mismatch: {name}")
    return raw, {"name": name, "shard": filename, "dtype": dtype,
                 "shape": list(expected_shape), "tensor_bytes": len(raw),
                 "tensor_sha256": digest, "header_sha256": sha_bytes(header_raw),
                 "tensor_data_offsets": [lo, hi], "absolute_file_offset": 8 + header_size + lo,
                 "full_shard_hash_verified": False,
                 "identity_limit": "weight range hash recorded now; selected scale hashes and index were frozen before execution"}


def case_passed(r):
    count = r.get("M", 0) * r.get("N", 0)
    return (count > 0 and r.get("compared_elements") == count
            and r.get("error_bound_violations") == 0
            and r.get("nonfinite_output_elements") == 0
            and r.get("quantization_passed") is True
            and r.get("inputs_unchanged") is True
            and r.get("composite_entry_bitwise_equal") is True
            and r.get("output_contract_passed") is True
            and r.get("reference_finite") is True
            and r.get("weight_and_scale_range_unchanged") is True)


def report_passed(report, common):
    expected = [(name, m, n, k) for name, ((n, k), _) in SCALE_AUDIT.items() for m in (1, 3)]
    expected.append(("synthetic_two_K_blocks_medium_partial_N", 129, 576, 256))
    cases = report.get("cases", [])
    observed = [(c.get("name"), c.get("M"), c.get("N"), c.get("K")) for c in cases]
    backends = report.get("backend_receipt", {}).get("actual_backends", [])
    return (observed == expected and all(case_passed(c) for c in cases)
            and report.get("source_unchanged") is True
            and report.get("checkpoint_index_unchanged") is True
            and common.COMPAT_BACKEND in backends and common.QUANT_BACKEND in backends
            and common.DEFAULT_BACKEND not in backends)


def plan(args):
    return {"schema": "r1-sm103-original-fp32-scale-compat-v1", "status": "plan_only_NOT_executed",
            "scope": "operator", "sources": source_receipts(args),
            "worker_sha256": sha_file(Path(__file__)), "output": str(args.output),
            "expected_hostname": args.expected_hostname, "checkpoint_root": str(args.checkpoint_root),
            "checkpoint_index_sha256": INDEX_SHA,
            "environment_required": {"R1_MAIN_FP8_COMPAT_GUARDED": "1", "BATCHGEN_R1_SM103_FP8_COMPAT": "1",
                                     "DISABLE_MMA_V5": "1", "CUDA_VISIBLE_DEVICES": "one assigned device"},
            "runtime_required": {"torch": "2.9.0+cu130", "triton": "3.5.0", "CUDA": "13.0", "SM": "10.3"},
            "external_limits_required": {"seconds": 900, "RSS_bytes": 8 * 1024**3, "GPU_bytes": 8 * 1024**3},
            "torch_allocator_cap_bytes": CAP_BYTES, "reference_N_chunk": N_CHUNK,
            "fixed_random_activation_seed": SEED,
            "quant_contract": {"scale": "max(max(block_amax,2^-16)*(1/448),1e-12)",
                               "scale_rel_error_bound": 2 * U32,
                               "FP8_rounding": "nearest finite E4M3FN, ties may differ only within 4*u32*abs(x/scale)",
                               "nonfinite_input": "rejected before quantizer; inherited main sanitizer is not qualified"},
            "real_cases": [{"name": name, "N": nk[0], "K": nk[1], "M": [1, 3],
                            "scale_sha256": digest, "numeric_contract": rounding_contract(nk[1])}
                           for name, (nk, digest) in SCALE_AUDIT.items()],
            "synthetic_cases": [{"M": 129, "N": 576, "K": 256,
                                 "purpose": "medium launch, partial N block and multiple non-power-of-two FP32 scale K blocks"}],
            "limitations": ["not full model/attention/grouped MoE/quality/performance qualification",
                            "selected local range reads do not verify entire checkpoint shards",
                            "same GPU arithmetic contract still needs actual SG hook runtime evidence"]}


def tensor_sha(tensor):
    import torch
    return sha_bytes(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("wrong explicit operator module loaded")
    return module


def quant_contract(torch, x_cpu, a_cpu, scale_cpu):
    m, k = x_cpu.shape
    kb = (k + 127) // 128
    if (tuple(a_cpu.shape) != (m, k) or a_cpu.dtype != torch.float8_e4m3fn
            or tuple(scale_cpu.shape) != (m, kb) or scale_cpu.dtype != torch.float32):
        return {"passed": False, "reason": "quantizer output dtype/shape"}
    x = x_cpu.to(torch.float64)
    q = a_cpu.to(torch.float64)
    s = scale_cpu.to(torch.float64)
    if not bool(torch.isfinite(x).all() and torch.isfinite(q).all() and torch.isfinite(s).all() and (s > 0).all()):
        return {"passed": False, "reason": "nonfinite activation/quantized value or nonpositive scale"}
    # Independent CPU FP64 formula, with no import/call of the tested quantizer.
    padded = torch.zeros((m, kb * 128), dtype=torch.float64)
    padded[:, :k] = x
    expected_s = (padded.view(m, kb, 128).abs().amax(dim=-1).clamp_min(2.0**-16) / 448.0).clamp_min(1e-12)
    scale_errors = (s - expected_s).abs()
    scale_violations = int((scale_errors > 2 * U32 * expected_s).sum())
    divisor = s.repeat_interleave(128, dim=1)[:, :k]
    ideal = (x / divisor).clamp(-448, 448)
    # Enumerating the finite FP8 value set avoids reusing a GPU conversion or
    # assuming CPU/GPU ties have identical intermediate division rounding.
    values = torch.arange(256, dtype=torch.int16).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)
    values = values[torch.isfinite(values)].unique().sort().values
    indices = torch.searchsorted(values, ideal.contiguous()).clamp(1, values.numel() - 1)
    lo, hi = values[indices - 1], values[indices]
    nearest_distance = torch.minimum((ideal - lo).abs(), (ideal - hi).abs())
    actual_distance = (q - ideal).abs()
    division_slack = 4 * U32 * ideal.abs() + 2.0**-149
    rounding_violations = int((actual_distance > nearest_distance + division_slack).sum())
    return {"passed": scale_violations == 0 and rounding_violations == 0,
            "compared_quantized_elements": m * k, "compared_scales": m * kb,
            "scale_bound_violations": scale_violations, "FP8_rounding_bound_violations": rounding_violations,
            "max_scale_relative_error": float((scale_errors / expected_s).max()),
            "max_FP8_rounding_excess": float((actual_distance - nearest_distance).clamp_min(0).max()),
            "quantized_sha256": tensor_sha(a_cpu), "scale_sha256": tensor_sha(scale_cpu)}


def reference_compare(torch, a, a_scale, w, w_scale, actual):
    """Independent CPU FP64 per-block dots, bounded memory and full output."""
    m, k = a.shape
    n = w.shape[0]
    kb = (k + 127) // 128
    a = a.to(torch.float64)
    a_scale = a_scale.to(torch.float64)
    w_scale = w_scale.to(torch.float64)
    contract = rounding_contract(k)
    gamma = contract["FP32_gamma"]
    compared = violations = nonfinite = 0
    maxerr = max_bound_ratio = sum_err2 = sum_ref2 = 0.0
    reference_finite = True
    reference_hash = hashlib.sha256()
    bound_hash = hashlib.sha256()
    for start in range(0, n, N_CHUNK):
        end = min(start + N_CHUNK, n)
        wc = w[start:end].to(torch.float64)
        columns = torch.arange(start, end) // 128
        reference = torch.zeros((m, end - start), dtype=torch.float64)
        absolute_sum = torch.zeros_like(reference)
        for block in range(kb):
            lo, hi = block * 128, min((block + 1) * 128, k)
            aq, wq = a[:, lo:hi], wc[:, lo:hi]
            scales = a_scale[:, block: block + 1] * w_scale[columns, block].unsqueeze(0)
            reference += (aq @ wq.T) * scales
            absolute_sum += (aq.abs() @ wq.abs().T) * scales.abs()
        bound = (gamma * (1 + UBF16) * absolute_sum + UBF16 * reference.abs()
                 + 2.0**-134 + (128 + 2 + kb) * 2.0**-149)
        observed = actual[:, start:end].to(torch.float64)
        error = (observed - reference).abs()
        reference_finite = reference_finite and bool(torch.isfinite(reference).all() and torch.isfinite(bound).all())
        nonfinite += int((~torch.isfinite(observed)).sum())
        violations += int((~torch.isfinite(error) | (error > bound)).sum())
        compared += m * (end - start)
        if bool(torch.isfinite(error).all()):
            maxerr = max(maxerr, float(error.max()))
            max_bound_ratio = max(max_bound_ratio, float((error / bound).max()))
            sum_err2 += float((error * error).sum())
            sum_ref2 += float((reference * reference).sum())
        reference_hash.update(reference.contiguous().numpy().tobytes())
        bound_hash.update(bound.contiguous().numpy().tobytes())
    return {"compared_elements": compared, "error_bound_violations": violations,
            "nonfinite_output_elements": nonfinite, "reference_finite": reference_finite,
            "max_absolute_error": maxerr, "max_error_to_bound_ratio": max_bound_ratio,
            "relative_L2": math.sqrt(sum_err2 / max(sum_ref2, sys.float_info.min)),
            "reference_FP64_chunk_stream_sha256": reference_hash.hexdigest(),
            "bound_FP64_chunk_stream_sha256": bound_hash.hexdigest(), "numeric_contract": contract}


def run_real_case(torch, common, quant, w_cpu, scale_cpu, name, m, case_seed):
    n, k = w_cpu.shape
    generator = torch.Generator(device="cpu").manual_seed(case_seed)
    x_cpu = torch.randn((m, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    if not bool(torch.isfinite(x_cpu.float()).all()):
        raise ValueError("Nonfinite qualification input rejected before main quantizer")
    x, w, scale = x_cpu.cuda(), w_cpu.cuda(), scale_cpu.cuda()
    before = [tensor_sha(t) for t in (x, w, scale)]
    a, a_scale = quant.per_token_blocked_quantize_bf16_to_fp8_1d(x)
    torch.cuda.synchronize()
    a_cpu, a_scale_cpu = a.cpu(), a_scale.cpu()
    q = quant_contract(torch, x_cpu, a_cpu, a_scale_cpu)
    if q["passed"] is not True:
        raise RuntimeError("Independent quantization contract failed: " + json.dumps(q))
    quant_before = [tensor_sha(t) for t in (a, a_scale)]
    out = common.w8a8_gemm(a, a_scale, w, scale)
    torch.cuda.synchronize()
    first = out.cpu()
    del out
    out2 = common.w8a16_gemm(w, scale, x)
    torch.cuda.synchronize()
    second = out2.cpu()
    record = {"name": name, "M": m, "N": n, "K": k, "activation_seed": case_seed,
              "activation_sha256": tensor_sha(x_cpu), "quantization": q, "quantization_passed": True,
              "output_contract_passed": tuple(out2.shape) == (m, n) and out2.dtype == torch.bfloat16 and out2.device == x.device,
              "composite_entry_bitwise_equal": tensor_sha(first) == tensor_sha(second),
              "output_sha256": tensor_sha(first),
              "inputs_unchanged": before == [tensor_sha(t) for t in (x, w, scale)]
                                    and quant_before == [tensor_sha(t) for t in (a, a_scale)]}
    record.update(reference_compare(torch, a_cpu, a_scale_cpu, w_cpu, scale_cpu, first))
    record["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    del x, w, scale, a, a_scale, out2
    torch.cuda.empty_cache()
    return record


def run_synthetic(torch, common):
    m, n, k = 129, 576, 256
    row = torch.arange(m, dtype=torch.int64)
    col = torch.arange(n, dtype=torch.int64)
    block = torch.arange(k, dtype=torch.int64) // 128
    a_cpu = (((row[:, None] + block[None, :]) % 7 + 1) * (1 - 2 * (block[None, :] % 2))).to(torch.float8_e4m3fn)
    w_cpu = (((col[:, None] + block[None, :]) % 3 + 1) * (1 - 2 * ((col[:, None] + block[None, :]) % 2))).to(torch.float8_e4m3fn)
    a_scale_cpu = ((1 + (row[:, None] + 3 * torch.arange(2)[None, :]) % 5) / 8).float()
    w_scale_cpu = ((1 + (torch.arange((n + 127) // 128)[:, None] + 2 * torch.arange(2)[None, :]) % 5) / 16).float()
    a, a_scale, w, w_scale = (t.cuda() for t in (a_cpu, a_scale_cpu, w_cpu, w_scale_cpu))
    before = [tensor_sha(t) for t in (a, a_scale, w, w_scale)]
    first = common.w8a8_gemm(a, a_scale, w, w_scale).cpu()
    second = common.w8a8_gemm(a, a_scale, w, w_scale).cpu()
    torch.cuda.synchronize()
    # Analytic independent sum over K blocks, using only row/column integers
    # and exact binary fractions, not the GPU inputs or GEMM implementation.
    expected = torch.zeros((m, n), dtype=torch.float64)
    for b in range(2):
        av = ((row + b) % 7 + 1).double() * (1 - 2 * (b % 2))
        wv = ((col + b) % 3 + 1).double() * (1 - 2 * ((col + b) % 2))
        sa = (1 + (row + 3 * b) % 5).double() / 8
        sw = (1 + (col // 128 + 2 * b) % 5).double() / 16
        expected += 128 * (av * sa)[:, None] * (wv * sw)[None, :]
    exact = tensor_sha(first) == tensor_sha(expected.to(torch.bfloat16))
    record = {"name": "synthetic_two_K_blocks_medium_partial_N", "M": m, "N": n, "K": k,
              "compared_elements": m * n, "error_bound_violations": 0 if exact else int((first != expected.to(torch.bfloat16)).sum()),
              "nonfinite_output_elements": int((~torch.isfinite(first.float())).sum()), "reference_finite": True,
              "quantization_passed": True, "quantization": {"scope": "not used; exact FP8 fixture"},
              "composite_entry_bitwise_equal": tensor_sha(first) == tensor_sha(second),
              "output_contract_passed": tuple(first.shape) == (m, n) and first.dtype == torch.bfloat16,
              "inputs_unchanged": before == [tensor_sha(t) for t in (a, a_scale, w, w_scale)],
              "weight_and_scale_range_unchanged": True, "synthetic_exact_BF16_match": exact,
              "input_sha256": before, "output_sha256": tensor_sha(first),
              "analytic_BF16_sha256": tensor_sha(expected.to(torch.bfloat16)),
              "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    del a, a_scale, w, w_scale
    torch.cuda.empty_cache()
    return record


def execute(args):
    if os.environ.get("R1_MAIN_FP8_COMPAT_GUARDED") != "1":
        raise RuntimeError("Execution requires the parent-owned FP8 compatibility resource guard")
    if os.environ.get("BATCHGEN_R1_SM103_FP8_COMPAT") != "1" or os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("Explicit SM103 compatibility/MMA environment required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one assigned CUDA device must be visible")
    if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
        raise RuntimeError("Expected allocated container identity")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton")}
    if versions != {"torch": "2.9.0+cu130", "triton": "3.5.0"}:
        raise RuntimeError("Frozen Torch2.9cu130/Triton3.5 required")
    report = plan(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Fresh exclusive output directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    report.update(status=FAIL, actual_versions=versions, cases=[], checkpoint_ranges=[], started_unix=time.time())
    started = time.monotonic()
    code = 1

    def save():
        temp = args.output / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output / "report.json")

    save()
    try:
        index = read_index(args.checkpoint_root)
        import torch
        torch.set_num_threads(1)
        if torch.version.cuda != "13.0":
            raise RuntimeError("Frozen CUDA13.0 required")
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError("Exactly one assigned SM103 required")
        torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        # Explicit isolated operator namespaces avoid loading BatchGen's model
        # package/ckpt converter and its unrelated optional native extensions.
        for name in ("batchgen", "batchgen.gemm", "batchgen_kernels", "batchgen_kernels.triton"):
            if name in sys.modules:
                raise RuntimeError("operator namespace unexpectedly loaded before explicit source binding")
            namespace = types.ModuleType(name)
            namespace.__path__ = []
            sys.modules[name] = namespace
        native = load_file("batchgen.gemm.w8a8", args.source)
        quant = load_file("batchgen_kernels.triton.fp8_quantize", args.quant_source)
        common = load_file("batchgen.gemm.r1_sm103_compat", args.compat_source)
        source_receipts(args)
        report["device"] = {"name": torch.cuda.get_device_name(0), "capability": [10, 3]}
        for number, (name, ((n, k), scale_sha)) in enumerate(SCALE_AUDIT.items()):
            torch.cuda.reset_peak_memory_stats()
            raw_w, rw = read_tensor_range(args.checkpoint_root, index, name + ".weight", {"F8_E4M3", "F8_E4M3FN"}, (n, k))
            raw_s, rs = read_tensor_range(args.checkpoint_root, index, name + ".weight_scale_inv", {"F32"}, ((n + 127) // 128, (k + 127) // 128), scale_sha)
            w = torch.frombuffer(bytearray(raw_w), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(n, k).clone()
            scale = torch.frombuffer(bytearray(raw_s), dtype=torch.float32).reshape((n + 127) // 128, (k + 127) // 128).clone()
            if not bool(torch.isfinite(w.float()).all() and torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("nonfinite original weight/scale or nonpositive scale")
            for m in (1, 3):
                record = run_real_case(torch, common, quant, w, scale, name, m, SEED + 100 * number + m)
                report["cases"].append(record)
                save()
            _, rw_after = read_tensor_range(args.checkpoint_root, index, name + ".weight", {"F8_E4M3", "F8_E4M3FN"}, (n, k), rw["tensor_sha256"])
            _, rs_after = read_tensor_range(args.checkpoint_root, index, name + ".weight_scale_inv", {"F32"}, ((n + 127) // 128, (k + 127) // 128), scale_sha)
            unchanged = rw == rw_after and rs == rs_after
            for record in report["cases"][-2:]:
                record["weight_and_scale_range_unchanged"] = unchanged
                record["passed"] = case_passed(record)
            report["checkpoint_ranges"].extend([rw, rs])
            del raw_w, raw_s, w, scale
            gc.collect()
            save()
        torch.cuda.reset_peak_memory_stats()
        synthetic = run_synthetic(torch, common)
        synthetic["passed"] = case_passed(synthetic)
        report["cases"].append(synthetic)
        report["source_unchanged"] = report["sources"] == source_receipts(args)
        report["checkpoint_index_unchanged"] = read_index(args.checkpoint_root) == index
        report["backend_receipt"] = common.backend_receipt()
        report["actual_main_entrypoint"] = str(Path(native.__file__).resolve())
        report["isolated_namespaces_used"] = True
        if report_passed(report, common):
            report["status"] = PASS
            code = 0
    except Exception:
        report["exception"] = traceback.format_exc()
    finally:
        report["finished_unix"] = time.time()
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    print(json.dumps({"status": report["status"], "scope": "operator", "report": str(args.output / "report.json")}))
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--compat-source", type=Path, required=True)
    p.add_argument("--compat-source-sha256", required=True)
    p.add_argument("--quant-source", type=Path, required=True)
    p.add_argument("--quant-source-sha256", required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute:
        return execute(args)
    print(json.dumps(plan(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
