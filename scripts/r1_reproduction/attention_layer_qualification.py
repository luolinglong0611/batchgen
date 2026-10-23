#!/usr/bin/env python3
"""Standalone full R1 attention-layer numerical qualification, never a benchmark.

Example (inside the allocated container, with installed CUDA kernel packages):
  DISABLE_MMA_V5=1 python attention_layer_qualification.py --config /models/DeepSeek-R1/config.json \
    --profile small --report /tmp/r1-attention-layer.json
  # To qualify the owned precision candidate, add:
  # --attention-implementation triton_precision
  # To replace deterministic random FP8 weights with checkpoint layer 0:
  # add --weights-shard FILE --weights-sha256 VERIFIED_SHA256

Uses actual SM103 backend, FP8 blockscale Triton projections, fused norms/RoPE,
and explicitly selected FA4/FlashInfer or owned precision attention primitives.
The oracle independently computes in PyTorch FP32 and
explicitly rounds at the same FP8/BF16 boundaries; it never consumes DUT outputs.
The paged-manager fixture exercises physical KV layout and appends, not the
production allocator, host offload, UCX, scheduling, or complete-model quality.
Local package namespaces avoid importing the engine or constructing a 671B model.
Numerical failures remain failures: the existing attention gate is never relaxed.
Synchronous assertions, reference computation, and first-use compilation make
elapsed time unsuitable for any throughput comparison.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib
import importlib.machinery
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import traceback
from types import ModuleType, SimpleNamespace

import torch

from attention_qualification import ATOL, RTOL, MAX_FAILURE_FRACTION, check, emit, versions


ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = ROOT / "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py"
BACKEND_SOURCE = ROOT / "batchgen/attention/mla/sm103_backend.py"
PRECISION_SOURCE = ROOT / "batchgen/attention/mla/triton_precision.py"
SHAPES = {
    "q_a_proj": (1536, 7168), "q_b_proj": (24576, 1536),
    "kv_a_proj_with_mqa": (576, 7168), "kv_b_proj": (32768, 512),
    "o_proj": (7168, 16384),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolated_backend():
    # These are real source modules, not kernel mocks. Only package __init__
    # side effects (engine imports/version preflight) are bypassed in this process.
    for name in ("batchgen", "batchgen.attention", "batchgen.attention.mla", "batchgen.gemm"):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = package
    return importlib.import_module("batchgen.attention.mla.sm103_backend")


def validate_config(config):
    expected = {
        "hidden_size": 7168, "q_lora_rank": 1536, "kv_lora_rank": 512,
        "num_attention_heads": 128, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Expected original R1 {name}={value}; got {config.get(name)}")
    if config.get("attention_bias", False):
        raise ValueError("The qualification backend does not implement attention bias")
    quant = config["quantization_config"]
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("Expected original block128 FP8 checkpoint quantization")
    rope = config["rope_scaling"]
    if rope.get("type", rope.get("rope_type")) != "yarn":
        raise ValueError("This test requires the original R1 YaRN configuration")


def original_layer_helpers():
    """Execute only the repo's norm/RoPE definitions, avoiding legacy FA3 imports."""
    from batchgen_kernels.common.mgn import fused_rmsnorm

    names = {
        "DeepseekV3RMSNorm", "DeepseekV3RotaryEmbedding", "DeepseekV3YarnRotaryEmbedding",
        "yarn_find_correction_dim", "yarn_find_correction_range", "yarn_get_mscale",
        "yarn_linear_ramp_mask",
    }
    parsed = ast.parse(MODEL_SOURCE.read_text())
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("The repository's R1 norm/RoPE definitions changed")
    namespace = {"torch": torch, "nn": torch.nn, "math": math,
                 "logger": logging.getLogger("r1_layer_test"), "fused_rmsnorm": fused_rmsnorm}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL_SOURCE), "exec"), namespace)
    return namespace


def random_fp8_weight(shape, device):
    rows, cols = shape
    # Independent 128x128 checkpoint-style quantization, including N=576 tail.
    padded_rows = ((rows + 127) // 128) * 128
    raw = torch.randn((padded_rows, cols), device=device, dtype=torch.float32) / math.sqrt(cols)
    blocks = raw.view(padded_rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    quantized = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape(padded_rows, cols)[:rows].contiguous(), scale.contiguous()


def load_weights(args, device, result):
    weights, scales, norms = {}, {}, {}
    if args.weights_shard:
        from safetensors import safe_open

        if not args.weights_sha256 or len(args.weights_sha256) != 64:
            raise ValueError("Checkpoint mode requires --weights-sha256 from the verified manifest")
        actual_digest = sha256_file(args.weights_shard)
        if actual_digest != args.weights_sha256.lower():
            raise ValueError("Checkpoint shard SHA256 does not match the provided verified digest")
        prefix = f"model.layers.{args.layer_index}.self_attn."
        with safe_open(str(args.weights_shard), framework="pt", device="cpu") as checkpoint:
            for name in SHAPES:
                weights[name] = checkpoint.get_tensor(prefix + name + ".weight").to(device)
                scales[name + ".weight_scale_inv"] = checkpoint.get_tensor(prefix + name + ".weight_scale_inv").to(device)
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                norms[name] = checkpoint.get_tensor(prefix + name + ".weight").to(device)
        result["weight_source"] = {"kind": "checkpoint", "path": str(args.weights_shard),
                                   "sha256": actual_digest, "layer_index": args.layer_index}
    else:
        for name, shape in SHAPES.items():
            weights[name], scales[name + ".weight_scale_inv"] = random_fp8_weight(shape, device)
        norms = {name: (1 + 0.05 * torch.randn(size, device=device)).bfloat16()
                 for name, size in (("q_a_layernorm", 1536), ("kv_a_layernorm", 512))}
        result["weight_source"] = {"kind": "deterministic_random_fp8", "seed": args.seed,
                                   "warning": "Kernel/geometry qualification, not pretrained-model quality"}
    for name, shape in SHAPES.items():
        weight, scale = weights[name], scales[name + ".weight_scale_inv"]
        if weight.shape != shape or weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Wrong checkpoint tensor shape/dtype for {name}")
        if scale.shape != ((shape[0] + 127) // 128, (shape[1] + 127) // 128) or scale.dtype != torch.float32:
            raise ValueError(f"Wrong blockscale shape/dtype for {name}")
        if not bool(torch.isfinite(weight.float()).all()) or not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError(f"Invalid weights or scales for {name}")
    for name, size in (("q_a_layernorm", 1536), ("kv_a_layernorm", 512)):
        if norms[name].shape != (size,) or norms[name].dtype != torch.bfloat16:
            raise ValueError(f"Wrong checkpoint norm tensor for {name}")
    return weights, scales, norms


def build_layer(config, weights, norms, device):
    helpers = original_layer_helpers()
    layer = SimpleNamespace(num_heads=128, q_head_dim=192, qk_nope_head_dim=128,
                            qk_rope_head_dim=64, kv_lora_rank=512, v_head_dim=128,
                            q_lora_rank=1536, single_sequence_prefill_chunk_size=0)
    for name, weight in weights.items():
        setattr(layer, name, SimpleNamespace(weight=weight))
    for name, size in (("q_a_layernorm", 1536), ("kv_a_layernorm", 512)):
        norm = helpers["DeepseekV3RMSNorm"](size, eps=config["rms_norm_eps"]).to(device=device, dtype=torch.bfloat16)
        norm.weight = torch.nn.Parameter(norms[name], requires_grad=False)
        setattr(layer, name, norm)
    rope = config["rope_scaling"]
    layer.rotary_emb = helpers["DeepseekV3YarnRotaryEmbedding"](
        64, max_position_embeddings=config["max_position_embeddings"], base=config["rope_theta"], device=device,
        scaling_factor=rope["factor"], original_max_position_embeddings=rope["original_max_position_embeddings"],
        beta_fast=rope.get("beta_fast", 32), beta_slow=rope.get("beta_slow", 1),
        mscale=rope.get("mscale", 1), mscale_all_dim=rope.get("mscale_all_dim", 0),
    )
    mscale = helpers["yarn_get_mscale"](rope["factor"], rope.get("mscale_all_dim", 0))
    layer.softmax_scale = 192**-0.5 * mscale**2
    return layer


def reference_yarn(config, length, device):
    """Independent formula; do not reuse the layer's cached cos/sin or helpers."""
    rope = config["rope_scaling"]
    theta, factor = config["rope_theta"], rope["factor"]
    def correction(rotations):
        return 64 * math.log(rope["original_max_position_embeddings"] / (rotations * 2 * math.pi)) / (2 * math.log(theta))
    low = max(math.floor(correction(rope.get("beta_fast", 32))), 0)
    high = min(math.ceil(correction(rope.get("beta_slow", 1))), 63)
    if low == high:
        high += 0.001
    frequencies = theta ** (-torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
    ramp = ((torch.arange(32, device=device, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    frequencies = frequencies * (1 - ramp) + frequencies / factor * ramp
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * frequencies
    def mscale(value):
        return 1.0 if factor <= 1 else 1.0 + 0.1 * value * math.log(factor)
    amplitude = mscale(rope.get("mscale", 1)) / mscale(rope.get("mscale_all_dim", 0))
    # The production R1 rotary module supplies BF16 cos/sin to BF16 inputs.
    return (angles.cos() * amplitude).bfloat16(), (angles.sin() * amplitude).bfloat16()


def reference_quantize(x):
    tokens, width = x.shape
    if width % 128:
        raise ValueError("R1 projection input width must be divisible by 128")
    blocks = x.float().reshape(tokens, width // 128, 128)
    scales = (blocks.abs().amax(-1).clamp_min(2**-16) * (1.0 / 448.0)).clamp_min(1e-12)
    quantized = (blocks / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape(tokens, width), scales


def reference_gemm(x, weight, weight_scale):
    """FP32 dot per K128 followed by original FP32 scales; no UE8M0 casting."""
    quantized, activation_scale = reference_quantize(x)
    result = torch.zeros((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32)
    weight_rows = torch.arange(weight.shape[0], device=x.device) // 128
    # Same mathematical blocking as the portable Triton kernel, computed using
    # IEEE FP32 cuBLAS (TF32 disabled), with no accelerated quantization imports.
    for block, offset in enumerate(range(0, x.shape[1], 128)):
        partial = quantized[:, offset:offset + 128].float() @ weight[:, offset:offset + 128].float().T
        scale = activation_scale[:, block, None] * weight_scale[weight_rows, block][None, :]
        result = result + partial * scale
    return result


def reference_norm(x, weight, eps):
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.float()).bfloat16()


def reference_rope(x, positions, cos, sin):
    pairs = x.float().reshape(*x.shape[:-1], 32, 2)
    c, s = cos[positions].float().unsqueeze(1), sin[positions].float().unsqueeze(1)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2).bfloat16()


class ReferenceLayer:
    def __init__(self, config, weights, scales, norms, cos, sin):
        self.config, self.weights, self.scales, self.norms = config, weights, scales, norms
        self.cos, self.sin = cos, sin
        rope = config["rope_scaling"]
        amplitude = 1 + 0.1 * rope.get("mscale_all_dim", 0) * math.log(rope["factor"]) if rope["factor"] > 1 else 1
        self.softmax_scale = 192**-0.5 * amplitude**2

    def projection(self, name, x):
        return reference_gemm(x, self.weights[name], self.scales[name + ".weight_scale_inv"])

    def qkv(self, hidden, positions):
        q = self.projection("q_a_proj", hidden).bfloat16()
        q = reference_norm(q, self.norms["q_a_layernorm"], self.config["rms_norm_eps"])
        q = self.projection("q_b_proj", q).bfloat16().view(-1, 128, 192)
        q_nope, q_pe = q.split((128, 64), -1)
        compressed = self.projection("kv_a_proj_with_mqa", hidden).bfloat16()
        latent = reference_norm(compressed[:, :512], self.norms["kv_a_layernorm"], self.config["rms_norm_eps"])
        q_pe = reference_rope(q_pe, positions, self.cos, self.sin)
        k_pe = reference_rope(compressed[:, 512:].unsqueeze(1), positions, self.cos, self.sin)
        return q_nope, q_pe, latent, k_pe

    def prefill(self, hidden, positions, boundaries):
        q_nope, q_pe, latent, k_pe = self.qkv(hidden, positions)
        kv = torch.cat((latent, k_pe.squeeze(1)), -1)
        expanded = self.projection("kv_b_proj", latent).bfloat16().view(-1, 128, 256)
        k_nope, values = expanded.split((128, 128), -1)
        queries = torch.cat((q_nope, q_pe), -1)
        keys = torch.cat((k_nope, k_pe.expand(-1, 128, -1)), -1)
        attended = torch.empty_like(values)
        for start, end in zip(boundaries, boundaries[1:]):
            q, k, v = [tensor[start:end].float().transpose(0, 1) for tensor in (queries, keys, values)]
            scores = (q @ k.transpose(-1, -2)) * self.softmax_scale
            causal = torch.ones((end-start, end-start), dtype=torch.bool, device=hidden.device).triu(1)
            probabilities = scores.masked_fill(causal, -torch.inf).softmax(-1)
            attended[start:end] = (probabilities @ v).transpose(0, 1).bfloat16()
        return self.projection("o_proj", attended.reshape(hidden.shape[0], -1)), kv

    def absorb_weight(self):
        weight = self.weights["kv_b_proj"].float()
        scale = self.scales["kv_b_proj.weight_scale_inv"]
        row_blocks = torch.arange(weight.shape[0], device=weight.device) // 128
        col_blocks = torch.arange(weight.shape[1], device=weight.device) // 128
        return (weight * scale[row_blocks[:, None], col_blocks[None, :]]).bfloat16().view(128, 256, 512)

    def decode(self, hidden, positions, histories):
        q_nope, q_pe, latent, k_pe = self.qkv(hidden, positions)
        kv = torch.cat((latent, k_pe.squeeze(1)), -1)
        absorb = self.absorb_weight()
        latent_queries = torch.einsum("bhd,hdc->bhc", q_nope.float(), absorb[:, :128].float()).bfloat16()
        query = torch.cat((latent_queries, q_pe), -1)
        attended = []
        for index, history in enumerate(histories):
            histories[index] = torch.cat((history, kv[index:index+1]), 0)
            keys = histories[index].float()
            probabilities = ((query[index].float() @ keys.T) * self.softmax_scale).softmax(-1)
            attended.append((probabilities @ keys[:, :512]).bfloat16())
        attended = torch.stack(attended)
        output = torch.einsum("bhc,hdc->bhd", attended.float(), absorb[:, 128:].float()).bfloat16()
        return self.projection("o_proj", output.reshape(hidden.shape[0], -1)), kv


class QualificationPagedCache:
    """A minimal independent shuffled-page fixture implementing backend methods."""
    def __init__(self, lengths, decode_steps, device):
        self.device = device
        width = (max(lengths) + decode_steps + 63) // 64
        # Odd width deliberately exercises the backend's even-width padding.
        width += 1 - width % 2
        pages = len(lengths) * width + 5
        self.cache = torch.full((pages, 64, 1, 576), float("nan"), dtype=torch.bfloat16, device=device)
        self.table = torch.randperm(pages, device=device)[:len(lengths)*width].reshape(len(lengths), width).int()
        self.append_calls = 0

    def initialize(self, packed_kv, boundaries):
        for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            positions = torch.arange(end-start, device=self.device)
            self.cache[self.table[row, positions // 64].long(), positions % 64, 0] = packed_kv[start:end]

    def get_layer_kv_with_page_table(self, layer_idx):
        if layer_idx != 0:
            raise ValueError("The fixture has exactly one attention layer")
        return self.cache, None, self.table

    def update_layer_decode_new_token(self, *, k_tensor, v_tensor, sequence_lengths, layer_idx, batch_slice):
        if v_tensor is not None or layer_idx != 0 or batch_slice is not None:
            raise ValueError("Unexpected paged append interface")
        rows = torch.arange(k_tensor.shape[0], device=self.device)
        pages = self.table[rows, sequence_lengths // 64].long()
        self.cache[pages, sequence_lengths % 64, 0] = k_tensor[:, 0, 0]
        self.append_calls += 1

    def gather(self, lengths):
        histories = []
        for row, length in enumerate(lengths):
            positions = torch.arange(length, device=self.device)
            histories.append(self.cache[self.table[row, positions // 64].long(), positions % 64, 0])
        return histories


def compare(actual, expected, case, result):
    # Continue independent numerical cases for diagnostics; never turn a failed
    # gate into a pass, nor feed DUT output to the reference calculation.
    try:
        check(actual, expected.float(), {"case": case}, result["cases"])
    except AssertionError:
        pass


def run(args, result):
    if os.environ.get("DISABLE_MMA_V5") != "1":
        raise RuntimeError("Set DISABLE_MMA_V5=1 before launch; SM103 Triton W8A8 otherwise aborts in LLVM")
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run only inside an allocated container")
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
    torch.cuda.reset_peak_memory_stats(device)
    result["device"] = {"name": torch.cuda.get_device_name(device), "capability": [10, 3], "index": args.device}
    config = json.loads(args.config.read_text())
    validate_config(config)
    result["config"] = config
    result["source_sha256"] = {"config": sha256_file(args.config), "backend": sha256_file(BACKEND_SOURCE),
                               "model_norm_rope": sha256_file(MODEL_SOURCE),
                               "triton_w8a8": sha256_file(ROOT / "batchgen/gemm/w8a8.py"),
                               "fp8_quantization": sha256_file(ROOT / "batchgen_kernels/triton/fp8_quantize.py"),
                               "native_fused_rope": sha256_file(ROOT / "batchgen_kernels/triton/fused_rmsnorm_rope.py")}
    backend = isolated_backend()
    if args.attention_implementation == "triton_precision":
        precision = importlib.import_module("batchgen.attention.mla.triton_precision")
        result["source_sha256"]["triton_precision"] = sha256_file(PRECISION_SOURCE)
        result["derived_kernel_source_sha256"] = precision.SOURCE_PROVENANCE_SHA256
    weights, scales, norms = load_weights(args, device, result)
    layer = build_layer(config, weights, norms, device)
    layer._r1_attention_implementation = args.attention_implementation
    lengths = [1, 63, 65] if args.profile == "small" else [1, 63, 64, 65, 129]
    result["lengths"] = lengths
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    max_length = max(lengths) + args.decode_steps
    cos, sin = reference_yarn(config, max_length, device)
    dut_cos, dut_sin = layer.rotary_emb(torch.empty(0, device=device, dtype=torch.bfloat16), seq_len=max_length)
    compare(dut_cos[:, :32], cos, "yarn_cos", result)
    compare(dut_sin[:, :32], sin, "yarn_sin", result)
    reference = ReferenceLayer(config, weights, scales, norms, cos, sin)
    hidden = torch.randn((boundaries[-1], 7168), device=device, dtype=torch.bfloat16)
    positions = torch.cat([torch.arange(length, device=device) for length in lengths])
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    emit({"event": "layer_prefill_start", "tokens": boundaries[-1], "shapes": SHAPES})
    expected, expected_kv = reference.prefill(hidden, positions, boundaries)
    actual, actual_kv = backend.mla_prefill_sm103_w8a16_prepacked(layer, hidden, positions, cu, max(lengths), len(lengths), scales)
    torch.cuda.synchronize(device)
    compare(actual, expected, "packed_prefill_output", result)
    compare(actual_kv, expected_kv, "packed_prefill_native_kv", result)
    # A distinct padded route with alternating left/right padding uses identical
    # live tokens and positions, and never passes padding into the oracle.
    padded = hidden.new_zeros((len(lengths), max(lengths), 7168))
    mask = torch.zeros(padded.shape[:2], device=device, dtype=torch.bool)
    padded_positions = torch.zeros(padded.shape[:2], device=device, dtype=torch.int64)
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        offset = max(lengths) - (end-start) if row % 2 else 0
        padded[row, offset:offset+end-start] = hidden[start:end]
        mask[row, offset:offset+end-start] = True
        padded_positions[row, offset:offset+end-start] = positions[start:end]
    padded_output, padded_kv = backend.mla_prefill_sm103_w8a16(layer, padded, mask, padded_positions, scales)
    compare(padded_output[mask], expected, "padded_prefill_output", result)
    compare(padded_kv[mask], expected_kv, "padded_prefill_native_kv", result)
    if bool(padded_output[~mask].any()) or bool(padded_kv[~mask].any()):
        raise AssertionError("Padded output/KV slots were not zeroed")
    manager = QualificationPagedCache(lengths, args.decode_steps, device)
    manager.initialize(actual_kv, boundaries)
    reference_histories = [expected_kv[start:end].clone() for start, end in zip(boundaries, boundaries[1:])]
    for row, history in enumerate(manager.gather(lengths)):
        compare(history, reference_histories[row], f"prefill_physical_cache_row_{row}", result)
    for step in range(args.decode_steps):
        emit({"event": "layer_decode_start", "step": step})
        decode_hidden = torch.randn((len(lengths), 7168), device=device, dtype=torch.bfloat16)
        positions = torch.tensor(lengths, device=device, dtype=torch.int64)
        next_lengths = [length + 1 for length in lengths]
        expected, expected_kv = reference.decode(decode_hidden, positions, reference_histories)
        actual, appended = backend.mla_decoding_sm103_bf16_with_pagekv(
            layer, decode_hidden.unsqueeze(1), positions[:, None],
            torch.tensor(next_lengths, device=device, dtype=torch.int32), max(next_lengths),
            scales, manager, 0, None,
        )
        torch.cuda.synchronize(device)
        compare(actual[:, 0], expected, f"decode_{step}_output", result)
        compare(appended[:, 0, 0], expected_kv, f"decode_{step}_native_kv", result)
        for row, history in enumerate(manager.gather(next_lengths)):
            compare(history, reference_histories[row], f"decode_{step}_physical_cache_row_{row}", result)
        lengths = next_lengths
    if manager.append_calls != args.decode_steps:
        raise AssertionError("Incorrect number of page-cache appends")
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result["status"] = "passed" if all(case["status"] == "passed" for case in result["cases"]) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attention-implementation", choices=("fa4_flashinfer", "triton_precision"), default="fa4_flashinfer")
    parser.add_argument("--profile", choices=("small", "standard"), default="small")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--weights-shard", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.decode_steps < 3:
        parser.error("At least three decode steps are required")
    result = {"status": "failed", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "versions": versions(), "cases": [],
              "attention_implementation": args.attention_implementation,
              "gate": {"atol": ATOL, "rtol": RTOL, "max_failure_fraction_exclusive": MAX_FAILURE_FRACTION},
              "projection_contract": "Original E4M3 weights and FP32 block128 scales; dynamic per-token block128 FP8 activations; Triton W8A8, no UE8M0 or weight requantization",
              "reference_contract": "Independent FP32 operations with explicit FP8/BF16 boundaries; no DUT values consumed",
              "scope": "One complete attention layer, synthetic paged-manager fixture, no scheduler/offload/model quality or performance claim",
              "compiler_environment": {name: os.environ.get(name) for name in ("TRITON_PTXAS_PATH", "DISABLE_MMA_V5", "CUDA_HOME")},
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
