"""Explicit admission controls for the R1 comparison, independent of model RoPE."""
import os
import re
import math


CONTEXT_ENV = "BATCHGEN_R1_CONTEXT_CAP"
GPU_KV_TOKENS_ENV = "BATCHGEN_R1_GPU_KV_TOKENS"


def effective_context_limit(model_max, model_identifier):
    if type(model_max) is not int or model_max <= 0:
        raise ValueError("The actual native model context must be a positive integer")
    raw = os.environ.get(CONTEXT_ENV)
    if raw is None:
        return model_max
    if ("deepseek-r1" not in str(model_identifier).lower()
            or os.environ.get("BATCHGEN_R1_SM103_MLA_COMPAT") != "flashinfer-fa2"
            or os.environ.get("BATCHGEN_R1_SM103_FP8_COMPAT") != "1"):
        raise ValueError("R1 context cap requires the explicit R1 compatibility profile")
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise ValueError(f"{CONTEXT_ENV} must be an explicit positive integer")
    limit = int(raw)
    if limit > model_max:
        raise ValueError("An admission cap cannot extend the native model context")
    return limit


def trajectory_capacity_bytes(context_cap, global_active_cap, page_tokens=64, token_bytes=4):
    """Exact existing QueryBook budget for a fixed global admission capacity.

    This does not add a scheduler or alter its defaults. The launch protocol
    supplies the returned byte budget through --input-ids-pool-size-gb and must
    still verify the worker's published total_capacity after initialization.
    """
    for value in (context_cap, global_active_cap, page_tokens, token_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("Context, active capacity and page/token sizes must be positive integers")
    return ((context_cap + page_tokens - 1) // page_tokens) * page_tokens * token_bytes * global_active_cap


def fixed_r1_gpu_kv_contract(*, model_identifier, kv_dtype, graphs_enabled,
                             device_capability, num_layers, kv_lora_rank,
                             qk_rope_head_dim):
    """Opt-in capacity contract, separate from the native memory-based policy.

    This controls allocated KV *data*, not scheduler policy or attention math.
    No environment override returns before examining any supplied attributes.
    The existing legacy size override cannot reliably reach the real decode
    initialization path and must not accompany this explicit control.
    """
    raw = os.environ.get(GPU_KV_TOKENS_ENV)
    if raw is None:
        return None
    if ("deepseek-r1" not in str(model_identifier).lower()
            or os.environ.get("BATCHGEN_R1_SM103_MLA_COMPAT") != "flashinfer-fa2"
            or os.environ.get("BATCHGEN_R1_SM103_FP8_COMPAT") != "1"):
        raise ValueError("Fixed R1 GPU KV requires the explicit R1 compatibility profile")
    if os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB") is not None:
        raise ValueError("Remove the legacy BATCHGEN_GPU_KV_CACHE_SIZE_GB override")
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise ValueError(GPU_KV_TOKENS_ENV + " must be an explicit positive integer")
    tokens = int(raw)
    if tokens % 64 or tokens >= 2**31:
        raise ValueError("Fixed R1 GPU KV tokens must be a page64 multiple below int32 range")
    if graphs_enabled is not False:
        raise ValueError("Fixed R1 GPU KV requires CUDA graphs explicitly OFF")
    if str(kv_dtype) not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("Fixed R1 GPU KV requires BF16 KV")
    if (not isinstance(device_capability, (tuple, list))
            or len(device_capability) != 2
            or any(type(x) is not int for x in device_capability)
            or tuple(device_capability) != (10, 3)):
        raise ValueError("Fixed R1 GPU KV requires actual CUDA SM103")
    if any(type(value) is not int or value != expected for value, expected in (
            (num_layers, 61), (kv_lora_rank, 512), (qk_rope_head_dim, 64))):
        raise ValueError("Fixed R1 GPU KV requires the actual 61-layer MLA512+RoPE64 geometry")
    data_bytes = tokens * 61 * 576 * 2
    return {"tokens_per_rank": tokens, "pages_per_layer": tokens // 64,
            "page_size_tokens": 64, "data_bytes_per_rank": data_bytes,
            "data_gib_per_rank": data_bytes / 1024**3,
            "origin": GPU_KV_TOKENS_ENV,
            "scope": "explicit_qualification_capacity_NOT_paper_or_performance_qualified"}


def fixed_r1_gpu_kv_available_bytes(*, free_bytes, total_bytes, memory_fraction):
    """Preserve the native fraction's free headroom; never shrink a target."""
    if (type(free_bytes) is not int or type(total_bytes) is not int
            or not 0 <= free_bytes <= total_bytes or total_bytes <= 0):
        raise ValueError("Invalid actual CUDA memory observation")
    if (type(memory_fraction) not in (int, float) or not math.isfinite(memory_fraction)
            or not 0 < memory_fraction <= 1):
        raise ValueError("Invalid explicit GPU memory fraction")
    return math.floor(free_bytes - total_bytes * (1 - memory_fraction))


def validate_fixed_r1_gpu_kv_allocator(config, contract):
    """Check the actual native allocator configuration, not a launch label."""
    expected = {"num_layers": 61, "num_pages": contract["pages_per_layer"],
                "page_size_tokens": 64, "num_k_heads": 1, "k_head_dim": 576,
                "num_v_heads": 0, "v_head_dim": 0}
    for field, value in expected.items():
        actual = getattr(config, field, None)
        if type(actual) is not int or actual != value:
            raise ValueError("Fixed R1 GPU KV allocator mismatch: " + field)
    if str(getattr(config, "kv_dtype", None)) not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("Fixed R1 GPU KV allocator dtype is not BF16")
    if getattr(config, "logical_to_physical_layer", None):
        raise ValueError("Fixed R1 GPU KV rejects remapped layer geometry")


def validate_fixed_r1_gpu_kv_ranks(reports, world_size):
    """Fail all explicit-profile ranks on any mismatch or local error.

    The launcher must supply the same opt-in environment on every rank: a
    process that omitted the flag follows the unchanged native path and cannot
    participate in this added collective.
    """
    if (type(world_size) is not int or world_size <= 0 or type(reports) is not list
            or len(reports) != world_size or any(type(r) is not dict for r in reports)
            or {r.get("rank") for r in reports} != set(range(world_size))
            or any(type(r.get("rank")) is not int for r in reports)):
        raise RuntimeError("Fixed R1 GPU KV rank reports are incomplete")
    errors = [{"rank": r["rank"], "error": r["error"]} for r in reports if r.get("error")]
    if errors:
        raise RuntimeError("Fixed R1 GPU KV collective validation failed: " + str(errors))
    first = reports[0]
    for report in reports:
        if (type(report.get("tokens_per_rank")) is not int
                or report.get("tokens_per_rank") != first.get("tokens_per_rank")
                or report.get("lifecycle") != first.get("lifecycle")):
            raise RuntimeError("Fixed R1 GPU KV rank token capacities/lifecycles differ")
    return first["tokens_per_rank"]
