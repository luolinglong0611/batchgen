#!/usr/bin/env python3
"""CPU-only arithmetic for larger common R1 KV candidates; never approval.

85% means estimated peak occupied KV allocator pages, not NVML utilization or
mem-fraction-static. Real allocation, headroom, ownership skew, growth curves,
kernel shapes and throughput must be measured on both engines separately.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

GIB = 1024**3
DP = 16
NODES = 2
PAGE = 64
CONTEXT = 32768
TOKEN_BYTES = 61 * 576 * 2
PAGE_BYTES = TOKEN_BYTES * PAGE
DEFAULT_TOKENS = (131072, 262144, 524288, 1048576, 2097152)
PROFILES = ((8192, 2048), (2048, 8192))
ROOT = Path(__file__).resolve().parents[2]


def ceil_div(a, b):
    return (a + b - 1) // b


def require_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def fraction(value, name):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(name + " must be an exact decimal string or Fraction")
    result = Fraction(value)
    if not 0 < result < 1:
        raise ValueError(name + " must lie strictly between0 and1")
    return result


def source_function(path, name):
    """Execute just the actual pure source function, never its Torch imports."""
    raw = Path(path).read_bytes()
    tree = ast.parse(raw)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely identify native pure capacity helper")
    node = matches[0]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[])
    namespace = {"math": math}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name], {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                              "function": name, "line": node.lineno}


def cap_row(tokens, per_rank, *, host_occupancy, host_ram_gib, host_reserve_gib,
            gpu_total_mib, memory_fraction):
    global_cap = per_rank * DP
    node_sequences = per_rank * (DP // NODES)
    full_sequence_pages = ceil_div(sum(PROFILES[0]), PAGE)
    needed_pages = node_sequences * full_sequence_pages
    # --host-kv-cache-size is a data-page budget, not total mapping bytes.
    minimum_host_gib = ceil_div(needed_pages * PAGE_BYTES, GIB)
    with_headroom_gib = ceil_div(needed_pages * PAGE_BYTES * host_occupancy.denominator,
                                 GIB * host_occupancy.numerator)
    available_pages = with_headroom_gib * GIB // PAGE_BYTES
    trajectory_bytes = global_cap * ceil_div(CONTEXT, PAGE) * PAGE * 4
    remaining_host_bytes = host_ram_gib * GIB - 675 * GIB - with_headroom_gib * GIB - trajectory_bytes
    occupancy = Fraction(per_rank * full_sequence_pages, tokens // PAGE)
    sg_data_bytes = (tokens + PAGE) * TOKEN_BYTES
    allowed_non_KV_bytes = gpu_total_mib * 1024**2 * memory_fraction - sg_data_bytes
    return {
        "global_active_cap": global_cap, "per_DP_rank_active_cap": per_rank,
        "predicted_terminal_occupied_pages_per_rank": per_rank * full_sequence_pages,
        "predicted_terminal_KV_pool_occupancy_percent": float(occupancy * 100),
        "predicted_terminal_free_KV_pool_percent": float((1 - occupancy) * 100),
        "fits_logical_KV_pages_at_terminal": occupancy <= 1,
        "BG_hostKV_balanced_full_sequence_pages_per_node": needed_pages,
        "BG_hostKV_balanced_minimum_data_gib_per_node": minimum_host_gib,
        "BG_hostKV_data_budget_gib_with_planned_page_headroom_per_node": with_headroom_gib,
        "BG_hostKV_resulting_free_page_percent": float(Fraction(available_pages - needed_pages, available_pages) * 100),
        "BG_hostKV_all_global_owners_on_one_node_minimum_data_gib": ceil_div(global_cap * full_sequence_pages * PAGE_BYTES, GIB),
        "BG_hostKV_balanced_scope": "Assumes balanced owner assignment; MAX_RANK_BSZ limits in-decode rows, not all host-owned prefetched/on-hold sequences. Actual per-node owner/page peaks required.",
        "BG_hostKV_metadata_alignment_mapping_bytes": "UNKNOWN_additional_to_data_budget; native backend adds header/free-stack/owner/link/sequence table and2MiB data alignment",
        "BG_input_trajectory_bytes_ON_EACH_NODE": trajectory_bytes,
        "BG_input_ids_pool_size_gib_ON_EACH_NODE": trajectory_bytes / GIB,
        "BG_input_pool_scope": "Each node replicates ALL global admitted trajectories; do not divide by2 or16",
        "SG_req_to_token_map_bytes_per_rank_candidate_lower_bound": global_cap * (CONTEXT + 4) * 4,
        "SG_req_to_token_map_scope": "Native ModelRunner passes global max-running as max_num_reqs to each local ReqToTokenPool; additional metadata not counted",
        "matching_candidate_fields_NOT_launch_authorization": {
            "SG_max_running_requests_global": global_cap,
            "SG_max_total_tokens_per_rank": tokens,
            "BG_BATCHGEN_R1_GPU_KV_TOKENS": str(tokens),
            "BG_input_ids_pool_size_gb": str(trajectory_bytes / GIB),
            "BG_BATCHGEN_MAX_DECODE_RANK_BSZ": str(per_rank),
            "BG_BATCHGEN_MAX_RANK_BSZ": str(per_rank),
            "BG_host_kv_cache_size_data_gib_per_node": with_headroom_gib,
        },
        "requires_raising_BG_default128_rank_limit": per_rank > 128,
        "BG_explicit_two_rank_limit_names_must_match_before_import": True,
        "BG_HOST_WEIGHT_RESERVATION_gib_per_node": 675,
        "host_ram_gib_per_node_assumed": host_ram_gib,
        "remaining_host_gib_after_named_reservations_NOT_measured_free": remaining_host_bytes / GIB,
        "unmodeled_host_reserve_gib_assumed": host_reserve_gib,
        "arithmetic_residual_after_unmodeled_reserve_gib": (remaining_host_bytes - host_reserve_gib * GIB) / GIB,
        "named_host_reservations_leave_assumed_reserve": remaining_host_bytes >= host_reserve_gib * GIB,
        "GPU_total_memory_mib_assumed_NOT_fresh": gpu_total_mib,
        "GPU_memory_fraction_parameter": str(memory_fraction),
        "maximum_nonKV_residency_gib_under_fraction_NOT_measured": float(allowed_non_KV_bytes / GIB),
        "GPU_memory_headroom_NOT_established": True,
        "minimum_submitted_requests_for_one_fully_populated_wave": global_cap,
        "total_request_count_for_throughput": "TBD_by_common_measured_shape_and_occupancy; old128 is not a cap",
        "allocation_qualified": False, "formal_test_permitted": False,
    }


def calculate(tokens, *, target="0.85", host_occupancy="0.85", host_ram_gib=2048,
              host_reserve_gib=256, gpu_total_mib=275040, memory_fraction="0.85", repo=ROOT):
    target, host_occupancy = fraction(target, "KV occupancy target"), fraction(host_occupancy, "host page occupancy")
    memory_fraction = fraction(memory_fraction, "GPU static memory fraction")
    for value, name in ((host_ram_gib, "host RAM"), (host_reserve_gib, "unmodeled host reserve"), (gpu_total_mib, "GPU total memory")):
        require_int(value, name)
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Provide nonempty distinct per-rank KV capacities")
    reserve, reserve_source = source_function(Path(repo) / "batchgen/worker/prefill.py", "compute_prefill_host_reservation")
    query_capacity, trajectory_source = source_function(Path(repo) / "batchgen/r1_alignment_controls.py", "trajectory_capacity_bytes")
    candidates = []
    for token_count in tokens:
        require_int(token_count, "KV tokens")
        if token_count % PAGE or token_count >= 2**31:
            raise ValueError("KV token capacity must be a page64 multiple below int32 range")
        pages = token_count // PAGE
        required = ceil_div(10240, PAGE)
        ideal = target * pages / required
        lower = ideal.numerator // ideal.denominator
        upper = ceil_div(ideal.numerator, ideal.denominator)
        if lower <= 0:
            raise ValueError("Candidate cannot hold one full-profile request below the target")
        # Report both sides; closest is an arithmetic candidate, never approval.
        closest = min((lower, upper), key=lambda c: (abs(Fraction(c * required, pages) - target), c))
        row = {"usable_GPU_KV_tokens_per_rank": token_count, "allocator_pages_per_rank": pages,
            "logical_KV_data_bytes_per_rank": token_count * TOKEN_BYTES,
            "logical_KV_data_gib_per_rank": token_count * TOKEN_BYTES / GIB,
            "SG_native_dummy_page_bytes_per_rank": PAGE * TOKEN_BYTES,
            "SG_native_KV_tensor_data_gib_per_rank": (token_count + PAGE) * TOKEN_BYTES / GIB,
            "ideal_target_per_rank_request_count": str(ideal),
            "target_bracketing_global_caps": [lower * DP, upper * DP],
            "closest_global_cap_arithmetic_ONLY": closest * DP,
            "options": []}
        for active in sorted(set((lower, upper))):
            option = cap_row(token_count, active, host_occupancy=host_occupancy, host_ram_gib=host_ram_gib,
                host_reserve_gib=host_reserve_gib, gpu_total_mib=gpu_total_mib, memory_fraction=memory_fraction)
            assert query_capacity(CONTEXT, option["global_active_cap"]) == option["BG_input_trajectory_bytes_ON_EACH_NODE"]
            profile_rows = []
            for prompt, output in PROFILES:
                effective_chunk = ceil_div(max(min(8192, output), 128), PAGE) * PAGE
                host_pages, rounded = reserve(prompt_length=prompt, kv_token_budget=prompt + output,
                    page_size=PAGE, chunk_size=effective_chunk, initial_gpu_page_buffer=32)
                # SequenceEntry initial GPU runway is capped by full token budget
                # and available host pages; it is not an extra32 pages at EOS.
                initial_gpu_pages = min(ceil_div(min(prompt + 1 + 32 * PAGE, prompt + output), PAGE), host_pages)
                profile_rows.append({"input_tokens": prompt, "output_tokens": output,
                    "maximum_total_tokens": prompt + output, "full_reservation_pages_per_sequence": required,
                    "BG_initial_host_reservation_pages_at_candidate_default_chunk": host_pages,
                    "BG_initial_host_reserved_tokens": rounded,
                    "BG_initial_GPU_runway_pages_per_sequence": initial_gpu_pages,
                    "SG_prompt_only_pages_per_sequence": ceil_div(prompt, PAGE),
                    "predicted_BG_initial_reserved_KV_percent": float(Fraction(active * initial_gpu_pages, pages) * 100),
                    "SG_prompt_only_occupied_KV_percent": float(Fraction(active * ceil_div(prompt, PAGE), pages) * 100),
                    "peak_is_not_time_average_or_sustained85percent": True})
            option["profile_reservations"] = profile_rows
            row["options"].append(option)
        candidates.append(row)
    tracked = {
        "batchgen/batchgen_worker.py": ("def _owns_host_kv", "def _max_decode_rank_bsz", "def _init_r1_fixed_gpu_kv", "reserved_tokens = [self.model_context_length]"),
        "batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py": ("def _init_decoding_padding_bsz", "def _init_ata_comms"),
        "batchgen/server/worker_manager.py": ("Host KV cache is per-node shared",),
        "batchgen/sequence.py": ("def get_gpu_pages_for_two_page_buffer", "self.kv_token_budget: int = prompt_length + max_decode_length"),
        "core/KV_Storage/host_paged_kv_backend.cpp": ("void HostPagedKVBackend::SharedState::ComputeOffsets",),
    }
    evidence = [reserve_source, trajectory_source]
    for name, patterns in tracked.items():
        path = Path(repo) / name;raw = path.read_bytes();lines = raw.decode().splitlines()
        anchors = {pattern: [i + 1 for i, line in enumerate(lines) if pattern in line] for pattern in patterns}
        if any(not value for value in anchors.values()): raise ValueError("Source capacity contract anchor moved: " + name)
        evidence.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "anchors": anchors})
    return {"schema": "r1-common-KV-capacity-arithmetic-v1", "status": "CPU_arithmetic_candidates_NOT_approved_or_executed",
        "meaning_of85percent": "Peak occupied allocator KV pages divided by usable pool pages; not static tensor allocation, GPU utilization, NVML allocated percentage, time-average occupancy or mem-fraction-static.",
        "target_KV_pool_occupancy": str(target), "host_page_occupancy_planning_ceiling": str(host_occupancy),
        "geometry": {"attention_DP": DP, "nodes": NODES, "context_cap": CONTEXT, "page_tokens": PAGE,
            "BF16_KV_bytes_per_token_all61layers": TOKEN_BYTES, "bytes_per_page_all_layers": PAGE_BYTES},
        "source_evidence_NOT_runtime_proof": evidence, "candidates": candidates,
        "scope_limits": ["HostKV bytes are per-node shared once; worker cudaHostRegister calls are not eight physical copies.",
            "HostKV still registers one whole range; weights chunk16 does not split HostKV registration.",
            "Host metadata/alignment, per-rank padded MoE workspace, model weights, NCCL, JIT and prefill peaks require actual measurement; arithmetic leftovers are not free-memory proof.",
            "GPU target additionally requires every rank's requested data allocation to fit observed free/headroom; SG must not silently profile-clamp below the common target.",
            "Changing caps/pools/buffer geometry requires new common admission, every-rank shape warmup, cache growth/release and correctness checks; no old pass is inherited.",
            "Current qualification remains active64 /131072tokens; these larger values are estimates only. No source, allocator or launcher is modified."],
        "formal_test_permitted": False, "GPU_HTTP_calls": False}


def markdown(report):
    lines = ["# Common R1 GPU KV capacity arithmetic", "", "CPU estimates only; no allocation, approval or performance result.", "",
        "|KV tokens/GPU|KV GiB/GPU|per-rank active|global active|predicted terminal KV%|HostKV min GiB/node|HostKV GiB/node with15% page reserve|input pool MiB ON EACH NODE|raise BG128?|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for candidate in report["candidates"]:
        for row in candidate["options"]:
            lines.append(f"|{candidate['usable_GPU_KV_tokens_per_rank']}|{candidate['logical_KV_data_gib_per_rank']:.6f}|{row['per_DP_rank_active_cap']}|{row['global_active_cap']}|{row['predicted_terminal_KV_pool_occupancy_percent']:.6f}|{row['BG_hostKV_balanced_minimum_data_gib_per_node']}|{row['BG_hostKV_data_budget_gib_with_planned_page_headroom_per_node']}|{row['BG_input_trajectory_bytes_ON_EACH_NODE']/1024**2:.3f}|{row['requires_raising_BG_default128_rank_limit']}|")
    lines += ["", report["meaning_of85percent"], "", "Both bracketing caps are shown because16-way integer concurrency cannot always hit85% exactly. Use actual occupancy and throughput to choose; no request-count limit is inferred from the old128 batch.", ""]
    lines += ["- " + text for text in report["scope_limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--target", default="0.85")
    parser.add_argument("--host-page-occupancy", default="0.85")
    parser.add_argument("--host-ram-gib", type=int, default=2048)
    parser.add_argument("--host-unmodeled-reserve-gib", type=int, default=256)
    parser.add_argument("--reported-gpu-memory-mib", type=int, default=275040)
    parser.add_argument("--memory-fraction", default="0.85")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = calculate(args.gpu_tokens, target=args.target, host_occupancy=args.host_page_occupancy,
        host_ram_gib=args.host_ram_gib, host_reserve_gib=args.host_unmodeled_reserve_gib,
        gpu_total_mib=args.reported_gpu_memory_mib, memory_fraction=args.memory_fraction, repo=args.repo)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "capacity.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with (args.output_dir / "capacity.md").open("x") as stream: stream.write(markdown(report))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
