"""Opt-in R1 page-boundary progress balancing; pure CPU planning only.

This changes which Host-backed sequences occupy the existing GPU working set.
It does not stream Host KV inside a layer, increase GPU capacity, or count
communication padding as useful rows. The worker retains all ownership,
collective allocation, append-drain and H2D-completion checks.
"""
from dataclasses import replace
from typing import Mapping


ENV_NAME = "BATCHGEN_R1_PAGE_ROTATION_TOKENS"


def rotation_quantum(environ: Mapping[str, str], *, page_size: int = 64) -> int:
    """Zero preserves the existing scheduler; positive values are page multiples."""
    value = environ.get(ENV_NAME, "0")
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"{ENV_NAME} must be0 or a positive integer page multiple")
    quantum = int(value)
    if quantum and (page_size <= 0 or quantum % page_size):
        raise ValueError(f"{ENV_NAME} must be a multiple of page size {page_size}")
    return quantum


def balance_gpu_working_set(req, original, *, quantum_tokens: int):
    """Keep Host planning intact and choose a capacity-bounded GPU working set.

    Lower progress buckets run first. Within a bucket existing GPU rows win,
    avoiding a full-history reload every boundary. Priority remains the first
    key, and immutable global IDs plus UUIDs make ties independent of dict order.
    A Host growth/eviction boundary retains its original complete transaction;
    balancing resumes at a later boundary with no pending Host mutation.
    """
    if not quantum_tokens:
        return original, None
    if (type(quantum_tokens) is not int or quantum_tokens <= 0 or quantum_tokens % 64
            or req.world_size != 8 or req.num_gpus_per_node != 8 or req.attn_tp_size != 1):
        raise ValueError("R1 page rotation requires single-node DP-attention EP8 and page64 quantum")
    if original.scheduler_error or original.host_growth_uuids or original.host_evicted_uuids:
        return original, {"skipped": "existing_Host_growth_eviction_or_scheduler_error"}
    if req.max_rank_bsz <= 0 or len(req.per_rank_total) != 8 or len(req.per_rank_free) != 8:
        raise ValueError("R1 page rotation requires an explicit row cap and all8 live GPU capacities")

    def integer(value, label, *, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid R1 page-rotation {label}: {value!r}")
        return value

    def metadata(uuid):
        meta = req.seq_meta.get(uuid)
        if meta is None:
            raise ValueError(f"R1 page-rotation sequence metadata missing: {uuid}")
        if type(meta.priority) is not int:
            raise ValueError("R1 page-rotation priority must be an integer")
        return integer(meta.global_idx, "global_idx"), meta.priority

    totals = [integer(n, "GPU total pages", minimum=1) for n in req.per_rank_total]
    budgets = [integer(n, "GPU free pages") for n in req.per_rank_free]
    completed = set(original.completed_uuids)
    active = set(original.active_uuids)
    if active & completed or set(req.decode_uuids) != active | completed:
        raise ValueError("R1 page-rotation active/completed coverage differs from decode snapshot")
    rows = {}
    for uuid in req.decode_uuids:
        state = req.global_seq_state[uuid]
        rank = integer(state.get("assigned_rank"), "assigned rank")
        if rank >= 8:
            raise ValueError("R1 page-rotation owner lies outside eight ranks")
        pages = integer(state.get("gpu_pages_allocated"), "resident pages")
        budgets[rank] += pages  # includes pages released by actual completions
        if uuid in completed:
            continue
        if not pages or state.get("completed"):
            raise ValueError("R1 page-rotation active row lacks valid resident KV")
        growth = integer(state.get("additional_pages_needed"), "extension pages")
        gid, priority = metadata(uuid)
        rows[uuid] = {"rank": rank, "pages": pages + growth, "resident": True,
                      "progress": integer(state.get("decoded_length"), "decoded length"),
                      "gid": gid, "priority": priority}
    if any(free > total for free, total in zip(budgets, totals)):
        raise ValueError("R1 page-rotation reclaimed GPU pages exceed actual pool capacity")
    for uuid, state in req.global_candidate_info.items():
        if uuid in rows or uuid in completed:
            raise ValueError("R1 page-rotation load candidate overlaps active/completed rows")
        if state.get("status") not in ("PREFILLED", "ON_HOLD"):
            raise ValueError("R1 page rotation may only load existing Host-backed states")
        rank = integer(state.get("assigned_rank"), "candidate owner")
        if rank >= 8:
            raise ValueError("R1 page-rotation candidate owner lies outside eight ranks")
        gid, priority = metadata(uuid)
        rows[uuid] = {"rank": rank,
                      "pages": integer(state.get("pages_needed"), "load pages", minimum=1),
                      "resident": False,
                      "progress": integer(state.get("decoded_length"), "candidate decoded length"),
                      "gid": gid, "priority": priority}

    def order(item):
        uuid, row = item
        return (-row["priority"], row["progress"] // quantum_tokens,
                not row["resident"], row["progress"], row["gid"], uuid)

    selected, used, counts = set(), [0] * 8, [0] * 8
    for uuid, row in sorted(rows.items(), key=order):
        rank, cost = row["rank"], row["pages"]
        if counts[rank] < req.max_rank_bsz and used[rank] + cost <= budgets[rank]:
            selected.add(uuid)
            used[rank] += cost
            counts[rank] += 1
    if rows and not selected:
        raise ValueError("No Host-backed row can fit the actual R1 GPU working-set budget")
    held = [uuid for uuid in original.active_uuids if uuid not in selected]
    kept = [uuid for uuid in original.active_uuids if uuid in selected]
    loaded = sorted((uuid for uuid in selected if not rows[uuid]["resident"]),
                    key=lambda uuid: (rows[uuid]["gid"], uuid))
    extension = [uuid for uuid in kept if req.global_seq_state[uuid]["additional_pages_needed"] > 0]
    decision = replace(original, onhold_uuids=held, seqs_needing_extension=extension,
                       new_load_uuids=loaded, decode_uuids_final=kept)
    return decision, {"quantum_tokens": quantum_tokens, "skipped": None,
        "actual_target_rows_per_rank": counts, "actual_target_GPU_pages_per_rank": used,
        "actual_available_GPU_pages_per_rank": budgets, "onhold_rows": len(held),
        "load_rows": len(loaded), "kept_rows": len(kept),
        "H2D_load_pages_per_rank": [sum(rows[u]["pages"] for u in loaded
                                         if rows[u]["rank"] == rank) for rank in range(8)],
        "loaded_progress_min": min((rows[u]["progress"] for u in loaded), default=None),
        "held_progress_max": max((rows[u]["progress"] for u in held), default=None),
        "unfinished_progress_min": min((r["progress"] for r in rows.values()), default=None),
        "unfinished_progress_max": max((r["progress"] for r in rows.values()), default=None),
        "unfinished_progress_gap": (max(r["progress"] for r in rows.values())
                                    - min(r["progress"] for r in rows.values())) if rows else 0,
        "Host_KV_or_trajectory_mutated": False}
