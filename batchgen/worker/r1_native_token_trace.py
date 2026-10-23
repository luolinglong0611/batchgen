"""Opt-in completion diagnostics from already materialized CPU token IDs.

No tokenizer, distributed operation, device transfer, or lifecycle mutation
belongs here. The worker calls this only for its assigned R1 sequences while
their trajectory reservations are still active.
"""

from __future__ import annotations

import hashlib
import json

import torch


MAX_GENERATED_IDS = 8192
PROMPT_HASH_ENCODING = "sha256:json-compact-integer-array:utf8"


def _validate_cpu_ids(tokens: torch.Tensor, name: str) -> int:
    # Check the device before any operation that could read tensor values.
    # Do not fix an unexpected device with .cpu() or .to(): this diagnostic
    # must never introduce a new device synchronization.
    if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be one-dimensional int32 or int64 IDs")
    return tokens.numel()


def completion_finish_reason(seq, model_context_length: int) -> str:
    """Mirror CompletionHandler's reason without its lifespan side effects."""
    if seq._rep_detected:
        return "repetition"
    if seq.decoded_length >= seq.max_decode_length:
        return "length"
    if seq.current_context_length >= model_context_length:
        return "length"
    if seq.eos_reached:
        return "stop"
    return "length"


def build_native_token_record(
    seq,
    *,
    rank: int,
    model_context_length: int,
    prompt_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
) -> dict:
    """Build a JSON record; generated IDs retain EOS and special tokens.

    The original prompt hash uses the full original prompt, independent of
    tensor dtype/strides and later eviction/re-entry. Only the first 8192
    generated IDs are converted to a Python list; counts expose truncation
    and any disagreement with sequence metadata rather than hiding it.
    """
    prompt_count = _validate_cpu_ids(prompt_tokens, "prompt_tokens")
    generated_count = _validate_cpu_ids(generated_tokens, "generated_tokens")
    if prompt_count != seq.original_prompt_length:
        raise ValueError("original prompt ID count differs from original_prompt_length")
    prompt_ids = prompt_tokens.tolist()
    generated_ids = generated_tokens[:MAX_GENERATED_IDS].tolist()
    if any(token < 0 for token in prompt_ids) or any(token < 0 for token in generated_ids):
        raise ValueError("native token IDs must be non-negative")
    prompt_bytes = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "r1_native_tokens_v1",
        "model_name": "deepseek-ai/DeepSeek-R1",
        "rank": rank,
        "assigned_rank": seq.assigned_rank,
        "uuid": seq.uuid,
        "batch_id": seq.batch_id,
        "global_idx": seq.global_idx,
        "original_prompt_length": seq.original_prompt_length,
        "original_prompt_ids_count": prompt_count,
        "original_prompt_ids_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_ids_hash_encoding": PROMPT_HASH_ENCODING,
        "generated_ids": generated_ids,
        "generated_ids_count": generated_count,
        "generated_ids_logged_count": len(generated_ids),
        "generated_ids_truncated": generated_count > MAX_GENERATED_IDS,
        "generated_ids_omitted_count": max(0, generated_count - MAX_GENERATED_IDS),
        "decoded_length": seq.decoded_length,
        "decoded_length_matches_ids_count": seq.decoded_length == generated_count,
        "finish_reason": completion_finish_reason(seq, model_context_length),
    }
