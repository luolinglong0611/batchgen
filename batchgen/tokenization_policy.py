"""Bound concurrent admission tokenizers without changing prompt ownership.

The legacy per-rank character guard remains the default. An explicit total
character budget permits several ranks to tokenize together on larger hosts;
this is a workload sizing control, not a character-to-RAM conversion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationPlan:
    groups: tuple[tuple[int, ...], ...]
    character_budget: int | None
    rank_limit: int
    explicit: bool


def _positive_integer(env, name, default):
    raw = env.get(name)
    if raw is None:
        return default
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw)


def admission_tokenization_plan(chars_by_rank, *, legacy_threshold, env=None):
    """Choose identical ascending rank waves from identical admission metadata.

    A rank's strided prompt partition is indivisible. A partition larger than
    the explicit budget therefore runs alone, preserving the legacy fallback.
    """
    env = os.environ if env is None else env
    chars = tuple(chars_by_rank)
    if legacy_threshold <= 0 or any(not isinstance(n, int) or n < 0 for n in chars):
        raise ValueError("Nonnegative rank character counts and positive threshold required")
    active = len(chars)
    if not active:
        return TokenizationPlan((), None, 0, False)
    budget = _positive_integer(env, "BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET", None)
    limit = min(active, _positive_integer(env, "BATCHGEN_TOKENIZER_PARALLEL_RANKS", active))
    explicit = budget is not None or "BATCHGEN_TOKENIZER_PARALLEL_RANKS" in env
    if budget is None:
        # An explicit rank limit alone only reduces the legacy concurrency;
        # expanding the old memory guard requires an explicit character budget.
        if max(chars) >= legacy_threshold:
            limit = 1
        budget = None
    groups, group, group_chars = [], [], 0
    for rank, count in enumerate(chars):
        if group and (len(group) >= limit or
                      (budget is not None and group_chars + count > budget)):
            groups.append(tuple(group))
            group, group_chars = [], 0
        group.append(rank)
        group_chars += count
    if group:
        groups.append(tuple(group))
    return TokenizationPlan(tuple(groups), budget, limit, explicit)
