from batchgen.tokenization_policy import admission_tokenization_plan
import pytest


def plan(chars, env=None):
    return admission_tokenization_plan(chars, legacy_threshold=4_000_000, env=env or {})


def test_default_retains_legacy_guard_and_parallel_small_admission():
    assert plan([6_000_000] * 8).groups == tuple((i,) for i in range(8))
    assert plan([3_000_000] * 8).groups == (tuple(range(8)),)


def test_explicit_budget_enables_whole_group_without_changing_ownership():
    p = plan([6_000_000] * 8, {"BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET": "64000000",
                               "BATCHGEN_TOKENIZER_PARALLEL_RANKS": "8"})
    assert p.groups == (tuple(range(8)),)
    assert p.explicit


def test_bounded_waves_and_oversized_partition_fallback():
    p = plan([6, 6, 30, 4, 4], {"BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET": "12",
                               "BATCHGEN_TOKENIZER_PARALLEL_RANKS": "3"})
    assert p.groups == ((0, 1), (2,), (3, 4))


def test_rank_cap_alone_does_not_disable_legacy_memory_guard():
    assert plan([6_000_000] * 8, {"BATCHGEN_TOKENIZER_PARALLEL_RANKS": "4"}).groups == tuple((i,) for i in range(8))
    assert plan([1] * 8, {"BATCHGEN_TOKENIZER_PARALLEL_RANKS": "3"}).groups == ((0, 1, 2), (3, 4, 5), (6, 7))


@pytest.mark.parametrize("value", ["0", "-1", "junk", "1.5", ""])
def test_invalid_budget_fails_before_any_collective(value):
    with pytest.raises(ValueError):
        plan([1, 2], {"BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET": value})
