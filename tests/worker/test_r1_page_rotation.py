"""Pure CPU scheduling contracts; no model, GPU, server or distributed run."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


rotation = load_file('_r1_rotation_cpu', 'batchgen/worker/r1_page_rotation.py')
continuous = load_file('_r1_rotation_continuous_cpu', 'batchgen/continuous_batching.py')


def snapshot(active, candidates=None, *, free=None, total=None, cap=384, priorities=None):
    """Rows are uuid:(owner,decoded,allocated_pages,additional_or_load_pages)."""
    candidates = candidates or {}
    state = {u: dict(assigned_rank=r, decoded_length=p, gpu_pages_allocated=pages,
        additional_pages_needed=extra, completed=False)
        for u, (r, p, pages, extra) in active.items()}
    waiting = {u: dict(assigned_rank=r, decoded_length=p, pages_needed=needed, status='PREFILLED')
        for u, (r, p, _, needed) in candidates.items()}
    meta = {u: SimpleNamespace(global_idx=i, priority=(priorities or {}).get(u, 0))
        for i, u in enumerate(sorted(set(active) | set(candidates)))}
    req = SimpleNamespace(world_size=8, num_gpus_per_node=8, attn_tp_size=1,
        max_rank_bsz=cap, decode_uuids=tuple(active), global_seq_state=state,
        global_candidate_info=waiting, seq_meta=meta,
        per_rank_free=tuple(free if free is not None else [0] * 8),
        per_rank_total=tuple(total if total is not None else [100] * 8))
    original = continuous.BoundaryDecisions(completed_uuids=[], active_uuids=list(active),
        host_growth_uuids=[], host_growth_pages=[], growth_feasible=False,
        host_evicted_uuids=[], onhold_uuids=[], seqs_needing_extension=[],
        new_load_uuids=[], decode_uuids_final=list(active))
    return req, original


def test_disabled_is_the_exact_legacy_object_without_new_constraints():
    req, original = snapshot({'a': (0, 10, 2, 0)})
    req.world_size = 2
    assert rotation.rotation_quantum({}) == 0
    assert rotation.rotation_quantum({rotation.ENV_NAME: '0'}) == 0
    result, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=0)
    assert result is original and summary is None


@pytest.mark.parametrize('value', ['-1', '1.5', 'true', '63', '65'])
def test_quantum_requires_an_explicit_page_multiple(value):
    with pytest.raises(ValueError):
        rotation.rotation_quantum({rotation.ENV_NAME: value})


def test_within_bucket_keeps_resident_KV_then_rotates_behind_progress_frontier():
    req, original = snapshot({'a': (0, 65, 5, 0), 'b': (0, 65, 5, 0)},
        {'waiting': (0, 1, 0, 5)}, total=[10] * 8)
    before = copy.deepcopy(req)
    plan, _ = rotation.balance_gpu_working_set(req, original, quantum_tokens=128)
    assert plan.onhold_uuids == plan.new_load_uuids == []
    plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert plan.new_load_uuids == ['waiting']
    assert len(plan.onhold_uuids) == 1
    assert summary['actual_target_rows_per_rank'][0] == 2
    assert summary['actual_target_GPU_pages_per_rank'][0] == 10
    assert req.__dict__ == before.__dict__  # no decoded length/KV/Host state mutation


def test_capacity_at_long_context_never_promises_all_Host_rows_or_384_valid_rows():
    # The actual full-pool geometry at a10240-token budget allows187 rows/rank.
    active, waiting = {}, {}
    for rank in range(8):
        for i in range(187):
            active[f'a-{rank}-{i:03}'] = (rank, 2041, 160, 0)
        for i in range(59):
            waiting[f'w-{rank}-{i:03}'] = (rank, 2041, 0, 160)
    req, original = snapshot(active, waiting, free=[12] * 8, total=[29932] * 8)
    plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert summary['actual_target_rows_per_rank'] == [187] * 8
    assert len(plan.decode_uuids_final) + len(plan.new_load_uuids) == 1496
    assert all(p <= 29932 for p in summary['actual_target_GPU_pages_per_rank'])
    assert not plan.new_load_uuids  # equal progress need not reload valid resident KV


def test_credit_completed_GPU_pages_can_admit_waiting_rows_in_same_boundary():
    req, original = snapshot({'done': (0, 256, 5, 0)},
                             {'waiting': (0, 1, 0, 5)}, total=[5] * 8)
    req.global_seq_state['done']['completed'] = True
    original.completed_uuids = ['done']
    original.active_uuids = original.decode_uuids_final = []
    plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert plan.completed_uuids == ['done'] and plan.new_load_uuids == ['waiting']
    assert plan.decode_uuids_final == []  # load joins only AFTER actual H2D completes
    assert summary['actual_target_rows_per_rank'][0] == 1


def test_Host_growth_eviction_and_errors_keep_the_existing_transaction():
    for field, value in [('host_growth_uuids', ['a']), ('host_evicted_uuids', ['a']),
                         ('scheduler_error', 'existing capacity failure')]:
        req, original = snapshot({'a': (0, 65, 4, 1)}, {'b': (0, 1, 0, 5)})
        setattr(original, field, value)
        before = copy.deepcopy(original)
        plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
        assert plan is original and plan == before
        assert summary['skipped'] == 'existing_Host_growth_eviction_or_scheduler_error'


def test_priority_capacity_and_canonical_ties_are_deterministic():
    req, original = snapshot({'high': (0, 500, 4, 1), 'low': (0, 500, 4, 1)},
        {'new': (0, 1, 0, 5)}, free=[2] + [0] * 7, total=[10] * 8,
        priorities={'high': 1})
    plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert plan.decode_uuids_final == ['high'] and plan.new_load_uuids == ['new']
    assert plan.onhold_uuids == ['low']
    assert plan.seqs_needing_extension == ['high']
    req.global_seq_state = dict(reversed(list(req.global_seq_state.items())))
    req.global_candidate_info = dict(reversed(list(req.global_candidate_info.items())))
    again, report = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert again == plan and report == summary


def test_row_cap_and_owner_budgets_remain_independent():
    active = {f'a{i}': (0, 65, 1, 0) for i in range(2)}
    waiting = {'other-rank': (1, 1, 0, 5), 'same-rank': (0, 1, 0, 1)}
    req, original = snapshot(active, waiting, free=[0] * 8, total=[5] * 8, cap=2)
    _, report = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
    assert report['actual_target_rows_per_rank'][0] == 2
    assert report['actual_target_rows_per_rank'][1] == 0  # rank0 credits cannot fund rank1
    req.per_rank_free = (5,) * 8
    with pytest.raises(ValueError, match='exceed actual pool'):
        rotation.balance_gpu_working_set(req, original, quantum_tokens=64)


def test_progress_balancing_prevents_initially_waiting_rows_starving_for_a_full_output():
    # Fixed physical working set10, Host inventory13 per rank. Progress is
    # advanced only for selected rows; no padding/dummy row earns a token.
    all_rows = {f'{rank}-{i:02}': dict(rank=rank, progress=1)
                for rank in range(8) for i in range(13)}
    resident = {u for u in all_rows if int(u.split('-')[1]) < 10}
    first_selected = {u: 0 for u in resident}
    for boundary in range(1, 5):
        for uuid in resident:
            all_rows[uuid]['progress'] += 64
        active = {u: (r['rank'], r['progress'], 1, 0) for u, r in all_rows.items() if u in resident}
        waiting = {u: (r['rank'], r['progress'], 0, 1) for u, r in all_rows.items() if u not in resident}
        req, original = snapshot(active, waiting, total=[10] * 8, cap=10)
        plan, summary = rotation.balance_gpu_working_set(req, original, quantum_tokens=64)
        resident = set(plan.decode_uuids_final) | set(plan.new_load_uuids)
        for uuid in resident:
            first_selected.setdefault(uuid, boundary)
        assert summary['actual_target_rows_per_rank'] == [10] * 8
        assert all(used <= cap for used, cap in zip(summary['actual_target_GPU_pages_per_rank'],
                                                  summary['actual_available_GPU_pages_per_rank']))
    assert len(first_selected) == 104
    assert max(first_selected.values()) == 1  # every initial waiter served by first quantum
    progress = [r['progress'] for r in all_rows.values()]
    assert max(progress) - min(progress) <= 128
