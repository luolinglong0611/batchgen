"""Exercise the actual assignment method without importing CUDA/native modules."""
import ast
from enum import IntEnum
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sequence_tree = ast.parse((ROOT / "batchgen/sequence.py").read_text())
status_node = next(node for node in sequence_tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "SequenceStatus")
namespace = {"IntEnum": IntEnum, "List": List, "logging": logging}
exec(compile(ast.fix_missing_locations(ast.Module(body=[status_node], type_ignores=[])),
             "<actual-sequence-status>", "exec"), namespace)
Status = namespace["SequenceStatus"]
worker_tree = ast.parse((ROOT / "batchgen/batchgen_worker.py").read_text())
assignment = next(node for node in ast.walk(worker_tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_assign_admitted_sequences_to_ranks")
exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
             "<actual-worker-assignment>", "exec"), namespace)


class Batch:
    def __init__(self, sequences):
        self.sequences = {seq.uuid: seq for seq in sequences}

    def __iter__(self):
        return iter(self.sequences.values())

    def get_sequence(self, uuid):
        return self.sequences.get(uuid)

    def assign_rank(self, uuid, rank):
        self.sequences[uuid].assigned_rank = rank


def sequence(uuid, length, rank=None, status=Status.QUEUEING):
    return SimpleNamespace(uuid=uuid, prompt_length=length,
                           assigned_rank=rank, status=status)


def assign(history, *, l2=True):
    pending = [sequence(f"new-{i}", 2048) for i in range(512)]
    worker = SimpleNamespace(world_size=16, rank=0,
                             global_batch=Batch(history + pending),
                             _prefill_prefix_pool_enabled=lambda: False)
    with patch.dict("os.environ", {"BATCHGEN_L2_BALANCE": str(int(l2))}):
        namespace["_assign_admitted_sequences_to_ranks"](
            worker, [seq.uuid for seq in pending])
    counts = tuple(sum(seq.assigned_rank == rank for seq in pending) for rank in range(16))
    return counts


class AssignmentHistoryTests(unittest.TestCase):
    def test_completed_large_prompt_does_not_bias_next_l2_batch(self):
        finished = [sequence("warmup", 8192, 0, Status.COMPLETED)]
        self.assertEqual(assign(finished), (32,) * 16)
        self.assertEqual(assign(finished), assign([]))

    def test_completed_history_does_not_bias_legacy_least_count(self):
        finished = [sequence(f"old-{i}", 8192, 0, Status.COMPLETED) for i in range(100)]
        self.assertEqual(assign(finished, l2=False), (32,) * 16)

    def test_live_work_still_counts_for_l2_balance(self):
        active = [sequence("active", 8192, 0, Status.IN_DECODE)]
        counts = assign(active)
        self.assertEqual(counts[0], 17)
        self.assertEqual(counts[1:], (33,) * 15)

    def test_evicted_work_is_resumable_and_still_counts(self):
        evicted = [sequence("evicted", 8192, 0, Status.EVICTED)]
        self.assertEqual(assign(evicted), assign([
            sequence("active", 8192, 0, Status.IN_DECODE)]))

    def test_unassigned_history_does_not_consume_a_rank(self):
        self.assertEqual(assign([sequence("unassigned", 16384)]), (32,) * 16)


if __name__ == "__main__":
    unittest.main()
