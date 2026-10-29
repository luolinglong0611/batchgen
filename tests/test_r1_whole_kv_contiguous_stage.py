"""CPU contracts for post-replay KV materialization; CUDA remains a test double."""
import ast
import gc
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/attention/mla/r1_decode_cuda_graph.py"
NAME = "_r1_whole_kv_contiguous_stage_contract"
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
candidate = importlib.util.module_from_spec(spec)
sys.modules[NAME] = candidate
spec.loader.exec_module(candidate)
ENV = "BATCHGEN_R1_WHOLE_KV_CONTIGUOUS_STAGE"


class Value:
    def __init__(self, data, *, owner=None, dtype="bfloat16", device="cuda:0", counter=None):
        self.data = data
        self.owner, self.dtype, self.device, self.counter = owner, dtype, device, counter

    @property
    def shape(self):
        return self.data.shape

    def clone(self):
        self.counter["clone"] += 1
        return Value(self.data.copy(), dtype=self.dtype, device=self.device, counter=self.counter)

    def __getitem__(self, index):
        return Value(self.data[index], owner=self, dtype=self.dtype, device=self.device, counter=self.counter)


def runtime():
    counts = {"clone": 0, "stack": 0}
    state = {"capturing": False}

    def stack(values, dim):
        assert dim == 0 and not state["capturing"]
        counts["stack"] += 1
        return Value(np.stack([x.data for x in values], axis=dim),
                     dtype=values[0].dtype, device=values[0].device, counter=counts)

    torch = SimpleNamespace(stack=stack, cuda=SimpleNamespace(
        is_current_stream_capturing=lambda: state["capturing"]))
    return torch, counts, state


class WholeKVStagingContracts(unittest.TestCase):
    def make_outputs(self, enabled, layers=(0, 1, 2), *, rows=3, strided=False):
        torch, counts, state = runtime()
        with patch.dict(os.environ, {ENV: str(int(enabled))}):
            outputs = candidate.WholeModelKVOutputs(layers, capture_stream=17)
        for layer in reversed(layers):
            data = (np.arange(rows * 576 * (2 if strided else 1), dtype=np.uint16)
                    .reshape(rows, 1, 1, -1) + layer * 3)
            if strided:
                data = data[..., ::2]
            outputs.record(layer, Value(data, counter=counts))
        return outputs, torch, counts, state

    def test_default_off_and_invalid_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(candidate.whole_kv_contiguous_stage_requested())
            self.assertFalse(candidate.WholeModelKVOutputs([0], 17).contiguous_staging)
        for value in ("", "true", "2", " 1", "01"):
            with self.subTest(value=value), patch.dict(os.environ, {ENV: value}):
                with self.assertRaises(ValueError):
                    candidate.WholeModelKVOutputs([0], 17)

    def test_default_preserves_per_layer_clone_and_callback_shape(self):
        outputs, torch, counts, _ = self.make_outputs(False, layers=tuple(range(61)))
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual(counts, {"clone": 61, "stack": 0})
        self.assertEqual([x[0] for x in calls], list(range(61)))
        self.assertTrue(all(x[1].shape == (3, 1, 1, 576) and x[2] is None for x in calls))

    def test_all61_layers_exact_bytes_one_allocation_and_immutable_sources(self):
        outputs, torch, counts, _ = self.make_outputs(True, layers=tuple(range(61)))
        original = {layer: value.data.copy() for layer, value in outputs.kv_by_layer.items()}
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual(counts, {"clone": 0, "stack": 1})
        self.assertEqual([x[0] for x in calls], list(range(61)))
        self.assertEqual(len({id(x[1].owner) for x in calls}), 1)
        for layer, staged, value in calls:
            self.assertIsNone(value)
            self.assertEqual(staged.data.tobytes(), original[layer].tobytes())
            self.assertFalse(np.shares_memory(staged.data, outputs.kv_by_layer[layer].data))
            self.assertEqual(outputs.kv_by_layer[layer].data.tobytes(), original[layer].tobytes())

    def test_deferred_views_retain_shared_allocation_until_last_task_releases(self):
        outputs, torch, _, _ = self.make_outputs(True)
        deferred = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda layer, tensor, value: deferred.append(tensor), replay_ordinal=1)
        allocation = weakref.ref(deferred[0].owner)
        gc.collect()
        self.assertIsNotNone(allocation())
        survivor = deferred.pop()
        deferred.clear()
        gc.collect()
        self.assertIsNotNone(allocation())
        del survivor
        gc.collect()
        self.assertIsNone(allocation())

    def test_later_replay_cannot_overwrite_prior_pending_host_sources(self):
        outputs, torch, counts, _ = self.make_outputs(True)
        prior, later = [], []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: prior.append(args), replay_ordinal=1)
            before = [value.data.tobytes() for _, value, _ in prior]
            for value in outputs.kv_by_layer.values():
                value.data.fill(99)
            outputs.stage_after_replay(lambda *args: later.append(args), replay_ordinal=2)
        self.assertEqual(counts["stack"], 2)
        self.assertEqual([value.data.tobytes() for _, value, _ in prior], before)
        self.assertTrue(all(np.all(value.data == 99) for _, value, _ in later))
        self.assertIsNot(prior[0][1].owner, later[0][1].owner)

    def test_noncontiguous_layer_outputs_materialize_exactly(self):
        outputs, torch, _, _ = self.make_outputs(True, layers=(5, 0, 8), strided=True)
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual([x[0] for x in calls], [5, 0, 8])
        for layer, value, _ in calls:
            self.assertTrue(value.data.flags.c_contiguous)
            self.assertEqual(value.data.tobytes(), outputs.kv_by_layer[layer].data.tobytes())

    def test_empty_ghost_rank_does_not_stack_or_publish(self):
        outputs, torch, counts, _ = self.make_outputs(True, layers=())
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual((counts, calls, outputs.last_staged_replay), ({"clone": 0, "stack": 0}, [], 1))

    def test_capture_rejected_before_any_materialization_or_append(self):
        outputs, torch, counts, state = self.make_outputs(True)
        state["capturing"] = True
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaises(RuntimeError):
                outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual((counts["stack"], calls, outputs.last_staged_replay), (0, [], 0))

    def test_duplicate_and_invalid_ordinal_never_publish_twice(self):
        outputs, torch, counts, _ = self.make_outputs(True)
        calls = []
        with patch.dict(sys.modules, {"torch": torch}):
            outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=3)
            for ordinal in (0, 1, 3, True, 4.0):
                with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                    outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=ordinal)
        self.assertEqual((counts["stack"], len(calls)), (1, 3))

    def test_missing_or_mismatched_layers_fail_before_callbacks(self):
        for kind in ("missing", "shape", "dtype", "device"):
            with self.subTest(kind=kind):
                outputs, torch, counts, _ = self.make_outputs(True)
                if kind == "missing":
                    del outputs.kv_by_layer[1]
                elif kind == "shape":
                    outputs.kv_by_layer[1].data = outputs.kv_by_layer[1].data[:1]
                else:
                    setattr(outputs.kv_by_layer[1], kind, "different")
                calls = []
                with patch.dict(sys.modules, {"torch": torch}), self.assertRaises(ValueError):
                    outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
                self.assertEqual((counts["stack"], calls, outputs.last_staged_replay), (0, [], 0))

    def test_staging_adds_no_stream_switch_host_wait_or_scalar_readback(self):
        tree = ast.parse(SOURCE.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WholeModelKVOutputs")
        stage = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "stage_after_replay")
        attributes = {n.func.attr for n in ast.walk(stage)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(attributes & {"stream", "synchronize", "item", "tolist", "cpu", "wait_event"})
        self.assertEqual(attributes & {"stack", "clone"}, {"stack", "clone"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
