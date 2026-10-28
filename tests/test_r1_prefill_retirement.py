"""Run real candidate offload/retirement methods on strict CPU test doubles."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WRAPPER = ROOT / "batchgen/models/deepseek/deepseekv3/wrappers.py"
BASE = ROOT / "batchgen/models/wrappers/attention.py"


def compile_methods(path, cls_name, method_names, namespace, *, base_class=False):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    if base_class:
        wrapper = ast.ClassDef(name="RealRetirement", bases=[], keywords=[], body=methods, decorator_list=[])
        methods = [wrapper]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Tensor:
    def __init__(self, rows, *, storage=None):
        self.rows = rows
        self.device = "cuda-test-double"
        self.storage = object() if storage is None else storage
    def __getitem__(self, index):
        return Tensor(len(range(self.rows)[index]), storage=self.storage)
    def unsqueeze(self, dim):
        return self
    def squeeze(self, dim):
        return self
    def to(self, device):
        return self


class Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class PrefillRetirementContracts(unittest.TestCase):
    def setup_real(self):
        ns = {"logging": logging}
        compile_methods(BASE, "AttnWrapperBase", {
            "retire_pending_prefill_offloads", "retire_pending_prefill_offloads_before_layer",
            "pin_prefill_offload_tensor", "track_prefill_offload_task",
        }, ns, base_class=True)
        state = ns["RealRetirement"]
        state.pending_prefill_offload_tasks = []
        state.pending_prefill_offload_tensors = []
        state.pending_prefill_offload_layer_idx = None
        wrapper_ns = {"AttnWrapperBase": state}
        compile_methods(WRAPPER, "DeepSeekAttnWrapper", {"_forward_prefill", "_offload_prepacked_kv"}, wrapper_ns)
        return state, wrapper_ns

    def make_wrapper(self, state, ns, *, layer=0, fail_wait=False):
        events = []
        class Task:
            def __init__(self, gid):
                self.gid = gid
            def wait(self):
                self.assert_pinned = bool(state.pending_prefill_offload_tensors)
                if not self.assert_pinned:
                    raise AssertionError("source released before native D2H completion")
                events.append(("wait", self.gid))
                if fail_wait:
                    raise RuntimeError("injected native D2H failure")
        def submit(**kwargs):
            gid = kwargs["sequence_ids"][0]
            events.append(("submit", gid, kwargs["layer_idx"], tuple(kwargs["sequence_lengths"])))
            return Task(gid)
        wrapper = SimpleNamespace(layer_idx=layer,
            prepack_cu_seqlens=[Scalar(0), Scalar(3), Scalar(5)],
            prepack_num_sequences=2, cur_batch=[32, 7],
            core_engine=SimpleNamespace(host_paged_kv_worker_view=SimpleNamespace(async_offload_layer_kv_to_host=submit)))
        wrapper._offload_prepacked_kv = lambda tensor: ns["_offload_prepacked_kv"](wrapper, tensor)
        return wrapper, events

    def test_every_submitted_task_retired_before_source_release(self):
        state, ns = self.setup_real()
        wrapper, events = self.make_wrapper(state, ns)
        source = Tensor(5)
        ns["_offload_prepacked_kv"](wrapper, source)
        self.assertEqual(events, [("submit", 32, 0, (3,)), ("submit", 7, 0, (2,))])
        self.assertEqual(len(state.pending_prefill_offload_tasks), 2)
        self.assertIs(state.pending_prefill_offload_tensors[0], source)
        self.assertEqual(state.retire_pending_prefill_offloads(reason="CPU test"), 2)
        self.assertEqual(events[-2:], [("wait", 32), ("wait", 7)])
        self.assertEqual(state.pending_prefill_offload_tasks, [])
        self.assertEqual(state.pending_prefill_offload_tensors, [])
        self.assertIsNone(state.pending_prefill_offload_layer_idx)

    def test_next_layer_prefill_cannot_compute_before_previous_native_waits(self):
        state, ns = self.setup_real()
        first, first_events = self.make_wrapper(state, ns, layer=0)
        first._offload_prepacked_kv(Tensor(5))
        second, second_events = self.make_wrapper(state, ns, layer=1)
        second.prepack_mode = True
        second.position_ids = Tensor(5)
        second.prepack_cu_seqlens = SimpleNamespace(to=lambda device: [0, 3, 5])
        # Offload needs the normal indexable cumulative-length object.
        class Cu:
            def to(self, device): return self
            def __getitem__(self, index): return [Scalar(0), Scalar(3), Scalar(5)][index]
        second.prepack_cu_seqlens = Cu()
        second.prepack_max_seqlen = 3
        second.weight_dequant_scale = {}
        def actual_prefill(*args):
            self.assertEqual(first_events[-2:], [("wait", 32), ("wait", 7)])
            self.assertFalse(state.pending_prefill_offload_tasks)
            self.assertFalse(state.pending_prefill_offload_tensors)
            return Tensor(5), Tensor(5)
        second.module = SimpleNamespace(prefill_attn_w8a16_prepacked=actual_prefill)
        result = ns["_forward_prefill"](second, Tensor(5))
        self.assertEqual(len(result), 3)
        self.assertEqual(second_events, [("submit", 32, 1, (3,)), ("submit", 7, 1, (2,))])
        self.assertEqual(state.pending_prefill_offload_layer_idx, 1)
        self.assertEqual(state.retire_pending_prefill_offloads(reason="end prefill"), 2)

    def test_native_error_propagates_without_clearing_pinned_source(self):
        state, ns = self.setup_real()
        wrapper, events = self.make_wrapper(state, ns, fail_wait=True)
        source = Tensor(5)
        wrapper._offload_prepacked_kv(source)
        with self.assertRaisesRegex(RuntimeError, "native D2H failure"):
            state.retire_pending_prefill_offloads(reason="end prefill")
        self.assertEqual(len(state.pending_prefill_offload_tasks), 2)
        self.assertIs(state.pending_prefill_offload_tensors[0], source)
        self.assertEqual(events[-1], ("wait", 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
