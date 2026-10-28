"""Candidate CPU contracts; no Torch, CUDA, model execution, or remote actions.

Real pure metadata planner + real candidate replay/callback routing execute;
accelerator/FI APIs are strict test doubles at their public boundaries.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "batchgen/attention/mla/r1_decode_cuda_graph.py"
NAME = "batchgen.attention.mla.r1_decode_cuda_graph"
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
candidate = importlib.util.module_from_spec(spec)
sys.modules[NAME] = candidate
spec.loader.exec_module(candidate)


class Value:
    def __init__(self, shape, *, dtype="bf16", device="cuda:0", data=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim = len(self.shape)
        self.data = data
        self.version = 0

    def data_ptr(self):
        return id(self)

    def clone(self):
        return Value(self.shape, dtype=self.dtype, device=self.device, data=self.version)

    def copy_(self, other, **kwargs):
        self.data = other
        return self


class FakeRecord:
    created = []

    def __init__(self, wrapper, hidden, batch, blocked, signature):
        self.signature, self.replays = signature, 0
        self.output = Value(hidden.shape)
        self.kv = Value((batch.rows, 1, 1, 576))
        self.created.append(self)

    def replay(self, hidden):
        self.replays += 1
        self.kv.version = self.replays
        return self.output, self.kv


class GraphMetadataContracts(unittest.TestCase):
    def test_fast_reservation_cache_reuses_and_invalidates_on_page_changes(self):
        class Pages:
            def __init__(self, values):
                self.values = values
                self.shape, self._version, self.reads = (len(values),), 0, 0
            def data_ptr(self):
                return id(self)
            def tolist(self):
                self.reads += 1
                return list(self.values)
        a, b = Pages([3, 9, 4]), Pages([8, 2, 7])
        manager = SimpleNamespace(_gpu_page_table_manager=object(),
            _sequences={91: SimpleNamespace(pages=a), 7: SimpleNamespace(pages=b)},
            get_page_table_version=lambda: 5)
        cache = candidate.MetadataPlanCache()
        first = cache.prepare(manager, [91, 7], [65, 127], context_cap=256, gpu_page_count=16)
        second = cache.prepare(manager, [91, 7], [66, 128], context_cap=256, gpu_page_count=16)
        self.assertIs(first.reservation_table, second.reservation_table)
        self.assertEqual((a.reads, b.reads, cache.hits, cache.rebuilds), (1, 1, 1, 1))
        self.assertEqual(second.small_arrays()["positions"].tolist(), [[65], [127]])
        # Only the small metadata builder is needed: token indices will be
        # generated on GPU from the full validated reservation table.
        self.assertNotIn("kv_indices", second.small_arrays())
        self.assertEqual(second.reservation_table.tolist(), [[3, 9, 4, -1], [8, 2, 7, -1]])
        a.values[0] = 1
        a._version += 1
        changed = cache.prepare(manager, [91, 7], [67, 128], context_cap=256, gpu_page_count=16)
        self.assertIsNot(changed.reservation_table, first.reservation_table)
        self.assertEqual(changed.pages[0], (1, 9))
        with self.assertRaises(ValueError):
            cache.prepare(manager, [91, 7], [193, 128], context_cap=256, gpu_page_count=16)
        # Allocator epoch change forces revalidation even with stable pointers.
        manager.get_page_table_version = lambda: 6
        cache.prepare(manager, [91, 7], [67, 128], context_cap=256, gpu_page_count=16)
        self.assertEqual(cache.rebuilds, 3)

    def test_nonmonotonic_rows_partial_page_and_remapped_working_set(self):
        plan = candidate.plan_metadata([91, 7], [65, 127], {91: [3, 9, 4], 7: [8, 2]},
                                        context_cap=256, gpu_page_count=16)
        arrays = plan.arrays()
        self.assertEqual(plan.positions, (64, 126))
        self.assertEqual(plan.kv_indptr, (0, 65, 192))
        self.assertEqual(arrays["page_table"].tolist(), [[3, 9, -1, -1], [8, 2, -1, -1]])
        expected = list(range(3*64, 4*64)) + [9*64] + list(range(8*64, 9*64)) + list(range(2*64, 3*64-1))
        self.assertEqual(arrays["kv_indices"].tolist(), expected)
        # Same IDs, new GPU physical pages: no stale address from the first
        # batch may survive the next production CPU metadata plan.
        remapped = candidate.plan_metadata([91, 7], [66, 128], {91: [1, 5], 7: [6, 0]},
                                            context_cap=256, gpu_page_count=16).arrays()
        self.assertEqual(remapped["kv_indices"][:64].tolist(), list(range(64, 128)))
        self.assertEqual(remapped["kv_indices"][64:66].tolist(), [320, 321])
        self.assertEqual(remapped["positions"].tolist(), [[65], [127]])

    def test_capacity_and_ownership_failures_before_staging(self):
        bad = [
            ([1, 1], [2, 2], {1: [0]}, 256, 4),
            ([1], [], {1: [0]}, 256, 4),
            ([1], [0], {1: [0]}, 256, 4),
            ([1], [257], {1: [0]*5}, 256, 4),
            ([1], [65], {1: [0]}, 256, 4),
            ([1], [1], {1: [-1]}, 256, 4),
            ([1], [1], {1: [4]}, 256, 4),
            ([1], [1], {1: [0]}, 256, 2**31),
        ]
        for gids, lengths, pages, cap, total in bad:
            with self.subTest(gids=gids, lengths=lengths, cap=cap, total=total):
                with self.assertRaises(ValueError):
                    candidate.plan_metadata(gids, lengths, pages, context_cap=cap, gpu_page_count=total)

    def test_empty_tail_and_microbatch_rows(self):
        self.assertEqual(candidate.slice_ranges(0, 64), ())
        self.assertEqual(candidate.slice_ranges(131, 64), ((0, 64), (64, 128), (128, 131)))
        self.assertEqual(candidate.slice_ranges(2, 1), ((0, 1), (1, 2)))
        self.assertEqual(candidate.slice_ranges(2, 0), ((0, 2),))
        empty = candidate.plan_metadata([], [], {}, context_cap=256, gpu_page_count=4).arrays()
        self.assertEqual(empty["page_table"].shape, (0, 4))
        self.assertEqual(empty["kv_indptr"].tolist(), [0])

    def test_opt_in_guard_does_not_change_eager_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(candidate.requested())
            self.assertFalse(candidate.validate_profile(model_identifier="other", enabled=False, disabled=True))
        env = {candidate.ENVIRONMENT: "1", "BATCHGEN_R1_SM103_FP8_COMPAT": "1",
               "BATCHGEN_R1_SM103_MLA_COMPAT": "flashinfer-fa2",
               "BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP": "1"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(candidate.validate_profile(model_identifier="deepseek-ai/DeepSeek-R1", enabled=True, disabled=False))
            for enabled, disabled in [(False, True), (False, False), (True, True)]:
                with self.assertRaises(ValueError):
                    candidate.validate_profile(model_identifier="deepseek-ai/DeepSeek-R1", enabled=enabled, disabled=disabled)


class FIWrapperBoundaryContracts(unittest.TestCase):
    def test_persistent_graph_buffers_and_full_plan_identity(self):
        class Wrapper:
            def __init__(self, workspace, **kwargs):
                self.kwargs = kwargs
                self._backend, self._use_cuda_graph = kwargs["backend"], kwargs["use_cuda_graph"]
                self.calls = []
            def plan(self, *args):
                self.calls.append(args)
                self._plan_info = [len(args[3]), len(args[2]), 17]
        torch = SimpleNamespace(uint8="uint8", int32=np.int32, bfloat16="bf16",
                                empty=lambda *shape, **kw: Value(shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape, dtype=kw.get("dtype")),
                                arange=lambda count, **kw: np.arange(count, dtype=np.int32),
                                from_numpy=lambda value: value)
        backend = SimpleNamespace(_flashinfer=lambda: SimpleNamespace(BatchMLAPagedAttentionWrapper=Wrapper), WORKSPACE_BYTES=128*1024**2)
        with patch.dict(sys.modules, {"torch": torch, "batchgen.attention.mla.common_flashinfer_backend": backend}):
            item = candidate._AttentionPlan(2, 256, 128, 0.1, "cuda:0")
            pointers = tuple(id(item.wrapper.kwargs[key]) for key in ("qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr"))
            a = candidate.plan_metadata([1, 2], [65, 127], {1: [3, 9], 2: [8, 2]}, context_cap=256, gpu_page_count=16).arrays()
            item.prepare(a, 0, 2)
            first = item.signature
            b = candidate.plan_metadata([1, 2], [66, 128], {1: [1, 5], 2: [6, 0]}, context_cap=256, gpu_page_count=16).arrays()
            item.prepare(b, 0, 2)
        self.assertTrue(item.wrapper.kwargs["use_cuda_graph"])
        self.assertEqual(item.wrapper.kwargs["backend"], "fa2")
        self.assertEqual(pointers, tuple(id(item.wrapper.kwargs[key]) for key in ("qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr")))
        self.assertNotEqual(first, item.signature)
        self.assertEqual(item.wrapper.calls[1][3].tolist(), [66, 128])
        self.assertEqual(item.wrapper.calls[1][2][:64].tolist(), list(range(64, 128)))
        # Retain the per-forward CPU storage without copying its full token
        # page list again. A later remap must still replace the earlier view.
        self.assertTrue(np.shares_memory(item.cpu_keepalive[2], b["kv_indices"]))
        self.assertTrue(np.shares_memory(item.cpu_keepalive[3], b["lengths"]))
        self.assertFalse(np.shares_memory(item.cpu_keepalive[2], a["kv_indices"]))

    def test_borrowed_subbatch_indices_keep_exact_offsets_after_parent_release(self):
        class Wrapper:
            def __init__(self, workspace, **kwargs):
                self._backend, self._use_cuda_graph = kwargs["backend"], kwargs["use_cuda_graph"]
            def plan(self, *args):
                self._plan_info = [len(args[3]), len(args[2]), 17]
        torch = SimpleNamespace(uint8="uint8", int32=np.int32, bfloat16="bf16",
                                empty=lambda *shape, **kw: Value(shape, dtype=kw.get("dtype")),
                                arange=lambda count, **kw: np.arange(count, dtype=np.int32),
                                from_numpy=lambda value: value)
        backend = SimpleNamespace(_flashinfer=lambda: SimpleNamespace(BatchMLAPagedAttentionWrapper=Wrapper), WORKSPACE_BYTES=128*1024**2)
        with patch.dict(sys.modules, {"torch": torch, "batchgen.attention.mla.common_flashinfer_backend": backend}):
            item = candidate._AttentionPlan(1, 256, 128, 0.1, "cuda:0")
            arrays = candidate.plan_metadata([1, 2], [65, 127], {1: [3, 9], 2: [8, 2]},
                                             context_cap=256, gpu_page_count=16).arrays()
            item.prepare(arrays, 1, 2)
            self.assertTrue(np.shares_memory(item.cpu_keepalive[2], arrays["kv_indices"]))
            del arrays
        self.assertEqual(item.cpu_keepalive[1].tolist(), [0, 127])
        self.assertEqual(item.cpu_keepalive[2].tolist(), list(range(8*64, 9*64)) + list(range(2*64, 3*64-1)))
        self.assertEqual(item.cpu_keepalive[3].tolist(), [127])


class ReplayHostContracts(unittest.TestCase):
    def setup_state(self, *, rows=2):
        batch = SimpleNamespace(rows=rows, stream=7, ranges=((0, 1), (1, 2)), plan_identity=((1, 2, 3),), micro_batch_size=1)
        blocked = Value((20, 64, 1, 576))
        manager = SimpleNamespace(get_layer_kv_with_page_table=lambda layer: (blocked, None, None))
        hidden = Value((rows, 1, 4))
        wrapper = SimpleNamespace(layer_idx=3, decode_attn_micro_batch_size=1)
        calls = []
        state = SimpleNamespace(kv_append_callback=lambda *args: calls.append(args))
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            current_stream=lambda device: SimpleNamespace(cuda_stream=7)))
        return batch, manager, hidden, wrapper, calls, state, torch

    def test_each_real_replay_appends_one_independent_merged_kv(self):
        batch, manager, hidden, wrapper, calls, state, torch = self.setup_state()
        FakeRecord.created = []
        token = candidate._FORWARD.set((batch, manager))
        try:
            with patch.dict(sys.modules, {"torch": torch, "batchgen.models.wrappers.attention": SimpleNamespace(AttnWrapperBase=state)}), patch.object(candidate, "_LayerGraph", FakeRecord), patch.object(candidate, "attention_source_signature", return_value=("testweights",), create=True):
                candidate.run_attention_graph(wrapper, hidden)
                candidate.run_attention_graph(wrapper, hidden)
                self.assertEqual(len(FakeRecord.created), 1)
                batch.plan_identity = ((1, 2, 4),)
                candidate.run_attention_graph(wrapper, hidden)
                self.assertEqual(len(FakeRecord.created), 2)
        finally:
            candidate._FORWARD.reset(token)
        self.assertEqual(len(calls), 3)
        self.assertEqual([args[1].data for args in calls], [1, 2, 1])
        self.assertEqual([args[1].shape for args in calls], [(2, 1, 1, 576)] * 3)
        self.assertTrue(all(args[0] == 3 and args[2] is None for args in calls))
        self.assertEqual(len({id(args[1]) for args in calls}), 3)

    def test_empty_rank_and_nonowning_stream_do_not_publish_kv(self):
        batch, manager, hidden, wrapper, calls, state, torch = self.setup_state(rows=0)
        token = candidate._FORWARD.set((None, manager))
        try:
            with patch.dict(sys.modules, {"torch": torch, "batchgen.models.wrappers.attention": SimpleNamespace(AttnWrapperBase=state)}):
                self.assertIs(candidate.run_attention_graph(wrapper, hidden)[0], hidden)
        finally:
            candidate._FORWARD.reset(token)
        self.assertEqual(calls, [])
        batch, manager, hidden, wrapper, calls, state, torch = self.setup_state()
        torch.cuda.current_stream = lambda device: SimpleNamespace(cuda_stream=99)
        token = candidate._FORWARD.set((batch, manager))
        try:
            with patch.dict(sys.modules, {"torch": torch, "batchgen.models.wrappers.attention": SimpleNamespace(AttnWrapperBase=state)}):
                with self.assertRaises(RuntimeError):
                    candidate.run_attention_graph(wrapper, hidden)
        finally:
            candidate._FORWARD.reset(token)
        self.assertEqual(calls, [])

    def test_capture_compute_has_no_host_or_scalar_readback_operations(self):
        tree = ast.parse(SOURCE.read_text())
        body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_attention")
        forbidden = {"item", "tolist", "cpu", "synchronize", "plan", "kv_append_callback"}
        operations = {node.func.attr for node in ast.walk(body) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertEqual(forbidden & operations, set())

    def test_whole_model_capture_collects_static_kv_and_stages_only_after_replay(self):
        batch, manager, hidden, wrapper, calls, state, torch = self.setup_state()
        torch.cuda.is_current_stream_capturing = lambda: True
        captured_kv = Value((2, 1, 1, 576))
        outputs = candidate.WholeModelKVOutputs([3], capture_stream=7)
        token = candidate._FORWARD.set((batch, manager))
        try:
            with patch.dict(sys.modules, {"torch": torch, "batchgen.models.wrappers.attention": SimpleNamespace(AttnWrapperBase=state)}), patch.object(candidate, "compute_attention", return_value=(hidden, captured_kv)):
                with candidate.whole_model_attention_scope(outputs):
                    self.assertIs(candidate.run_attention_graph(wrapper, hidden)[0], hidden)
                self.assertEqual(calls, [])
                with self.assertRaises(RuntimeError):
                    outputs.stage_after_replay(state.kv_append_callback, replay_ordinal=1)
                torch.cuda.is_current_stream_capturing = lambda: False
                captured_kv.version = 7
                outputs.stage_after_replay(state.kv_append_callback, replay_ordinal=1)
                with self.assertRaises(ValueError):
                    outputs.stage_after_replay(state.kv_append_callback, replay_ordinal=1)
                captured_kv.version = 8
                outputs.stage_after_replay(state.kv_append_callback, replay_ordinal=2)
        finally:
            candidate._FORWARD.reset(token)
        self.assertEqual([call[1].data for call in calls], [7, 8])

    def test_missing_whole_model_layers_fail_before_any_host_append(self):
        outputs = candidate.WholeModelKVOutputs([0, 1], capture_stream=7)
        outputs.record(0, Value((2, 1, 1, 576)))
        calls = []
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaises(ValueError):
                outputs.stage_after_replay(lambda *args: calls.append(args), replay_ordinal=1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
