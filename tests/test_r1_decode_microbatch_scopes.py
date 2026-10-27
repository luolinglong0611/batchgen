"""Independent CPU contracts for declared MLA row slices and shared plans.

No Torch import: versioned alias geometry and scalar/plan counters model the
contracts; real CUDA numerical parity belongs to the separate GPU fixture.
"""
import contextlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "r1_microbatch_test_fixture", ROOT / "tests/test_r1_fa2_forward_scope.py")
fixture_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture_module
spec.loader.exec_module(fixture_module)
backend = fixture_module.backend
positions = importlib.import_module(fixture_module.PACKAGE + ".decode_position_validation")
micro = importlib.import_module(fixture_module.PACKAGE + ".decode_microbatch_scope")


class Tensor(fixture_module.Tensor):
    runtime = None

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.origin.pointer)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key += (slice(None),) * (self.ndim - len(key))
        shape, strides, offset = [], [], self.offset
        for size, stride, part in zip(self.shape, self._strides, key):
            if isinstance(part, int):
                offset += part * stride
                continue
            start, stop, step = part.indices(size)
            offset += start * stride
            shape.append(len(range(start, stop, step)))
            strides.append(stride * step)
        data = self.data[key[0]] if self.data is not None else None
        value = Tensor(shape, data=data, dtype=self.dtype, device=self.device,
                       origin=self.origin, strides=strides, label=self.label)
        value.offset = offset
        return value

    def mutate(self, data):
        self.data = data
        self.origin.version += 1

    def max(self):
        self.runtime.counts["position_max"] += 1
        values = [x[0] if isinstance(x, list) else x for x in self.data]
        maximum = max(values)
        runtime = self.runtime

        def item():
            runtime.counts["position_item"] += 1
            return maximum

        return SimpleNamespace(item=item)


class MicrobatchScopeContracts(unittest.TestCase):
    def setUp(self):
        self.runtime = fixture_module.Runtime()
        Tensor.runtime = self.runtime
        self.table = Tensor((131, 1), data=[[i] for i in range(131)], dtype="int32")
        self.lengths = Tensor((131,), data=[63] * 131, dtype="int32")
        self.pos = Tensor((131, 1), data=[[62]] * 131, dtype="int64")
        self.manager = fixture_module.Manager(self.table)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.runtime.install())
        self.stack.enter_context(patch.object(positions, "_torch", return_value=self.runtime.torch))
        self.addCleanup(self.stack.close)

    def tearDown(self):
        self.assertIsNone(positions._DECODE_POSITION_STATE.get())
        self.assertIsNone(backend._DECODE_FORWARD_STATE.get())

    @contextlib.contextmanager
    def outer(self):
        with positions.mla_decode_position_scope(self.pos) as ps:
            with backend.decode_forward_scope(self.manager) as fs:
                yield ps, fs

    def attend(self, start, end, *, table=None, lengths=None):
        with micro.decode_attention_microbatch_scope(
                self.manager, self.pos, self.lengths, start, end) as (pos, lens):
            self.assertEqual(positions.decode_position_max(pos), 62)
            query = Tensor((end - start, 1, 4, 576), label="query")
            cache = Tensor((131, 64, 1, 576), label="cache")
            return fixture_module.attend((query, cache,
                                          self.table[start:end] if table is None else table,
                                          lens if lengths is None else lengths))

    def test_64_64_3_reuse_three_plans_and_four_scalar_reads_for_61_layers(self):
        with self.outer() as (ps, fs):
            for _ in range(61):
                for start, end in ((0, 64), (64, 128), (128, 131)):
                    self.attend(start, end)
            pchildren = list(ps.slice_states.values())
            fchildren = [item[0] for item in fs.slice_states.values()]
            self.assertEqual([x.tensor.shape[0] for x in pchildren], [64, 64, 3])
        self.assertEqual(self.runtime.counts["position_max"], 4)
        self.assertEqual(self.runtime.counts["position_item"], 4)
        self.assertEqual(self.runtime.counts["plan"], 3)
        self.assertEqual(self.runtime.counts["run"], 183)
        for state in pchildren:
            self.assertIsNone(state.tensor)
            self.assertIsNone(state.maximum)
        for state in fchildren:
            self.assertIsNone(state.prepared)
            self.assertIsNone(state.manager)
            self.assertIsNone(state.length_tensor)
            self.assertIsNone(state.expected_table_view)
        self.assertFalse(ps.slice_states)
        self.assertFalse(fs.slice_states)

    def test_fullpath_still_rejects_equal_valued_alias(self):
        with self.outer():
            with self.assertRaisesRegex(ValueError, "identity"):
                positions.decode_position_max(self.pos[:])
            self.runtime.counts.clear()
            q = Tensor((131, 1, 4, 576))
            cache = Tensor((131, 64, 1, 576))
            with self.assertRaisesRegex(ValueError, "belong"):
                fixture_module.attend((q, cache, self.table[:], self.lengths))

    def test_wrong_page_slice_or_clone_rejected(self):
        for bad_table in (self.table[1:65], Tensor((64, 1), data=[[i] for i in range(64)], dtype="int32")):
            with self.subTest(offset=bad_table.offset), self.outer():
                with self.assertRaisesRegex(ValueError, "declared"):
                    self.attend(0, 64, table=bad_table)

    def test_parent_lengths_replacement_or_alias_rejected_across_chunks(self):
        with self.outer():
            self.attend(0, 64)
            self.lengths = self.lengths[:]
            with self.assertRaisesRegex(ValueError, "Full cache lengths"):
                self.attend(64, 128)

    def test_normal_table_mutation_between_different_chunks_rejected(self):
        with self.outer():
            self.attend(0, 64)
            self.table.mutate([[i + 1] for i in range(131)])
            with self.assertRaisesRegex(ValueError, "page table|page-table"):
                self.attend(64, 128)

    def test_table_identity_change_between_different_chunks_rejected(self):
        with self.outer():
            self.attend(0, 64)
            self.manager._gpu_page_table_manager.gpu_table = Tensor(
                (131, 1), data=[[i] for i in range(131)], dtype="int32")
            with self.assertRaisesRegex(ValueError, "page table|page-table"):
                self.attend(64, 128)

    def test_generation_change_rejected(self):
        with self.outer():
            self.attend(0, 64)
            self.manager.generation += 1
            with self.assertRaisesRegex(ValueError, "generation"):
                self.attend(64, 128)

    def test_position_alias_mutation_outside_current_slice_rejected(self):
        with self.assertRaisesRegex(ValueError, "version"):
            with self.outer():
                with micro.decode_attention_microbatch_scope(
                        self.manager, self.pos, self.lengths, 0, 64):
                    self.pos[128:131].mutate([[63]] * 3)

    def test_stream_switch_after_first_chunk_rejected(self):
        with self.outer():
            self.attend(0, 64)
            self.runtime.stream.cuda_stream += 1
            with self.assertRaisesRegex(ValueError, "stream"):
                self.attend(64, 128)

    def test_slice_exception_restores_parent_and_clears_children_on_exit(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with self.outer() as (ps, fs):
                self.attend(0, 64)
                with micro.decode_attention_microbatch_scope(
                        self.manager, self.pos, self.lengths, 64, 128):
                    child_pos = positions._DECODE_POSITION_STATE.get()
                    child_fa = backend._DECODE_FORWARD_STATE.get()
                    raise RuntimeError("injected child failure")
        self.assertIsNone(child_pos.tensor)
        self.assertIsNone(child_fa.manager)
        self.assertFalse(ps.slice_states)
        self.assertFalse(fs.slice_states)

    def test_initial_child_bind_failure_does_not_retain_position_tensor(self):
        original = positions._DecodePositionState.bind
        failed = []

        def bind(state, tensor):
            original(state, tensor)
            if state.is_slice:
                failed.append(state)
                raise RuntimeError("injected initial bind failure")

        with self.assertRaisesRegex(RuntimeError, "initial bind"):
            with self.outer(), patch.object(positions._DecodePositionState, "bind", bind):
                self.attend(0, 64)
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0].tensor)
        self.assertIsNone(failed[0].maximum)

    def test_bad_ranges_and_mismatched_metadata_rows_fail_before_attention(self):
        for start, end in ((-1, 64), (64, 64), (0, 132), (True, 64)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                with micro.decode_attention_microbatch_scope(
                        self.manager, self.pos, self.lengths, start, end):
                    self.fail("invalid range reached body")
        with self.assertRaisesRegex(ValueError, "rows"):
            with micro.decode_attention_microbatch_scope(
                    self.manager, self.pos[:130], self.lengths, 0, 64):
                self.fail("misaligned metadata reached body")
        self.assertEqual(self.runtime.counts["run"], 0)


if __name__ == "__main__":
    unittest.main()
