"""Position-scope lifecycle contracts without importing Torch or model code."""
from collections import Counter
import gc
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mla_decode_position_validation_contract",
    ROOT / "batchgen/attention/mla/decode_position_validation.py",
)
positions = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = positions
spec.loader.exec_module(positions)


class Device:
    def __init__(self, kind="cuda", index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    """Shared version counters model normal Torch tensor/view invalidation."""
    def __init__(self, values, runtime, *, origin=None, version=0):
        self.values = list(values)
        self.runtime = runtime
        self.shape = (len(values), 1)
        self.strides = (1, 1)
        self.offset = 0
        self.layout = "strided"
        self.dtype = "int64"
        self.device = Device()
        self.origin = origin if origin is not None else self
        self.version = version

    @property
    def _version(self):
        if self.origin.version is None:
            raise RuntimeError("Inference tensors do not track version counter")
        return self.origin.version

    def stride(self):
        return self.strides

    def storage_offset(self):
        return self.offset

    def numel(self):
        return math.prod(self.shape)

    def max(self):
        self.runtime.counts["max"] += 1
        if self.runtime.fail_max:
            raise RuntimeError("injected maximum failure")
        # Just like Torch's reduction, an empty input has no maximum.
        value = max(self.values)
        runtime = self.runtime

        class Scalar:
            def item(self):
                runtime.counts["item"] += 1
                if runtime.fail_item:
                    raise RuntimeError("injected scalar readback failure")
                if runtime.on_item is not None:
                    runtime.on_item()
                return value

        return Scalar()

    def mutate(self, values):
        self.values[:] = values
        if self.origin.version is not None:
            self.origin.version += 1

    def view(self):
        result = Tensor(self.values, self.runtime, origin=self.origin)
        result.values = self.values
        return result


class Runtime:
    def __init__(self):
        self.counts = Counter()
        self.stream = 101
        self.capturing = False
        self.fail_max = False
        self.fail_item = False
        self.on_item = None
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: self.capturing,
                current_stream=lambda device: SimpleNamespace(cuda_stream=self.stream),
            ),
        )

    def tensor(self, values=(8191, 8195, 8193), **kwargs):
        return Tensor(values, self, **kwargs)


class DecodePositionScopeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.torch_patch = patch.object(positions, "_torch", return_value=self.runtime.torch)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def tearDown(self):
        self.assertIsNone(positions._DECODE_POSITION_STATE.get())

    def test_61_layers_read_once_at_entry_before_projections(self):
        tensor = self.runtime.tensor()
        with positions.mla_decode_position_scope(tensor):
            self.assertEqual(self.runtime.counts, {"max": 1, "item": 1})
            for layer in range(61):
                self.runtime.counts["projection"] += 1
                self.assertEqual(positions.decode_position_max(tensor), 8195, layer)
        self.assertEqual(self.runtime.counts, {"max": 1, "item": 1, "projection": 61})

    def test_next_step_reads_new_values_in_same_tensor(self):
        tensor = self.runtime.tensor()
        for step in range(3):
            tensor.mutate([8191 + step, 8195 + step, 8193 + step])
            with positions.mla_decode_position_scope(tensor):
                self.assertEqual(positions.decode_position_max(tensor), 8195 + step)
        self.assertEqual(self.runtime.counts, {"max": 3, "item": 3})

    def test_later_layer_still_rejects_its_shorter_rotary_cache(self):
        tensor = self.runtime.tensor()

        def layer(cos_sequence_length):
            if positions.decode_position_max(tensor) >= cos_sequence_length:
                raise ValueError("position exceeds this layer's cosine cache")

        with positions.mla_decode_position_scope(tensor):
            layer(16384)
            layer(8196)
            with self.assertRaisesRegex(ValueError, "cosine cache"):
                layer(8195)
        self.assertEqual(self.runtime.counts, {"max": 1, "item": 1})

    def test_unscoped_calls_keep_fresh_reduction_and_values(self):
        tensor = self.runtime.tensor()
        self.assertEqual(positions.decode_position_max(tensor), 8195)
        tensor.mutate([10, 20, 30])
        self.assertEqual(positions.decode_position_max(tensor), 30)
        self.assertEqual(self.runtime.counts, {"max": 2, "item": 2})

    def test_equal_valued_replacement_tensor_is_rejected(self):
        tensor = self.runtime.tensor()
        replacement = self.runtime.tensor()
        with positions.mla_decode_position_scope(tensor):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                positions.decode_position_max(replacement)

    def test_in_place_mutation_is_rejected_without_second_readback(self):
        tensor = self.runtime.tensor()
        with positions.mla_decode_position_scope(tensor):
            tensor.mutate([10, 20, 30])
            with self.assertRaisesRegex(ValueError, "version changed"):
                positions.decode_position_max(tensor)
        self.assertEqual(self.runtime.counts, {"max": 1, "item": 1})

    def test_same_data_restored_still_invalidates_version(self):
        tensor = self.runtime.tensor()
        initial = list(tensor.values)
        with positions.mla_decode_position_scope(tensor):
            tensor.mutate([1, 2, 3])
            tensor.mutate(initial)
            with self.assertRaisesRegex(ValueError, "version changed"):
                positions.decode_position_max(tensor)

    def test_view_replacement_is_rejected(self):
        tensor = self.runtime.tensor()
        view = tensor.view()
        with positions.mla_decode_position_scope(tensor):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                positions.decode_position_max(view)

    def test_mutation_through_alias_is_rejected(self):
        tensor = self.runtime.tensor()
        view = tensor.view()
        with positions.mla_decode_position_scope(tensor):
            view.mutate([1, 2, 3])
            with self.assertRaisesRegex(ValueError, "version changed"):
                positions.decode_position_max(tensor)

    def test_normal_versioned_view_can_be_scope_owner(self):
        tensor = self.runtime.tensor()
        view = tensor.view()
        with positions.mla_decode_position_scope(view):
            self.assertEqual(positions.decode_position_max(view), 8195)
            tensor.mutate([1, 2, 3])
            with self.assertRaisesRegex(ValueError, "version changed"):
                positions.decode_position_max(view)

    def test_geometry_dtype_device_and_offset_changes_are_rejected(self):
        mutations = {
            "shape": (1, 3), "strides": (3, 1), "offset": 1,
            "layout": "sparse", "dtype": "int32", "device": Device(index=1),
        }
        for attribute, value in mutations.items():
            with self.subTest(attribute=attribute):
                tensor = self.runtime.tensor()
                with positions.mla_decode_position_scope(tensor):
                    setattr(tensor, attribute, value)
                    with self.assertRaisesRegex(ValueError, "metadata or version changed"):
                        positions.decode_position_max(tensor)

    def test_changed_stream_is_rejected(self):
        tensor = self.runtime.tensor()
        with positions.mla_decode_position_scope(tensor):
            self.runtime.stream += 1
            with self.assertRaisesRegex(ValueError, "stream changed"):
                positions.decode_position_max(tensor)

    def test_nested_scope_rejected_without_disturbing_outer(self):
        tensor = self.runtime.tensor()
        with positions.mla_decode_position_scope(tensor):
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with positions.mla_decode_position_scope(tensor):
                    self.fail("nested scope entered")
            self.assertEqual(positions.decode_position_max(tensor), 8195)
        self.assertEqual(self.runtime.counts, {"max": 1, "item": 1})

    def test_state_clears_references_and_scalar_after_body_exception(self):
        tensor = self.runtime.tensor()
        reference = weakref.ref(tensor)
        with self.assertRaisesRegex(RuntimeError, "layer failed"):
            with positions.mla_decode_position_scope(tensor) as state:
                raise RuntimeError("layer failed")
        self.assertTrue(all(value is None for value in vars(state).values()))
        del tensor
        gc.collect()
        self.assertIsNone(reference())
        next_tensor = self.runtime.tensor([5])
        with positions.mla_decode_position_scope(next_tensor):
            self.assertEqual(positions.decode_position_max(next_tensor), 5)

    def test_state_clears_references_after_normal_exit(self):
        tensor = self.runtime.tensor()
        reference = weakref.ref(tensor)
        with positions.mla_decode_position_scope(tensor) as state:
            self.assertIs(state.tensor, tensor)
        self.assertTrue(all(value is None for value in vars(state).values()))
        del tensor
        gc.collect()
        self.assertIsNone(reference())

    def test_scope_holds_strong_reference_until_exit(self):
        tensor = self.runtime.tensor()
        reference = weakref.ref(tensor)
        with positions.mla_decode_position_scope(tensor):
            del tensor
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertEqual(positions.decode_position_max(reference()), 8195)
        gc.collect()
        self.assertIsNone(reference())

    def test_failed_entry_cleans_state_and_allows_new_forward(self):
        tensor = self.runtime.tensor()
        for flag in ("fail_max", "fail_item"):
            with self.subTest(flag=flag):
                setattr(self.runtime, flag, True)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    with positions.mla_decode_position_scope(tensor):
                        self.fail("failed readback entered scope")
                self.assertIsNone(positions._DECODE_POSITION_STATE.get())
                setattr(self.runtime, flag, False)
                with positions.mla_decode_position_scope(tensor):
                    self.assertEqual(positions.decode_position_max(tensor), 8195)

    def test_mutation_during_entry_readback_is_rejected(self):
        tensor = self.runtime.tensor()
        self.runtime.on_item = lambda: tensor.mutate([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "version changed"):
            with positions.mla_decode_position_scope(tensor):
                self.fail("changed positions entered scope")

    def test_empty_batch_enters_without_reduction(self):
        tensor = self.runtime.tensor([])
        with positions.mla_decode_position_scope(tensor):
            # The real attention returns early for this empty batch.
            self.assertEqual(self.runtime.counts, {})
        self.assertEqual(self.runtime.counts, {})

    def test_empty_helper_fails_without_inventing_maximum(self):
        tensor = self.runtime.tensor([])
        with positions.mla_decode_position_scope(tensor):
            with self.assertRaisesRegex(ValueError, "Empty MLA positions"):
                positions.decode_position_max(tensor)
        self.assertEqual(self.runtime.counts, {})

    def test_inference_tensor_scope_rejected_but_unscoped_path_preserved(self):
        tensor = self.runtime.tensor(version=None)
        with self.assertRaisesRegex(RuntimeError, "normal versioned tensor"):
            with positions.mla_decode_position_scope(tensor):
                self.fail("unversioned tensor entered scope")
        self.assertEqual(self.runtime.counts, {})
        self.assertEqual(positions.decode_position_max(tensor), 8195)
        self.assertEqual(self.runtime.counts, {"max": 1, "item": 1})

    def test_capture_rejected_at_entry_and_on_later_layer(self):
        tensor = self.runtime.tensor()
        self.runtime.capturing = True
        with self.assertRaisesRegex(ValueError, "eager CUDA"):
            with positions.mla_decode_position_scope(tensor):
                self.fail("capture entered scope")
        self.assertEqual(self.runtime.counts, {})
        self.runtime.capturing = False
        with positions.mla_decode_position_scope(tensor):
            self.runtime.capturing = True
            with self.assertRaisesRegex(ValueError, "eager CUDA"):
                positions.decode_position_max(tensor)

    def test_cpu_scope_rejected_but_unscoped_path_preserved(self):
        tensor = self.runtime.tensor()
        tensor.device = Device("cpu")
        with self.assertRaisesRegex(ValueError, "eager CUDA"):
            with positions.mla_decode_position_scope(tensor):
                self.fail("CPU tensor entered scope")
        self.assertEqual(self.runtime.counts, {})
        self.assertEqual(positions.decode_position_max(tensor), 8195)

    def test_negative_positions_preserve_existing_max_semantics(self):
        # This optimization must not silently add a new minimum-position check.
        tensor = self.runtime.tensor([-3, -1, -2])
        with positions.mla_decode_position_scope(tensor):
            self.assertEqual(positions.decode_position_max(tensor), -1)


if __name__ == "__main__":
    unittest.main()
