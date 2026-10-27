"""CPU control-flow tests; real BF16 arithmetic is qualified on CUDA separately."""

import ast
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "batchgen/attention/mla/decode_weight_cache.py"


class FakeTensor:
    next_pointer = 100

    def __init__(self, value=1, shape=(8, 4), dtype="fp8", device=None):
        FakeTensor.next_pointer += 100
        self.storage = SimpleNamespace(pointer=FakeTensor.next_pointer, value=value)
        self.shape = shape
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.strides = (shape[1], 1)
        self.offset = 0
        self.version = 0
        self.versioned = True

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def _version(self):
        if not self.versioned:
            raise RuntimeError("Inference tensors do not track version counter")
        return self.version

    @property
    def data(self):
        return self

    @data.setter
    def data(self, other):
        self.storage = other.storage
        self.shape, self.strides = other.shape, other.strides
        self.dtype, self.device = other.dtype, other.device

    def data_ptr(self):
        return self.storage.pointer + self.offset

    def stride(self):
        return self.strides

    def storage_offset(self):
        return self.offset

    def detach(self):
        alias = FakeTensor(shape=self.shape, dtype=self.dtype, device=self.device)
        alias.storage = self.storage
        alias.strides, alias.offset = self.strides, self.offset
        return alias


class FakeBaseWrapper:
    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent, weight_dequant_scale):
        self.module = module
        self.persistent = persistent
        self.events = []

    def apply_weights(self, weights):
        self.events.append(("apply", self.module._mla_decode_weight_cache_entry))
        for name, tensor in weights.items():
            getattr(self.module, name).weight.data = tensor

    def clear_weights(self):
        self.events.append(("clear", self.module._mla_decode_weight_cache_entry))

    def _apply(self, fn, recurse=True):
        self.events.append(("move", self.module._mla_decode_weight_cache_entry))
        fn(self.module.kv_b_proj.weight)
        return self


class DecodeWeightCacheTests(unittest.TestCase):
    def setUp(self):
        self.torch = ModuleType("torch")
        self.torch.grad_enabled = False
        self.torch.stream = 7
        self.torch.bfloat16 = "bf16"
        self.torch.is_grad_enabled = lambda: self.torch.grad_enabled
        self.torch.cuda = SimpleNamespace(current_stream=lambda device: SimpleNamespace(
            cuda_stream=self.torch.stream))
        spec = importlib.util.spec_from_file_location("mla_weight_cache_under_test", SOURCE)
        self.cache = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": self.torch}):
            spec.loader.exec_module(self.cache)
        self.weight = FakeTensor()
        self.scale = FakeTensor(value=2, shape=(1, 1), dtype="fp32")
        self.module = SimpleNamespace(
            training=False, num_heads=2, qk_nope_head_dim=2,
            v_head_dim=2, kv_lora_rank=4,
            kv_b_proj=SimpleNamespace(weight=self.weight),
        )
        self.scales = {"kv_b_proj.weight_scale_inv": self.scale}
        self.calls = []

        def dequantize(weight, scale):
            self.calls.append((weight, scale))
            return FakeTensor(value=(weight.storage.value, scale.storage.value),
                              shape=weight.shape, dtype="bf16", device=weight.device)
        self.dequantize = dequantize

    def enable(self):
        self.cache.configure_mla_decode_weight_cache(self.module)

    def get(self, dequantize=None):
        return self.cache.get_mla_decode_weight(
            self.module, self.scales, dequantize or self.dequantize)

    def test_disabled_by_default_and_returns_original_result(self):
        first, second = self.get(), self.get()
        self.assertIsNot(first, second)
        self.assertEqual(len(self.calls), 2)
        self.assertIs(self.calls[0][0], self.weight.data)
        self.assertIs(self.calls[0][1], self.scale)

    def test_enabled_reuses_exact_matrix_and_retains_storage_aliases(self):
        self.enable()
        first = self.get()
        self.assertIs(self.get(), first)
        self.assertEqual(len(self.calls), 1)
        entry = self.module._mla_decode_weight_cache_entry
        self.assertIs(entry[1], self.weight)
        self.assertIs(entry[2], self.scale)
        self.assertIs(entry[3].storage, self.weight.storage)
        self.assertIs(entry[4].storage, self.scale.storage)

    def test_normal_weight_and_scale_mutations_recompute(self):
        self.enable()
        first = self.get()
        self.weight.storage.value, self.weight.version = 3, 1
        second = self.get()
        self.assertIsNot(first, second)
        self.assertEqual(second.storage.value, (3, 2))
        self.scale.storage.value, self.scale.version = 4, 1
        third = self.get()
        self.assertIsNot(second, third)
        self.assertEqual(third.storage.value, (3, 4))
        self.assertIs(self.get(), third)

    def test_data_storage_rebind_recomputes_and_keeps_old_alias_alive(self):
        self.enable()
        first = self.get()
        entry = self.module._mla_decode_weight_cache_entry
        old_storage = self.weight.storage
        self.weight.data = FakeTensor(value=9)
        self.assertIs(entry[3].storage, old_storage)
        second = self.get()
        self.assertIsNot(first, second)
        self.assertEqual(second.storage.value, (9, 2))

    def test_parameter_and_scale_identity_changes_recompute_same_storage(self):
        self.enable()
        first = self.get()
        replacement = self.weight.detach()
        self.module.kv_b_proj.weight = replacement
        second = self.get()
        self.assertIsNot(first, second)
        self.scales["kv_b_proj.weight_scale_inv"] = self.scale.detach()
        self.assertIsNot(second, self.get())

    def test_metadata_and_geometry_changes_invalidate(self):
        changes = (
            lambda: setattr(self.weight, "shape", (4, 8)),
            lambda: setattr(self.weight, "strides", (1, 8)),
            lambda: setattr(self.weight, "dtype", "other_fp8"),
            lambda: setattr(self.weight, "offset", 2),
            lambda: setattr(self.module, "num_heads", 1),
            lambda: setattr(self.module, "kv_lora_rank", 8),
        )
        for change in changes:
            with self.subTest(change=change):
                self.setUp()
                self.enable()
                first = self.get()
                change()
                self.assertIsNot(first, self.get())

    def test_device_change_rebuilds_and_drops_old_entry(self):
        self.enable()
        first = self.get()
        new_device = SimpleNamespace(type="cuda", index=1)
        self.weight.device = self.scale.device = new_device
        second = self.get()
        self.assertIsNot(first, second)
        self.assertIs(self.module._mla_decode_weight_cache_entry[-1], second)

    def test_stream_switch_rebuilds_bounded_single_entry(self):
        self.enable()
        first = self.get()
        self.torch.stream = 8
        second = self.get()
        self.assertIsNot(first, second)
        self.assertIs(self.get(), second)
        self.torch.stream = 7
        third = self.get()
        self.assertIsNot(first, third)
        self.assertIsNot(second, third)
        self.assertEqual(len(self.calls), 3)
        self.assertIs(self.module._mla_decode_weight_cache_entry[-1], third)

    def test_unversioned_input_clears_cache_and_always_recomputes(self):
        for which in ("weight", "scale"):
            with self.subTest(which=which):
                self.setUp()
                self.enable()
                first = self.get()
                getattr(self, which).versioned = False
                second, third = self.get(), self.get()
                self.assertIsNot(first, second)
                self.assertIsNot(second, third)
                self.assertIsNone(self.module._mla_decode_weight_cache_entry)

    def test_training_grad_or_cpu_falls_back_and_clears_old_entry(self):
        switches = (
            lambda: setattr(self.module, "training", True),
            lambda: setattr(self.torch, "grad_enabled", True),
            lambda: setattr(self.weight, "device", SimpleNamespace(type="cpu")),
        )
        for switch in switches:
            with self.subTest(switch=switch):
                self.setUp()
                self.enable()
                self.get()
                switch()
                self.assertIsNot(self.get(), self.get())
                self.assertIsNone(self.module._mla_decode_weight_cache_entry)

    def test_clear_preserves_opt_in_and_disable_drops_entry(self):
        self.enable()
        first = self.get()
        self.cache.clear_mla_decode_weight_cache(self.module)
        second = self.get()
        self.assertIsNot(first, second)
        self.assertIs(self.get(), second)
        self.cache.configure_mla_decode_weight_cache(self.module, enabled=False)
        self.assertIsNone(self.module._mla_decode_weight_cache_entry)
        self.assertIsNot(self.get(), self.get())

    def test_unsafe_data_mutation_requires_explicit_clear(self):
        self.enable()
        self.get()
        # A .data/external write does not advance the source version counter.
        self.weight.storage.value = 42
        self.cache.clear_mla_decode_weight_cache(self.module)
        self.assertEqual(self.get().storage.value, (42, 2))

    def test_dequantizer_replacement_recomputes(self):
        self.enable()
        first = self.get()
        other = lambda weight, scale: self.dequantize(weight, scale)
        self.assertIsNot(first, self.get(other))

    def make_wrapper(self, persistent=True):
        # Execute the real class methods against a CPU fake base; importing the
        # package normally requires CUDA extensions and torch on this host.
        path = ROOT / "batchgen/models/deepseek/deepseekv3/wrappers.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "DeepSeekAttnWrapper")
        option_parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == "_read_decode_attention_micro_batch_size")
        methods = {"__init__", "_register_fp8_weights", "_unregister_fp8_weights",
                   "apply_weights", "clear_weights", "_apply"}
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name in methods]
        for node in ast.walk(cls):
            if isinstance(node, ast.arg):
                node.annotation = None
            elif isinstance(node, ast.FunctionDef):
                node.returns = None
        namespace = {
            "os": os,
            "logging": logging,
            "AttnWrapperBase": FakeBaseWrapper,
            "configure_mla_decode_weight_cache": self.cache.configure_mla_decode_weight_cache,
            "clear_mla_decode_weight_cache": self.cache.clear_mla_decode_weight_cache,
        }
        exec(compile(ast.Module(body=[option_parser, cls], type_ignores=[]), str(path), "exec"), namespace)
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            setattr(self.module, name, SimpleNamespace(weight=FakeTensor()))
        with patch.dict(os.environ, {"BATCHGEN_DECODE_ATTN_MICRO_BATCH_SIZE": "0"}):
            return namespace["DeepSeekAttnWrapper"](
                self.module, 0, None, None, None, persistent=persistent)

    def test_persistent_wrapper_enables_only_after_register_and_disables_on_unregister(self):
        wrapper = self.make_wrapper()
        self.assertFalse(self.module._mla_decode_weight_cache_enabled)
        wrapper._register_fp8_weights()
        self.assertTrue(self.module._mla_decode_weight_cache_enabled)
        self.get()
        wrapper._unregister_fp8_weights()
        self.assertFalse(self.module._mla_decode_weight_cache_enabled)
        self.assertIsNone(self.module._mla_decode_weight_cache_entry)

    def test_nonpersistent_wrapper_never_enables_cache(self):
        wrapper = self.make_wrapper(persistent=False)
        wrapper._register_fp8_weights()
        self.assertFalse(self.module._mla_decode_weight_cache_enabled)
        self.assertIsNot(self.get(), self.get())

    def test_wrapper_invalidates_before_load_clear_and_device_move(self):
        wrapper = self.make_wrapper()
        wrapper._register_fp8_weights()
        self.get()
        wrapper.apply_weights({"kv_b_proj": FakeTensor(value=17)})
        self.assertEqual(self.get().storage.value, (17, 2))
        wrapper.clear_weights()
        self.get()
        wrapper._apply(lambda tensor: None)
        self.assertEqual(wrapper.events, [("apply", None), ("clear", None), ("move", None)])
        self.assertIsNone(self.module._mla_decode_weight_cache_entry)


if __name__ == "__main__":
    unittest.main()
