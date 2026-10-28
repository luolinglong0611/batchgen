"""CPU storage-lifetime/version checks for external CUDA Graph inputs."""
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "batchgen/attention/mla/r1_decode_cuda_graph.py"
name = "r1_graph_lifetime_candidate"
spec = importlib.util.spec_from_file_location(name, SOURCE)
candidate = importlib.util.module_from_spec(spec)
sys.modules[name] = candidate
spec.loader.exec_module(candidate)


class Storage:
    def data_ptr(self):
        return id(self)


class Tensor:
    def __init__(self, storage=None, version=0):
        self.storage = storage if storage is not None else Storage()
        self._version = version
        self.shape = (4, 4)
        self.device = "cuda:0"
        self.dtype = "FP8-or-BF16-test-double"
    def data_ptr(self): return self.storage.data_ptr()
    def untyped_storage(self): return self.storage
    def stride(self): return (4, 1)
    def storage_offset(self): return 0
    def detach(self): return Tensor(self.storage, self._version)


def wrapper():
    children = {key: SimpleNamespace(weight=Tensor()) for key in (
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        "q_a_layernorm", "kv_a_layernorm")}
    children["rotary_emb"] = SimpleNamespace(cos_cached=Tensor(), sin_cached=Tensor())
    return SimpleNamespace(module=SimpleNamespace(**children),
                           weight_dequant_scale={"kv_b_proj.weight_scale_inv": Tensor()})


class GraphLifetimeContracts(unittest.TestCase):
    def test_normal_inplace_and_parameter_storage_rebind_change_signature(self):
        value = wrapper()
        first = candidate.attention_source_signature(value)
        value.module.q_a_proj.weight._version += 1
        second = candidate.attention_source_signature(value)
        self.assertNotEqual(first, second)
        value.module.kv_b_proj.weight.storage = Storage()
        third = candidate.attention_source_signature(value)
        self.assertNotEqual(second, third)
        value.weight_dequant_scale["kv_b_proj.weight_scale_inv"]._version += 1
        self.assertNotEqual(third, candidate.attention_source_signature(value))

    def test_capture_detach_alias_survives_cache_replacement_and_data_rebind(self):
        expanded = Tensor()
        previous = weakref.ref(expanded.storage)
        collector = {}
        token = candidate._CAPTURE_INPUTS.set(collector)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: True))
        try:
            with patch.dict(sys.modules, {"torch": torch}):
                candidate._retain_capture_inputs(expanded, expanded)
            self.assertEqual(len(collector), 1)
            # Retaining only the mutable source Tensor would be insufficient:
            # a .data rebind changes the source object's storage in place.
            expanded.storage = Storage()
            del expanded
            gc.collect()
            self.assertIsNotNone(previous())
            self.assertIs(next(iter(collector.values())).storage, previous())
            collector.clear()
            gc.collect()
            self.assertIsNone(previous())
        finally:
            candidate._CAPTURE_INPUTS.reset(token)

    def test_warmup_does_not_retain_an_unused_second_expanded_copy(self):
        collector = {}
        token = candidate._CAPTURE_INPUTS.set(collector)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        try:
            with patch.dict(sys.modules, {"torch": torch}):
                candidate._retain_capture_inputs(Tensor(), Tensor())
            self.assertEqual(collector, {})
        finally:
            candidate._CAPTURE_INPUTS.reset(token)

    def test_whole_model_collector_owns_capture_inputs_until_graph_release(self):
        model_outputs = candidate.WholeModelKVOutputs([0], capture_stream=7)
        expanded = Tensor()
        previous = weakref.ref(expanded.storage)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: True))
        forward = candidate._FORWARD.set((object(), object()))
        try:
            with patch.dict(sys.modules, {"torch": torch}), candidate.whole_model_attention_scope(model_outputs):
                candidate._retain_capture_inputs(expanded)
            self.assertIsNone(candidate._CAPTURE_INPUTS.get())
            del expanded
            gc.collect()
            self.assertIsNotNone(previous())
            model_outputs.capture_inputs.clear()
            gc.collect()
            self.assertIsNone(previous())
        finally:
            candidate._FORWARD.reset(forward)


if __name__ == "__main__":
    unittest.main(verbosity=2)
