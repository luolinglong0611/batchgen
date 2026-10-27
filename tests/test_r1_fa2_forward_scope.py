"""CPU lifecycle contracts for FA2 preparation; no Torch/CUDA/model import.

Fake tensors expose only metadata and view semantics. These tests establish
preparation ownership, invalidation and dispatch; GPU output/LSE equivalence is
checked separately by the bounded primitive.
"""
import ast
from collections import Counter
import contextlib
import copy
import gc
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "r1_fa2_forward_scope_contract_namespace"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "batchgen/attention/mla")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".common_flashinfer_backend",
    ROOT / "batchgen/attention/mla/common_flashinfer_backend.py",
)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
reference = sys.modules[PACKAGE + ".common_sdpa_backend"]


class Device:
    def __init__(self, index=0):
        self.type, self.index = "cuda", index

    def __eq__(self, other):
        return isinstance(other, Device) and self.index == other.index

    def __hash__(self):
        return hash((self.type, self.index))

    def __str__(self):
        return "cuda:" + str(self.index)


def contiguous_strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class Tensor:
    """No allocation: the backing identity survives views and metadata updates."""
    _next_pointer = 1024

    def __init__(self, shape, *, data=None, dtype="bf16", device=None,
                 version=0, label=None, origin=None, strides=None, signature=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.layout = "strided"
        self.device = device or Device()
        self.data = copy.deepcopy(data)
        self.version = version
        self.label = label
        self.origin = origin or self
        self._strides = tuple(strides or contiguous_strides(self.shape))
        self.signature = signature
        self.offset = 0
        self.recorded_streams = []
        if origin is None:
            self.pointer = Tensor._next_pointer
            Tensor._next_pointer += 1024
        else:
            self.pointer = origin.pointer

    @property
    def _version(self):
        if self.origin.version is None:
            raise RuntimeError("Inference tensors do not track version counter")
        return self.origin.version

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self.offset

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self._strides == contiguous_strides(self.shape)

    def is_inference(self):
        return self.origin.version is None

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def mutate(self, data):
        self.data = copy.deepcopy(data)
        if self.version is not None:
            self.version += 1

    def new_empty(self, shape, dtype=None):
        return Tensor(shape, dtype=dtype or self.dtype, device=self.device)

    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // math.prod(x for x in shape if x != -1)
        if math.prod(shape) != self.numel():
            raise AssertionError("Fake view changed element count")
        return Tensor(shape, dtype=self.dtype, device=self.device, origin=self.origin,
                      label=self.label, signature=self.signature)

    def unsqueeze(self, dimension):
        if dimension < 0:
            dimension += self.ndim + 1
        shape = list(self.shape)
        shape.insert(dimension, 1)
        return Tensor(shape, dtype=self.dtype, device=self.device, origin=self.origin,
                      label=self.label, signature=self.signature)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            index = key.index(Ellipsis)
            key = key[:index] + (slice(None),) * (self.ndim - len(key) + 1) + key[index + 1:]
        key += (slice(None),) * (self.ndim - len(key))
        shape, strides = [], []
        for size, stride, part in zip(self.shape, self._strides, key):
            if isinstance(part, int):
                continue
            start, stop, step = part.indices(size)
            shape.append(len(range(start, stop, step)))
            strides.append(stride * step)
        return Tensor(shape, dtype=self.dtype, device=self.device, origin=self.origin,
                      label=self.label, strides=strides, signature=self.signature)


class Manager:
    def __init__(self, table):
        self.generation = 0
        self._gpu_page_table_manager = SimpleNamespace(gpu_table=table)

    def get_page_table_version(self):
        return self.generation


class Runtime:
    def __init__(self):
        self.counts = Counter()
        self.stream = SimpleNamespace(cuda_stream=101, device=Device())
        self.plans = []
        self.runs = []
        self.workspace_refs = []
        self.fail_plan = False
        self.fail_run = False
        self.capture = False
        self.torch = SimpleNamespace(
            Tensor=Tensor, bfloat16="bf16", uint8="uint8", int32="int32",
            int64="int64", float32="fp32", empty=self.empty, tensor=self.tensor,
            inference_mode=lambda *args: contextlib.nullcontext(),
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: self.capture,
                current_stream=lambda *args, **kwargs: self.stream,
                current_device=lambda: self.stream.device.index,
            ),
        )
        self.flashinfer = SimpleNamespace(BatchMLAPagedAttentionWrapper=self.wrapper)

    def empty(self, shape, **kwargs):
        self.counts["workspace"] += 1
        value = Tensor((shape,) if isinstance(shape, int) else shape, **kwargs)
        self.workspace_refs.append(weakref.ref(value))
        return value

    def tensor(self, values, **kwargs):
        self.counts["h2d"] += 1
        data = values.tolist() if hasattr(values, "tolist") else list(values)
        return Tensor((len(data),), data=data, **kwargs)

    def cpu_ints(self, tensor, name, ndim):
        self.counts["snapshot:" + name] += 1
        if not isinstance(tensor, Tensor) or tensor.ndim != ndim or tensor.dtype not in ("int32", "int64"):
            raise ValueError("invalid integer tensor")
        return tensor.tolist()

    def wrapper(self, workspace, **kwargs):
        self.counts["wrapper"] += 1
        assert kwargs == {"backend": "fa2", "use_cuda_graph": False}
        runtime = self

        class Wrapper:
            _backend = "fa2"

            def __init__(self):
                self.workspace = workspace
                self.metadata = None

            def plan(self, *args):
                runtime.counts["plan"] += 1
                if runtime.fail_plan:
                    raise RuntimeError("injected plan failure")
                self.metadata = [copy.deepcopy(value.data) for value in args[:4]]
                runtime.plans.append(copy.deepcopy(self.metadata))

            def run(self, query, rope, cache, cache_rope, **kwargs):
                runtime.counts["run"] += 1
                if runtime.fail_run:
                    raise RuntimeError("injected run failure")
                assert kwargs == {"return_lse": True}
                assert query.origin is rope.origin and cache.origin is cache_rope.origin
                signature = (query.origin.label, cache.origin.label, copy.deepcopy(self.metadata))
                runtime.runs.append(signature)
                batch, heads = query.shape[:2]
                return (Tensor((batch, heads, 512), signature=signature, device=query.device),
                        Tensor((batch, heads), dtype="fp32", signature=signature, device=query.device))

        return Wrapper()

    @contextlib.contextmanager
    def install(self):
        with patch.object(backend, "_torch", return_value=self.torch), \
                patch.object(backend, "_flashinfer", return_value=self.flashinfer), \
                patch.object(backend, "_cpu_ints", side_effect=self.cpu_ints):
            yield


def fixture(*, lengths=(63, 65), table=((3, -1), (6, 1)), version=0):
    return (
        Tensor((len(lengths), 1, 4, 576), label="query"),
        Tensor((12, 64, 1, 576), label="cache"),
        Tensor((len(table), len(table[0])), data=table, dtype="int32", version=version),
        Tensor((len(lengths),), data=list(lengths), dtype="int32", version=version),
    )


def attend(values, scale=0.125, *, metadata=None, causal=True):
    query, cache, table, lengths = values
    if metadata is None:
        metadata, splits = backend.get_mla_metadata(lengths, query.shape[2], 1)
    else:
        splits = None
    return backend.flash_mla_with_kvcache(query, cache, table, lengths, 512,
                                          metadata, splits, scale, causal)


class ForwardScopeContracts(unittest.TestCase):
    def test_61_layers_prepare_once_but_run_every_distinct_query_and_cache(self):
        runtime, values = Runtime(), fixture()
        manager = Manager(values[2])
        outputs = []
        with runtime.install(), backend.decode_forward_scope(manager):
            for layer in range(61):
                query = Tensor(values[0].shape, label="query-" + str(layer))
                cache = Tensor(values[1].shape, label="cache-" + str(layer))
                output, lse = attend((query, cache, *values[2:]))
                self.assertEqual(output.signature[:2], (query.label, cache.label))
                self.assertEqual(lse.signature, output.signature)
                outputs.append(output)
        self.assertEqual(runtime.counts["plan"], 1)
        self.assertEqual(runtime.counts["wrapper"], 1)
        self.assertEqual(runtime.counts["workspace"], 1)
        self.assertEqual(runtime.counts["h2d"], 4)
        self.assertEqual(runtime.counts["run"], 61)
        self.assertEqual(runtime.counts["snapshot:cache_seqlens"], 1)
        self.assertEqual(runtime.counts["snapshot:block_table"], 1)
        self.assertEqual(len({id(output) for output in outputs}), 61)

    def test_unscoped_calls_prepare_fresh_every_time(self):
        runtime, values = Runtime(), fixture()
        with runtime.install():
            for _ in range(3):
                attend(values)
        for key in ("plan", "wrapper", "workspace", "run"):
            self.assertEqual(runtime.counts[key], 3, key)
        self.assertEqual(runtime.counts["h2d"], 12)

    def test_new_scope_sees_same_pointer_length_page_and_row_order_changes(self):
        runtime, values = Runtime(), fixture()
        manager = Manager(values[2])
        pointers = tuple(value.data_ptr() for value in values)
        transitions = [([63, 65], [[3, -1], [6, 1]]),
                       ([64, 65], [[3, -1], [6, 1]]),
                       ([65, 65], [[3, 2], [6, 1]]),
                       ([65, 65], [[3, 4], [6, 1]]),
                       ([65, 65], [[6, 1], [3, 4]])]
        with runtime.install():
            for lengths, table in transitions:
                values[3].mutate(lengths)
                values[2].mutate(table)
                manager.generation += 1
                with backend.decode_forward_scope(manager):
                    output, _ = attend(values)
                expected = reference.token_page_indices(reference.plan_paged_decode(
                    values[0].shape, values[1].shape, table, lengths, max(lengths)))
                self.assertEqual(output.signature[2][2], expected["kv_indices"])
                self.assertEqual(output.signature[2][3], lengths)
                self.assertEqual(tuple(value.data_ptr() for value in values), pointers)
        self.assertEqual(runtime.counts["plan"], len(transitions))

    def test_in_scope_length_page_and_manager_generation_mutation_rejected(self):
        for mutation in ("length", "table", "manager"):
            with self.subTest(mutation=mutation):
                runtime, values = Runtime(), fixture()
                manager = Manager(values[2])
                with runtime.install(), backend.decode_forward_scope(manager):
                    attend(values)
                    if mutation == "length":
                        values[3].mutate([64, 65])
                    elif mutation == "table":
                        values[2].mutate([[4, -1], [6, 1]])
                    else:
                        manager.generation += 1
                    with self.assertRaises((ValueError, RuntimeError)):
                        attend(values)
                self.assertEqual(runtime.counts["run"], 1)

    def test_in_scope_stream_device_geometry_scale_or_head_change_rejected(self):
        for mutation in ("stream", "device", "query_shape", "cache_shape", "query_stride",
                         "cache_stride", "dtype", "table_offset", "length_offset",
                         "table_identity", "length_identity", "scale", "heads", "causal", "capture"):
            with self.subTest(mutation=mutation):
                runtime, values = Runtime(), list(fixture())
                manager = Manager(values[2])
                scale = 0.125
                causal = True
                with runtime.install(), backend.decode_forward_scope(manager):
                    attend(values)
                    if mutation == "stream":
                        runtime.stream = SimpleNamespace(cuda_stream=102, device=Device())
                    elif mutation == "device":
                        for value in values:
                            value.device = Device(1)
                        runtime.stream = SimpleNamespace(cuda_stream=101, device=Device(1))
                    elif mutation == "query_shape":
                        values[0] = Tensor((2, 2, 4, 576))
                    elif mutation == "cache_shape":
                        values[1] = Tensor((13, 64, 1, 576))
                    elif mutation == "query_stride":
                        values[0]._strides = (5000, 2304, 576, 1)
                    elif mutation == "cache_stride":
                        values[1]._strides = (38000, 576, 576, 1)
                    elif mutation == "dtype":
                        values[0].dtype = "fp32"
                    elif mutation == "table_offset":
                        values[2].offset = 1
                    elif mutation == "length_offset":
                        values[3].offset = 1
                    elif mutation == "table_identity":
                        values[2] = Tensor(values[2].shape, data=values[2].data,
                                           dtype="int32", origin=values[2])
                        manager._gpu_page_table_manager.gpu_table = values[2]
                    elif mutation == "length_identity":
                        values[3] = Tensor(values[3].shape, data=values[3].data,
                                           dtype="int32", origin=values[3])
                    elif mutation == "heads":
                        values[0] = Tensor((2, 1, 8, 576))
                    elif mutation == "causal":
                        causal = False
                    elif mutation == "capture":
                        runtime.capture = True
                    else:
                        scale = 0.25
                    with self.assertRaises((ValueError, RuntimeError)):
                        attend(values, scale, causal=causal)
                self.assertEqual(runtime.counts["run"], 1)

    def test_inference_length_without_version_fails_closed_before_plan(self):
        runtime, values = Runtime(), fixture(version=None)
        manager = Manager(values[2])
        with runtime.install(), backend.decode_forward_scope(manager):
            with self.assertRaises(RuntimeError):
                attend(values)
        self.assertEqual(runtime.counts["plan"], 0)
        self.assertEqual(runtime.counts["run"], 0)

    def test_inference_table_uses_manager_generation_and_rejects_rebuild(self):
        runtime, values = Runtime(), fixture()
        values[2].version = None
        manager = Manager(values[2])
        with runtime.install(), backend.decode_forward_scope(manager):
            attend(values)
            attend(values)
            self.assertEqual(runtime.counts["plan"], 1)
            values[2].mutate([[4, -1], [6, 1]])
            manager.generation += 1
            with self.assertRaises(ValueError):
                attend(values)
        self.assertEqual(runtime.counts["run"], 2)

    def test_manager_must_own_the_exact_table_object(self):
        runtime, values = Runtime(), fixture()
        foreign = Tensor(values[2].shape, data=values[2].data, dtype="int32")
        manager = Manager(foreign)
        with runtime.install(), backend.decode_forward_scope(manager):
            with self.assertRaises((ValueError, RuntimeError)):
                attend(values)
        self.assertEqual(runtime.counts["run"], 0)

    def test_metadata_returned_to_caller_cannot_poison_cached_lengths(self):
        runtime, values = Runtime(), fixture()
        manager = Manager(values[2])
        with runtime.install(), backend.decode_forward_scope(manager):
            metadata, _ = backend.get_mla_metadata(values[3], 4, 1)
            metadata["lengths"][0] = 1
            metadata["num_heads"] = 900
            valid, _ = backend.get_mla_metadata(values[3], 4, 1)
            self.assertEqual(valid["lengths"], [63, 65])
            self.assertEqual(valid["num_heads"], 4)
            with self.assertRaises(ValueError):
                attend(values, metadata=metadata)
            output, _ = attend(values, metadata=valid)
            self.assertEqual(output.signature[2][3], [63, 65])
        self.assertEqual(runtime.counts["run"], 1)

    def test_nested_scope_rejected_and_outer_scope_survives(self):
        runtime, values = Runtime(), fixture()
        manager = Manager(values[2])
        with runtime.install(), backend.decode_forward_scope(manager):
            attend(values)
            with self.assertRaises((ValueError, RuntimeError)):
                with backend.decode_forward_scope(manager):
                    self.fail("nested scope entered")
            attend(values)
        self.assertEqual(runtime.counts["plan"], 1)
        with runtime.install(), backend.decode_forward_scope(manager):
            attend(values)
        self.assertEqual(runtime.counts["plan"], 2)

    def test_exception_cleanup_does_not_reuse_previous_forward(self):
        for failure in ("body", "plan", "run"):
            with self.subTest(failure=failure):
                runtime, values = Runtime(), fixture()
                manager = Manager(values[2])
                runtime.fail_plan = failure == "plan"
                runtime.fail_run = failure == "run"
                with runtime.install():
                    with self.assertRaises(RuntimeError):
                        with backend.decode_forward_scope(manager):
                            attend(values)
                            raise RuntimeError("injected body failure")
                    runtime.fail_plan = runtime.fail_run = False
                    with backend.decode_forward_scope(manager):
                        attend(values)
                self.assertEqual(runtime.counts["plan"], 2)

    def test_completed_scopes_do_not_retain_workspace(self):
        runtime, values = Runtime(), fixture()
        manager = Manager(values[2])
        with runtime.install():
            for _ in range(8):
                with backend.decode_forward_scope(manager):
                    attend(values)
        gc.collect()
        self.assertEqual(len(runtime.workspace_refs), 8)
        self.assertTrue(all(reference() is None for reference in runtime.workspace_refs))

    def test_retained_scope_object_releases_resources_even_after_exception(self):
        for fail in (False, True):
            with self.subTest(exception=fail):
                runtime = Runtime()

                def complete_scope():
                    values = fixture()
                    manager = Manager(values[2])
                    refs = [weakref.ref(manager), weakref.ref(values[2]), weakref.ref(values[3])]
                    state = None
                    with runtime.install():
                        try:
                            with backend.decode_forward_scope(manager) as state:
                                attend(values)
                                if fail:
                                    raise RuntimeError("injected exit failure")
                        except RuntimeError:
                            if not fail:
                                raise
                    return state, refs

                state, references = complete_scope()
                self.assertIsNotNone(state)
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
                self.assertTrue(all(reference() is None for reference in runtime.workspace_refs))


class VectorizedMetadataRegression(unittest.TestCase):
    def test_existing_74_exact_fixtures_still_match_scalar_reference(self):
        import numpy as np
        cases = []
        def plan(lengths, tables=None, capacity=24576):
            if tables is None:
                tables = [list(range(i * 200, i * 200 + (length + 63) // 64))
                          for i, length in enumerate(lengths)]
            return reference.plan_paged_decode((len(lengths), 1, 128, 576),
                                               (capacity, 64, 1, 576), tables,
                                               lengths, max(lengths, default=1))
        cases.append(plan([]))
        cases.append(plan([1, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193]))
        cases.append(plan([65, 129, 63], [[8, 2, -999], [90, 7, 23, 999999], [42, -1]], 100))
        cases.append(plan([65, 64, 1], [[8, 2], [8], [8]], 100))
        cases.extend(plan([length], [[2**25 - 1]], 2**25) for length in (64, 1))
        cases.append([{"row": 0, "length": 0, "physical_pages": []}])
        rng = random.Random(20261027)
        for _ in range(60):
            lengths = [rng.randint(1, 400) for _ in range(rng.randint(1, 20))]
            tables = [rng.sample(range(5000), (length + 63) // 64) + [-123, 999999]
                      for length in lengths]
            cases.append(plan(lengths, tables, 5000))
        for step in (0, 1, 19, 63, 64, 2047, 2048):
            length = 8192 + step
            tables = [list(range(i * 170, i * 170 + (length + 63) // 64)) for i in range(131)]
            cases.append(plan([length] * 131, tables))
        self.assertEqual(len(cases), 74)
        for number, metadata in enumerate(cases):
            with self.subTest(case=number):
                expected = reference.token_page_indices(metadata)
                actual = backend._token_page_indices_array(metadata)
                self.assertEqual(actual.keys(), expected.keys())
                for key in ("query_indptr", "kv_indptr", "kv_indices", "kv_lengths"):
                    self.assertEqual(actual[key].dtype, np.int32)
                    self.assertTrue(actual[key].flags.c_contiguous)
                    self.assertEqual(actual[key].tolist(), expected[key])
                self.assertEqual(actual["allocator_page_size"], 64)
                self.assertEqual(actual["kernel_page_size"], 1)

    def test_int32_overflow_and_planner_rejections_unchanged(self):
        for length, pages in ((1, [2**25]), (65, [0, 2**25]), (64, [2**25])):
            plan = reference.plan_paged_decode((1, 1, 128, 576),
                (2**25 + 1, 64, 1, 576), [pages], [length], length)
            for converter in (reference.token_page_indices, backend._token_page_indices_array):
                with self.subTest(length=length, converter=converter.__name__), \
                        self.assertRaisesRegex(ValueError, "FlashInfer int32 token metadata would overflow"):
                    converter(plan)
        with self.assertRaisesRegex(ValueError, "FlashInfer int32 token metadata would overflow"):
            backend._token_page_indices_array([{"row": 0, "length": 2**31, "physical_pages": [0]}])
        for lengths, table in (([65], [[2, 2]]), ([1], [[-1]]), ([1], [[10]]),
                               ([65], [[1]]), ([0], [[]])):
            with self.subTest(lengths=lengths, table=table), self.assertRaises(ValueError):
                reference.plan_paged_decode((1, 1, 128, 576), (10, 64, 1, 576),
                                           table, lengths, max(lengths, default=1))


class WorkerScopeSelectionContracts(unittest.TestCase):
    """Execute the real method AST without importing the GPU worker module."""

    def harness(self):
        source = ROOT / "batchgen/batchgen_worker.py"
        tree = ast.parse(source.read_text())
        methods = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_glm5_decode_model_forward"]
        self.assertEqual(len(methods), 1)
        method = copy.deepcopy(methods[0])
        method.decorator_list = []
        compiled = ast.Module(body=[*ast.parse("from __future__ import annotations").body,
                                    method], type_ignores=[])
        selector_spec = importlib.util.spec_from_file_location(
            "r1_fa2_worker_selection_contract", ROOT / "batchgen/attention/mla/compat_selection.py")
        selector = importlib.util.module_from_spec(selector_spec)
        selector_spec.loader.exec_module(selector)
        events, manager, mask, positions, result = [], object(), object(), object(), object()
        active = [False]

        @contextlib.contextmanager
        def scope(actual_manager):
            self.assertIs(actual_manager, manager)
            events.append("scope-enter")
            active[0] = True
            try:
                yield
            finally:
                active[0] = False
                events.append("scope-exit")

        def model(tokens, **kwargs):
            self.assertEqual(kwargs, {"attention_mask": mask, "position_ids": positions,
                                      "use_cache": False})
            events.append(("model", active[0], tokens))
            return result

        namespace = {"AttnWrapperBase": SimpleNamespace(gpu_paged_kv_manager=manager),
                     "Attn_Wrapper": SimpleNamespace(attention_mask=mask, position_ids=positions)}
        exec(compile(ast.fix_missing_locations(compiled), str(source), "exec"), namespace)
        modules = {"batchgen.attention.mla.compat_selection": selector,
                   "batchgen.attention.mla.common_flashinfer_backend":
                       SimpleNamespace(decode_forward_scope=scope)}
        worker = SimpleNamespace(model=model, model_name="DeepSeek-R1")
        return namespace[method.name], worker, modules, events, result

    def test_only_explicit_flashinfer_fa2_selector_enters_scope(self):
        for setting in (None, "0", "flashinfer-fa2"):
            with self.subTest(setting=setting):
                forward, worker, modules, events, result = self.harness()
                environment = {} if setting is None else {"BATCHGEN_R1_SM103_MLA_COMPAT": setting}
                with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, modules):
                    self.assertIs(forward(worker, "tokens"), result)
                if setting == "flashinfer-fa2":
                    self.assertEqual(events, ["scope-enter", ("model", True, "tokens"), "scope-exit"])
                else:
                    self.assertEqual(events, [("model", False, "tokens")])

    def test_native_selector_does_not_import_compat_backend(self):
        import builtins
        forward, worker, modules, events, result = self.harness()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "batchgen.attention.mla.common_flashinfer_backend":
                raise AssertionError("Native selector imported compatibility backend")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, modules), \
                patch.object(builtins, "__import__", side_effect=guarded_import):
            self.assertIs(forward(worker, "tokens"), result)

    def test_model_exception_always_exits_scope(self):
        forward, worker, modules, events, _ = self.harness()
        worker.model = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failure"))
        with patch.dict(os.environ, {"BATCHGEN_R1_SM103_MLA_COMPAT": "flashinfer-fa2"}, clear=True), \
                patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "model failure"):
            forward(worker, "tokens")
        self.assertEqual(events, ["scope-enter", "scope-exit"])

    def test_unknown_selector_fails_before_model(self):
        forward, worker, modules, events, _ = self.harness()
        with patch.dict(os.environ, {"BATCHGEN_R1_SM103_MLA_COMPAT": "auto"}, clear=True), \
                patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            forward(worker, "tokens")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
