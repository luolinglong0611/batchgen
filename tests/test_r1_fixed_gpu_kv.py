"""Pure CPU contract and execution of the actual worker initialization methods."""
import ast
import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_fixed_kv_controls", ROOT / "batchgen/r1_alignment_controls.py")
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)
ENV = {controls.GPU_KV_TOKENS_ENV: "131072",
       "BATCHGEN_R1_SM103_MLA_COMPAT": "flashinfer-fa2",
       "BATCHGEN_R1_SM103_FP8_COMPAT": "1"}
PARAMS = dict(model_identifier="deepseek-ai/DeepSeek-R1", kv_dtype="bfloat16",
              graphs_enabled=False, device_capability=(10, 3), num_layers=61,
              kv_lora_rank=512, qk_rope_head_dim=64)


def config(pages=2048):
    return SimpleNamespace(num_layers=61, num_pages=pages, page_size_tokens=64,
        num_k_heads=1, k_head_dim=576, num_v_heads=0, v_head_dim=0,
        kv_dtype="torch.bfloat16", logical_to_physical_layer=None)


def actual_method(name, namespace):
    source = ROOT / "batchgen/batchgen_worker.py"
    tree = ast.parse(source.read_text())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(candidates) != 1:
        raise AssertionError("Expected one actual worker method: " + name)
    node = copy.deepcopy(candidates[0])
    node.decorator_list = []
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


class ContractTests(unittest.TestCase):
    def test_absent_control_is_noop_even_for_unavailable_attributes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(controls.fixed_r1_gpu_kv_contract(**dict.fromkeys(PARAMS)))

    def test_exact_tokens_geometry_and_bytes(self):
        with patch.dict(os.environ, ENV, clear=True):
            value = controls.fixed_r1_gpu_kv_contract(**PARAMS)
        self.assertEqual(value["pages_per_layer"], 2048)
        self.assertEqual(value["data_bytes_per_rank"], 9210691584)
        self.assertEqual(value["data_gib_per_rank"], 8.578125)
        self.assertIn("NOT_paper", value["scope"])

    def test_malformed_fractional_nonpage_and_overflow_tokens_fail(self):
        for raw in ("", "0", "-64", "131072.0", " 131072", "0131072", "65", str(2**31)):
            with self.subTest(raw=raw), patch.dict(os.environ, {**ENV, controls.GPU_KV_TOKENS_ENV: raw}, clear=True):
                with self.assertRaises(ValueError):
                    controls.fixed_r1_gpu_kv_contract(**PARAMS)

    def test_wrong_or_incomplete_profile_cannot_enable_control(self):
        cases = [dict(PARAMS, **override) for override in (
            {"model_identifier": "deepseek-ai/DeepSeek-V3"}, {"graphs_enabled": True},
            {"graphs_enabled": 0}, {"kv_dtype": "float8_e4m3fn"},
            {"device_capability": (10, 0)}, {"device_capability": (True, 3)},
            {"num_layers": 60}, {"kv_lora_rank": 511}, {"qk_rope_head_dim": 128},
            {"num_layers": 61.0})]
        for values in cases:
            with self.subTest(values=values), patch.dict(os.environ, ENV, clear=True), self.assertRaises(ValueError):
                controls.fixed_r1_gpu_kv_contract(**values)
        for removed in ("BATCHGEN_R1_SM103_MLA_COMPAT", "BATCHGEN_R1_SM103_FP8_COMPAT"):
            with patch.dict(os.environ, {k: v for k, v in ENV.items() if k != removed}, clear=True), self.assertRaises(ValueError):
                controls.fixed_r1_gpu_kv_contract(**PARAMS)
        with patch.dict(os.environ, {**ENV, "BATCHGEN_GPU_KV_CACHE_SIZE_GB": "8.578125"}, clear=True), self.assertRaises(ValueError):
            controls.fixed_r1_gpu_kv_contract(**PARAMS)

    def test_available_budget_keeps_headroom_and_rejects_invalid_observations(self):
        self.assertEqual(controls.fixed_r1_gpu_kv_available_bytes(free_bytes=80, total_bytes=100, memory_fraction=.5), 30)
        for changes in ({"free_bytes": True}, {"free_bytes": 101}, {"free_bytes": -1},
                        {"memory_fraction": True}, {"memory_fraction": float("nan")}, {"memory_fraction": 1.1}):
            values = dict(free_bytes=80, total_bytes=100, memory_fraction=.5, **{})
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                controls.fixed_r1_gpu_kv_available_bytes(**values)

    def test_actual_allocator_must_match_every_geometry_field(self):
        with patch.dict(os.environ, ENV, clear=True):
            value = controls.fixed_r1_gpu_kv_contract(**PARAMS)
        controls.validate_fixed_r1_gpu_kv_allocator(config(), value)
        for field, wrong in (("num_pages", 2047), ("num_layers", 62), ("num_k_heads", 2),
                             ("page_size_tokens", 1), ("kv_dtype", "torch.float16"),
                             ("logical_to_physical_layer", [0, 1]), ("k_head_dim", 575)):
            actual = config()
            setattr(actual, field, wrong)
            with self.subTest(field=field), self.assertRaises(ValueError):
                controls.validate_fixed_r1_gpu_kv_allocator(actual, value)

    def test_collective_reports_reject_missing_duplicate_error_or_disagreement(self):
        reports = [dict(rank=i, tokens_per_rank=131072, lifecycle="create", error=None) for i in range(2)]
        self.assertEqual(controls.validate_fixed_r1_gpu_kv_ranks(reports, 2), 131072)
        for mutation in ({"rank": 0}, {"error": "insufficient available memory"},
                         {"tokens_per_rank": 65536}, {"lifecycle": "reuse"}, {"tokens_per_rank": 131072.0}):
            changed = copy.deepcopy(reports)
            changed[1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                controls.validate_fixed_r1_gpu_kv_ranks(changed, 2)
        with self.assertRaises(RuntimeError):
            controls.validate_fixed_r1_gpu_kv_ranks(reports[:1], 2)


class WorkerPathTests(unittest.TestCase):
    def make_worker(self, lifecycle="create", *, free_gib=50, peer_error=None, actual_pages=2048):
        events = []
        gpu = SimpleNamespace(get_device_capability=lambda _: (10, 3),
            synchronize=lambda _: events.append("sync"), empty_cache=lambda: events.append("empty_cache"),
            mem_get_info=lambda _: (free_gib * 1024**3, 64 * 1024**3))
        def gather(output, report):
            events.append(("gather", copy.deepcopy(report)))
            for rank in range(2):
                output[rank] = {**report, "rank": rank}
            if peer_error is not None:
                output[1]["error"] = peer_error
        namespace = {"os": os, "logging": logging, "torch": SimpleNamespace(cuda=gpu),
                     "dist": SimpleNamespace(all_gather_object=gather)}
        initialize = actual_method("_init_gpu_kv_with_actual_size", namespace)
        fixed = actual_method("_init_r1_fixed_gpu_kv", namespace)
        worker = SimpleNamespace(rank=0, world_size=2, local_rank=0, torch_device="cuda:0",
            huggingface_ckpt_name=PARAMS["model_identifier"], _phase="decode",
            engine_config=SimpleNamespace(Basic_Config=SimpleNamespace(kv_dtype="bfloat16", enable_cuda_graphs=False)),
            # Actual initializer returns a compact native ModelConfig here;
            # only the materialized module has the HF MLA/RoPE dimensions.
            model_config=SimpleNamespace(num_hidden_layers=61, compressed_kv_dim=576),
            model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61, kv_lora_rank=512, qk_rope_head_dim=64)),
            gpu_memory_frac=.85, gpu_kv_cache_size_gb=None, gpu_paged_kv_cache_manager=None,
            _gpu_kv_sizing_provisional=True, _gpu_kv_cached_pool_gb=None)
        def manager(initialized):
            return SimpleNamespace(is_initialized=initialized, config=config(), uses_fixed_address_pool=True,
                get_stats=lambda: SimpleNamespace(num_total_pages=actual_pages))
        if lifecycle != "create":
            worker.gpu_paged_kv_cache_manager = manager(lifecycle == "reuse")
            worker.gpu_kv_cache_size_gb = 8.578125
        def create():
            events.append("create")
            worker.gpu_paged_kv_cache_manager = manager(True)
        def remap():
            events.append("remap")
            worker.gpu_paged_kv_cache_manager.is_initialized = True
        worker._initialize_gpu_kv_manager_fixed_size = create
        worker._remap_reused_kv_pool = remap
        worker._init_r1_fixed_gpu_kv = lambda: fixed(worker)
        return initialize, worker, events

    def module_context(self):
        host = SimpleNamespace(build_gpu_kv_config_fixed_size=lambda **_: config())
        return patch.dict(sys.modules, {"batchgen.r1_alignment_controls": controls,
            "batchgen.kv_cache.host_kv_mananger_config": host})

    def test_actual_entry_uses_explicit_capacity_and_marks_it_not_provisional(self):
        initialize, worker, events = self.make_worker()
        with patch.dict(os.environ, ENV, clear=True), self.module_context():
            initialize(worker)
        self.assertEqual(worker.gpu_kv_cache_size_gb, 8.578125)
        self.assertEqual(worker._r1_fixed_gpu_kv_tokens, 131072)
        self.assertFalse(worker._gpu_kv_sizing_provisional)
        self.assertEqual(worker._gpu_kv_cached_pool_gb, 8.578125)
        self.assertIn("create", events)
        self.assertEqual(len([x for x in events if isinstance(x, tuple)]), 2)
        self.assertIn("actual_allocator_capacity_verified", worker._r1_fixed_gpu_kv_receipt["status"])

    def test_actual_fixed_address_remap_preserves_existing_allocator_and_budget(self):
        initialize, worker, events = self.make_worker("remap")
        original = worker.gpu_paged_kv_cache_manager
        with patch.dict(os.environ, ENV, clear=True), self.module_context():
            initialize(worker)
        self.assertIs(worker.gpu_paged_kv_cache_manager, original)
        self.assertIn("remap", events)
        self.assertNotIn("create", events)
        self.assertEqual(worker.gpu_kv_cache_size_gb, 8.578125)

    def test_actual_already_initialized_path_revalidates_without_allocation_or_free_query(self):
        initialize, worker, events = self.make_worker("reuse", free_gib=0)
        with patch.dict(os.environ, ENV, clear=True), self.module_context():
            initialize(worker)
        self.assertEqual(len(events), 2)  # only two small validation collectives
        self.assertEqual(worker.gpu_kv_cache_size_gb, 8.578125)

    def test_actual_insufficient_available_memory_does_not_silently_min_or_allocate(self):
        initialize, worker, events = self.make_worker(free_gib=10)
        with patch.dict(os.environ, ENV, clear=True), self.module_context(), self.assertRaisesRegex(RuntimeError, "does not fit"):
            initialize(worker)
        self.assertIsNone(worker.gpu_kv_cache_size_gb)
        self.assertNotIn("create", events)
        self.assertFalse(hasattr(worker, "_r1_fixed_gpu_kv_receipt"))

    def test_actual_peer_error_prevents_local_allocation(self):
        initialize, worker, events = self.make_worker(peer_error="peer cannot fit")
        with patch.dict(os.environ, ENV, clear=True), self.module_context(), self.assertRaisesRegex(RuntimeError, "peer cannot fit"):
            initialize(worker)
        self.assertNotIn("create", events)

    def test_actual_capacity_change_or_allocator_count_mismatch_fails(self):
        for changes in ({"_r1_fixed_gpu_kv_tokens": 65536}, {"_phase": "prefill"}):
            initialize, worker, events = self.make_worker()
            worker.__dict__.update(changes)
            with self.subTest(changes=changes), patch.dict(os.environ, ENV, clear=True), self.module_context(), self.assertRaises(RuntimeError):
                initialize(worker)
            self.assertNotIn("create", events)
        initialize, worker, events = self.make_worker(actual_pages=2047)
        with patch.dict(os.environ, ENV, clear=True), self.module_context(), self.assertRaisesRegex(RuntimeError, "page count"):
            initialize(worker)
        self.assertFalse(hasattr(worker, "_r1_fixed_gpu_kv_receipt"))

    def test_actual_model_config_is_authoritative_not_compact_or_loaded_config(self):
        initialize, worker, events = self.make_worker()
        worker.loaded_model_config = SimpleNamespace(num_hidden_layers=61, kv_lora_rank=512, qk_rope_head_dim=64)
        worker.model.config.kv_lora_rank = 511
        with patch.dict(os.environ, ENV, clear=True), self.module_context(), self.assertRaisesRegex(RuntimeError, "geometry"):
            initialize(worker)
        self.assertNotIn("create", events)

    def test_native_path_has_no_added_collective_or_fixed_helper_when_env_absent(self):
        initialize, worker, events = self.make_worker("reuse")
        worker._init_r1_fixed_gpu_kv = lambda: self.fail("native default called fixed profile")
        with patch.dict(os.environ, {}, clear=True):
            initialize(worker)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
