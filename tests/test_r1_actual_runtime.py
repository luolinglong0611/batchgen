"""No accelerator imports: actual-metadata and conservative aggregation tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_actual_runtime_tests", ROOT / "scripts/r1_reproduction/r1_actual_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class Tensor:
    def __init__(self, shape=(128, 128), dtype="torch.float8_e4m3fn", device="cuda:0", meta=False):
        self.shape, self.dtype, self.device, self.is_meta = shape, dtype, device, meta

    def stride(self):
        return tuple(__import__("math").prod(self.shape[index+1:]) for index in range(len(self.shape)))

    def element_size(self):
        return 1 if "float8" in self.dtype else 2

    def cpu(self):
        raise AssertionError("Collector must never read tensor values")

    def item(self):
        raise AssertionError("Collector must never read tensor values")


class Model:
    def __init__(self, meta=False):
        self.model = SimpleNamespace(layers=[])
        self.lm_head = SimpleNamespace(weight=Tensor(), tp_size=1)
        self.weight = Tensor(meta=meta)

    def named_parameters(self):
        yield "lm_head.weight", self.weight


def environment(directory):
    return {"R1_RUNTIME_AUDIT_DIR": str(directory), "R1_RUNTIME_RUN_ID": "run-1",
            "R1_RUNTIME_DEPLOYMENT_SHA": "a" * 64}


def rank_receipts(size=16):
    result = []
    for engine in ("BG", "SG"):
        for rank in range(size):
            result.append({"schema": runtime.SCHEMA, "engine": engine, "phase": "decode", "run_id": "run-1",
                "deployment_sha256": ("a" if engine == "BG" else "b") * 64,
                "input_manifest_binding": "c" * 64, "observed_at_utc": "2026-10-26T10:00:00Z",
                "rank": runtime._observed(rank, "actual global rank"),
                "world_size": runtime._observed(size, "actual group world size"),
                "parameters": {name: runtime._missing("No actual proof") for name in (*runtime.REQUIRED, *runtime.ADDITIONAL_CONTROLLED)},
                "hardware": {"uuid": runtime._observed("GPU-" + str(rank), "actual CUDA property"),
                    "node_hardware_id": runtime._observed("DMI-" + str(rank // 8), "actual DMI property")},
                "receipt_path": "/owned/" + engine + "/rank-" + str(rank) + ".json"})
    return result


def proc_files(values):
    def read(path, *args, **kwargs):
        value = values.get(str(path), FileNotFoundError(str(path)))
        if isinstance(value, Exception):
            raise value
        return value
    return patch.object(Path, "read_text", autospec=True, side_effect=read)


def memory_files():
    return {"/proc/meminfo": "MemTotal:       2147483648 kB\nMemAvailable: 100 kB\n",
            "/proc/self/cgroup": "0::/jobs/run\n",
            "/proc/self/mountinfo": "42 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            "/sys/fs/cgroup/jobs/run/memory.max": "max\n",
            "/sys/fs/cgroup/jobs/memory.max": "max\n",
            "/sys/fs/cgroup/memory.max": "max\n"}


class RuntimeVersionAndHostEvidence(unittest.TestCase):
    def test_loaded_linux_runtime_api_is_bound_to_path_raw_version_and_noload(self):
        library_path = "/installed/nvidia/cuda_runtime/lib/libcudart.so.13"
        maps = (f"1000-2000 r--p 00000000 08:01 21 {library_path}\n"
                f"2000-3000 r-xp 00001000 08:01 21 {library_path}\n")
        def query(pointer):
            runtime.ctypes.cast(pointer, runtime.ctypes.POINTER(runtime.ctypes.c_int)).contents.value = 13000
            return 0
        api = Mock(side_effect=query)
        forbidden = Mock(side_effect=AssertionError("Must not initialize/query a device"))
        torch = SimpleNamespace(version=SimpleNamespace(cuda="99.9"), cuda=SimpleNamespace(
            is_initialized=lambda: False, init=forbidden, cudart=forbidden, get_device_properties=forbidden))
        with patch.object(sys, "platform", "linux"), proc_files({"/proc/self/maps": maps}), \
                patch.object(os, "RTLD_NOLOAD", 4, create=True), patch.object(os, "RTLD_LOCAL", 0, create=True), \
                patch.object(runtime.ctypes, "CDLL", return_value=SimpleNamespace(cudaRuntimeGetVersion=api)) as loader, \
                patch.dict(sys.modules, {"torch": torch}):
            versions = runtime._versions()
        record = versions["cuda_runtime_version"]
        self.assertEqual(record["value"], "13.0")
        self.assertEqual(record["raw_version"], 13000)
        self.assertEqual(record["evidence_level"], "runtime")
        self.assertEqual(versions["torch_cuda_build_NOT_runtime"]["value"], "99.9")
        self.assertEqual(record["loaded_runtime_libraries"][0]["path"], library_path)
        self.assertEqual(record["loaded_runtime_libraries"][0]["mapped_inode"], "21")
        loader.assert_called_once_with(library_path, mode=4)
        api.assert_called_once()
        forbidden.assert_not_called()

    def test_absent_pre12_unversioned_or_deleted_runtime_never_loads_or_initializes(self):
        for path in (None, "/cuda/libcudart.so.11.0", "/cuda/libcudart.so", "/cuda/libcudart.so.13 (deleted)"):
            maps = "" if path is None else f"1000-2000 r-xp 0 08:01 21 {path}\n"
            with self.subTest(path=path), patch.object(sys, "platform", "linux"), \
                    patch.object(os, "RTLD_NOLOAD", 4, create=True), proc_files({"/proc/self/maps": maps}), \
                    patch.object(runtime.ctypes, "CDLL", side_effect=AssertionError("No load")):
                record = runtime._cuda_runtime_version()
            self.assertEqual(record["evidence_level"], "missing")
            self.assertIsNone(record["value"])

    def test_api_failure_missing_symbol_noload_failure_and_conflicting_dsos_stay_missing(self):
        paths = ["/one/libcudart.so.12", "/two/libcudart.so.13"]
        maps = "".join(f"1000-2000 r-xp 0 08:01 {index} {path}\n" for index, path in enumerate(paths, 20))
        for mode in ("return_error", "missing_symbol", "noload_failed", "conflicting_versions"):
            def load(path, mode_flag=None, **kwargs):
                if mode == "missing_symbol":
                    return SimpleNamespace()
                if mode == "noload_failed":
                    raise OSError("Library was unloaded")
                def query(pointer):
                    runtime.ctypes.cast(pointer, runtime.ctypes.POINTER(runtime.ctypes.c_int)).contents.value = 12080 if path == paths[0] else 13000
                    return 1 if mode == "return_error" else 0
                return SimpleNamespace(cudaRuntimeGetVersion=Mock(side_effect=query))
            with self.subTest(mode=mode), patch.object(sys, "platform", "linux"), \
                    patch.object(os, "RTLD_NOLOAD", 4, create=True), patch.object(os, "RTLD_LOCAL", 0, create=True), \
                    proc_files({"/proc/self/maps": maps}), patch.object(runtime.ctypes, "CDLL", side_effect=load):
                record = runtime._cuda_runtime_version()
            self.assertEqual(record["evidence_level"], "missing")
            self.assertEqual(len(record["loaded_runtime_libraries"]), 2)
            if mode == "conflicting_versions":
                self.assertIn("different versions", record["reason"])

    def test_unlimited_records_physical_ceiling_not_available_or_reserved_bytes(self):
        with proc_files(memory_files()):
            record = runtime._host_limit()
        self.assertEqual(record["value"], 2**41)
        self.assertEqual(record["evidence_level"], "runtime")
        self.assertEqual(record["cgroup_memory_limit"]["status"], "unlimited")
        self.assertIsNone(record["cgroup_memory_limit"]["effective_limit_bytes"])
        self.assertEqual(record["physical_memory_bytes"]["value"], 2**41)
        self.assertIn("NOT_available_reserved_or_guaranteed", record["value_semantics"])
        self.assertIn("/sys/fs/cgroup/jobs/run/memory.max", record["evidence"])

    def test_process_and_ancestor_limits_are_both_checked_not_just_mount_root(self):
        for child, parent, expected in ((8 * 2**30, 4 * 2**30, 4 * 2**30),
                                        (2 * 2**30, 4 * 2**30, 2 * 2**30),
                                        ("max", 4 * 2**30, 4 * 2**30)):
            files = memory_files()
            files["/sys/fs/cgroup/jobs/run/memory.max"] = str(child)
            files["/sys/fs/cgroup/jobs/memory.max"] = str(parent)
            with self.subTest(child=child, parent=parent), proc_files(files):
                record = runtime._host_limit()
            self.assertEqual(record["value"], expected)
            self.assertEqual(record["cgroup_memory_limit"]["status"], "finite")
            self.assertEqual(record["cgroup_memory_limit"]["effective_limit_bytes"], expected)
            self.assertEqual(len(record["cgroup_memory_limit"]["observations"]), 3)

    def test_missing_or_hidden_ancestor_and_unknown_membership_keep_observations_but_not_budget(self):
        for mode in ("missing_parent", "hidden_parent", "unreadable_root", "bad_value", "no_membership"):
            files = memory_files()
            if mode == "missing_parent": del files["/sys/fs/cgroup/jobs/memory.max"]
            elif mode == "hidden_parent":
                files["/proc/self/mountinfo"] = "42 23 0:26 /jobs /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
                files["/sys/fs/cgroup/run/memory.max"] = "max"
            elif mode == "unreadable_root": files["/sys/fs/cgroup/memory.max"] = PermissionError("denied")
            elif mode == "bad_value": files["/sys/fs/cgroup/jobs/memory.max"] = "unknown"
            else: del files["/proc/self/cgroup"]
            with self.subTest(mode=mode), proc_files(files):
                record = runtime._host_limit()
            self.assertIsNone(record["value"])
            self.assertEqual(record["evidence_level"], "missing")
            self.assertEqual(record["physical_memory_bytes"]["value"], 2**41)
            self.assertEqual(record["cgroup_memory_limit"]["status"], "unknown")

    def test_actual_v2_root_without_memory_max_is_not_confused_with_missing_parent(self):
        files = memory_files()
        del files["/sys/fs/cgroup/memory.max"]
        with proc_files(files):
            record = runtime._host_limit()
        self.assertEqual(record["cgroup_memory_limit"]["status"], "unlimited")
        self.assertEqual(record["cgroup_memory_limit"]["observations"][-1]["status"], "hierarchy_root_without_limit_file")

    def test_v1_uses_kernel_hierarchical_limit_including_inherited_limit(self):
        unlimited = ((1 << 63) - 1) // 4096 * 4096
        for inherited in (4 * 2**30, unlimited):
            files = memory_files()
            files.update({"/proc/self/cgroup": "0::/\n5:memory:/jobs/run\n",
                "/proc/self/mountinfo": "42 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
                "/sys/fs/cgroup/memory/jobs/run/memory.limit_in_bytes": str(unlimited),
                "/sys/fs/cgroup/memory/jobs/run/memory.stat": f"hierarchical_memory_limit {inherited}\n"})
            with self.subTest(inherited=inherited), proc_files(files), patch.object(os, "sysconf", return_value=4096):
                record = runtime._host_limit()
            self.assertEqual(record["value"], 2**41 if inherited == unlimited else inherited)
            self.assertEqual(record["cgroup_memory_limit"]["status"], "unlimited" if inherited == unlimited else "finite")

    def test_no_physical_data_is_not_replaced_by_a_pool_or_available_memory(self):
        files = memory_files()
        files["/proc/meminfo"] = "MemAvailable: 1234567 kB\n"
        with proc_files(files):
            unlimited = runtime._host_limit()
        self.assertIsNone(unlimited["value"])
        self.assertEqual(unlimited["cgroup_memory_limit"]["status"], "unlimited")
        files["/sys/fs/cgroup/jobs/run/memory.max"] = str(4 * 2**30)
        with proc_files(files):
            finite = runtime._host_limit()
        self.assertEqual(finite["value"], 4 * 2**30)
        self.assertEqual(finite["physical_memory_bytes"]["evidence_level"], "missing")


class RuntimeCollectorContracts(unittest.TestCase):
    def test_disabled_collectors_do_not_read_objects_or_query_cuda_or_write(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("Disabled collector must return before introspection")

        with patch.dict(os.environ, {}, clear=True), patch.object(runtime, "_hardware", side_effect=AssertionError("No query")):
            self.assertIsNone(runtime.record_bg(Forbidden(), "decode"))
            self.assertIsNone(runtime.record_sg(Forbidden(), Forbidden(), Forbidden(), "decode"))

    def test_partial_switches_are_disabled_and_malformed_complete_switches_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            env = environment(folder)
            for key in env:
                partial = dict(env)
                del partial[key]
                with patch.dict(os.environ, partial, clear=True):
                    self.assertIsNone(runtime.record_bg(None, "decode"))
            for key, bad in (("R1_RUNTIME_RUN_ID", "../escape"), ("R1_RUNTIME_DEPLOYMENT_SHA", "main"),
                             ("R1_RUNTIME_AUDIT_DIR", "relative")):
                with patch.dict(os.environ, dict(env, **{key: bad}), clear=True), self.assertRaises(ValueError):
                    runtime.record_bg(None, "decode")

    def test_all_missing_objects_produce_explicit_missing_receipt_without_claiming_alignment(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            receipt = runtime.record_bg(None, "decode")
            self.assertEqual(receipt["model"]["status"], "missing")
            self.assertEqual(receipt["rank"]["evidence_level"], "missing")
            self.assertEqual(receipt["parameters"]["actual_attention_tp"]["evidence_level"], "missing")
            self.assertFalse(receipt["formal_test_permitted"])
            self.assertTrue(Path(receipt["receipt_path"]).is_file())
            self.assertEqual(json.loads(Path(receipt["receipt_path"]).read_text())["model"], receipt["model"])

    def test_configuration_intent_never_becomes_effective_cap_graph_tp_or_page_geometry(self):
        runner = SimpleNamespace(rank=0, world_size=16)
        args = SimpleNamespace(tp_size=16, ep_size=16, page_size=64, max_running_requests=512,
                               context_length=32768, disable_cuda_graph=True, mem_fraction_static=0.85)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            receipt = runtime.record_sg(Model(), runner, args, "decode")
            for field in ("active_cap", "context_cap", "graph_scope", "actual_attention_tp", "page_size", "gpu_memory_budget"):
                self.assertEqual(receipt["parameters"][field]["evidence_level"], "missing", field)
            for field in ("tp_size", "max_running_requests", "disable_cuda_graph", "mem_fraction_static"):
                self.assertEqual(receipt["configured_NOT_effective"][field]["evidence_level"], "source", field)

    def test_tensor_metadata_and_tp_are_actual_and_never_hash_or_read_values(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            owner = SimpleNamespace(model=Model(), rank=0, world_size=16, model_context_length=163840)
            first = runtime.record_bg(owner, "decode")
            second = runtime.record_bg(owner, "decode")
            self.assertNotEqual(first["receipt_path"], second["receipt_path"])
            parameter = first["model"]["parameters"][0]
            self.assertEqual(parameter["shape"], [128, 128])
            self.assertEqual(parameter["strides"], [128, 1])
            self.assertFalse(parameter["values_hashed"])
            self.assertEqual(first["parameters"]["lm_head_tp"]["value"], 1)
            self.assertEqual(first["parameters"]["context_cap"]["value"], 163840)
            self.assertEqual(first["parameters"]["weights_tensor_hash"]["evidence_level"], "missing")

    def test_meta_weights_and_unmaterialized_kv_geometry_are_not_upgraded(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            kv = SimpleNamespace(config=SimpleNamespace(page_size_tokens=64), _k_cache=None)
            owner = SimpleNamespace(model=Model(meta=True), rank=0, world_size=16, gpu_paged_kv_cache_manager=kv)
            receipt = runtime.record_bg(owner, "decode")
            self.assertEqual(receipt["parameters"]["weight_dtype"]["evidence_level"], "missing")
            self.assertEqual(receipt["parameters"]["page_size"]["evidence_level"], "source")
            self.assertEqual(receipt["parameters"]["kv_dtype"]["evidence_level"], "missing")

    def test_actual_bg_cache_geometry_must_match_declared_page_size(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            kv = SimpleNamespace(config=SimpleNamespace(page_size_tokens=64),
                                 _k_cache=Tensor((61, 2, 64, 1, 576), "torch.bfloat16"))
            owner = SimpleNamespace(model=Model(), rank=0, world_size=16, gpu_paged_kv_cache_manager=kv)
            receipt = runtime.record_bg(owner, "decode")
            self.assertEqual(receipt["parameters"]["page_size"]["value"], 64)
            self.assertEqual(receipt["parameters"]["kv_dtype"]["value"], "bfloat16")
            kv.config.page_size_tokens = 1
            receipt = runtime.record_bg(owner, "decode")
            self.assertEqual(receipt["parameters"]["page_size"]["evidence_level"], "missing")

    def test_sparse_module_with_missing_expert_attributes_is_recorded_without_expected_values(self):
        model = Model()
        model.model.layers = [SimpleNamespace(self_attn=SimpleNamespace(num_heads=128),
            mlp=SimpleNamespace(shared_experts=SimpleNamespace(tp_size=1)))]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, environment(folder), clear=True):
            receipt = runtime.record_bg(SimpleNamespace(model=model, rank=0, world_size=16), "decode")
            self.assertEqual(receipt["parameters"]["routed_ep"]["evidence_level"], "missing")
            self.assertEqual(receipt["parameters"]["shared_expert_tp"]["value"], 1)
            self.assertEqual(receipt["parameters"]["actual_attention_tp"]["evidence_level"], "missing")


class RuntimeAggregationContracts(unittest.TestCase):
    def test_host_limit_semantics_paths_and_runtime_library_evidence_survive_aggregation(self):
        with proc_files(memory_files()):
            host = runtime._host_limit()
        receipts = rank_receipts()
        for receipt in receipts:
            receipt["parameters"]["host_memory_bytes"] = copy.deepcopy(host)
            version = runtime._observed("13.0", "mapped DSO cudaRuntimeGetVersion")
            version.update(raw_version=13000, value_semantics="loaded runtime API", loaded_runtime_libraries=[
                {"path": "/actual/libcudart.so.13", "raw_version": 13000, "mapping_evidence": "/proc/self/maps"}])
            receipt["parameters"]["cuda_runtime_version"] = version
        output = runtime.aggregate(receipts)
        for engine in ("BG", "SG"):
            parameters = output["engines"][engine]["parameters"]
            self.assertEqual(parameters["host_memory_bytes"]["value"], 2**41)
            self.assertIn("NOT_available_reserved_or_guaranteed", parameters["host_memory_bytes"]["value_semantics"])
            actual = parameters["host_memory_bytes"]["per_rank_observations"]["0"]
            self.assertEqual(actual["cgroup_memory_limit"]["status"], "unlimited")
            self.assertIn("/sys/fs/cgroup/jobs/run/memory.max", actual["evidence"])
            actual = parameters["cuda_runtime_version"]["per_rank_observations"]["0"]
            self.assertEqual(actual["loaded_runtime_libraries"][0]["path"], "/actual/libcudart.so.13")
            self.assertEqual(actual["raw_version"], 13000)
        self.assertFalse(output["formal_test_permitted"])
        receipts[0]["parameters"]["host_memory_bytes"]["value_semantics"] = "different enforced limit semantics"
        output = runtime.aggregate(receipts)
        self.assertEqual(output["engines"]["BG"]["parameters"]["host_memory_bytes"]["evidence_level"], "missing")

    def test_all_16_ranks_are_required_and_no_correctness_pass_is_synthesized(self):
        receipts = rank_receipts()
        result = runtime.aggregate(receipts)
        self.assertEqual(result["schema"], "r1-alignment-observations-v1")
        self.assertEqual(result["engines"]["BG"]["parameters"]["gpu_count"]["value"], 16)
        self.assertEqual(result["engines"]["SG"]["parameters"]["node_count"]["value"], 2)
        self.assertEqual(result["correctness"], {})
        self.assertFalse(result["formal_test_permitted"])
        self.assertIn("phase scheduling", result["treatments_NOT_controlled"][0])
        for bad in (receipts[:-1], receipts + [copy.deepcopy(receipts[0])]):
            with self.assertRaisesRegex(ValueError, "all unique ranks"):
                runtime.aggregate(bad)

    def test_different_source_config_missing_or_unevidenced_values_cannot_become_runtime(self):
        for mode in ("different", "source", "missing", "no_evidence"):
            receipts = rank_receipts()
            for receipt in receipts:
                receipt["parameters"]["active_cap"] = runtime._observed(512, "actual admission controller")
            target = receipts[0]["parameters"]["active_cap"]
            if mode == "different":
                target["value"] = 513
            elif mode == "source":
                target["evidence_level"] = "source"
            elif mode == "missing":
                target["value"] = None
            else:
                target["evidence"] = []
            with self.subTest(mode=mode):
                result = runtime.aggregate(receipts)
                self.assertEqual(result["engines"]["BG"]["parameters"]["active_cap"]["evidence_level"], "missing")
                self.assertEqual(result["engines"]["SG"]["parameters"]["active_cap"]["value"], 512)

    def test_prefill_decode_runs_duplicate_devices_and_stale_bindings_are_not_hidden(self):
        for mode in ("phase", "run", "source", "world", "rank_bool", "rank_source"):
            receipts = rank_receipts()
            target = receipts[0]
            if mode == "phase": target["phase"] = "prefill"
            elif mode == "run": target["run_id"] = "another-run"
            elif mode == "source": target["deployment_sha256"] = "d" * 64
            elif mode == "world": target["world_size"]["value"] = 8
            elif mode == "rank_bool": target["rank"]["value"] = False
            else: target["rank"]["evidence_level"] = "source"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runtime.aggregate(receipts)
        receipts = rank_receipts()
        receipts[0]["hardware"]["uuid"]["value"] = "GPU-1"
        result = runtime.aggregate(receipts)
        self.assertEqual(result["engines"]["BG"]["parameters"]["gpu_count"]["evidence_level"], "missing")

    def test_device_and_node_identity_need_actual_runtime_evidence_not_names_or_rank_count(self):
        receipts = rank_receipts()
        for receipt in receipts:
            receipt["hardware"]["uuid"]["evidence_level"] = "source"
            receipt["hardware"]["node_hardware_id"] = runtime._missing("No actual DMI identity")
            receipt["hostname"] = "fake-host-" + str(receipt["rank"]["value"])
        result = runtime.aggregate(receipts)
        for engine in ("BG", "SG"):
            for name in ("gpu_count", "gpu_uuids", "node_count", "rank_device_mapping"):
                self.assertEqual(result["engines"][engine]["parameters"][name]["evidence_level"], "missing")

    def test_aggregated_metadata_alone_cannot_pass_existing_alignment_gate(self):
        spec = importlib.util.spec_from_file_location("r1_actual_runtime_gate_test", ROOT / "scripts/r1_reproduction/alignment_audit.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        output = gate.audit(runtime.aggregate(rank_receipts()), as_of_utc="2026-10-26T10:00:01Z")
        self.assertFalse(output["gate"]["formal_test_permitted"])
        self.assertTrue(all(not row["qualified"] for row in output["correctness"]))
        self.assertEqual(output["gate"]["required_parameter_count"], 43)


class WrappedMathMetadata(unittest.TestCase):
    def test_known_wrapper_preserves_outer_missing_and_records_actual_inner_storage(self):
        wrapper_type = type("DeepSeekAttnWrapper", (), {
            "__module__": "batchgen.models.deepseek.deepseekv3.wrappers"})
        wrapper = wrapper_type()
        wrapper.module = SimpleNamespace(tp_size=1, w_kc=Tensor((128, 128, 512), dtype="torch.bfloat16"),
            kv_a_proj_with_mqa=SimpleNamespace(weight=Tensor((576, 7168)), weight_scale_inv=Tensor((5, 56))))
        observed = runtime._module(wrapper)
        self.assertIsNone(observed["attributes"]["tp_size"]["value"])
        inner = observed["wrapped_module"]
        self.assertEqual(inner["attributes"]["tp_size"]["value"], 1)
        self.assertEqual(inner["weight_metadata"]["w_kc"]["shape"], [128, 128, 512])
        self.assertIn("kv_a_proj_with_mqa", inner["weight_metadata"])
        self.assertIn("not proof", observed["wrapped_module_scope"])

    def test_unknown_wrappers_are_not_guessed_and_cycles_remain_missing(self):
        self.assertNotIn("wrapped_module", runtime._module(SimpleNamespace(module=Tensor())))
        wrapper_type = type("DeepSeekExpertWrapper", (), {
            "__module__": "batchgen.models.deepseek.deepseekv3.wrappers"})
        wrapper = wrapper_type()
        wrapper.module = wrapper
        self.assertEqual(runtime._module(wrapper)["wrapped_module"]["status"], "missing")

    def test_loaded_math_receipts_are_observed_without_importing_unloaded_backends(self):
        dense = "scripts.r1_reproduction.sglang_r1_dense_bf16"
        rope = "batchgen_kernels.triton.fused_rmsnorm_rope"
        with patch.dict(sys.modules, {
                dense: SimpleNamespace(dense_backend_receipt=lambda: {"actual_calls": 3}),
                rope: SimpleNamespace(layout_receipt=lambda: {"native_interleaved": {"actual_calls": 61}})}):
            receipts = runtime._loaded_receipts()
        self.assertEqual(receipts[dense]["actual_calls"], 3)
        self.assertEqual(receipts[rope]["native_interleaved"]["actual_calls"], 61)


if __name__ == "__main__":
    unittest.main()
