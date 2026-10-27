#!/usr/bin/env python3
"""Opt-in per-rank R1 metadata receipts and conservative offline aggregation.

Call record_bg(worker, phase) after constructing the phase's actual model, or
record_sg(model, model_runner, server_args, phase). Three explicit environment
variables are required: R1_RUNTIME_AUDIT_DIR, R1_RUNTIME_RUN_ID, and
R1_RUNTIME_DEPLOYMENT_SHA. Without all three, both entrypoints return None
before reading model attributes or importing/querying accelerator libraries.

This collector runs no model, allocation, collective, synchronization, tensor
value hash or GPU-to-host tensor copy. Existing initialized CUDA metadata may
be queried only in enabled mode. Missing attributes remain missing. Configured
intent is retained separately and never upgraded to an effective observation.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import uuid


SCHEMA = "r1-actual-runtime-rank-v1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
ABSENT = object()
REQUIRED = (
    "model_revision", "weights_tensor_hash", "tokenizer_hash", "input_manifest_hash", "weight_dtype",
    "kv_dtype", "actual_attention_tp", "attention_dp", "routed_ep", "expert_tp", "shared_expert_tp",
    "dense_tp", "lm_head_tp", "graph_scope", "attention_kernel", "moe_kernel", "context_cap",
    "active_cap", "prefill_budget", "host_memory_bytes", "nccl_version", "warmup_protocol", "timing_protocol",
    "gpu_uuids", "node_count", "gpu_count", "page_size", "gpu_memory_budget", "python_version",
    "torch_version", "triton_version", "cuda_runtime_version", "driver_version", "gemm_kernel",
    "quantization_policy", "generation_policy", "stop_policy", "prefix_reuse", "speculative_decoding",
)
TREATMENTS = (
    "phase scheduling and actual batch trajectory", "weight residency and offload/streaming",
    "prefill pure-DP versus resident-EP strategy", "KV transfer/placement and page reuse",
)
ADDITIONAL_CONTROLLED = ("kernel_page_size", "rank_device_mapping", "attention_group_memberships",
                         "routed_expert_assignments")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list, range) and len(value) <= 100000:
        return [_json(item) for item in value]
    if type(value) is dict and len(value) <= 100000:
        return {str(key): _json(item) for key, item in value.items()}
    raise ValueError("Attribute is not bounded finite Python metadata")


def _get(obj, path):
    try:
        for part in path.split("."):
            obj = obj.get(part, ABSENT) if type(obj) is dict else getattr(obj, part, ABSENT)
            if obj is ABSENT:
                return ABSENT
        return obj
    except Exception:
        return ABSENT


def _read(obj, *paths):
    for path in paths:
        value = _get(obj, path)
        if value is not ABSENT and value is not None:
            try:
                return {"value": _json(value), "evidence_level": "runtime", "evidence": [path]}
            except ValueError:
                continue
    return _missing("Missing bounded attribute: " + ", ".join(paths))


def _missing(reason):
    return {"value": None, "evidence_level": "missing", "evidence": [], "reason": reason}


def _observed(value, evidence):
    return {"value": _json(value), "evidence_level": "runtime", "evidence": [evidence]}


def tensor_metadata(tensor):
    """Read only tensor metadata, never its values, pointer or storage bytes."""
    if tensor is ABSENT or tensor is None:
        return {"status": "missing"}
    try:
        shape = [int(value) for value in tensor.shape]
        stride = [int(value) for value in tensor.stride()]
        return {"status": "observed", "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": shape, "strides": stride, "device": str(tensor.device),
                "numel": math.prod(shape), "element_size": int(tensor.element_size()),
                "is_meta": bool(getattr(tensor, "is_meta", False)), "values_hashed": False}
    except Exception as error:
        return {"status": "missing", "reason": "Tensor metadata unavailable: " + type(error).__name__}


def _source(value):
    try:
        path = inspect.getsourcefile(getattr(value, "__func__", value))
        return {"path": str(Path(path).resolve()), "sha256": _sha(path)} if path else {"status": "missing"}
    except Exception as error:
        return {"status": "missing", "reason": type(error).__name__}


def _group(group):
    if group is ABSENT or group is None:
        return {"status": "missing", "reason": "No actual group object"}
    record = {name: _read(group, name) for name in ("rank", "rank_in_group", "world_size", "ranks")}
    if record["ranks"]["value"] is None:
        dist = sys.modules.get("torch.distributed")
        try:
            if dist is not None and dist.is_initialized():
                ranks = dist.get_process_group_ranks(group)
                record["ranks"] = _observed(ranks, "torch.distributed.get_process_group_ranks(actual_group)")
        except Exception as error:
            record["membership_error"] = type(error).__name__
    record["status"] = "observed_object"
    return record


def _module(module, _depth=0, _seen=()):
    if module is ABSENT or module is None:
        return {"status": "missing"}
    if id(module) in _seen or _depth > 2:
        return {"status": "missing", "reason": "Wrapper cycle or observation depth limit"}
    result = {"status": "observed", "class": type(module).__module__ + "." + type(module).__qualname__,
              "class_source": _source(type(module)),
              "attributes": {name: _read(module, name) for name in (
                  "tp_size", "tp_rank", "ep_size", "ep_rank", "moe_ep_size", "num_heads", "num_local_heads",
                  "num_kv_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                  "routed_expert_start_idx", "routed_expert_end_idx", "experts_per_rank", "num_local_experts",
                  "start_expert_id", "end_expert_id", "enable_ep_offloading", "_r1_local_shared_tp1")},
              "groups": {name: _group(_get(module, name)) for name in ("tp_group", "ep_group", "comm.group")},
              "entrypoints": {name: _source(_get(module, name)) for name in (
                  "forward", "_forward_impl", "deepgemm_forward", "prefill_attn",
                  "prefill_attn_w8a16", "prefill_attn_prepacked", "prefill_attn_w8a16_prepacked",
                  "decoding_attn_mode_3_bf16")
                  if _get(module, name) is not ABSENT},
              "weight_metadata": {}}
    start, stop = (_get(module, name) for name in ("routed_expert_start_idx", "routed_expert_end_idx"))
    if type(start) is int and type(stop) is int and 0 <= start <= stop <= 100000:
        result["routed_expert_ids"] = _observed(list(range(start, stop)), "actual module routed_expert_start_idx/end_idx")
    else:
        result["routed_expert_ids"] = _missing("No explicit contiguous routed expert interval; EP rank alone is insufficient")
    for name in ("weight", "weight_scale", "weight_scale_inv", "q_absorb", "out_absorb", "w_kc", "w_vc", "w13_weight", "w2_weight",
                 "w13_weight_scale_inv", "w2_weight_scale_inv"):
        value = _get(module, name)
        if value is not ABSENT:
            result["weight_metadata"][name] = tensor_metadata(value)
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "gate_up_proj", "gate_proj", "up_proj", "down_proj"):
        child = _get(module, name)
        if child is not ABSENT:
            result["weight_metadata"][name] = {"tp_size": _read(child, "tp_size"),
                "weight": tensor_metadata(_get(child, "weight")),
                "weight_scale_inv": tensor_metadata(_get(child, "weight_scale_inv"))}
    if result["class"] in (
            "batchgen.models.deepseek.deepseekv3.wrappers.DeepSeekExpertWrapper",
            "batchgen.models.deepseek.deepseekv3.wrappers.DeepSeekAttnWrapper"):
        result["wrapped_module"] = _module(_get(module, "module"), _depth + 1, (*_seen, id(module)))
        result["wrapped_module_scope"] = (
            "Actual stored inner object and bound methods after forward; not proof that every method ran, "
            "nor that streamed weights remained materialized. Outer missing fields are not filled from this object.")
    return result


def _model_metadata(model):
    layers = _get(model, "model.layers")
    if layers is ABSENT:
        layers = _get(model, "layers")
    result = {"status": "missing" if model is ABSENT or model is None else "observed", "layers": [],
              "lm_head": _module(_get(model, "lm_head")), "parameters": [], "parameter_walk_complete": False}
    if layers is not ABSENT and layers is not None:
        try:
            for index, layer in enumerate(layers):
                mlp = _get(layer, "mlp")
                result["layers"].append({"layer_index": index, "attention": _module(_get(layer, "self_attn")),
                    "mlp": _module(mlp), "shared": _module(_get(mlp, "shared_experts")),
                    "routed": _module(_get(mlp, "experts"))})
        except Exception as error:
            result["layer_walk_error"] = type(error).__name__
    try:
        for name, tensor in model.named_parameters():
            result["parameters"].append({"name": name, **tensor_metadata(tensor)})
        result["parameter_walk_complete"] = bool(result["parameters"]) and all(
            entry["status"] == "observed" for entry in result["parameters"])
    except Exception as error:
        result["parameter_walk_error"] = type(error).__name__
    return result


def _known_groups():
    result = {}
    # Consult only modules the running engine has already imported. These are
    # local group metadata getters, never collectives or group constructors.
    for module_name in ("sglang.srt.distributed.parallel_state", "sglang.srt.layers.dp_attention"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for getter in ("get_tp_group", "get_attention_tp_group", "get_attention_dp_group", "get_moe_tp_group", "get_moe_ep_group"):
            function = getattr(module, getter, None)
            if callable(function):
                try:
                    result[module_name + "." + getter] = _group(function())
                except Exception as error:
                    result[module_name + "." + getter] = {"status": "missing", "reason": type(error).__name__}
    return result


def _loaded_receipts():
    result = {}
    for module_name, function in (
        ("batchgen.gemm.r1_sm103_compat", "backend_receipt"),
        ("scripts.r1_reproduction.sglang_r1_alignment_runtime", "linear_backend_receipt"),
        ("scripts.r1_reproduction.sglang_r1_attention_compat", "receipt"),
        ("scripts.r1_reproduction.sglang_r1_dense_bf16", "dense_backend_receipt"),
        ("batchgen_kernels.triton.fused_rmsnorm_rope", "layout_receipt"),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            try:
                result[module_name] = _json(getattr(module, function)())
            except Exception as error:
                result[module_name] = {"status": "unavailable", "reason": type(error).__name__}
    return result


def _hardware(device):
    result = {"uuid": _missing("No initialized CUDA metadata"), "device_total_bytes": _missing("No initialized CUDA metadata"),
              "capability": _missing("No initialized CUDA metadata"), "node_hardware_id": _missing("No readable DMI product UUID")}
    path = Path("/sys/devices/virtual/dmi/id/product_uuid")
    try:
        value = path.read_text().strip()
        if value:
            result["node_hardware_id"] = _observed(value, str(path))
    except OSError:
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return result
        props = torch.cuda.get_device_properties(device)
        actual_uuid = getattr(props, "uuid", None)
        if actual_uuid:
            result["uuid"] = _observed(str(actual_uuid), "torch.cuda.get_device_properties(actual_device).uuid")
        result["device_total_bytes"] = _observed(int(props.total_memory), "CUDA actual device total_memory")
        result["capability"] = _observed([props.major, props.minor], "CUDA actual device properties, not NVML marketing name")
        result["current_usage_NOT_budget"] = {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
                                            "reserved_bytes": int(torch.cuda.memory_reserved(device))}
    except Exception as error:
        result["query_error"] = type(error).__name__
    return result


def _cuda_runtime_version():
    """Query only an already-mapped CUDA 12+ DSO; never load/init CUDA.

    NVIDIA documents cudaRuntimeGetVersion as non-initializing since CUDA 12:
    https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html
    Earlier/unversioned DSOs remain unknown rather than risking initialization.
    """
    observations = []
    result = _missing("No safely queryable already-loaded CUDA runtime")
    result["loaded_runtime_libraries"] = observations
    if sys.platform != "linux" or not hasattr(os, "RTLD_NOLOAD"):
        result["reason"] = "Loaded-DSO version query requires Linux RTLD_NOLOAD"
        return result
    try:
        mappings = Path("/proc/self/maps").read_text()
    except OSError as error:
        result["reason"] = "Cannot inspect loaded runtime mappings: " + type(error).__name__
        return result
    candidates = {}
    for line in mappings.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or "libcudart.so" not in fields[5]:
            continue
        path = fields[5]
        # Keep the mapped device/inode as well as path; repeated segments are
        # one DSO. Deleted/renamed/unversioned libraries must not be reopened.
        candidates[(fields[3], fields[4], path)] = path
    for (device, inode, path), _ in sorted(candidates.items()):
        item = {"path": path, "mapped_device": device, "mapped_inode": inode,
                "mapping_evidence": "/proc/self/maps", "status": "unverified"}
        observations.append(item)
        match = re.fullmatch(r"libcudart\.so\.(\d+)(?:\.\d+)*", Path(path).name)
        if not path.startswith("/") or match is None or int(match[1]) < 12:
            item["reason"] = "Not an identifiable CUDA 12+ mapped DSO; API not called"
            continue
        try:
            library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            query = library.cudaRuntimeGetVersion
            query.argtypes = [ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            raw = ctypes.c_int()
            code = query(ctypes.byref(raw))
            item.update(api="cudaRuntimeGetVersion", return_code=code,
                        raw_version=raw.value, loader_mode="RTLD_NOLOAD|RTLD_LOCAL")
            if code != 0 or raw.value < 12000 or raw.value % 10:
                item["reason"] = "Runtime version API failed or returned an invalid/unsupported version"
                continue
            item.update(status="observed", version=f"{raw.value // 1000}.{raw.value % 1000 // 10}")
        except (OSError, AttributeError, TypeError, ValueError) as error:
            item["reason"] = "Existing-DSO query unavailable: " + type(error).__name__
    if observations and all(item["status"] == "observed" for item in observations):
        values = {item["raw_version"] for item in observations}
        if len(values) == 1:
            result = _observed(observations[0]["version"],
                "; ".join(item["path"] + " mapped in /proc/self/maps; RTLD_NOLOAD cudaRuntimeGetVersion"
                          for item in observations))
            result.update(raw_version=observations[0]["raw_version"], loaded_runtime_libraries=observations,
                          value_semantics="loaded CUDA runtime API major.minor; not Torch build CUDA")
        else:
            result["reason"] = "Multiple loaded CUDA runtimes report different versions"
    return result


def _versions():
    result = {"python_version": _observed(platform.python_version(), "running CPython platform.python_version()")}
    for module_name, field in (("torch", "torch_version"), ("triton", "triton_version")):
        module = sys.modules.get(module_name)
        version = getattr(module, "__version__", None) if module else None
        result[field] = _observed(str(version), "loaded " + module_name + ".__version__") if version else _missing("Module not loaded")
    torch = sys.modules.get("torch")
    result["torch_cuda_build_NOT_runtime"] = _read(torch, "version.cuda")
    result["cuda_runtime_version"] = _cuda_runtime_version()
    result["nccl_version"] = _missing("NCCL runtime version unavailable")
    try:
        if torch and torch.cuda.is_initialized():
            version = torch.cuda.nccl.version()
            result["nccl_version"] = _observed(".".join(map(str, version)), "loaded torch.cuda.nccl.version()")
    except Exception:
        pass
    result["driver_version"] = _missing("No readable loaded NVIDIA kernel driver version")
    try:
        path = Path("/proc/driver/nvidia/version")
        match = re.search(r"Kernel Module[^\n]*?\b(\d{3,}(?:\.\d+)+)", path.read_text())
        if match:
            result["driver_version"] = _observed(match.group(1), str(path))
    except OSError:
        pass
    return result


def _cgroup_memory_limit():
    """Resolve this process, not the mount root; retain incomplete evidence."""
    result = {"status": "unknown", "observations": [], "effective_limit_bytes": None,
              "evidence": ["/proc/self/cgroup", "/proc/self/mountinfo"]}
    try:
        memberships = []
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if "memory" in controllers.split(","):
                memberships.append(("v1", path))
            elif hierarchy == "0" and not controllers:
                memberships.append(("v2", path))
        # Hybrid hierarchies can have both; the explicit v1 memory controller
        # owns memory accounting in that case.
        explicit = [entry for entry in memberships if entry[0] == "v1"]
        memberships = explicit or memberships
        if len(memberships) != 1:
            raise ValueError("No unique process memory-controller membership")
        version, process_path = memberships[0]
        result.update(version=version, process_cgroup_path=process_path)
        if not process_path.startswith("/") or ".." in Path(process_path).parts:
            raise ValueError("Unresolvable process cgroup path")
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            left, right = before.split(), after.split()
            if ((version == "v2" and right[0] == "cgroup2") or
                    (version == "v1" and right[0] == "cgroup" and "memory" in right[2].split(","))):
                unescape = lambda value: re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
                root, mount = map(unescape, (left[3], left[4]))
                try:
                    relative = Path(process_path).relative_to(root)
                except ValueError:
                    continue
                if Path(root).is_absolute() and Path(mount).is_absolute():
                    mounts.append((root, mount, relative))
        if not mounts:
            raise ValueError("No mapped memory-controller mount for the process cgroup")
        root, mount, relative = min(mounts, key=lambda item: (len(Path(item[0]).parts), item[1]))
        result.update(mount_root=root, mount_point=mount, resolved_process_path=str(Path(mount) / relative))
        current = Path(mount) / relative
        if version == "v1":
            # The kernel-provided hierarchical limit includes inherited v1
            # limits, unlike blindly reading memory.limit_in_bytes at a root.
            local_path = current / "memory.limit_in_bytes"
            local = local_path.read_text().strip()
            result["observations"].append({"path": str(local_path), "raw": local})
            stat_path = current / "memory.stat"
            stats = dict(line.split() for line in stat_path.read_text().splitlines())
            raw = stats.get("hierarchical_memory_limit", "")
            result["observations"].append({"path": str(stat_path), "field": "hierarchical_memory_limit", "raw": raw})
            if not raw.isdigit() or int(raw) <= 0 or not local.isdigit() or int(local) <= 0:
                raise ValueError("Invalid/missing v1 local or hierarchical memory limit")
            limit = min(int(local), int(raw))
            page = os.sysconf("SC_PAGE_SIZE")
            unlimited = ((1 << 63) - 1) // page * page
            result.update(status="unlimited" if limit == unlimited else "finite",
                          effective_limit_bytes=None if limit == unlimited else limit,
                          limit_semantics="kernel v1 hierarchical_memory_limit intersected with process local limit")
            return result
        if root != "/":
            # Still inspect visible ancestors, but never call a subtree mount
            # unrestricted when potentially stricter ancestors are hidden.
            result["reason"] = "Mount hides ancestors above its root; effective limit unknown"
        complete, finite = root == "/", []
        while True:
            path = current / "memory.max"
            try:
                raw = path.read_text().strip()
                observation = {"path": str(path), "raw": raw}
                result["observations"].append(observation)
                if raw == "max":
                    observation["status"] = "unlimited"
                elif raw.isdigit() and int(raw) > 0:
                    finite.append(int(raw))
                    observation.update(status="finite", limit_bytes=int(raw))
                else:
                    complete = False
                    observation["status"] = "invalid"
            except OSError as error:
                # The actual cgroup-v2 hierarchy root has no memory.max file.
                is_root = current == Path(mount) and root == "/" and isinstance(error, FileNotFoundError)
                result["observations"].append({"path": str(path),
                    "status": "hierarchy_root_without_limit_file" if is_root else "unreadable",
                    "error": type(error).__name__})
                complete = complete and is_root
            if current == Path(mount):
                break
            current = current.parent
        if complete:
            result.update(status="finite" if finite else "unlimited",
                          effective_limit_bytes=min(finite) if finite else None,
                          limit_semantics="minimum memory.max over process cgroup and visible hierarchy ancestors")
        else:
            result.setdefault("reason", "Incomplete ancestor observations; effective limit unknown")
    except (OSError, ValueError, IndexError) as error:
        result["reason"] = "Cannot prove process cgroup memory limit: " + str(error)
    return result


def _host_limit():
    physical = _missing("No observed host physical MemTotal")
    try:
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", Path("/proc/meminfo").read_text(), re.MULTILINE)
        if match and int(match[1]) > 0:
            physical = _observed(int(match[1]) * 1024, "/proc/meminfo MemTotal (kB * 1024)")
    except OSError:
        pass
    cgroup = _cgroup_memory_limit()
    # Preserve the existing positive-integer parameter schema. This value is
    # an observed ceiling, never available/reserved/guaranteed allocatable RAM.
    if cgroup["status"] == "finite":
        ceiling = cgroup["effective_limit_bytes"]
        if physical["value"] is not None:
            ceiling = min(ceiling, physical["value"])
        result = _observed(ceiling, "process cgroup effective memory limit; intersected with observed physical MemTotal when available")
        semantics = "finite_cgroup_ceiling_intersect_physical_NOT_available_reserved_or_guaranteed"
    elif cgroup["status"] == "unlimited" and physical["value"] is not None:
        result = _observed(physical["value"], "observed cgroup unlimited; /proc/meminfo MemTotal physical ceiling only")
        semantics = "physical_ceiling_cgroup_unlimited_NOT_available_reserved_or_guaranteed"
    else:
        result = _missing("Effective host ceiling unverified; retain observed physical/cgroup metadata")
        semantics = "unknown_effective_ceiling"
    result.update(value_semantics=semantics, physical_memory_bytes=physical, cgroup_memory_limit=cgroup)
    result["evidence"].extend(cgroup["evidence"] + [item["path"] for item in cgroup["observations"]])
    return result


def _settings(obj, paths):
    result = {}
    for path in paths:
        record = _read(obj, path)
        if record["value"] is not None:
            record["evidence_level"] = "source"
            record["reason"] = "Observed configuration attribute; enforcement/dispatch not established"
        result[path] = record
    return result


def _enabled():
    directory, run, deployment = (os.environ.get(name) for name in (
        "R1_RUNTIME_AUDIT_DIR", "R1_RUNTIME_RUN_ID", "R1_RUNTIME_DEPLOYMENT_SHA"))
    if not directory or not run or not deployment:
        return None
    if not Path(directory).is_absolute() or not TOKEN.fullmatch(run) or not HASH.fullmatch(deployment):
        raise ValueError("Runtime audit requires absolute directory, safe run ID and SHA256 deployment binding")
    return Path(directory), run, deployment


def _common_value(records, label):
    if not records or any(record.get("evidence_level") != "runtime" or record.get("value") is None
                          or not record.get("evidence") or not all(type(item) is str and item.strip()
                              for item in record["evidence"]) for record in records):
        return _missing("Incomplete per-module evidence for " + label)
    values = [json.dumps(record["value"], sort_keys=True, allow_nan=False) for record in records]
    if len(set(values)) != 1 or len({type(record["value"]) for record in records}) != 1:
        return _missing("Nonuniform per-module evidence for " + label)
    return _observed(records[0]["value"], "all relevant observed module attributes: " + label)


def _tp(module):
    if module.get("status") != "observed":
        return _missing("No actual module")
    direct = module["attributes"]["tp_size"]
    if type(direct["value"]) is int and direct["value"] > 0:
        return direct
    sizes = []
    for value in module.get("weight_metadata", {}).values():
        if type(value) is dict and type(value.get("tp_size", {}).get("value")) is int:
            sizes.append(value["tp_size"])
    return _common_value(sizes, "actual child linear.tp_size")


def _collect(engine, model, owner, phase, settings):
    enabled = _enabled()
    if enabled is None:
        return None
    directory, run, deployment = enabled
    if not TOKEN.fullmatch(phase):
        raise ValueError("Safe explicit phase name required")
    metadata = _model_metadata(model)
    rank = _read(owner, "rank", "global_rank", "args.global_rank")
    world = _read(owner, "world_size", "parallel_manager.world_size", "args.world_size")
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_initialized():
            rank = _observed(int(dist.get_rank()), "initialized torch.distributed global rank")
            world = _observed(int(dist.get_world_size()), "initialized torch.distributed world size")
    except Exception:
        pass
    device = _get(owner, "device")
    if device is ABSENT:
        device = None
    parameters = {name: _missing("No qualified effective runtime observation") for name in (*REQUIRED, *ADDITIONAL_CONTROLLED)}
    versions = _versions()
    parameters.update({name: record for name, record in versions.items() if name in parameters})
    hardware = _hardware(device)
    parameters["host_memory_bytes"] = _host_limit()
    layers = metadata["layers"]
    if layers:
        parameters["actual_attention_tp"] = _common_value([_tp(layer["attention"]) for layer in layers], "attention TP")
        sparse = [layer for layer in layers if layer["shared"]["status"] == "observed"]
        dense = [layer for layer in layers if layer["shared"]["status"] != "observed"
                 and layer["routed"]["status"] != "observed"]
        parameters["shared_expert_tp"] = _common_value([_tp(layer["shared"]) for layer in sparse], "shared TP")
        parameters["dense_tp"] = _common_value([_tp(layer["mlp"]) for layer in dense], "dense TP")
        parameters["expert_tp"] = _common_value([_tp(layer["routed"]) for layer in sparse], "routed expert TP")
        ep_records = []
        for layer in sparse:
            candidates = [layer["routed"].get("attributes", {}).get("ep_size", _missing("No actual routed module")), layer["mlp"]["attributes"]["ep_size"],
                          layer["mlp"]["attributes"]["moe_ep_size"]]
            ep_records.append(next((entry for entry in candidates if type(entry["value"]) is int), _missing("No actual EP size")))
        parameters["routed_ep"] = _common_value(ep_records, "routed EP")
    parameters["lm_head_tp"] = _tp(metadata["lm_head"])
    if metadata["parameter_walk_complete"] and not any(entry["is_meta"] for entry in metadata["parameters"]):
        dtypes = sorted({entry["dtype"] for entry in metadata["parameters"] if "scale" not in entry["name"]})
        if dtypes:
            parameters["weight_dtype"] = _observed("mixed:" + ",".join(dtypes) if len(dtypes) > 1 else dtypes[0],
                                                    "complete model.named_parameters metadata, scales excluded by name")
    receipts = _loaded_receipts()
    common_gemm = receipts.get("batchgen.gemm.r1_sm103_compat", {})
    if common_gemm.get("actual_backends"):
        parameters["gemm_kernel"] = _observed(json.dumps(common_gemm["actual_backends"], sort_keys=True),
                                               "already-loaded common GEMM completed dispatch receipt")
    sg_attention = receipts.get("scripts.r1_reproduction.sglang_r1_attention_compat", {})
    mla_plans = [wrapper.get("last_plan", {}) for wrapper in sg_attention.get("wrappers", [])
                 if wrapper.get("kind") == "mla" and wrapper.get("completed_run_api_calls", 0) > 0]
    if mla_plans and all(type(plan.get("page_size")) is int for plan in mla_plans):
        parameters["kernel_page_size"] = _common_value(
            [_observed(plan["page_size"], "actual completed FI MLA plan/run receipt") for plan in mla_plans], "FI kernel page size")
    kv_owner = _get(owner, "gpu_paged_kv_cache_manager") if engine == "BG" else _get(owner, "token_to_kv_pool")
    allocator = _get(owner, "token_to_kv_pool_allocator") if engine == "SG" else kv_owner
    page = _read(allocator, "config.page_size_tokens", "page_size")
    parameters["page_size"] = dict(page)
    kv = {name: _read(allocator, name) for name in ("config.num_pages", "config.page_size_tokens", "page_size", "size", "_is_initialized")}
    cache = _get(kv_owner, "_k_cache")
    if cache is ABSENT:
        for name in ("kv_buffer", "k_buffer"):
            value = _get(kv_owner, name)
            if isinstance(value, (list, tuple)) and value:
                cache = value[0]
                break
    kv["cache_tensor"] = tensor_metadata(cache)
    if kv["cache_tensor"]["status"] == "observed" and not kv["cache_tensor"]["is_meta"]:
        parameters["kv_dtype"] = _observed(kv["cache_tensor"]["dtype"], "actual allocated KV tensor metadata")
        tensor = kv["cache_tensor"]
        kv["materialized_first_or_combined_cache_bytes_NOT_total_budget"] = tensor["numel"] * tensor["element_size"]
        if engine == "BG" and (len(tensor["shape"]) != 5 or tensor["shape"][2] != page["value"]):
            parameters["page_size"] = _missing("Allocator geometry does not match actual BG cache shape")
    elif page["value"] is not None:
        parameters["page_size"].update(evidence_level="source",
            reason="Allocator declaration exists but materialized cache geometry is unverified")
    parameters["context_cap"] = _read(owner, "model_context_length") if engine == "BG" else _read(owner, "max_req_len", "context_len")
    known_groups = _known_groups()
    attention_groups = {name.rsplit(".", 1)[-1]: record.get("ranks") for name, record in known_groups.items()
                        if name.endswith((".get_attention_tp_group", ".get_attention_dp_group"))}
    if attention_groups and all(record and record["evidence_level"] == "runtime" for record in attention_groups.values()):
        parameters["attention_group_memberships"] = _observed(
            {name: record["value"] for name, record in attention_groups.items()}, "actual already-constructed attention groups")
    sparse_layers = [layer for layer in layers if layer["shared"]["status"] == "observed"]
    assignments = {}
    for layer in sparse_layers:
        candidates = [layer["mlp"]["routed_expert_ids"], layer["routed"].get("routed_expert_ids", _missing("No actual routed module"))]
        assignments[str(layer["layer_index"])] = next((record for record in candidates if record["evidence_level"] == "runtime"),
                                                      _missing("No explicit expert ownership metadata"))
    if assignments and all(record["evidence_level"] == "runtime" for record in assignments.values()):
        parameters["routed_expert_assignments"] = _observed(
            {name: record["value"] for name, record in assignments.items()}, "every observed sparse layer's explicit routed expert IDs")
    for getter, field in (("get_attention_tp_group", "actual_attention_tp"), ("get_attention_dp_group", "attention_dp"),
                          ("get_moe_tp_group", "expert_tp"), ("get_moe_ep_group", "routed_ep")):
        matching = [record["ranks"] for name, record in known_groups.items()
                    if name.endswith("." + getter) and "ranks" in record]
        members = _common_value(matching, getter + " observed members")
        value = members["value"]
        if type(value) is list and value and all(type(item) is int for item in value) and len(set(value)) == len(value):
            previous = parameters[field]["value"]
            if previous is None or previous == len(value):
                parameters[field] = _observed(len(value), getter + " actual member count")
            else:
                parameters[field] = _missing("Actual module TP/EP metadata conflicts with group membership")
    manager = _get(owner, "parallel_manager")
    scales = _get(manager, "dequant_scale")
    manager_scales = {str(name): tensor_metadata(value) for name, value in scales.items()} if type(scales) is dict else {"status": "missing"}
    # Never use max_pool_size, max-running CLI intent, buffer rows or KV capacity
    # as proof that a global active-request admission limit is enforced.
    receipt = {"schema": SCHEMA, "engine": engine, "phase": phase,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run, "deployment_sha256": deployment,
        "input_manifest_binding": os.environ.get("R1_RUNTIME_INPUT_MANIFEST_SHA256"),
        "binding_provenance": "coordinator supplied deployment/input identifiers; loaded source identities retained separately",
        "collector_source": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
        "hostname": socket.gethostname(), "pid": os.getpid(), "rank": rank, "world_size": world,
        "model": metadata, "parameters": parameters, "configured_NOT_effective": settings,
        "hardware": hardware, "versions": versions, "KV": kv,
        "manager_scale_metadata": manager_scales,
        "graph_state_NOT_full_scope_proof": {"fields": {name: _read(owner, name) for name in (
            "_whole_model_graph", "_decode_graph_captures", "_glm5_whole_model_graph")},
            "configuration": _settings(owner, ("engine_config.Basic_Config.enable_cuda_graphs",)),
            "manager_objects_present": {name: (_missing("Attribute absent") if _get(owner, name) is ABSENT
                else _observed(_get(owner, name) is not None, "actual owner." + name + " object presence"))
                for name in ("_cuda_graph_manager", "cuda_graph_runner", "piecewise_cuda_graph_runner")},
            "reason": "Flags/manager presence alone do not prove every execution scope was eager"},
        "observed_groups": {name: _group(_get(owner, name)) for name in ("tp_group", "ep_group", "comm.group", "parallel_manager.comm.group")},
        "loaded_engine_groups": known_groups,
        "existing_dispatch_receipts": receipts,
        "treatments_NOT_controlled": {"categories": list(TREATMENTS),
            "phase": phase, "manager": _settings(_get(owner, "parallel_manager"), (
                "enable_ep_offloading", "local_routed_experts", "host_routed_experts", "num_tokens_per_rank"))},
        "scope": "metadata only; no tensor-value equivalence, enforcement, correctness or formal qualification",
        "formal_test_permitted": False}
    directory.mkdir(parents=True, exist_ok=True)
    label = str(rank["value"]) if type(rank["value"]) is int else "missing-rank"
    path = directory / (engine + "." + run + "." + phase + ".rank-" + label + "." + uuid.uuid4().hex + ".json")
    receipt["receipt_path"] = str(path)
    with path.open("x") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return receipt


def record_bg(worker, phase):
    if _enabled() is None:
        return None
    model = _get(worker, "model")
    if model is ABSENT or model is None:
        model = _get(worker, "parallel_manager.model")
    settings = _settings(worker, (
        "engine_config.Basic_Config.enable_cuda_graphs", "engine_config.Basic_Config.max_context_length",
        "args.disable_cuda_graphs", "args.enable_cuda_graph", "gpu_memory_frac", "gpu_kv_cache_size_gb",
        "_max_pool_size", "max_context_length", "max_input_length", "max_decoding_length",
        "engine_config.Module_Batching_Config.attn_prefill_micro_batch_size", "engine_config.TP_Config.tp_size",
        "engine_config.EP_Config.ep_size", "host_kv_cache_size", "global_host_kv_cache_size_gb"))
    return _collect("BG", model, worker, phase, settings)


def record_sg(model, model_runner=None, server_args=None, phase="unknown"):
    if _enabled() is None:
        return None
    settings = _settings(server_args, (
        "tp_size", "dp_size", "ep_size", "enable_dp_attention", "moe_dense_tp_size", "enable_dp_lm_head",
        "disable_cuda_graph", "enable_piecewise_cuda_graph", "context_length", "max_running_requests",
        "max_total_tokens", "mem_fraction_static", "chunked_prefill_size", "page_size", "attention_backend",
        "moe_runner_backend", "disable_radix_cache", "speculative_algorithm"))
    return _collect("SG", model, model_runner, phase, settings)


def aggregate(receipts, expected_ranks=16):
    """Emit audit input without promoting configuration, partial ranks or fixtures.

    Require one receipt per rank per engine for one explicit phase. Call this
    separately for prefill/decode; phase-dependent topology must not be merged.
    Metadata agreement is not correctness qualification; gate input retains
    empty correctness and missing model/tokenizer/input value hashes.
    """
    if type(expected_ranks) is not int or expected_ranks <= 0:
        raise ValueError("Expected rank count must be a positive integer")
    if not receipts or any(row.get("schema") != SCHEMA or row.get("engine") not in ("BG", "SG") for row in receipts):
        raise ValueError("Expected nonempty runtime rank receipts")
    for key in ("run_id", "phase"):
        if len({row.get(key) for row in receipts}) != 1:
            raise ValueError("Never mix run IDs or prefill/decode phases")
    result = {"schema": "r1-alignment-observations-v1", "engines": {}, "correctness": {},
              "additional_required_parameters": list(ADDITIONAL_CONTROLLED),
              "qualification_expected_bindings": {}, "aggregation": {"expected_ranks": expected_ranks,
                  "run_id": receipts[0]["run_id"], "phase": receipts[0]["phase"],
                  "scope": "complete per-rank metadata agreement only_NOT_model_qualified", "engines": {}},
              "treatments_NOT_controlled": list(TREATMENTS), "formal_test_permitted": False}
    for engine in ("BG", "SG"):
        rows = [row for row in receipts if row.get("engine") == engine]
        ranks = [row.get("rank", {}).get("value") for row in rows]
        if (len(rows) != expected_ranks or any(type(rank) is not int for rank in ranks)
                or set(ranks) != set(range(expected_ranks))
                or any(row.get("rank", {}).get("evidence_level") != "runtime" for row in rows)):
            raise ValueError(engine + " requires exactly all unique ranks 0..N-1")
        rows.sort(key=lambda row: row["rank"]["value"])
        if any(type(row.get("world_size", {}).get("value")) is not int
               or row["world_size"]["value"] != expected_ranks
               or row["world_size"].get("evidence_level") != "runtime" for row in rows):
            raise ValueError(engine + " lacks matching actual world-size evidence")
        deployments = {row.get("deployment_sha256") for row in rows}
        bindings = {row.get("input_manifest_binding") for row in rows}
        if len(deployments) != 1 or any(type(value) is not str or not HASH.fullmatch(value) for value in deployments) or len(bindings) != 1:
            raise ValueError(engine + " deployment/input bindings disagree or are invalid")
        parameters = {}
        for name in (*REQUIRED, *ADDITIONAL_CONTROLLED):
            records = [row.get("parameters", {}).get(name, _missing("Rank lacks field")) for row in rows]
            record = _common_value(records, name)
            if name in ("host_memory_bytes", "cuda_runtime_version"):
                # Numeric/version agreement must not erase path/API evidence
                # or the difference between physical and enforced ceilings.
                semantics = {item.get("value_semantics") for item in records}
                if name == "host_memory_bytes" and len(semantics) > 1:
                    record = _missing("Nonuniform host-memory ceiling semantics across ranks")
                elif len(semantics) == 1 and None not in semantics:
                    record["value_semantics"] = next(iter(semantics))
                record["per_rank_observations"] = {
                    str(row["rank"]["value"]): item for row, item in zip(rows, records)}
            record["evidence"] = [row.get("receipt_path", "unlocated-receipt") + "#/parameters/" + name for row in rows]
            parameters[name] = record
        uuids = [row.get("hardware", {}).get("uuid", {}).get("value") for row in rows]
        if (all(type(value) is str and value for value in uuids) and len(set(uuids)) == expected_ranks
                and all(row["hardware"]["uuid"].get("evidence_level") == "runtime" for row in rows)):
            references = ";".join(row.get("receipt_path", "unlocated-receipt") + "#/hardware/uuid" for row in rows)
            parameters["gpu_uuids"] = _observed(sorted(uuids), references)
            parameters["gpu_count"] = _observed(expected_ranks, references)
        for name in ("attention_group_memberships", "routed_expert_assignments"):
            records = [row.get("parameters", {}).get(name, _missing("Rank lacks field")) for row in rows]
            if all(_common_value([record], name)["evidence_level"] == "runtime" for record in records):
                parameters[name] = _observed({str(row["rank"]["value"]): record["value"] for row, record in zip(rows, records)},
                                             "complete per-rank controlled-layout receipts")
        if parameters["gpu_uuids"]["evidence_level"] == "runtime":
            parameters["rank_device_mapping"] = _observed(
                {str(row["rank"]["value"]): value for row, value in zip(rows, uuids)}, "actual per-rank CUDA UUID receipts")
        nodes = [row.get("hardware", {}).get("node_hardware_id", {}).get("value") for row in rows]
        if (all(type(value) is str and value for value in nodes)
                and all(row["hardware"]["node_hardware_id"].get("evidence_level") == "runtime" for row in rows)):
            parameters["node_count"] = _observed(len(set(nodes)), "all rank receipts observed DMI product UUIDs, not inferred from rank count")
        result["engines"][engine] = {"observed_at_utc": min(row["observed_at_utc"] for row in rows),
            "binding": {"source_sha256": next(iter(deployments)), "input_manifest_sha256": next(iter(bindings))},
            "parameters": parameters}
        result["aggregation"]["engines"][engine] = {"rank_count": len(rows), "ranks": [row["rank"]["value"] for row in rows],
            "receipt_paths": [row.get("receipt_path") for row in rows],
            "missing_or_nonuniform_parameters": [name for name, record in parameters.items() if record["evidence_level"] != "runtime"]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--expected-ranks", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate([json.loads(path.read_text()) for path in args.receipt], args.expected_ranks)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
