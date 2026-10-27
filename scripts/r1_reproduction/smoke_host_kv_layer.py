#!/usr/bin/env python3
"""Check one production-sized BF16 MLA Host KV layer with the native worker view.

The default is a CPU-only plan. Explicit execution allocates/registers one
37.77--41.97 GiB host layer and tests only its first, middle and final pages.
This is compatibility evidence, not a full pool, eight-worker, model, scheduler,
accuracy or performance qualification. No LD_PRELOAD audit library is needed.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

GIB = 1024**3
MODEL_LAYERS = 61
PAGE_TOKENS = 64
KV_DIM = 576
PAGE_BYTES = PAGE_TOKENS * KV_DIM * 2
GPU_ALLOCATOR_LIMIT_BYTES = GIB // 2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(host_kv_gib: int) -> dict:
    pages = host_kv_gib * GIB // (MODEL_LAYERS * PAGE_BYTES)
    return {
        "equivalent_full_pool_gib": host_kv_gib,
        "equivalent_model_layers": MODEL_LAYERS,
        "tested_layers": 1,
        "pages_per_layer": pages,
        "page_tokens": PAGE_TOKENS,
        "kv_dim": KV_DIM,
        "dtype": "bfloat16",
        "registered_layer_bytes": pages * PAGE_BYTES,
        "tested_page_ids": [0, pages // 2, pages - 1],
    }


def host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def owned_kv_memfds() -> dict:
    result = {}
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if "memfd:batchgen_kv" in target:
            result[entry.name] = target
    return result


def is_mapped(address: int) -> bool:
    for line in Path("/proc/self/maps").read_text().splitlines():
        low, high = (int(part, 16) for part in line.split()[0].split("-"))
        if low <= address < high:
            return True
    return False


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cuda_runtime(torch):
    major = torch.version.cuda.split(".", 1)[0]
    runtime = ctypes.CDLL(f"libcudart.so.{major}")
    runtime.cudaHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    runtime.cudaHostGetFlags.restype = ctypes.c_int
    runtime.cudaGetLastError.argtypes = []
    runtime.cudaGetLastError.restype = ctypes.c_int
    return runtime


def worker(args) -> int:
    spec = geometry(args.host_kv_gib)
    result = {
        "schema": "batchgen-single-host-kv-layer-smoke-v1",
        "status": "failed_single_layer_compatibility",
        "scope": "One full native Host KV layer; three sentinel pages only. "
                 "NOT full-pool, eight-worker, model, scheduler, accuracy or performance qualification.",
        "hostname": socket.gethostname(),
        "geometry": spec,
        "registration_chunk_gib": args.chunk_gib,
        "started_unix": time.time(),
        "checks": [],
        "cleanup": {},
    }
    view = None
    base = None
    torch = None
    old_fds = owned_kv_memfds()
    try:
        require(socket.gethostname() == args.expected_hostname, "Unexpected hostname")
        require(Path("/.dockerenv").is_file(), "Execute inside the allocated container")
        require(host_available_bytes() >= spec["registered_layer_bytes"] + args.host_free_floor_gib * GIB,
                "Insufficient Host memory for this layer plus the requested free-memory floor")
        require(not os.environ.get("LD_PRELOAD"), "Run this smoke without LD_PRELOAD")
        os.environ["BATCHGEN_HOST_REGISTER_CHUNK_GIB"] = str(args.chunk_gib)
        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo))
        import torch as torch_module
        import batchgen.core_engine as core

        torch = torch_module
        native = Path(core.__file__).resolve()
        require(native.is_relative_to(repo), "Native extension must come from this repository")
        native_sha = sha(native)
        if args.native_sha256:
            require(native_sha == args.native_sha256, "Native extension SHA256 differs")
        result["native"] = {"path": str(native), "sha256": native_sha}
        result["script_sha256"] = sha(Path(__file__))
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(GPU_ALLOCATOR_LIMIT_BYTES / total, device)
        torch.cuda.reset_peak_memory_stats(device)
        result["runtime"] = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": args.device, "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
        }
        runtime = cuda_runtime(torch)
        config = core.HostPagedKVConfig()
        config.shm_name = "/batchgen_layer_smoke_" + str(os.getpid())
        for key, value in {
            "num_layers": 1, "num_pages": spec["pages_per_layer"],
            "page_size_tokens": PAGE_TOKENS, "num_k_heads": 1,
            "k_head_dim": KV_DIM, "num_v_heads": 0, "v_head_dim": 0,
            "k_element_size_bytes": 2, "v_element_size_bytes": 0,
            "sequence_table_capacity": 16, "alignment_bytes": 64,
        }.items():
            setattr(config, key, value)
        config.logger_name = "SingleHostKVLayerSmoke"
        config.enable_thp = True
        view = core.MLAHostPagedKVWorkerView(config)
        view.initialize(args.device, True)
        base = view.data_base_address()
        require(view.k_page_ptr(0, spec["pages_per_layer"] - 1) + PAGE_BYTES ==
                base + spec["registered_layer_bytes"], "Unexpected native layer geometry")
        flags = ctypes.c_uint()
        require(runtime.cudaHostGetFlags(ctypes.byref(flags), ctypes.c_void_p(base)) == 0,
                "The native Host KV mapping is not CUDA registered")
        result["cuda_host_flags"] = flags.value
        require(host_available_bytes() >= args.host_free_floor_gib * GIB,
                "Host memory floor was crossed after registration")
        ids = [71, 72, 73]
        view.register_sequences(ids)
        for sid, page in zip(ids, spec["tested_page_ids"]):
            view.set_sequence_pages(sid, [page])

        def pattern(shape, seed):
            count = 1
            for dimension in shape:
                count *= dimension
            values = torch.arange(count, dtype=torch.int32).reshape(shape)
            return ((values + seed) % 127 - 63).to(torch.bfloat16)

        def check(name, actual, expected):
            require(torch.equal(actual.contiguous().view(torch.uint8),
                                expected.contiguous().view(torch.uint8)), name + " byte mismatch")
            result["checks"].append({"name": name, "bytes": expected.numel() * expected.element_size(),
                                     "byte_equal": True})

        source = pattern((3, PAGE_TOKENS, 1, KV_DIM), 101).to(device)
        # The native call owns the tensor until its real task has completed.
        view.async_offload_layer_kv_to_host(0, ids, source, None, [PAGE_TOKENS] * 3).wait()
        expected = {sid: source[index].cpu().reshape(1, 1, PAGE_TOKENS, 1, KV_DIM)
                    for index, sid in enumerate(ids)}
        for sid in ids:
            check("native_D2H_" + str(sid), view.read_sequence_kv_to_cpu(sid)[0], expected[sid])
        appended = pattern((3, 1, 1, KV_DIM), 303).to(device)
        view.async_append_decode_kv_to_host_batched_kernel(
            [(0, appended, None)], ids, [PAGE_TOKENS - 1] * 3).wait()
        for index, sid in enumerate(ids):
            expected[sid][0, 0, -1] = appended[index, 0].cpu()
            check("native_append_" + str(sid), view.read_sequence_kv_to_cpu(sid)[0], expected[sid])
        destination = torch.empty((1, 3, 1, PAGE_TOKENS, 1, KV_DIM), dtype=torch.bfloat16, device=device)
        pointers = torch.tensor([[[destination[0, index, 0].data_ptr()] for index in range(3)]],
                                dtype=torch.int64)
        view.async_load_layer_paged_kv_to_device(
            torch.tensor(ids, dtype=torch.int64), torch.ones(3, dtype=torch.int64), pointers, None).wait()
        torch.cuda.synchronize(device)
        for index, sid in enumerate(ids):
            check("native_paged_H2D_" + str(sid), destination[:, index].cpu(), expected[sid])
        result["tested_host_pages"] = [
            {"sequence": sid, "page": page, "address": view.k_page_ptr(0, page), "bytes": PAGE_BYTES}
            for sid, page in zip(ids, spec["tested_page_ids"])
        ]
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        require(result["torch_peak_reserved_bytes"] < GIB, "Torch GPU allocation exceeded one GiB")
        view.shutdown()
        require(runtime.cudaHostGetFlags(ctypes.byref(flags), ctypes.c_void_p(base)) != 0,
                "Host KV mapping remained CUDA registered after shutdown")
        runtime.cudaGetLastError()  # Clear the expected lookup error after unregister.
        result["cleanup"]["native_unregistration_verified"] = True
        result["status"] = "passed_single_layer_compatibility_NOT_model_qualified"
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        if view is not None:
            try:
                view.shutdown()
            except BaseException as error:
                result["cleanup"]["shutdown_error"] = str(error)
                result["status"] = "failed_single_layer_compatibility"
            view = None
            gc.collect()
        result["cleanup"]["owned_memfds_restored"] = owned_kv_memfds() == old_fds
        result["cleanup"]["owned_mapping_removed"] = base is None or not is_mapped(base)
        if not (result["cleanup"]["owned_memfds_restored"] and result["cleanup"]["owned_mapping_removed"]):
            result["status"] = "failed_single_layer_compatibility"
        result["finished_unix"] = time.time()
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return 0 if result["status"].startswith("passed_") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-kv-gib", type=int, choices=(2304, 2432, 2560), default=2560)
    parser.add_argument("--chunk-gib", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host-free-floor-gib", type=int, default=192)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--native-sha256")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.chunk_gib <= 0 or args.host_free_floor_gib < 0 or args.device < 0 or args.timeout_seconds <= 0:
        parser.error("Chunk, device, timeout and memory floor are outside their valid ranges")
    if geometry(args.host_kv_gib)["registered_layer_bytes"] > args.chunk_gib * GIB:
        parser.error("One complete layer must fit within the registration chunk")
    if not args.execute:
        print(json.dumps({"status": "plan_only_NO_CUDA", "geometry": geometry(args.host_kv_gib),
                          "scope": "single-layer compatibility only", "chunk_gib": args.chunk_gib}, indent=2))
        return 0
    if args.expected_hostname != socket.gethostname() or args.output is None:
        parser.error("Execution requires this container's exact --expected-hostname and a fresh --output")
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence and choose a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args._worker:
        return worker(args)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        return child.wait(timeout=args.timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        # The direct, unreaped child handle prevents signaling a recycled PID.
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        if not args.output.exists():
            with args.output.open("x") as stream:
                json.dump({"schema": "batchgen-single-host-kv-layer-smoke-v1",
                           "status": "failed_or_interrupted_single_layer_compatibility",
                           "error": type(error).__name__, "cleanup": "native cleanup NOT verified",
                           "child_exit_code": child.returncode}, stream, indent=2)
                stream.write("\n")
        return 124 if isinstance(error, subprocess.TimeoutExpired) else 130


if __name__ == "__main__":
    raise SystemExit(main())
