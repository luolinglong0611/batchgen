#!/usr/bin/env python3
"""Run the unchanged main converter and verify every tensor byte independently.

No model, CUDA operation, repacking, scale rounding, or checkpoint mutation.
The output must be a new, explicit local directory. Default is plan-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import time


INDEX_SHA = "a2c5d238ab1e8301a117336e9797e0d3551cb8f6dc9dc787318b23d8f54c9248"
DTYPES = {"F8_E4M3": ("float8_e4m3fn", 1), "BF16": ("bfloat16", 2),
          "F32": ("float32", 4), "F16": ("float16", 2), "I64": ("int64", 8)}
BLOCK = 8 * 1024**2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_shard(source, output, expected_names):
    """Compare raw safetensors data ranges with the converter's independent map."""
    with source.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size < 64 * 1024**2:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    meta = json.loads((output / (source.stem + ".json")).read_text())
    if set(header) != set(expected_names) or set(meta["state_dict"]) != set(header):
        raise ValueError("Tensor inventory differs from the frozen original index")
    binary = output / (source.stem + ".bin")
    ranges = []
    for name, original in header.items():
        converted = meta["state_dict"][name]
        dtype, itemsize = DTYPES[original["dtype"]]
        start, end = original["data_offsets"]
        count = 1
        for dim in original["shape"]:
            count *= dim
        if (converted["dtype"] != dtype or converted["shape"] != original["shape"]
                or converted["byte_size"] != count * itemsize or end-start != count * itemsize):
            raise ValueError("Converted tensor metadata changed: " + name)
        ranges.append((converted["offset"], converted["offset"] + end-start))
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError("Converted storage has a gap or overlap")
        cursor = end
    if cursor != binary.stat().st_size or cursor != meta["total_byte_size"]:
        raise ValueError("Converted storage length differs")
    records = []
    with source.open("rb") as src, binary.open("rb") as dst:
        for name, original in header.items():
            start, end = original["data_offsets"]
            converted = meta["state_dict"][name]
            src.seek(8+size+start)
            dst.seek(converted["offset"])
            digest = hashlib.sha256()
            remaining = end-start
            while remaining:
                n = min(BLOCK, remaining)
                a, b = src.read(n), dst.read(n)
                if len(a) != n or a != b:
                    raise ValueError("Tensor bytes differ: " + name)
                digest.update(a)
                remaining -= n
            records.append({"name": name, "bytes": end-start, "original_and_converted_sha256": digest.hexdigest()})
    return {"shard": source.name, "tensor_count": len(records), "bytes": cursor,
            "metadata_sha256": sha(output / (source.stem + ".json")), "tensors": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--converter-sha256", required=True)
    ap.add_argument("--expected-hostname", required=True)
    ap.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if socket.gethostname() != a.expected_hostname:
        raise ValueError("Unexpected host")
    converter = a.repo / "batchgen/ckpt_converter/ckpt_converter.py"
    if sha(converter) != a.converter_sha256:
        raise ValueError("Converter source changed")
    index_path = a.input / "model.safetensors.index.json"
    if sha(index_path) != INDEX_SHA:
        raise ValueError("Original index changed")
    index = json.loads(index_path.read_text())["weight_map"]
    shards = {}
    for name, filename in index.items():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Unsafe shard name")
        shards.setdefault(filename, []).append(name)
    total = sum((a.input / name).stat().st_size for name in shards)
    report = {"schema": "r1-main-checkpoint-byte-verification-v1", "status": "plan_only",
              "scope": "all original tensor bytes; no inference or performance qualification",
              "hostname": socket.gethostname(), "index_sha256": INDEX_SHA,
              "converter_sha256": a.converter_sha256, "source_bytes": total,
              "shards": len(shards), "tensor_count": len(index), "workers": a.workers,
              "marlin_repack": False, "full_model_qualified": False, "formal_test_permitted": False}
    if not a.execute:
        print(json.dumps(report, indent=2))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("R1_CHECKPOINT_CPU_ONLY") != "1":
        raise ValueError("Explicit CUDA-hidden CPU-only launch required")
    if a.output.exists() or a.report_dir.exists() or a.output.resolve().is_relative_to(a.input.resolve()):
        raise ValueError("Fresh local output/report directories required")
    if shutil.disk_usage(a.output.parent).free < total + 30 * 1024**3:
        raise ValueError("Insufficient local output storage")
    a.report_dir.mkdir()
    report.update(status="running", started_unix=time.time(), completed_shards=0, verified_bytes=0)
    atomic_json(a.report_dir / "status.json", report)
    try:
        spec = importlib.util.spec_from_file_location("original_main_converter", converter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        if torch.cuda.is_initialized():
            raise ValueError("Unexpected CUDA initialization")
        a.output.mkdir()
        cvt = module.ckpt_converter()

        def work(filename):
            cvt.convert(str(a.input / filename), str(a.output), marlin=False)
            result = verify_shard(a.input / filename, a.output, shards[filename])
            atomic_json(a.report_dir / (Path(filename).stem + ".verified.json"), result)
            return result

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(work, name) for name in sorted(shards)]
            for future in as_completed(futures):
                result = future.result()
                report["completed_shards"] += 1
                report["verified_bytes"] += result["bytes"]
                report["last_verified_shard"] = result["shard"]
                atomic_json(a.report_dir / "status.json", report)
        if sha(converter) != a.converter_sha256 or sha(index_path) != INDEX_SHA or torch.cuda.is_initialized():
            raise ValueError("Source changed or CUDA unexpectedly initialized")
        report.update(status="passed_all_tensor_bytes_unchanged_NOT_model_qualified", cuda_initialized=False)
    except BaseException as e:
        report.update(status="failed", error=type(e).__name__ + ": " + str(e))
        raise
    finally:
        report["finished_unix"] = time.time()
        atomic_json(a.report_dir / "status.json", report)


if __name__ == "__main__":
    main()
