#!/usr/bin/env python3
"""Seal existing R1 conversion evidence and independently hash original files.

Default is a read-only plan, without huge-file hashing, waits, writes, model or
Torch imports. --execute hashes original files with at most four CPU threads
while the existing conversion job finishes. It never converts/repackages,
changes original/converted data, starts a model or claims model qualification.
The final identity is a weight identity; tokenizer/config/semantic qualification
remain separate. An external process wall limit is still required for stalled
filesystem I/O: internal deadlines are checked between bounded reads/polls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import threading
import time


MANIFEST_SHA256 = "d9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253"
INDEX_SHA256 = "a2c5d238ab1e8301a117336e9797e0d3551cb8f6dc9dc787318b23d8f54c9248"
EXPECTED_SHARDS = 163
EXPECTED_TENSORS = 91991
EXPECTED_FILE_BYTES = 688586727753
SUCCESS = "passed_all_tensor_bytes_unchanged_NOT_model_qualified"
SEALED = "sealed_original_weight_identity_and_conversion_evidence_NOT_model_qualified"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ITEM_BYTES = {"float8_e4m3fn": 1, "bfloat16": 2, "float32": 4, "float16": 2, "int64": 8}
SOURCE_DTYPES = {"F8_E4M3": "float8_e4m3fn", "BF16": "bfloat16", "F32": "float32",
                 "F16": "float16", "I64": "int64"}
BLOCK = 8 * 1024**2


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON key: " + name)
        result[name] = value
    return result


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("Unexpectedly large JSON metadata: " + str(path))
    return json.loads(path.read_text(), object_pairs_hook=object_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def small_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, label, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError("Invalid integer " + label)
    return value


def checked_sha(value, label):
    if type(value) is not str or not SHA_RE.fullmatch(value):
        raise ValueError("Invalid SHA256 " + label)
    return value


def source_header(path, expected_names):
    """Derive actual payload bytes from independent original tensor headers.

    Index metadata.total_size can describe an older BF16 checkpoint. It is
    deliberately not used for payload sizes or acceptance. Header identity
    is later bound to the same byte stream as the full frozen file SHA256.
    """
    path = Path(path)
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated original safetensors length prefix: " + path.name)
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= 64 * 1024**2 or 8 + size > file_bytes:
            raise ValueError("Invalid original safetensors header length: " + path.name)
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Truncated original safetensors JSON header: " + path.name)
    header = json.loads(raw, object_pairs_hook=object_unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite header: " + value)))
    if type(header) is not dict:
        raise ValueError("Original safetensors header must be an object")
    header.pop("__metadata__", None)
    if set(header) != set(expected_names):
        raise ValueError("Original safetensors header inventory differs from the bound index: " + path.name)
    records, intervals = {}, []
    for name, entry in header.items():
        if type(entry) is not dict or entry.get("dtype") not in SOURCE_DTYPES or type(entry.get("shape")) is not list:
            raise ValueError("Invalid original safetensors dtype/shape: " + str(name))
        shape = entry["shape"]
        count = math.prod(positive(value, "original tensor dimension") for value in shape)
        dtype = SOURCE_DTYPES[entry["dtype"]]
        size_bytes = count * ITEM_BYTES[dtype]
        offsets = entry.get("data_offsets")
        if type(offsets) is not list or len(offsets) != 2:
            raise ValueError("Invalid original safetensors data_offsets")
        start, stop = (positive(value, "original data offset", zero=True) for value in offsets)
        if stop <= start or stop - start != size_bytes:
            raise ValueError("Original dtype/shape/data_offsets byte count mismatch: " + name)
        intervals.append((start, stop))
        records[name] = {"dtype": dtype, "shape": shape, "bytes": size_bytes, "data_offsets": offsets}
    cursor = 0
    for first, last in sorted(intervals):
        if first != cursor:
            raise ValueError("Original safetensors tensor extents overlap or have gaps")
        cursor = last
    if cursor != file_bytes - 8 - len(raw):
        raise ValueError("Original safetensors header extents do not cover the exact file payload")
    return {"header_bytes": 8 + len(raw), "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
            "tensor_bytes": cursor, "tensor_count": len(records), "tensors": records}


def inventory(source, manifest, *, manifest_sha=MANIFEST_SHA256, index_sha=INDEX_SHA256,
              expected_shards=EXPECTED_SHARDS, expected_tensors=EXPECTED_TENSORS,
              expected_file_bytes=EXPECTED_FILE_BYTES):
    """Small-file/size plan; test-only arguments allow independent tiny fixtures.

    Production CLI always calls this with the pinned defaults above; no CLI
    switch can weaken its manifest, index, shard/tensor or total-byte contract.
    """
    source, manifest = Path(source).resolve(), Path(manifest).resolve()
    if small_sha(manifest) != manifest_sha:
        raise ValueError("Frozen model manifest SHA256 mismatch")
    entries = load_json(manifest)
    if type(entries) is not list:
        raise ValueError("Frozen model manifest must be a list")
    files = {}
    for entry in entries:
        if type(entry) is not dict or type(entry.get("Path")) is not str:
            raise ValueError("Invalid frozen manifest entry")
        name = entry["Path"]
        if name in files:
            raise ValueError("Duplicate frozen manifest path: " + name)
        files[name] = entry
    index_path = source / "model.safetensors.index.json"
    if small_sha(index_path) != index_sha:
        raise ValueError("Original index SHA256 mismatch")
    index_entry = files.get(index_path.name)
    if (not index_entry or index_entry.get("Sha256") != index_sha
            or type(index_entry.get("Size")) is not int or index_entry["Size"] != index_path.stat().st_size):
        raise ValueError("Frozen manifest does not bind the original index hash and size")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or len(weight_map) != expected_tensors:
        raise ValueError("Original index tensor count mismatch")
    index_declared_bytes = index.get("metadata", {}).get("total_size")
    shards = {}
    for tensor, name in weight_map.items():
        if (type(tensor) is not str or not tensor or type(name) is not str or Path(name).name != name
                or not name.endswith(".safetensors")):
            raise ValueError("Unsafe tensor/shard entry in original index")
        shards.setdefault(name, []).append(tensor)
    listed_shards = {name for name in files if name.endswith(".safetensors")}
    if len(shards) != expected_shards or set(shards) != listed_shards:
        raise ValueError("Frozen manifest and index shard inventories disagree")
    total, tensor_bytes, headers = 0, 0, {}
    for name in shards:
        entry = files[name]
        checked_sha(entry.get("Sha256"), name)
        size = positive(entry.get("Size"), "source size " + name)
        if entry.get("Name") != name or Path(name).name != name or (source / name).stat().st_size != size:
            raise ValueError("Original source file name/size mismatch: " + name)
        total += size
        headers[name] = source_header(source / name, shards[name])
        tensor_bytes += headers[name]["tensor_bytes"]
    if total != expected_file_bytes:
        raise ValueError("Original source total file bytes mismatch")
    return {"source": str(source), "manifest_path": str(manifest), "manifest_sha256": manifest_sha,
            "index_path": str(index_path), "index_sha256": index_sha, "shards": shards,
            "files": {name: files[name] for name in sorted(shards)}, "tensor_count": len(weight_map),
            "tensor_bytes": tensor_bytes, "source_file_bytes": total,
            "index_declared_total_size_NOT_authoritative": index_declared_bytes,
            "payload_byte_provenance": "independent original safetensors dtype/shape/data_offsets; full-file SHA binding required at execution",
            "source_header_shards": headers, "weight_map": weight_map}


def file_identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def original_file_sha(path, expected, deadline, stop):
    """Hash every original file byte; fail if size/stat identity changes."""
    path = Path(path)
    before = file_identity(path)
    if before[2] != expected["Size"]:
        raise ValueError("Original file size changed before hashing: " + path.name)
    digest, count, header_digest = hashlib.sha256(), 0, hashlib.sha256()
    header_remaining = expected.get("header_bytes", 0)
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before:
            raise ValueError("Original file identity changed while opening: " + path.name)
        while True:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Checkpoint seal deadline or sibling failure")
            block = stream.read(BLOCK)
            if not block:
                break
            count += len(block)
            if count > expected["Size"]:
                raise ValueError("Original source grew while hashing: " + path.name)
            digest.update(block)
            if header_remaining:
                part = block[:header_remaining]
                header_digest.update(part)
                header_remaining -= len(part)
    if file_identity(path) != before or count != expected["Size"] or digest.hexdigest() != expected["Sha256"]:
        raise ValueError("Original file full SHA256/size/stat mismatch: " + path.name)
    if "header_sha256" in expected and (header_remaining or header_digest.hexdigest() != expected["header_sha256"]):
        raise ValueError("Independently parsed header differs from full-file hash stream: " + path.name)
    return {"name": path.name, "bytes": count, "sha256": digest.hexdigest(),
            "stable_stat": list(before), "frozen_manifest_match": True,
            "source_header_sha256": header_digest.hexdigest() if "header_sha256" in expected else None}


def conversion_status(report_dir, plan, *, require_complete=False):
    path = Path(report_dir) / "status.json"
    if not path.is_file():
        if require_complete:
            raise ValueError("Conversion job success receipt is absent")
        return None
    report = load_json(path)
    if (report.get("schema") != "r1-main-checkpoint-byte-verification-v1"
            or report.get("index_sha256") != plan["index_sha256"]
            or report.get("source_bytes") != plan["source_file_bytes"]
            or report.get("shards") != len(plan["shards"])
            or report.get("tensor_count") != plan["tensor_count"]
            or report.get("marlin_repack") is not False):
        raise ValueError("Conversion job inventory/source/byte contract mismatch")
    checked_sha(report.get("converter_sha256"), "conversion source")
    if report.get("status") == "failed":
        raise ValueError("Existing conversion job failed; it will not be rerun")
    if report.get("status") != SUCCESS:
        if require_complete:
            raise ValueError("Existing conversion job has not succeeded")
        return report
    if (type(report.get("completed_shards")) is not int or report["completed_shards"] != len(plan["shards"])
            or type(report.get("verified_bytes")) is not int or report["verified_bytes"] != plan["tensor_bytes"]
            or report.get("cuda_initialized") is not False):
        raise ValueError("Successful conversion receipt has incomplete counts/bytes or CUDA state")
    return report


def verified_tensor_records(plan, converted, report_dir, deadline=None):
    """Bind all existing verified records to exact current metadata and index."""
    converted, report_dir = Path(converted).resolve(), Path(report_dir).resolve()
    records, shard_records, seen = [], [], set()
    total_bytes = 0
    expected_verified = {Path(name).stem + ".verified.json" for name in plan["shards"]}
    if {path.name for path in report_dir.glob("*.verified.json")} != expected_verified:
        raise ValueError("Missing/extra conversion per-shard verification receipts")
    for name in sorted(plan["shards"]):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint metadata sealing deadline exceeded")
        stem = Path(name).stem
        verified_path, metadata_path = report_dir / (stem + ".verified.json"), converted / (stem + ".json")
        verified_sha, metadata_sha = small_sha(verified_path), small_sha(metadata_path)
        verified, metadata = load_json(verified_path), load_json(metadata_path)
        if metadata.get("file_name") != stem + ".bin":
            raise ValueError("Converted metadata file_name is not the original shard stem .bin: " + name)
        if metadata_sha != verified.get("metadata_sha256"):
            raise ValueError("Converted metadata SHA256 changed after tensor byte verification: " + name)
        tensors, listed = metadata.get("state_dict"), verified.get("tensors")
        if type(tensors) is not dict or type(listed) is not list or verified.get("shard") != name:
            raise ValueError("Malformed or wrong per-shard conversion receipt")
        expected_names = set(plan["shards"][name])
        if set(tensors) != expected_names or len(listed) != len(expected_names) or verified.get("tensor_count") != len(expected_names):
            raise ValueError("Per-shard tensor inventory/count mismatch")
        receipt_names, intervals, shard_bytes = set(), [], 0
        for entry in listed:
            tensor_name = entry.get("name")
            if type(tensor_name) is not str or tensor_name not in expected_names or tensor_name in seen or tensor_name in receipt_names:
                raise ValueError("Duplicate, missing or index-misassigned tensor: " + str(tensor_name))
            receipt_names.add(tensor_name)
            meta = tensors[tensor_name]
            dtype, shape = meta.get("dtype"), meta.get("shape")
            if dtype not in ITEM_BYTES or type(shape) is not list:
                raise ValueError("Unknown tensor dtype/shape")
            count = math.prod(positive(value, "tensor dimension") for value in shape)
            size = positive(meta.get("byte_size"), "tensor byte_size")
            offset = positive(meta.get("offset"), "tensor offset", zero=True)
            if size != count * ITEM_BYTES[dtype] or type(entry.get("bytes")) is not int or entry["bytes"] != size:
                raise ValueError("Tensor shape/dtype/byte counts disagree")
            original = plan["source_header_shards"][name]["tensors"][tensor_name]
            if dtype != original["dtype"] or shape != original["shape"] or size != original["bytes"]:
                raise ValueError("Converted metadata disagrees with independent original safetensors header")
            if ("dtype" in entry and entry["dtype"] != dtype) or ("shape" in entry and entry["shape"] != shape):
                raise ValueError("Verified tensor dtype/shape disagrees with hash-bound metadata")
            raw = checked_sha(entry.get("original_and_converted_sha256"), "verified raw tensor")
            intervals.append((offset, offset + size))
            shard_bytes += size
            records.append({"name": tensor_name, "dtype": dtype, "shape": shape, "raw_sha256": raw})
            seen.add(tensor_name)
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("Converted metadata has overlapping/gapped tensor extents")
            cursor = end
        binary = converted / (stem + ".bin")
        if (receipt_names != expected_names or type(verified.get("bytes")) is not int
                or verified["bytes"] != shard_bytes or metadata.get("total_byte_size") != shard_bytes
                or cursor != shard_bytes or binary.stat().st_size != shard_bytes):
            raise ValueError("Converted shard inventory/total/bin byte count mismatch")
        if small_sha(verified_path) != verified_sha or small_sha(metadata_path) != metadata_sha:
            raise ValueError("Per-shard receipts/metadata changed while sealing")
        shard_records.append({"name": name, "metadata_sha256": metadata_sha, "verified_record_sha256": verified_sha,
                              "tensor_count": len(listed), "tensor_bytes": shard_bytes})
        total_bytes += shard_bytes
    if seen != set(plan["weight_map"]) or len(records) != plan["tensor_count"] or total_bytes != plan["tensor_bytes"]:
        raise ValueError("Global complete tensor inventory or bytes mismatch")
    records.sort(key=lambda item: item["name"])
    return {"records": records, "weight_tensor_hash": hashlib.sha256(canonical(records)).hexdigest(),
            "tensor_count": len(records), "tensor_bytes": total_bytes, "shards": shard_records}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(args):
    value = inventory(args.input, args.manifest)
    return {"schema": "r1-checkpoint-seal-v1", "status": "plan_only_NOT_hashed_or_sealed",
            "source_directory": str(args.input.resolve()), "converted_directory": str(args.converted.resolve()),
            "conversion_report_directory": str(args.conversion_report_dir.resolve()),
            "report_directory": str(args.report_dir.resolve()), "expected_hostname": args.expected_hostname,
            "manifest_sha256": value["manifest_sha256"], "index_sha256": value["index_sha256"],
            "expected_shards": EXPECTED_SHARDS, "expected_tensors": EXPECTED_TENSORS,
            "source_file_bytes_to_read": value["source_file_bytes"], "tensor_bytes": value["tensor_bytes"],
            "index_declared_total_size_NOT_authoritative": value["index_declared_total_size_NOT_authoritative"],
            "payload_byte_provenance": value["payload_byte_provenance"],
            "source_header_summary": {name: {key: row[key] for key in ("header_bytes", "header_sha256", "tensor_bytes", "tensor_count")}
                                      for name, row in value["source_header_shards"].items()},
            "workers": args.workers, "timeout_seconds": args.timeout_seconds,
            "worker_sha256": small_sha(__file__),
            "checks": ["full original163 file SHA256 AND Size versus frozen manifest", "completed existing conversion job",
                       "metadata file_name and its already-verified SHA256", "complete unique index-bound tensor inventory and bytes",
                       "canonical sorted name,dtype,shape,raw_sha256 aggregate"],
            "converted_byte_proof": "completed prepare job's raw tensor byte comparisons; this sealer rechecks metadata/bin size, not a second full converted-bin read",
            "requires_environment": {"R1_CHECKPOINT_SEAL_CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""},
            "originals_or_conversion_modified": False, "model_loaded": False,
            "full_model_qualified": False, "formal_test_permitted": False}


def execute(args):
    if (socket.gethostname() != args.expected_hostname or os.environ.get("R1_CHECKPOINT_SEAL_CPU_ONLY") != "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Named host and explicit CUDA-hidden CPU-only seal guard required")
    if type(args.workers) is not int or not 1 <= args.workers <= 4 or args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("Bounded positive CPU workers, timeout and poll period required")
    report = plan(args)
    inputs = inventory(args.input, args.manifest)
    if not args.report_dir.is_absolute() or args.report_dir.exists() or args.report_dir.is_symlink():
        raise FileExistsError("Fresh absolute seal report directory required")
    for protected in (args.input, args.converted, args.conversion_report_dir):
        if args.report_dir.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Seal output must be separate from original, converted and conversion evidence trees")
    args.report_dir.mkdir(parents=True, exist_ok=False)
    report.update(status="hashing_original_files_waiting_for_existing_conversion", started_unix=time.time(),
                  completed_original_files=0, hashed_original_bytes=0, original_files=[])
    atomic_json(args.report_dir / "status.json", report)
    deadline, stop = time.monotonic() + args.timeout_seconds, threading.Event()
    exit_code = 1
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(original_file_sha, args.input / name, dict(entry,
                           header_bytes=inputs["source_header_shards"][name]["header_bytes"],
                           header_sha256=inputs["source_header_shards"][name]["header_sha256"]), deadline, stop): name
                       for name, entry in inputs["files"].items()}
            try:
                while pending:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Source hashing/conversion wait deadline exceeded")
                    conversion_status(args.conversion_report_dir, inputs)
                    done, _ = wait(pending, timeout=min(args.poll_seconds, max(0.001, deadline-time.monotonic())), return_when=FIRST_COMPLETED)
                    for future in done:
                        name = pending.pop(future)
                        result = future.result()
                        atomic_json(args.report_dir / (Path(name).stem + ".source-hash.json"), result)
                        report["original_files"].append(result)
                        report["completed_original_files"] += 1
                        report["hashed_original_bytes"] += result["bytes"]
                        report["last_completed_original_file"] = name
                        atomic_json(args.report_dir / "status.json", report)
            finally:
                if pending:
                    stop.set()
                    for future in pending:
                        future.cancel()
        while True:
            status = conversion_status(args.conversion_report_dir, inputs)
            if status is not None and status.get("status") == SUCCESS:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Existing conversion job did not succeed before the seal deadline")
            report["status"] = "original_files_verified_waiting_for_conversion"
            atomic_json(args.report_dir / "status.json", report)
            time.sleep(min(args.poll_seconds, max(0, deadline-time.monotonic())))
        status_path = args.conversion_report_dir / "status.json"
        status_sha = small_sha(status_path)
        conversion_status(args.conversion_report_dir, inputs, require_complete=True)
        converted = verified_tensor_records(inputs, args.converted, args.conversion_report_dir, deadline)
        for row in report["original_files"]:
            if list(file_identity(args.input / row["name"])) != row["stable_stat"]:
                raise ValueError("Original source changed after its full file hash")
        for row in converted["shards"]:
            stem = Path(row["name"]).stem
            if (small_sha(args.converted / (stem + ".json")) != row["metadata_sha256"]
                    or small_sha(args.conversion_report_dir / (stem + ".verified.json")) != row["verified_record_sha256"]):
                raise ValueError("Converted metadata/verified records changed before final seal publication")
        if (small_sha(args.manifest) != inputs["manifest_sha256"] or small_sha(Path(inputs["index_path"])) != inputs["index_sha256"]
                or small_sha(status_path) != status_sha or small_sha(__file__) != report["worker_sha256"]):
            raise ValueError("Manifest/index/conversion-success receipt/worker changed during sealing")
        original_rows = sorted(({key: row[key] for key in ("name", "bytes", "sha256")} for row in report["original_files"]), key=lambda row: row["name"])
        identity = {"frozen_model_manifest_sha256": inputs["manifest_sha256"], "index_sha256": inputs["index_sha256"],
                    "original_weight_files_hash": hashlib.sha256(canonical(original_rows)).hexdigest(),
                    "weight_tensor_hash": converted["weight_tensor_hash"], "tensor_count": converted["tensor_count"],
                    "tensor_bytes": converted["tensor_bytes"], "source_file_bytes": inputs["source_file_bytes"]}
        atomic_json(args.report_dir / "tensor-identity-records.json", converted["records"])
        report.update(status=SEALED, original_file_identity_verified=True, conversion_evidence_sealed=True,
                      original_files=original_rows, conversion_success_report_sha256=status_sha,
                      converted_metadata_and_record_bindings=converted["shards"], identity=identity,
                      model_weight_identity_sha256=hashlib.sha256(canonical(identity)).hexdigest(),
                      weight_tensor_hash=converted["weight_tensor_hash"], weights_tensor_hash=converted["weight_tensor_hash"],
                      tensor_identity_records_sha256=small_sha(args.report_dir / "tensor-identity-records.json"))
        exit_code = 0
    except BaseException as error:
        stop.set()
        report.update(status="failed_checkpoint_seal_NOT_model_qualified", error_type=type(error).__name__, error=str(error))
    finally:
        report["finished_unix"] = time.time()
        atomic_json(args.report_dir / "status.json", report)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--conversion-report-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        return execute(args)
    print(json.dumps(plan(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
