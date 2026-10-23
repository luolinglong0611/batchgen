#!/usr/bin/env python3
"""Convert pinned, verified R1 OSS shards on CPU, one shard at a time.

Run inside an allocated container. Originals stay under /mnt/fuse/oss. This
program imports the explicitly pinned repository converter by file path, never
imports the batchgen package, and always calls convert(..., marlin=False).
Only one safetensors shard is loaded at a time; CUDA is hidden before import.

Example:
  python convert_verified_shards.py --manifest /work/r1_model_manifest.json \
    --source /mnt/fuse/oss/USER/models/DeepSeek-R1 \
    --destination /home/admin/models/R1/converted \
    --converter-source /home/admin/batchgen/batchgen/ckpt_converter/ckpt_converter.py \
    --converter-sha256 <sha256-of-reviewed-converter> --wait-seconds 14400

The manifest SHA256 and R1 weight revision are fixed below. Download receipts
must match those pins; each new conversion also rereads the complete source SHA.
Tensor-by-tensor validation rereads the source, checks its SHA again, and compares
every original tensor byte to its output offset. No full source shard is copied
to local disk. At least source_size + 20 GiB free local space is required before
each new conversion.

The destination contains ONLY .bin and loader metadata .json files. Sibling
*.staging and *.receipts directories hold temporary outputs and provenance.
Publication is atomic PER FILE: bin first, metadata last. It is not an atomic
whole-model transaction: do not launch a loader until all shards are complete.
Failures retain temporary outputs. Receipt-verified outputs can be resumed;
unreceipted complete pairs are fully revalidated, while orphan files fail closed.
--wait-seconds bounds the wall-clock window for waiting on downloads, not the
duration of a conversion already running. Exit 2 means downloads remain pending.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time


MANIFEST_SHA256 = "d9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253"
WEIGHT_REVISION = "20f2865710d161dc4a32c4e8c3d7c42c7b9852f8"
INDEX_NAME = "model.safetensors.index.json"
RESERVE_BYTES = 20 * 1024**3
CHUNK_BYTES = 8 * 1024**2
DTYPES = {
    "F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2),
    "F8_E4M3": ("float8_e4m3fn", 1), "I64": ("int64", 8),
    "I32": ("int32", 4), "I8": ("int8", 1), "U8": ("uint8", 1),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def event(name, **fields):
    print(json.dumps({"event": name, "utc": utc_now(), **fields}, sort_keys=True), flush=True)


def digest(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            result.update(block)
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=unique_object)


def integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def regular_file(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")


def validate_manifest(path: Path):
    regular_file(path)
    if digest(path) != MANIFEST_SHA256:
        raise ValueError("Manifest SHA256 differs from the pinned R1 manifest")
    entries = read_json(path)
    selected = {}
    for entry in entries:
        name = entry["Path"]
        if name != INDEX_NAME and not name.endswith(".safetensors"):
            continue
        if name != INDEX_NAME and not re.fullmatch(r"model-\d{5}-of-000163\.safetensors", name):
            raise ValueError(f"Unexpected shard name: {name}")
        if name in selected or entry["Type"] != "blob" or entry["Revision"] != WEIGHT_REVISION:
            raise ValueError(f"Unpinned or duplicate file: {name}")
        integer(entry["Size"], "manifest Size")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["Sha256"]):
            raise ValueError(f"Invalid SHA256 for {name}")
        selected[name] = entry
    if INDEX_NAME not in selected or len(selected) != 164:
        raise ValueError("Pinned model requires its index and 163 weight shards")
    return selected


def download_ready(source: Path, entry: dict):
    path = source / entry["Path"]
    receipt_path = source / (entry["Path"] + ".verified.json")
    if not path.exists() or not receipt_path.exists():
        return False
    regular_file(path)
    regular_file(receipt_path)
    try:
        receipt = read_json(receipt_path)
    except json.JSONDecodeError:
        # Download receipt publication can briefly expose a partial JSON write.
        return False
    expected = {"sha256": entry["Sha256"], "size": entry["Size"], "revision": entry["Revision"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Download receipt does not match pinned manifest: {path}")
    if path.stat().st_size != entry["Size"]:
        raise ValueError(f"Receipted source has wrong size: {path}")
    return True


def verify_source(path: Path, entry: dict):
    regular_file(path)
    before = path.stat()
    if before.st_size != entry["Size"] or digest(path) != entry["Sha256"]:
        raise ValueError(f"Source failed complete SHA256/size verification: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Source changed while hashing: {path}")


def safetensors_header(path: Path):
    """Parse/validate a header without safetensors, torch, or tensor allocation."""
    regular_file(path)
    file_size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors length prefix")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= min(64 * 1024**2, file_size - 8):
            raise ValueError("Invalid or oversized safetensors header")
        header_bytes = source.read(header_length)
    header = json.loads(header_bytes, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    payload_size = file_size - 8 - header_length
    tensors, intervals = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            if not isinstance(item, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in item.items()):
                raise ValueError("Invalid safetensors __metadata__")
            continue
        if not isinstance(item, dict) or item.get("dtype") not in DTYPES:
            raise ValueError(f"Unsupported tensor dtype/header: {name}")
        shape = item.get("shape")
        if not isinstance(shape, list):
            raise ValueError(f"Invalid shape: {name}")
        for dimension in shape:
            integer(dimension, f"{name} shape")
        offsets = item.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid data_offsets: {name}")
        begin, end = [integer(value, f"{name} offset") for value in offsets]
        dtype, width = DTYPES[item["dtype"]]
        byte_size = math.prod(shape) * width
        if begin > end or end > payload_size or end - begin != byte_size:
            raise ValueError(f"Shape/offset/byte-size inconsistency: {name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "source_offset": begin, "byte_size": byte_size}
        intervals.append((begin, end, name))
    if not tensors:
        raise ValueError("Empty safetensors shard")
    cursor = 0
    for begin, end, name in sorted(intervals):
        if begin != cursor:
            raise ValueError(f"Gap/overlap in safetensors data: {name}")
        cursor = end
    if cursor != payload_size:
        raise ValueError("Trailing unreferenced safetensors data")
    return tensors, prefix + header_bytes, payload_size


def verify_converted(source: Path, binary: Path, metadata_path: Path, expected_keys: set,
                     expected_source_sha256: str):
    """Verify every original tensor byte against its converted file offset."""
    regular_file(binary)
    regular_file(metadata_path)
    tensors, prefix, payload_size = safetensors_header(source)
    metadata = read_json(metadata_path)
    if set(tensors) != expected_keys:
        raise ValueError("Safetensors keys disagree with pinned index mapping")
    if not isinstance(metadata, dict) or set(metadata) != {"file_name", "state_dict", "total_byte_size"}:
        raise ValueError("Unexpected converter metadata schema")
    if metadata["file_name"] != binary.name or not isinstance(metadata["state_dict"], dict):
        raise ValueError("Metadata file_name/state_dict mismatch")
    if set(metadata["state_dict"]) != expected_keys:
        raise ValueError("Converted tensor keys disagree with pinned index mapping")
    total = integer(metadata["total_byte_size"], "total_byte_size")
    if total != payload_size or binary.stat().st_size != total:
        raise ValueError("Converted binary length/metadata size mismatch")
    intervals = []
    for name, tensor in tensors.items():
        converted = metadata["state_dict"][name]
        if not isinstance(converted, dict) or set(converted) != {"dtype", "shape", "offset", "byte_size"}:
            raise ValueError(f"Unexpected converted tensor metadata: {name}")
        offset = integer(converted["offset"], f"{name} converted offset")
        size = integer(converted["byte_size"], f"{name} converted byte_size")
        shape = converted["shape"]
        if not isinstance(shape, list):
            raise ValueError(f"Invalid converted shape: {name}")
        for dimension in shape:
            integer(dimension, f"{name} converted shape")
        if converted["dtype"] != tensor["dtype"] or shape != tensor["shape"] or size != tensor["byte_size"]:
            raise ValueError(f"Converted dtype/shape/byte_size differs: {name}")
        if offset + size > total:
            raise ValueError(f"Converted tensor offset out of bounds: {name}")
        intervals.append((offset, offset + size, name))
    cursor = 0
    for begin, end, name in sorted(intervals):
        if begin != cursor:
            raise ValueError(f"Gap/overlap in converted data: {name}")
        cursor = end
    if cursor != total:
        raise ValueError("Unreferenced bytes in converted binary")
    source_hash = hashlib.sha256(prefix)
    tensor_bytes_compared = 0
    with source.open("rb") as original, binary.open("rb") as converted:
        original.seek(len(prefix))
        # Source order preserves the full source SHA even if converter tensor
        # order differs. Converted offsets are independently checked above.
        for name, tensor in sorted(tensors.items(), key=lambda pair: (pair[1]["source_offset"], pair[1]["byte_size"], pair[0])):
            converted.seek(metadata["state_dict"][name]["offset"])
            remaining = tensor["byte_size"]
            while remaining:
                amount = min(remaining, CHUNK_BYTES)
                original_bytes = original.read(amount)
                converted_bytes = converted.read(amount)
                if len(original_bytes) != amount or len(converted_bytes) != amount:
                    raise ValueError(f"Short read during tensor validation: {name}")
                source_hash.update(original_bytes)
                if original_bytes != converted_bytes:
                    raise ValueError(f"Tensor bytes differ: {name}, byte {tensor['byte_size'] - remaining}")
                remaining -= amount
                tensor_bytes_compared += amount
        if original.read(1):
            raise ValueError("Source size changed during comparison")
    if source_hash.hexdigest() != expected_source_sha256:
        raise ValueError("Source SHA256 changed or differs during tensor comparison")
    return {"tensor_count": len(tensors), "tensor_bytes_compared": tensor_bytes_compared,
            "binary_sha256": digest(binary), "metadata_sha256": digest(metadata_path),
            "binary_size": total, "source_sha256_verified_after_comparison": source_hash.hexdigest()}


def fsync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, content: dict):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as output:
        json.dump(content, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def validate_converter_pin(source: Path, expected_sha256: str):
    regular_file(source)
    if tuple(source.parts[-3:]) != ("batchgen", "ckpt_converter", "ckpt_converter.py"):
        raise ValueError("Converter must be the repository batchgen/ckpt_converter/ckpt_converter.py")
    if digest(source) != expected_sha256:
        raise ValueError("Converter source SHA256 differs from the explicitly reviewed pin")


def checked_converter(source: Path, expected_sha256: str):
    validate_converter_pin(source, expected_sha256)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    spec = importlib.util.spec_from_file_location("r1_verified_cpu_ckpt_converter", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load converter source")
    module = importlib.util.module_from_spec(spec)
    # Compile the verified source bytes, bypassing timestamp-based .pyc caches.
    # importlib provides the module context without importing batchgen/__init__.
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Converter changed before module load")
    exec(compile(raw, str(source), "exec"), module.__dict__)
    module.torch.set_num_threads(1)
    return module.ckpt_converter()


@contextmanager
def exclusive_lock(path: Path):
    if path.is_symlink():
        raise ValueError("Lock path must not be a symlink")
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another converter owns this destination") from error
        yield


def validated_existing(receipt_path: Path, binary: Path, metadata: Path, entry: dict, converter_hash: str):
    if not receipt_path.exists():
        return False
    regular_file(receipt_path)
    receipt = read_json(receipt_path)
    for key, expected in {"source_sha256": entry["Sha256"], "source_size": entry["Size"],
                          "source_revision": WEIGHT_REVISION, "converter_sha256": converter_hash,
                          "manifest_sha256": MANIFEST_SHA256, "marlin": False}.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Existing conversion receipt has wrong {key}")
    regular_file(binary)
    regular_file(metadata)
    if binary.stat().st_size != receipt["validation"]["binary_size"]:
        raise ValueError("Previously converted binary changed size")
    if digest(binary) != receipt["validation"]["binary_sha256"] or digest(metadata) != receipt["validation"]["metadata_sha256"]:
        raise ValueError("Previously converted files changed SHA256")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Pinned original files and downloader .verified.json receipts, under /mnt/fuse/oss")
    parser.add_argument("--destination", required=True, type=Path, help="Container-local converted directory, outside /mnt/fuse")
    parser.add_argument("--converter-source", required=True, type=Path)
    parser.add_argument("--converter-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=0, help="Window to wait for index/shard download receipts; conversion itself can run longer")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--limit", type=int, help="Convert only first N sorted shards, e.g. first-shard qualification")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        parser.error("Run inside the allocated container")
    if not all(math.isfinite(value) for value in (args.wait_seconds, args.poll_seconds)) or args.wait_seconds < 0 or not 0 < args.poll_seconds <= 60:
        parser.error("wait must be nonnegative and poll must be within (0, 60] seconds")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", args.converter_sha256):
        parser.error("converter-sha256 must be a lowercase SHA256")
    source, destination, converter_source = (path.resolve() for path in (args.source, args.destination, args.converter_source))
    if not source.is_relative_to(Path("/mnt/fuse/oss")) or not source.is_dir():
        parser.error("source must be an existing OSS directory under /mnt/fuse/oss")
    if destination.is_relative_to(Path("/mnt/fuse")) or destination == source or source.is_relative_to(destination):
        parser.error("destination must be container-local and separate from the source")
    if converter_source.is_relative_to(Path("/mnt/fuse")):
        parser.error("converter-source must be the reviewed container-local repository source")
    validate_converter_pin(converter_source, args.converter_sha256)
    entries = validate_manifest(args.manifest)
    shards = sorted(name for name in entries if name.endswith(".safetensors"))
    if args.limit is not None:
        shards = shards[:args.limit]
    destination.mkdir(parents=True, exist_ok=True)
    receipts = destination.with_name(destination.name + ".receipts")
    staging = destination.with_name(destination.name + ".staging")
    for directory in (receipts, staging):
        if directory.is_symlink():
            raise ValueError("Receipt/staging directory must not be a symlink")
        directory.mkdir(exist_ok=True)
    if staging.stat().st_dev != destination.stat().st_dev:
        raise ValueError("Staging and destination must share a filesystem for atomic publication")
    allowed = {Path(name).stem + suffix for name in entries if name.endswith(".safetensors") for suffix in (".bin", ".json")}
    for path in destination.iterdir():
        if path.name not in allowed or path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected file in loader directory: {path}")
    deadline = time.monotonic() + args.wait_seconds
    with exclusive_lock(destination.with_name(destination.name + ".lock")):
        index_entry = entries[INDEX_NAME]
        while not download_ready(source, index_entry):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                event("pending_index", file=INDEX_NAME)
                return 2
            event("waiting_index", file=INDEX_NAME, remaining_wait_seconds=round(remaining, 1))
            time.sleep(min(args.poll_seconds, remaining))
        verify_source(source / INDEX_NAME, index_entry)
        index = read_json(source / INDEX_NAME)
        by_shard = defaultdict(set)
        for tensor_name, filename in index["weight_map"].items():
            if not isinstance(tensor_name, str) or filename not in entries or not filename.endswith(".safetensors"):
                raise ValueError("Invalid tensor/shard mapping in index")
            by_shard[filename].add(tensor_name)
        if set(by_shard) != {name for name in entries if name.endswith(".safetensors")}:
            raise ValueError("Pinned index does not cover exactly the manifest shards")
        converter = None
        pending = list(shards)
        while pending:
            progressed = False
            for name in list(pending):
                entry = entries[name]
                binary = destination / (Path(name).stem + ".bin")
                metadata = destination / (Path(name).stem + ".json")
                receipt_path = receipts / (name + ".converted.json")
                if validated_existing(receipt_path, binary, metadata, entry, args.converter_sha256):
                    event("already_converted_verified", file=name)
                    pending.remove(name)
                    progressed = True
                    continue
                if not download_ready(source, entry):
                    continue
                if binary.exists() != metadata.exists():
                    raise ValueError(f"Orphan published file for {name}; inspect retained staging output before recovery")
                started = time.monotonic()
                verify_source(source / name, entry)
                if digest(converter_source) != args.converter_sha256:
                    raise ValueError("Converter source changed during campaign")
                temporary = None
                recovered = binary.exists()
                if recovered:
                    validation = verify_converted(source / name, binary, metadata, by_shard[name], entry["Sha256"])
                else:
                    free = shutil.disk_usage(destination).free
                    if free < entry["Size"] + RESERVE_BYTES:
                        raise ValueError(f"Insufficient free space before {name}: {free} < shard size + 20 GiB")
                    if converter is None:
                        converter = checked_converter(converter_source, args.converter_sha256)
                    temporary = Path(tempfile.mkdtemp(prefix=Path(name).stem + ".", dir=staging))
                    event("convert_started", file=name, temporary=str(temporary), source_size=entry["Size"])
                    try:
                        converter.convert(str(source / name), str(temporary), marlin=False)
                        tmp_binary, tmp_metadata = temporary / binary.name, temporary / metadata.name
                        if {path.name for path in temporary.iterdir()} != {binary.name, metadata.name}:
                            raise ValueError("Converter emitted unexpected temporary files")
                        validation = verify_converted(source / name, tmp_binary, tmp_metadata, by_shard[name], entry["Sha256"])
                        for path in (tmp_binary, tmp_metadata):
                            with path.open("rb") as published:
                                os.fsync(published.fileno())
                        # Metadata is the visibility boundary used by the loader.
                        # Never overwrite an existing publication unexpectedly.
                        if binary.exists() or metadata.exists():
                            raise ValueError("Output appeared during conversion")
                        os.replace(tmp_binary, binary)
                        fsync_directory(destination)
                        os.replace(tmp_metadata, metadata)
                        fsync_directory(destination)
                    except BaseException as error:
                        event("conversion_failed", file=name, temporary=str(temporary), error=f"{type(error).__name__}: {error}")
                        raise
                receipt = {"schema_version": 1, "verified_utc": utc_now(), "source_file": str(source / name),
                           "source_sha256": entry["Sha256"], "source_size": entry["Size"],
                           "source_revision": WEIGHT_REVISION, "manifest_sha256": MANIFEST_SHA256,
                           "converter_source": str(converter_source), "converter_sha256": args.converter_sha256,
                           "marlin": False, "cuda_visible_devices": "", "workers": 1,
                           "recovered_existing_output": recovered, "validation": validation}
                atomic_json(receipt_path, receipt)
                if temporary is not None:
                    temporary.rmdir()  # Empty successful temp directory only.
                event("converted_verified", file=name, seconds=round(time.monotonic() - started, 2),
                      receipt=str(receipt_path), tensor_count=validation["tensor_count"], recovered=recovered)
                pending.remove(name)
                progressed = True
            if pending and not progressed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    event("pending_downloads", remaining=len(pending), files=pending)
                    return 2
                event("waiting_downloads", remaining=len(pending), remaining_wait_seconds=round(remaining, 1))
                time.sleep(min(args.poll_seconds, remaining))
        event("complete", selected_shards=len(shards), full_model=len(shards) == 163,
              destination=str(destination), receipts=str(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
