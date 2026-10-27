#!/usr/bin/env python3
"""Audit/derive the unchanged first128 real requests; never run a benchmark.

Default plan reads only the frozen small manifest and asset/source identities.
--execute is CPU/tokenizer-only: it hashes the four original artifacts, retains
their deterministic prefix including duplicates, and checks current HF versus
actual BG single/native-batch tokenization. No HTTP, subprocess, dataset fitting,
trimming, padding, model loading, downloads, or performance release is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import types

MANIFEST_SHA = "8dc0c57dbe130c496b85cbe5563ca71df7d58638d7ad9ae5b7044d9e3945baa7"
TOKENIZER_PINS = {"tokenizer.json": "ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d",
                  "tokenizer_config.json": "99e80c96269ca4430f1e67f67b5766cb096cbac96fed7a7c611c6ee088b3a039"}
PROFILES = {
    "input8192-output2048": (8192, 2048,
        "c52bec745989c2d170b1681bc474605a39c3b534bdc89330155164303259a4ed",
        "edccb1acbb7fd3358ce03d521258d1acf0c11f9f8e15bb9dbe9f61f545ec9b3f",
        "b706b4e5d37f7630a57308cb85d9eecfc4d2608abe129dd5a18a71fdfd9340e9"),
    "input2048-output8192": (2048, 8192,
        "e5f907b18a65f3580ab43563bf893c38f74e6e30cd5b9b67fdf884926340fe9a",
        "398528c8c26bc6667dadb3f7d1497509da56f2104c4a345356f7821c6642f4ed",
        "50f2778508a7a087b28dd56048767cc0040d31124098eac7bfd204c0eb1ad024"),
}
REMEDIATION = [
    "Keep the unchanged original prompt and report its actual current token length as a different condition; this does not meet exact8192/2048 and requires explicit methodology review.",
    "Only if original native IDs are valid and a real native-ID path exists on both engines, separately review that transport with exact input proof; do not fabricate IDs from response usage.",
    "Separately review regeneration from the same original LongBench source rows/selection with the current tokenizer; any context-boundary edit is a new, explicitly changed payload. This tool never trims or inserts filler.",
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def input_ids(value):
    if (type(value) is not list or not value
            or any(type(item) is not int or not 0 <= item < 129280 for item in value)):
        raise ValueError("Invalid native token IDs")
    return value


def safe_child(parent, name):
    if type(name) is not str or Path(name).name != name:
        raise ValueError("Source manifest artifact must be a local basename")
    result = parent / name
    if result.is_symlink() or result.resolve().parent != parent.resolve():
        raise ValueError("Source artifact escapes the frozen bundle")
    return result


def inspect_manifest(path, *, expected_sha=MANIFEST_SHA, source_count=6000, profiles=PROFILES):
    """Test-only arguments support tiny fixtures; production CLI has no overrides."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Original6000 manifest SHA256 differs")
    manifest = decode(raw)
    if (manifest.get("count_per_profile") != source_count
            or type(manifest.get("count_per_profile")) is not int
            or manifest.get("same_sample_order_across_profiles") is not True):
        raise ValueError("Frozen manifest count or paired selection differs")
    artifacts = {}
    for item in manifest.get("artifacts", []):
        key = f"input{item.get('input_tokens')}-output{item.get('max_output_tokens')}"
        if key in artifacts:
            raise ValueError("Duplicate source profile")
        artifacts[key] = item
    if set(artifacts) != set(profiles):
        raise ValueError("Both exact original profiles are required")
    for key, (tokens, output, requests_sha, provenance_sha, _) in profiles.items():
        item = artifacts[key]
        if (item.get("requests_sha256") != requests_sha or item.get("provenance_sha256") != provenance_sha
                or type(item.get("input_tokens")) is not int or type(item.get("max_output_tokens")) is not int):
            raise ValueError("Original artifact hashes/lengths differ: " + key)
        for field in ("requests", "provenance"):
            safe_child(path.parent, item.get(field)).stat()
    return manifest, artifacts


def hashed_rows(path, expected_sha, expected_count):
    path = Path(path)
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if len(line) > 4 * 1024**2:
                raise ValueError("Unbounded source JSONL row")
            if line.strip():
                count += 1
                yield decode(line)
    after = path.stat()
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity
            or digest.hexdigest() != expected_sha or count != expected_count):
        raise ValueError("Original source artifact SHA/count/stat changed: " + str(path))


def load_prefix(directory, artifact, *, source_count=6000, count=128):
    """Read/hash all source bytes; retain only the ordered prefix in memory."""
    requests = hashed_rows(safe_child(directory, artifact["requests"]), artifact["requests_sha256"], source_count)
    provenance = hashed_rows(safe_child(directory, artifact["provenance"]), artifact["provenance_sha256"], source_count)
    kept, seen = [], set()
    for ordinal, pair in enumerate(itertools.zip_longest(requests, provenance)):
        request, record = pair
        if request is None or record is None:
            raise ValueError("Source request/provenance row counts differ")
        custom_id = request.get("custom_id")
        if (type(custom_id) is not str or custom_id != f"r1-{ordinal:06d}"
                or custom_id in seen or record.get("custom_id") != custom_id):
            raise ValueError("Source paired custom-ID order/uniqueness differs")
        seen.add(custom_id)
        body = request.get("body")
        if (request.get("method") != "POST" or request.get("url") != "/v1/completions"
                or type(body) is not dict or type(body.get("prompt")) is not str
                or body.get("model") != "deepseek-ai/DeepSeek-R1"):
            raise ValueError("Not the original R1 completion prompt envelope")
        allowed = {"model", "prompt", "max_tokens", "temperature", "top_p", "top_k", "stream", "n", "ignore_eos", "repetition_penalty"}
        if set(body) - allowed or body.get("stream", False) is not False or body.get("n", 1) != 1:
            raise ValueError("Unexpected source stop/transport/generation field")
        ids = input_ids(record.get("input_token_ids"))
        if (len(ids) != artifact["input_tokens"] or record.get("input_tokens") != len(ids)
                or record.get("input_token_ids_sha256") != value_sha(ids)
                or record.get("max_output_tokens") != artifact["max_output_tokens"]
                or type(body.get("max_tokens")) is not int or body["max_tokens"] != artifact["max_output_tokens"]):
            raise ValueError("Original provenance token identity/length differs")
        if (type(record.get("dataset")) is not str or type(record.get("original_id")) is not str
                or type(record.get("source_row_sha256")) is not str or len(record["source_row_sha256"]) != 64):
            raise ValueError("Missing original dataset/row provenance")
        if ordinal < count:
            kept.append((request, record))
    if len(kept) != count:
        raise ValueError("Not enough source requests for the fixed prefix")
    return kept


def effective_sha(rows):
    digest = hashlib.sha256()
    for request, _ in rows:
        digest.update(request["custom_id"].encode() + b"\0" + canonical(request["body"]).encode() + b"\n")
    return digest.hexdigest()


def validate_tokens(rows, hf, service, expected_length):
    prompts = [request["body"]["prompt"] for request, _ in rows]
    batch = service(prompts, return_tensors=None, padding=False, truncation=False,
                    return_attention_mask=False)["input_ids"]
    if type(batch) is not list or len(batch) != len(rows):
        raise ValueError("Native BG batch encoding returned the wrong row count")
    results, failures = [], []
    for (request, record), worker_ids in zip(rows, batch):
        prompt = request["body"]["prompt"]
        hf_ids = input_ids(hf.encode(prompt, add_special_tokens=True, truncation=False))
        single_ids = input_ids(service.encode(prompt, add_special_tokens=True))
        worker_ids = input_ids(worker_ids)
        original_ids = record["input_token_ids"]
        row = {"custom_id": request["custom_id"], "prompt_sha256": value_sha(prompt),
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source_request_body_sha256": value_sha(request["body"]),
            "expected_length": expected_length, "HF_length": len(hf_ids),
            "BG_single_length": len(single_ids), "BG_worker_length": len(worker_ids),
            "original_native_ids_sha256": value_sha(original_ids), "current_native_ids_sha256": value_sha(hf_ids),
            "HF_BG_single_identical": hf_ids == single_ids, "HF_BG_worker_identical": hf_ids == worker_ids,
            "original_native_ids_reused": original_ids == hf_ids, "prompt_changed": False}
        results.append(row)
        if not (len(hf_ids) == expected_length and hf_ids == single_ids == worker_ids == original_ids):
            failures.append(row)
    return results, failures


def load_service_tokenizer(repo):
    """Load only the actual tokenizer classes; registry imports stay inert."""
    sources = {"batchgen.config.base_tokenizer": repo / "batchgen/config/base_tokenizer.py",
        "batchgen.config.fast_tokenizer": repo / "batchgen/config/fast_tokenizer.py",
        "_r1_formal128_service_tokenizer": repo / "batchgen/models/deepseek/deepseekv3/tokenizer.py"}
    # Hash the same immutable bytes that are compiled, never a later read or
    # an imported .pyc. Final path rechecks below retain the freeze contract.
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    source_hashes = {str(path.relative_to(repo)): hashlib.sha256(source_bytes[name]).hexdigest()
                     for name, path in sources.items()}
    names = ["batchgen", "batchgen.config", "batchgen.config.tokenizer_registry", *sources]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name, path in (("batchgen", repo / "batchgen"), ("batchgen.config", repo / "batchgen/config")):
            package = types.ModuleType(name); package.__path__ = [str(path)]; sys.modules[name] = package
        registry = types.ModuleType("batchgen.config.tokenizer_registry")
        registry.register_tokenizer = lambda kind: (lambda cls: cls)
        sys.modules[registry.__name__] = registry
        for name, path in sources.items():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module
            exec(compile(source_bytes[name], str(path), "exec"), module.__dict__)
        service = module.DeepSeekV3Tokenizer()  # Its real bundled TOKENIZER_DIR.
    finally:
        for name, previous in saved.items():
            if previous is None: sys.modules.pop(name, None)
            else: sys.modules[name] = previous
    return service, source_hashes


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows: stream.write(canonical(row) + "\n")


def policy_envelope(request, output):
    body = dict(request["body"])
    body.update(max_tokens=output, temperature=0.0, top_k=1, top_p=1.0,
                ignore_eos=True, stream=False, n=1, repetition_penalty=1.0)
    return {**request, "body": body}


def guard_output(output, protected_directories):
    resolved = Path(output).resolve()
    if Path(output).exists():
        raise FileExistsError("Output must be a fresh directory")
    for directory in protected_directories:
        directory = Path(directory).resolve()
        if resolved == directory or directory in resolved.parents:
            raise ValueError("Output cannot be inside original workload/tokenizer assets")


def prepare(args):
    manifest_path = args.source_manifest.resolve(strict=True)
    manifest, artifacts = inspect_manifest(manifest_path)
    repo = args.repo.resolve(strict=True)
    bundled = repo / "batchgen/models/deepseek/deepseekv3"
    tokenizer = args.tokenizer_dir.resolve(strict=True)
    source = bundled / "tokenizer.py"
    if sha(source) != args.bg_tokenizer_source_sha256:
        raise ValueError("Current production BG tokenizer source differs from explicit binding")
    for directory in (bundled, tokenizer):
        for name, expected in TOKENIZER_PINS.items():
            if sha(directory / name) != expected:
                raise ValueError("Original tokenizer/config identity mismatch: " + str(directory / name))
    guard_output(args.output, (manifest_path.parent, tokenizer, bundled))
    report = {"schema": "r1-formal128-derivation-v1", "status": "plan_only_NOT_executed",
        "source_manifest": {"path": str(manifest_path), "sha256": MANIFEST_SHA},
        "selection": "unchanged deterministic prefix128 of original6000; duplicates retained; no replacement or search",
        "count_per_profile": 128, "repetitions_per_engine_per_profile": 3, "summary_statistic": "median",
        "profiles": {key: {"input_tokens": values[0], "output_tokens": values[1], "selected_count": 128,
            "source_requests_sha256_NOT_yet_verified": values[2], "source_provenance_sha256_NOT_yet_verified": values[3],
            "original_first128_effective_payload_sha256_target": values[4], "current_tokenization": "NOT_executed"}
            for key, values in PROFILES.items()}, "source_dataset_revision": manifest.get("dataset_revision"),
        "inherited_source_limitations": manifest.get("limitations", []),
        "source_sampling": manifest.get("sampling"), "source_seed": manifest.get("seed"),
        "source_prompt_format": manifest.get("prompt_format"),
        "tokenizer_assets": dict(TOKENIZER_PINS), "BG_tokenizer_source_sha256": sha(source),
        "generation_policy": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0,
            "seed": None, "seed_semantics": "greedy; no per-request random seed introduced"},
        "stop_policy": {"ignore_eos": True, "profile_exact_output_target": True, "additional_stop": None},
        "required_runtime_controls": {"prefix_reuse": False, "speculative_decoding": False,
            "global_active_cap_candidate": 64, "actual_controls_verified": False},
        "current_tokenizer_contract_verified": False, "source_artifact_full_hashes_verified": False,
        "GPU_HTTP_model_calls": False, "output_length_runtime_qualified": False,
        "model_quality_qualified": False, "formal_test_permitted": False}
    if not args.execute:
        return report
    if (os.environ.get("R1_FORMAL128_CPU_ONLY") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or args.expected_hostname != socket.gethostname()):
        raise ValueError("Execute requires explicit CPU-only environment and exact hostname binding")
    # Imports happen only after the CPU guard. No engine imports/model constructors.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized")
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True, trust_remote_code=False)
    service, source_hashes = load_service_tokenizer(repo)
    if source_hashes["batchgen/models/deepseek/deepseekv3/tokenizer.py"] != args.bg_tokenizer_source_sha256:
        raise ValueError("Compiled BG tokenizer bytes differ from explicit source binding")
    report["BG_tokenizer_sources"] = source_hashes
    report["libraries"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows, paired_selection = {}, None
    try:
        for key, artifact in artifacts.items():
            rows = load_prefix(manifest_path.parent, artifact)
            original_effective = effective_sha(rows)
            if original_effective != PROFILES[key][4]:
                raise ValueError("Original first128 effective payload hash differs: " + key)
            selection = [(record["dataset"], record["original_id"], record["source_row_sha256"]) for _, record in rows]
            if paired_selection is not None and selection != paired_selection:
                raise ValueError("Original profile source-sample selections differ")
            paired_selection = selection
            checks, failures = validate_tokens(rows, hf, service, artifact["input_tokens"])
            report["profiles"][key] = {"input_tokens": artifact["input_tokens"], "output_tokens": artifact["max_output_tokens"],
                "source_artifact": artifact, "original_first128_effective_payload_sha256": original_effective,
                "selected_custom_ids": [request["custom_id"] for request, _ in rows],
                "unique_source_samples": len(set(selection)), "selection_sha256": value_sha(selection),
                "tokenization_checks": checks, "tokenization_mismatches": failures}
            all_rows[key] = rows
        report["source_artifact_full_hashes_verified"] = True
        matched = not any(row["tokenization_mismatches"] for row in report["profiles"].values())
        report["current_tokenizer_contract_verified"] = matched
        report["status"] = "prepared_unchanged_first128_candidate_NOT_formal_released" if matched else "blocked_current_tokenizer_contract_NOT_formal_released"
        if not matched:
            report["remediation_options_NOT_automatically_applied"] = REMEDIATION
        else:
            for key, rows in all_rows.items():
                generated = [policy_envelope(request, PROFILES[key][1]) for request, _ in rows]
                bg, sg = args.output / (key + ".bg.requests.jsonl"), args.output / (key + ".sg.requests.jsonl")
                write_rows(bg, generated); write_rows(sg, generated)
                assert sha(bg) == sha(sg)
                native = args.output / (key + ".native-inputs.jsonl")
                write_rows(native, [{"ordinal": i, "custom_id": request["custom_id"],
                    "source_provenance": record, "prompt_sha256": value_sha(request["body"]["prompt"]),
                    "prompt_utf8_sha256": hashlib.sha256(request["body"]["prompt"].encode("utf-8")).hexdigest(),
                    "source_request_body_sha256": value_sha(request["body"]),
                    "generation_changes": {field: {"source": request["body"].get(field, {"status": "absent"}), "candidate": value}
                        for field, value in generated[i]["body"].items()
                        if field != "prompt" and (field not in request["body"] or request["body"][field] != value)},
                    "input_ids": record["input_token_ids"], "input_ids_sha256": value_sha(record["input_token_ids"])}
                    for i, (request, record) in enumerate(rows)])
                report["profiles"][key]["prepared_artifacts"] = {"BG": {"name": bg.name, "sha256": sha(bg)},
                    "SG": {"name": sg.name, "sha256": sha(sg)}, "native_inputs": {"name": native.name, "sha256": sha(native)}}
                report["profiles"][key]["parameter_changes_from_original_body"] = {
                    "top_k": "explicit1", "repetition_penalty": "explicit1.0", "temperature": "explicit0.0",
                    "top_p": "explicit1.0", "ignore_eos": "explicitTrue", "prompt": "byte-for-byte original string unchanged"}
        # Reject source/assets changed during tokenizer execution or output.
        if sha(manifest_path) != MANIFEST_SHA or any(sha(repo / name) != expected for name, expected in source_hashes.items()):
            raise ValueError("Frozen manifest or actual tokenizer sources changed during derivation")
        for directory in (bundled, tokenizer):
            if any(sha(directory / name) != expected for name, expected in TOKENIZER_PINS.items()):
                raise ValueError("Tokenizer assets changed during derivation")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only tokenizer operation initialized CUDA")
        report["CUDA_initialized"] = False
    except Exception as error:
        report.update(status="failed_derivation_NOT_formal_released", error={"type": type(error).__name__, "message": str(error)},
                      prepared_payloads_usable=False)
    report["tool_sha256"] = sha(__file__)
    write_json(args.output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bg-tokenizer-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(args)
        print(canonical(report if not args.execute else {"status": report["status"], "report": str(args.output / "manifest.json"), "formal_test_permitted": False}))
        return 0 if report["status"] in ("plan_only_NOT_executed", "prepared_unchanged_first128_candidate_NOT_formal_released") else 2
    except Exception as error:
        print(canonical({"status": "failed_closed_NOT_formal_released", "error": type(error).__name__ + ": " + str(error), "formal_test_permitted": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
