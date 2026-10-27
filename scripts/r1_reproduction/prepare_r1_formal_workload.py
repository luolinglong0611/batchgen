#!/usr/bin/env python3
"""CPU-only ordered prefix-N derivation from the original frozen6000 bundle.

Explicit N must be a positive multiple of16, at most6000. The frozen128 tool is
SHA-bound and reused without modification. No endpoint, HTTP, model execution,
fitting, trimming, filler, sample replacement, benchmark or release exists here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import types

HELPER_SHA256 = "3f8d312f216215cd9a6cadce05efdc5a414828cf5d9de6906fdddd7c53a13aef"
HELPER_PATH = Path(__file__).with_name("prepare_r1_formal128.py")
ROOT = Path(__file__).resolve().parents[2]


def checked_count(value):
    if type(value) is not int or value <= 0 or value > 6000 or value % 16:
        raise ValueError("Explicit count must be a positive multiple of16 at most6000")
    return value


def pinned_helper():
    raw = HELPER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA256:
        raise ValueError("Frozen128 helper SHA256 changed; do not silently inherit a new implementation")
    module = types.ModuleType("_r1_frozen128_workload_contract")
    module.__file__ = str(HELPER_PATH.resolve())
    # Execute/hash the same source bytes, without Python bytecode-cache reuse.
    exec(compile(raw, str(HELPER_PATH), "exec"), module.__dict__)
    return module


def recheck_inputs(base, manifest_path, artifacts, directories, repo, sources):
    if base.sha(manifest_path) != base.MANIFEST_SHA:
        raise ValueError("Original manifest changed during derivation")
    for artifact in artifacts.values():
        for key in ("requests", "provenance"):
            path = base.safe_child(manifest_path.parent, artifact[key])
            if base.sha(path) != artifact[key + "_sha256"]:
                raise ValueError("Original request/provenance changed after complete source read: " + str(path))
    for directory in directories:
        for name, expected in base.TOKENIZER_PINS.items():
            if base.sha(directory / name) != expected:
                raise ValueError("Original tokenizer assets changed during derivation")
    if any(base.sha(repo / name) != expected for name, expected in sources.items()):
        raise ValueError("Actual BG tokenizer source changed during derivation")
    if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError("Frozen128 helper changed during derivation")


def prepare(args):
    count = checked_count(args.count)
    base = pinned_helper()
    manifest_path = args.source_manifest.resolve(strict=True)
    manifest, artifacts = base.inspect_manifest(manifest_path)
    repo = args.repo.resolve(strict=True)
    tokenizer = args.tokenizer_dir.resolve(strict=True)
    bundled = repo / "batchgen/models/deepseek/deepseekv3"
    source = bundled / "tokenizer.py"
    if base.sha(source) != args.bg_tokenizer_source_sha256:
        raise ValueError("Explicit current BG tokenizer source identity differs")
    for directory in (tokenizer, bundled):
        for name, expected in base.TOKENIZER_PINS.items():
            if base.sha(directory / name) != expected:
                raise ValueError("Original HF/BG tokenizer/config identity differs")
    base.guard_output(args.output, (manifest_path.parent, tokenizer, repo / "batchgen", repo / "batchgen_kernels", repo / "core"))
    report = {"schema": "r1-formal-ordered-prefix-N-derivation-v1", "status": "plan_only_NOT_executed",
        "count_per_profile": count, "selection_rule": "unchanged firstN ordered rows of original6000; no skip, shuffle, deduplication or replacement",
        "count_source": "explicit caller value; final common capacity and measurement protocol determine N outside this tool",
        "source_manifest": {"path": str(manifest_path), "sha256": base.MANIFEST_SHA},
        "frozen_helper": {"path": str(HELPER_PATH.resolve()), "sha256": HELPER_SHA256},
        "tool_sha256": base.sha(__file__),
        "source_dataset_revision": manifest.get("dataset_revision"), "source_sampling": manifest.get("sampling"),
        "source_seed": manifest.get("seed"), "source_prompt_format": manifest.get("prompt_format"),
        "inherited_source_limitations": manifest.get("limitations", []),
        "tokenizer_assets": dict(base.TOKENIZER_PINS), "BG_tokenizer_source_sha256": args.bg_tokenizer_source_sha256,
        "generation_policy": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0,
                              "ignore_eos": True, "stream": False, "n": 1},
        "required_runtime_controls": {"prefix_reuse": False, "speculative_decoding": False, "actual_controls_verified": False},
        "profiles": {key: {"input_tokens": value[0], "output_tokens": value[1], "selected_count": count,
            "source_artifact": artifacts[key], "original_prefixN_effective_payload_sha256": None,
            "new_policy_prefixN_effective_payload_sha256": None, "tokenization": "NOT_executed"}
            for key, value in base.PROFILES.items()},
        "source_artifact_full_hashes_verified": False, "current_allN_tokenizer_contract_verified": False,
        "source_artifacts_rehashed_after_derivation": False,
        "output_length_runtime_qualified": False, "model_quality_qualified": False,
        "formal_test_permitted": False, "GPU_HTTP_model_calls": False}
    if not args.execute:
        return report
    if (os.environ.get("R1_FORMAL_WORKLOAD_CPU_ONLY") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or args.expected_hostname != socket.gethostname()):
        raise ValueError("Execute requires explicit CPU-only environment and exact allocated hostname")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized")
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True, trust_remote_code=False)
    service, source_hashes = base.load_service_tokenizer(repo)
    if source_hashes["batchgen/models/deepseek/deepseekv3/tokenizer.py"] != args.bg_tokenizer_source_sha256:
        raise ValueError("Compiled production tokenizer bytes differ from explicit binding")
    report["BG_tokenizer_sources"] = source_hashes
    report["libraries"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}
    args.output.mkdir(parents=True, exist_ok=False)
    selected, paired_selection = {}, None
    try:
        for key, artifact in artifacts.items():
            rows = base.load_prefix(manifest_path.parent, artifact, count=count)
            if len(rows) != count:
                raise ValueError("Prefix loader did not return all explicit N rows")
            original_effective = base.effective_sha(rows)
            # This optional nested consistency check is never used as N's hash.
            if count >= 128 and base.effective_sha(rows[:128]) != base.PROFILES[key][4]:
                raise ValueError("Original nested first128 historical source binding differs")
            selection = [(row["dataset"], row["original_id"], row["source_row_sha256"]) for _, row in rows]
            if paired_selection is not None and selection != paired_selection:
                raise ValueError("Both profiles must retain the same original source-sample order")
            paired_selection = selection
            checks, failures = base.validate_tokens(rows, hf, service, artifact["input_tokens"])
            report["profiles"][key].update(original_prefixN_effective_payload_sha256=original_effective,
                selected_custom_ids=[request["custom_id"] for request, _ in rows],
                unique_source_samples=len(set(selection)), selection_sha256=base.value_sha(selection),
                tokenization_checks=checks, tokenization_mismatches=failures,
                tokenization="passed_exact_original_native_IDs_and_lengths" if not failures else "blocked")
            selected[key] = rows
        report["source_artifact_full_hashes_verified"] = True
        recheck_inputs(base, manifest_path, artifacts, (tokenizer, bundled), repo, source_hashes)
        if base.sha(__file__) != report["tool_sha256"]:
            raise ValueError("Derivation tool source changed during execution")
        report["source_artifacts_rehashed_after_derivation"] = True
        matched = not any(row["tokenization_mismatches"] for row in report["profiles"].values())
        report["current_allN_tokenizer_contract_verified"] = matched
        if not matched:
            report.update(status="blocked_current_allN_tokenizer_contract_NOT_formal_released",
                          remediation_options_NOT_automatically_applied=base.REMEDIATION)
        else:
            for key, rows in selected.items():
                prepared = [base.policy_envelope(request, artifacts[key]["max_output_tokens"]) for request, _ in rows]
                new_effective = base.effective_sha([(request, previous[1]) for request, previous in zip(prepared, rows)])
                bg, sg = args.output / (key + ".bg.requests.jsonl"), args.output / (key + ".sg.requests.jsonl")
                base.write_rows(bg, prepared); base.write_rows(sg, prepared)
                if base.sha(bg) != base.sha(sg):
                    raise ValueError("Prepared backend bytes differ")
                native = args.output / (key + ".native-inputs.jsonl")
                base.write_rows(native, [{"ordinal": i, "custom_id": request["custom_id"], "source_provenance": record,
                    "source_request_body_sha256": base.value_sha(request["body"]),
                    "prompt_utf8_sha256": hashlib.sha256(request["body"]["prompt"].encode("utf-8")).hexdigest(),
                    "input_ids": record["input_token_ids"], "input_ids_sha256": base.value_sha(record["input_token_ids"]),
                    "generation_changes": {field: {"source": request["body"].get(field, {"status": "absent"}), "candidate": value}
                        for field, value in prepared[i]["body"].items()
                        if field != "prompt" and (field not in request["body"] or request["body"][field] != value)}}
                    for i, (request, record) in enumerate(rows)])
                report["profiles"][key].update(new_policy_prefixN_effective_payload_sha256=new_effective,
                    prepared_artifacts={"BG": {"name": bg.name, "sha256": base.sha(bg)},
                        "SG": {"name": sg.name, "sha256": base.sha(sg)},
                        "native_inputs": {"name": native.name, "sha256": base.sha(native)}},
                    original_prompt_and_native_IDs_unchanged=True)
            # Detect a source change during output too. Failed output is unusable.
            recheck_inputs(base, manifest_path, artifacts, (tokenizer, bundled), repo, source_hashes)
            if base.sha(__file__) != report["tool_sha256"]:
                raise ValueError("Derivation tool source changed during output")
            report["status"] = "prepared_unchanged_ordered_prefixN_candidate_NOT_formal_released"
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only preparation initialized CUDA")
        report["CUDA_initialized"] = False
    except Exception as error:
        report.update(status="failed_derivation_NOT_formal_released", prepared_payloads_usable=False,
                      error={"type": type(error).__name__, "message": str(error)})
    base.write_json(args.output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--bg-tokenizer-source-sha256", required=True)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(args)
        result = report if not args.execute else {"status": report["status"], "report": str(args.output / "manifest.json"), "formal_test_permitted": False}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0 if report["status"] in ("plan_only_NOT_executed", "prepared_unchanged_ordered_prefixN_candidate_NOT_formal_released") else 2
    except Exception as error:
        print(json.dumps({"status": "failed_closed_NOT_formal_released", "error": type(error).__name__ + ": " + str(error), "formal_test_permitted": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
