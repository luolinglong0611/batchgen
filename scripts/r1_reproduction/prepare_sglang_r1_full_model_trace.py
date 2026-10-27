#!/usr/bin/env python3
"""Compose the pinned existing SG alignment candidate with opt-in trace bootstrap.

Default plan-only. --execute writes a fresh isolated candidate and diffs; never
modifies installed SG, imports accelerators, connects remotely or starts a model.
"""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


bridge = load("r1_trace_source_bridge", "sglang_r1_full_model_trace.py")
shared = load("r1_trace_existing_shared_generator", "prepare_sglang_r1_local_shared.py")


def prepare(source_root):
    sources = {}
    for path, expected in bridge.STOCK.items():
        actual = bridge.sha(source_root / path)
        if actual != expected:
            raise ValueError("Exact installed stock SG source SHA mismatch: " + path)
        sources[path] = {"path": str((source_root / path).resolve()), "sha256": actual}
    original = (source_root / "models/deepseek_v2.py").read_bytes()
    alignment = shared.patched_source(original)
    combined = alignment + bridge.BOOTSTRAP.encode()
    import hashlib
    if hashlib.sha256(combined).hexdigest() != bridge.COMBINED_MODEL_SHA256:
        raise ValueError("Existing alignment candidate changed; review a new exact combined model admission before deploying trace")
    compile(combined, "deepseek_v2.py.r1-shared-common-and-trace", "exec")
    return original, alignment, combined, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Stock installed sglang/srt directory")
    p.add_argument("--output-dir", type=Path); p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    original, alignment, combined, sources = prepare(args.source_root)
    import hashlib
    digest = lambda data: hashlib.sha256(data).hexdigest()
    diff = lambda before, after, a, b: "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True), fromfile=a, tofile=b))
    receipt = {"status": "plan_only_NOT_executed", "scope": "full_model", "stock_sources": sources,
        "stock_model_sha256": digest(original), "existing_alignment_model_sha256": digest(alignment),
        "combined_candidate_model_sha256": digest(combined),
        "existing_alignment_generator_sha256": bridge.sha(ROOT / "prepare_sglang_r1_local_shared.py"),
        "trace_generator_sha256": bridge.sha(__file__), "bridge_sha256": bridge.sha(ROOT / "sglang_r1_full_model_trace.py"),
        "full_model_helper_sha256": bridge.sha(ROOT / "r1_full_model_trace.py"),
        "required_environment_additions": {"R1_FULL_MODEL_TRACE": "1", "R1_SG_TRACE_MODEL_SHA256": digest(combined)},
        "required_existing_environment": "retain the shared/common-kernel candidate settings and complete full-model trace suite/deployment environment",
        "new_arithmetic_or_sampling_changes": False, "installed_source_changed": False,
        "actual_GPU_or_full_model_qualification": False,
        "alignment_to_combined_diff": diff(alignment, combined, "deepseek_v2.py.alignment", "deepseek_v2.py.alignment-and-trace"),
        "stock_to_combined_diff": diff(original, combined, "deepseek_v2.py.stock", "deepseek_v2.py.alignment-and-trace")}
    if args.execute:
        if args.output_dir is None or not args.output_dir.is_absolute():
            p.error("--execute requires an absolute fresh --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "deepseek_v2.py.stock").write_bytes(original)
        (args.output_dir / "deepseek_v2.py.alignment").write_bytes(alignment)
        (args.output_dir / "deepseek_v2.py").write_bytes(combined)
        (args.output_dir / "stock-to-combined.patch").write_text(receipt["stock_to_combined_diff"])
        (args.output_dir / "alignment-to-trace.patch").write_text(receipt["alignment_to_combined_diff"])
        receipt["status"] = "isolated_combined_source_prepared_NOT_deployed_or_qualified"
        (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
