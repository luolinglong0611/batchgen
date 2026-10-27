#!/usr/bin/env python3
"""Bind the native SG allocator probe to the common owned-process guard."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import traceback


PASS = "passed_native_page64_token_address_contract_NOT_model_qualified"
EXPECTED = {
    "sglang/srt/mem_cache/allocator.py": "4c40242b7dafc98cfbc28c396e964b96bcdbdca1d55edc3d8752ad19683a908a",
    "sglang/srt/mem_cache/memory_pool.py": "e5f75d3b14b1c375da839894abf3e6fd6b61a65ccec0ad3dab2dc4844a4a32fb",
    "sglang/srt/layers/attention/flashinfer_mla_backend.py": "f24df800a605d1d349d1d11c1a78921ace252778a2a10e980492d5fb83abad2d",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if sha(args.source) != args.source_sha256:
        raise ValueError("Native probe source binding mismatch")
    report = {"schema": "r1-native-sg-page64-guarded-v1", "status": "plan_only_NOT_executed",
              "scope": "operator", "source_sha256": args.source_sha256,
              "expected_installed_sources": EXPECTED, "full_model_qualified": False,
              "formal_test_permitted": False}
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if os.environ.get("R1_SG_ALLOCATOR_GUARDED") != "1":
        raise RuntimeError("Native SG allocator requires the owning parent guard")
    if args.output.exists() or not args.output.is_absolute():
        raise FileExistsError("Fresh absolute diagnostic directory required")
    args.output.mkdir(parents=True)
    code = 1
    report["status"] = "failed_native_page64_qualification"
    try:
        distribution = importlib.metadata.distribution("sglang")
        installed = {}
        for name, expected in EXPECTED.items():
            path = Path(distribution.locate_file(name))
            if sha(path) != expected:
                raise ValueError("Installed SG source mismatch: " + name)
            installed[name] = {"path": str(path), "sha256": expected}
        if importlib.metadata.version("sgl-kernel") != "0.3.17.post1":
            raise ValueError("Frozen SG CUDA13 kernel is required")
        import torch
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location("source_bound_sg_page64_probe", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = module.execute(args)
        if sha(args.source) != args.source_sha256:
            raise ValueError("Native probe changed during execution")
        for name, record in installed.items():
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Installed SG source changed during execution")
        report.update(native)
        report.update(installed_sources=installed, source_unchanged=True, scope="operator",
                      full_model_qualified=False, formal_test_permitted=False)
        if native["status"] == PASS and native["free_and_reuse_passed"] and all(
                value["uniform_oracle_passed"] and value["nan_tail_equals_zero_tail_bitwise"]
                for value in native["numerical_diagnostics"]):
            code = 0
    except BaseException as error:
        report.update(status="failed_native_page64_qualification", error=repr(error), traceback=traceback.format_exc())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
