#!/usr/bin/env python3
"""Execute the source-bound native MGN dispatch contracts under a parent guard."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import traceback
import unittest


PASS = "passed_isolated_native_mgn_dispatch_contract_NOT_model_qualified"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--native-extension", type=Path, required=True)
    p.add_argument("--native-extension-sha256", required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if sha(args.source) != args.source_sha256:
        raise ValueError("Native test source SHA mismatch")
    if sha(args.native_extension) != args.native_extension_sha256:
        raise ValueError("AOT MGN binary SHA mismatch")
    report = {"schema": "r1-native-mgn-dispatch-qualification-v1", "status": "plan_only_NOT_executed",
              "scope": "operator", "source_sha256": args.source_sha256,
              "full_model_qualified": False, "formal_test_permitted": False}
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if os.environ.get("R1_MAIN_MGN_GUARDED") != "1":
        raise RuntimeError("An owning parent guard is required")
    repo = args.source.resolve().parents[1]
    if not (repo / "batchgen_kernels").is_dir():
        raise ValueError("The bound test must belong to the actual qualified checkout")
    if args.output.exists() or not args.output.is_absolute():
        raise FileExistsError("Fresh absolute output directory required")
    args.output.mkdir(parents=True)
    report["status"] = "failed_native_mgn_dispatch_qualification"
    code = 1
    try:
        import socket
        if socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
            raise RuntimeError("Assigned container identity mismatch")
        import torch
        if torch.__version__ != "2.9.0+cu130" or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
            raise RuntimeError("Frozen Torch and one assigned SM103 GPU required")
        torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        sys.path.insert(0, str(repo))
        os.environ["BATCHGEN_RUN_CUDA_TESTS"] = "1"
        spec = importlib.util.spec_from_file_location("bound_native_mgn_contracts", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.NativeDispatchContract)
        with contextlib.redirect_stdout(stream):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        torch.cuda.synchronize()
        from batchgen_kernels.common.mgn import _get_mgn
        extension = Path(_get_mgn().__file__).resolve()
        if (extension != args.native_extension.resolve()
                or sha(extension) != args.native_extension_sha256):
            raise RuntimeError("Native MGN extension differs from the bound AOT binary")
        report.update(tests_run=result.testsRun, skips=len(result.skipped),
                      failures=len(result.failures), errors=len(result.errors), native_log=stream.getvalue(),
                      native_extension={"path": str(extension), "sha256": sha(extension)},
                      dispatch_source={"path": str(module.SOURCE), "sha256": sha(module.SOURCE)},
                      peak_torch_bytes=torch.cuda.max_memory_allocated(),
                      source_unchanged=sha(args.source) == args.source_sha256)
        if result.wasSuccessful() and result.testsRun == 1 and not result.skipped and report["source_unchanged"]:
            report["status"] = PASS
            code = 0
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
