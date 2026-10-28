"""Verify the tested R1 source bytes with Python's standard library only.

Default: source and test hashes. Add --check-binaries inside the original Linux
GPU environment to require all four pinned native artifacts as well. This does
not build extensions, import Torch, start a service or imply GPU qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent /
        "frozen/r1-sm103-hotpath-20261028/SOURCE_FREEZE.json")
    parser.add_argument("--check-binaries", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "r1-sm103-hotpath-source-freeze-v1":
        raise ValueError("Unknown R1 source manifest schema")
    expected = dict(manifest["runtime_sources"])
    expected.update(manifest["CPU_test_sources"])
    if args.check_binaries:
        expected.update(manifest["binary_artifacts"])
    errors = {}
    for relative, wanted in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes checkout: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != wanted:
            errors[relative] = {"expected": wanted, "actual": actual}
    print(json.dumps({"status": "failed" if errors else "verified",
        "version_tag": manifest["version_tag"], "files_checked": len(expected),
        "binary_artifacts_checked": args.check_binaries, "mismatches": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
