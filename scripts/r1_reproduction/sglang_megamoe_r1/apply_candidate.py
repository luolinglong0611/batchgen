"""Exact-byte private source overlay; plan by default, explicit --execute.

Use only a NEW isolated v0.5.22 source tree. This never discovers or modifies an
installed old SGLang environment. Each before and after source is SHA checked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

D = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def apply(repo, *, execute=False):
    pins = json.loads((D / "PINS.json").read_text())
    rows = []
    modes = set()
    for row in pins["files"]:
        source, target = D / "candidate" / row["path"], repo / row["path"]
        if sha(source) != row["candidate_sha256"]:
            raise RuntimeError("Candidate bytes changed: " + row["path"])
        actual = sha(target)
        if actual == row["base_sha256"]:
            mode = "unpatched"
        elif actual == row["candidate_sha256"]:
            mode = "already_exact_candidate"
        else:
            raise RuntimeError("Refusing unknown target bytes: " + str(target))
        modes.add(mode)
        rows.append(dict(row, current_sha256=actual, target=str(target), mode=mode))
    if len(modes) != 1:
        raise RuntimeError("Refusing partially applied bridge; inspect the isolated tree")
    if execute and modes == {"unpatched"}:
        for row in rows:
            target = Path(row["target"])
            # Preflight all three before any mutation; verify again for a race.
            if sha(target) != row["base_sha256"]:
                raise RuntimeError("Target changed during private overlay")
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                temp = Path(f.name)
                f.write((D / "candidate" / row["path"]).read_bytes())
            os.chmod(temp, target.stat().st_mode & 0o777 if target.exists() else 0o644)
            os.replace(temp, target)
        for row in rows:
            if sha(Path(row["target"])) != row["candidate_sha256"]:
                raise RuntimeError("Private overlay postflight failed")
    return {"executed": execute, "upstream_commit": pins["upstream_commit"],
            "files": rows, "scope": "isolated_new_v0522_source_only"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(json.dumps(apply(args.repo.resolve(), execute=args.execute), indent=2))
