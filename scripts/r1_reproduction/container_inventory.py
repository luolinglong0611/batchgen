#!/usr/bin/env python3
"""Read-only inventory, restricted to an explicitly named experiment container.

Run inside each allocated container, using its exact pod hostname:
  python3 container_inventory.py --expected-hostname POD_NAME > inventory.json
No Torch import, model loading, package installation, or process termination occurs.
Keep the output private: it includes internal addresses, mounts and process names.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None


def command(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def inventory(expected_hostname):
    hostname = socket.gethostname()
    if platform.system() != "Linux" or hostname != expected_hostname:
        raise RuntimeError("Refusing inventory outside the explicitly named Linux container")
    cgroup = read("/proc/1/cgroup") or ""
    mounts = read("/proc/self/mountinfo") or ""
    evidence = {
        "dockerenv": Path("/.dockerenv").exists(),
        "containerenv": Path("/run/.containerenv").exists(),
        "container_cgroup": any(x in cgroup for x in ("kubepods", "docker", "containerd")),
        "overlay_root": any(" / / " in line and " - overlay " in line
                            for line in mounts.splitlines()),
    }
    if not any(evidence.values()):
        raise RuntimeError("Container evidence missing; refusing host inventory")
    packages = {}
    for name in ("torch", "triton", "transformers", "sglang", "vllm", "flash-attn",
                 "flash-attn-4", "flash-mla", "sgl-deep-gemm", "cuda-python", "libucx-cu12"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commands = {
        "gpu": ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        "gpu_processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "nvcc": ["nvcc", "--version"],
        "memory": ["free", "-b"],
        "disk": ["df", "-h", "/", "/tmp", "/mnt/fuse"],
        "network": ["ip", "-br", "addr"],
        "rdma": ["rdma", "link"],
    }
    memory_paths = (
        "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    fuse_entries = []
    if Path("/mnt/fuse").is_dir():
        try:
            with os.scandir("/mnt/fuse") as entries:
                for entry in entries:
                    fuse_entries.append(entry.name)
                    if len(fuse_entries) >= 100:
                        break
        except OSError:
            pass
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        status = read(entry / "status") or ""
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        rss = fields.get("VmRSS", "0 kB").split()[0]
        processes.append({"pid": int(entry.name), "name": fields.get("Name", "").strip(),
                          "rss_kib": int(rss) if rss.isdigit() else 0})
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": hostname, "container_evidence": evidence,
        "python": sys.version, "packages": packages,
        "limits": {path: read(path) for path in memory_paths},
        "fuse_top_level": sorted(fuse_entries),
        "largest_processes": sorted(processes, key=lambda x: x["rss_kib"], reverse=True)[:15],
        "commands": {name: command(argv) for name, argv in commands.items()},
        "qualification": "Inventory only; does not certify kernel support or free capacity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    try:
        result = inventory(args.expected_hostname)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
