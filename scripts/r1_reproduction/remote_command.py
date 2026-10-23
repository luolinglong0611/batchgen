#!/usr/bin/env python3
"""Run a Python file inside one named Pouch container via existing local auth.

This never installs an agent, refreshes auth, or falls back to browser login.
The caller must supply the verified container ID and its exact hostname.
"""
import argparse
import base64
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsh", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{12,64}", args.container):
        parser.error("Use a verified Pouch container ID, not a name or shell expression")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    nonce = uuid.uuid4().hex
    begin, end = "R1_BEGIN_" + nonce, "R1_END_" + nonce
    code = (
        "import socket\nfrom pathlib import Path\n"
        f"assert socket.gethostname() == {args.hostname!r}, 'Unexpected container hostname'\n"
        "assert Path('/.dockerenv').exists(), 'Container marker missing'\n"
        f"print({begin!r}, flush=True)\n"
        f"exec(compile({args.script.read_text()!r}, '<r1-remote-script>', 'exec'))\n"
        f"print({end!r}, flush=True)\n"
    )
    encoded = base64.b64encode(zlib.compress(code.encode())).decode()
    bootstrap = f"import base64,zlib;exec(zlib.decompress(base64.b64decode({encoded!r})))"
    if len(bootstrap.encode()) > 100_000:
        parser.error('Payload is too large for a single remote command; split the deployment')
    remote = shlex.join(["pouch", "exec", args.container, "python3", "-c", bootstrap])
    argv = [args.wsh, "--transport", "direct", "--auth-profile", "default",
            "--no-auto-auth-refresh", "--idle-timeout", str(args.timeout),
            "--deadline", str(args.timeout), args.target, "-c", remote]
    try:
        result = subprocess.run(argv, capture_output=True, timeout=args.timeout + 20)
    except subprocess.TimeoutExpired as exc:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        args.capture.write_bytes((exc.stdout or b"") + (exc.stderr or b""))
        print("Transport timed out; remote process state must be checked before retrying", file=sys.stderr)
        return 124
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    args.capture.write_bytes(result.stdout + result.stderr)
    output = result.stdout.decode(errors="replace")
    output = re.sub(r"\x1b\][^\x07]*\x07|\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    output = output.replace("\r", "")
    start = output.find(begin)
    finish = output.find(end, start + len(begin)) if start >= 0 else -1
    if start >= 0:
        print(output[start + len(begin):finish if finish >= 0 else None].strip())
    else:
        print(output[-2000:], file=sys.stderr)
    if result.stderr:
        print(result.stderr.decode(errors="replace"), file=sys.stderr)
    if result.returncode != 0 or finish < 0:
        return result.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
