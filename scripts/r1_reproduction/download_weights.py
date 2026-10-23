#!/usr/bin/env python3
"""Fetch a pinned ModelScope file manifest into the user's persistent model store.

Uses bounded local staging, verifies size + SHA256 before publishing each file,
and never executes code from the checkpoint. Run inside an allocated container.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlencode


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_persistent(path, expected, timeout=300):
    """Object-backed mounts can acknowledge close before reads see the object."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if path.stat().st_size != expected['Size']:
                raise OSError('Persistent size mismatch: ' + path.name)
            if digest(path) != expected['Sha256']:
                raise OSError('Persistent hash mismatch: ' + path.name)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--staging', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--exclude-file', action='append', default=[])
    ap.add_argument('--weight-start', type=int, default=0)
    ap.add_argument('--weight-end', type=int)
    args = ap.parse_args()
    if not Path('/.dockerenv').exists():
        ap.error('Run inside the allocated container')
    files = json.loads(args.manifest.read_text())
    files = [f for f in files if f['Type'] == 'blob' and '/' not in f['Path']]
    for f in files:
        if Path(f['Path']).name != f['Path'] or f['Path'] in ('.', '..'):
            ap.error('Unsafe manifest filename')
        for k, n in [('Sha256', 64), ('Revision', 40)]:
            if len(f[k]) != n or any(c not in '0123456789abcdef' for c in f[k]):
                ap.error('Manifest must contain pinned revisions and hashes')
    if not 1 <= args.workers <= 8:
        ap.error('workers must be 1..8')
    # Metadata first; then weights. Bound local scratch space to concurrent files.
    files.sort(key=lambda f: (f['Path'].endswith('.safetensors'), f['Path']))
    weights = [f for f in files if f['Path'].endswith('.safetensors')]
    if args.weight_start < 0 or (args.weight_end is not None and args.weight_end < args.weight_start):
        ap.error('Invalid weight slice')
    selected = {f['Path'] for f in weights[args.weight_start:args.weight_end]}
    files = [f for f in files if (not f['Path'].endswith('.safetensors') or f['Path'] in selected) and f['Path'] not in args.exclude_file]
    if args.limit:
        files = files[:args.limit]
    args.destination.mkdir(parents=True, exist_ok=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    needed = sum(sorted([f['Size'] for f in files], reverse=True)[:args.workers]) + 20 * 1024**3
    if shutil.disk_usage(args.staging).free < needed:
        ap.error('Insufficient local staging capacity')
    manifest_copy = args.destination / 'download-manifest.json'
    if manifest_copy.exists():
        if manifest_copy.read_text() != args.manifest.read_text():
            ap.error('Persistent manifest differs; use a new destination')
    else:
        manifest_copy.write_text(args.manifest.read_text())

    def fetch(f):
        name = f['Path']
        dest = args.destination / name
        receipt = args.destination / (name + '.verified.json')
        if dest.exists() and receipt.exists():
            old = json.loads(receipt.read_text())
            if dest.stat().st_size == f['Size'] and old.get('sha256') == f['Sha256']:
                print(json.dumps({'event': 'already_verified', 'file': name}), flush=True)
                return
        tmp = args.staging / (name + '.partial')
        url = 'https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-R1/repo?' + urlencode({'Revision': f['Revision'], 'FilePath': name})
        start = time.monotonic()
        if tmp.exists() and tmp.stat().st_size > f['Size']:
            raise RuntimeError('Staging file exceeds expected size: ' + name)
        if not tmp.exists() or tmp.stat().st_size < f['Size']:
            subprocess.run(['curl', '--continue-at', '-', '-fL', '--silent', '--show-error', '--connect-timeout', '15', '--max-time', '7200', '--retry', '3', '--retry-delay', '3', '--output', str(tmp), url], check=True, timeout=29000)
        if tmp.stat().st_size != f['Size'] or digest(tmp) != f['Sha256']:
            raise RuntimeError('Downloaded file failed verification: ' + name)
        publishing = args.destination / (name + '.publishing')
        # Recover an interrupted publication without copying several GB again.
        # The local verified source stays intact until the final object is verified.
        if not publishing.exists() or publishing.stat().st_size != f['Size']:
            shutil.copyfile(tmp, publishing)
        verify_persistent(publishing, f)
        os.replace(publishing, dest)
        verify_persistent(dest, f)
        receipt.write_text(json.dumps({'sha256': f['Sha256'], 'size': f['Size'], 'revision': f['Revision']}))
        tmp.unlink()
        print(json.dumps({'event': 'verified', 'file': name, 'bytes': f['Size'], 'seconds': round(time.monotonic()-start, 2)}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(fetch, files))
    print(json.dumps({'event': 'complete', 'files': len(files)}), flush=True)


if __name__ == '__main__':
    main()
