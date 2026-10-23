#!/usr/bin/env python3
"""Run the real R1 startup preflight without starting workers or loading weights."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('batchgen/runtime_preflight.py',
           'batchgen/models/deepseek/deepseekv3/kernel_dispatch.py',
           'batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py',
           'batchgen/attention/mla/sm103_backend.py',
           'batchgen/attention/mla/triton_precision.py')


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Use a fresh report path; retain old failure records')
    report = {'status': 'failed', 'scope': 'Actual startup dependency/origin/AOT preflight only; no model worker or weights',
              'hostname': socket.gethostname(), 'model': args.model,
              'environment': {key: os.environ.get(key) for key in (
                  'BATCHGEN_R1_SM103_EXPERIMENTAL', 'BATCHGEN_R1_SM103_ATTENTION',
                  'DISABLE_MMA_V5', 'BATCHGEN_W8A16_DEQUANT', 'TORCH_CUDA_CHANNEL', 'TRITON_PTXAS_PATH')}}
    start = time.monotonic()
    preflight = None
    try:
        if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
            raise RuntimeError('Require the explicitly allocated container')
        sys.path.insert(0, str(ROOT))
        import torch
        torch.cuda.set_device(args.device)
        if torch.cuda.get_device_capability(args.device) != (10, 3):
            raise RuntimeError('This qualification is restricted to SM103')
        logging.basicConfig(level=logging.INFO)
        import batchgen.runtime_preflight as preflight
        before = hashes()
        report['source_sha256'] = before
        report['model_type'] = preflight.run_runtime_preflight(SimpleNamespace(
            model=args.model, enable_cuda_graph=False, enable_prefix_cache=False,
        ))
        if hashes() != before:
            raise RuntimeError('Runtime source changed while preflight was running; rerun against a stable snapshot')
        report.update(status='passed', torch=torch.__version__, cuda=torch.version.cuda,
                      cuda_device=args.device, device_name=torch.cuda.get_device_name(args.device),
                      cuda_allocated_bytes=torch.cuda.memory_allocated(args.device),
                      root=str(preflight._batchgen_root()))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if preflight is not None:
            report['manifest'] = dict(preflight._manifest)
        report['seconds'] = time.monotonic() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
