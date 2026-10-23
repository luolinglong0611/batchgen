#!/usr/bin/env python3
"""Qualify the actual native Parameter_Server using one verified converted shard.

This allocates only this shard in anonymous host memory; no huge pages or host
settings are changed. It does not qualify a complete model or GPU DMA/offload.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--converted', type=Path, required=True)
    ap.add_argument('--receipts', type=Path, required=True)
    ap.add_argument('--shard', default='model-00001-of-000163')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if not Path('/.dockerenv').exists():
        ap.error('Run inside allocated container')
    import torch
    import libucx
    libucx.load_library()
    from batchgen import core_engine
    metadata_path = args.converted / (args.shard + '.json')
    bin_path = args.converted / (args.shard + '.bin')
    receipt_path = args.receipts / (args.shard + '.safetensors.converted.json')
    receipt = json.loads(receipt_path.read_text())
    meta = json.loads(metadata_path.read_text())
    size = bin_path.stat().st_size
    assert meta['total_byte_size'] == size and size < 8 * 1024**3
    assert meta['file_name'] == bin_path.name
    with tempfile.TemporaryDirectory(prefix='loader-qualification-', dir=args.converted.parent) as tmp:
        view = Path(tmp)
        os.link(bin_path, view / bin_path.name)
        os.link(metadata_path, view / metadata_path.name)
        ps = core_engine.Parameter_Server(False, False)
        start = time.monotonic()
        ps.Init('/r1_loader_qualification', '/r1_loader_metadata', ((size + 4095) // 4096) * 4096, str(view), {})
        tensors = ps.get_skeleton_state_dict()
        assert set(tensors) == set(meta['state_dict'])
        weights_fd = ps.weights_memfd_fd()
        assert weights_fd >= 0 and os.fstat(weights_fd).st_size >= size
        checks = []
        with bin_path.open('rb') as source:
            for name, tensor in sorted(tensors.items()):
                record = meta['state_dict'][name]
                assert list(tensor.shape) == record['shape']
                assert str(tensor.dtype).removeprefix('torch.') == record['dtype']
                raw = tensor.contiguous().view(torch.uint8).numpy().reshape(-1)
                assert raw.size == record['byte_size']
                source.seek(record['offset'])
                expected = hashlib.sha256()
                actual = hashlib.sha256()
                for offset in range(0, raw.size, 8 * 1024**2):
                    block = raw[offset:offset + 8 * 1024**2]
                    actual.update(memoryview(block))
                    data = source.read(len(block))
                    assert len(data) == len(block)
                    expected.update(data)
                assert expected.digest() == actual.digest(), name
                checks.append({'name': name, 'bytes': int(raw.size), 'sha256': actual.hexdigest()})
        report = {'status': 'passed', 'scope': 'Actual C++ Parameter_Server single-shard O_DIRECT + anonymous memfd load and byte equality; not GPU DMA or full model', 'core_engine': core_engine.__file__, 'torch': torch.__version__, 'shard': args.shard, 'bytes': size, 'tensor_count': len(checks), 'seconds': time.monotonic()-start, 'hugetlbfs': False, 'thp': False, 'receipt': receipt, 'tensors': checks}
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('tensors', 'receipt')}), flush=True)
        del raw, tensor, tensors
        gc.collect()
        del ps


if __name__ == '__main__':
    main()
