#!/usr/bin/env python3
"""Qualify native R1 BF16 MLA host offload/reload; no model or network benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def run(args, report):
    if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Run only inside the specified allocated container')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import libucx
    libucx.load_library()
    import torch
    from batchgen.models.engine_loader import core_engine as bg
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError('Qualification targets SM103')
    torch.manual_seed(20261023)
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = f'/r1_qualification_{os.getpid()}'
    cfg.num_layers, cfg.num_pages, cfg.page_size_tokens = 61, 32, 64
    cfg.num_k_heads, cfg.k_head_dim = 1, 576
    cfg.num_v_heads, cfg.v_head_dim = 0, 0
    cfg.k_element_size_bytes, cfg.v_element_size_bytes = 2, 0
    cfg.sequence_table_capacity, cfg.alignment_bytes = 16, 64
    cfg.enable_thp = False
    ids, lengths = [17, 29, 43, 71], [1, 64, 65, 129]
    pages = [(length+63)//64 for length in lengths]
    view = bg.MLAHostPagedKVWorkerView(cfg)
    initialized = False
    try:
        view.initialize(args.device, True)
        initialized = True
        view.register_sequences(ids)
        view.allocate_pages_for_sequences(list(zip(ids, lengths)))
        source = torch.randn(61, sum(lengths), 1, 576, device=device, dtype=torch.bfloat16)
        # Queue writes and D2H on the caller stream, retaining source until completion.
        for layer in range(61):
            task = view.async_offload_packed_layer_kv_to_host(layer, ids, source[layer], None, lengths)
            task.wait()
        expected = source.cpu()
        offset = 0
        for sid, length in zip(ids, lengths):
            actual, unused = view.read_sequence_kv_to_cpu(sid)
            assert unused.numel() == 0
            actual = actual.reshape(61, -1, 1, 576)[:, :length]
            assert torch.equal(actual, expected[:, offset:offset+length]), f'D2H mismatch {sid}'
            offset += length
        report['cases'].append({'name': 'all_61_layers_packed_D2H', 'status': 'passed', 'exact_bf16': True})
        destination = torch.full((61, len(ids), max(pages), 64, 1, 576), float('nan'), device=device, dtype=torch.bfloat16)
        pointers = torch.tensor([[[destination[l, s, p].data_ptr() for p in range(max(pages))]
                                 for s in range(len(ids))] for l in range(61)], dtype=torch.int64)
        task = view.async_load_layer_paged_kv_to_device(torch.tensor(ids), torch.tensor(pages), pointers, None)
        task.wait()
        torch.cuda.synchronize(device)
        offset = 0
        for s, length in enumerate(lengths):
            actual = destination[:, s].reshape(61, -1, 1, 576)[:, :length].cpu()
            assert torch.equal(actual, expected[:, offset:offset+length]), f'H2D mismatch {ids[s]}'
            offset += length
        report['cases'].append({'name': 'all_61_layers_paged_H2D', 'status': 'passed', 'exact_bf16': True})
        report.update(status='passed', torch=torch.__version__, core_engine_path=bg.__file__,
                      core_engine_sha256=hashlib.sha256(Path(bg.__file__).read_bytes()).hexdigest(),
                      geometry={'layers':61,'head_dim':576,'page_tokens':64,'lengths':lengths,'host_pool_pages':32},
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
    finally:
        if initialized:
            view.release_sequence_pages(ids)
            view.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = {'status':'failed','cases':[], 'hostname':socket.gethostname(),
              'scope':'single GPU native CPU offload/reload, no cross-node transfer or inference',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.monotonic()
    try:
        run(args, report)
    except Exception:
        report['traceback'] = traceback.format_exc()
    report['seconds_not_benchmark'] = time.monotonic()-start
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
