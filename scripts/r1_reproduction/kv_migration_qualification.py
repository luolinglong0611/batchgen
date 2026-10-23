#!/usr/bin/env python3
"""Exercise the real worker migration method with native host KV and real QueryBook.

Two ranks, one per allocated node. The harness bypasses the model constructor;
no model weights, scheduler decisions, EP dispatch or inference are qualified.
"""
import argparse
import faulthandler
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace


def run(args, result):
    if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Unexpected container')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    print('migration_stage: imports',flush=True)
    import libucx
    libucx.load_library()
    import torch
    import torch.distributed as dist
    from batchgen.models.engine_loader import core_engine as bg
    print('migration_stage: core_loaded',flush=True)
    from batchgen.batchgen_worker import BatchGenWorker
    print('migration_stage: worker_loaded',flush=True)
    from batchgen.sequence import SequenceEntry
    from batchgen.query_book import QueryBook
    print('migration_stage: cuda_init',flush=True)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError('SM103 required')
    print('migration_stage: gloo_init',flush=True)
    dist.init_process_group('gloo', init_method='tcp://'+args.init_address,
                            rank=args.rank, world_size=2, timeout=timedelta(seconds=180))
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = f'/r1_migration_{os.getpid()}'
    cfg.num_layers, cfg.num_pages, cfg.page_size_tokens = 61, 160, 64
    cfg.num_k_heads, cfg.k_head_dim, cfg.num_v_heads, cfg.v_head_dim = 1, 576, 0, 0
    cfg.k_element_size_bytes, cfg.v_element_size_bytes = 2, 0
    cfg.sequence_table_capacity, cfg.alignment_bytes, cfg.enable_thp = 16, 64, False
    view = bg.MLAHostPagedKVWorkerView(cfg)
    view.initialize(0, True)
    worker = BatchGenWorker.__new__(BatchGenWorker)
    worker.rank, worker.world_size = args.rank, 2
    worker.enable_prefix_cache = False
    worker.model_context_length = 16384
    worker.host_paged_kv_worker_view = view
    worker._trajectory_book = QueryBook(1024*1024)
    worker._get_or_create_gloo_group()  # real collective, invoked on both ranks
    try:
        for ordinal, length in enumerate([1, 65, 2048, 8192]):
            sender, receiver = ordinal % 2, 1-ordinal % 2
            sid, uuid, pages = 500+ordinal, f'r1-migration-{ordinal}', (length+63)//64
            decoded = min(3, length-1)
            prompt = length-decoded
            seq = SequenceEntry(uuid, sid, prompt, 16, text='qualification only')
            seq.decoded_length, seq.host_pages_allocated = decoded, pages
            seq.kv_token_budget = length+16
            worker.global_batch = SimpleNamespace(get_sequence=lambda key: seq if key==uuid else None)
            expected = ((torch.arange(61*length*576, dtype=torch.int64).reshape(61,length,1,576)%251)-125).to(torch.bfloat16)/8
            trajectory = torch.arange(length, dtype=torch.int32) % 120000
            if args.rank == sender:
                view.register_sequences([sid]);view.allocate_pages_for_sequences([(sid,length)])
                source = expected.to('cuda:0')
                for layer in range(61):
                    view.async_offload_packed_layer_kv_to_host(layer,[sid],source[layer],None,[length]).wait()
                slot = worker._trajectory_book.bind(uuid, worker.model_context_length)
                worker._trajectory_book.write_prompt(slot, trajectory[:prompt])
                for token in trajectory[prompt:]:worker._trajectory_book.append_token(slot, int(token))
                seq._buffer_slot = slot
            dist.barrier()
            start = time.monotonic()
            worker._execute_single_kv_migration(uuid,sender,receiver)
            dist.barrier()  # production caller also completes all transfers before releasing trajectory
            if args.rank == receiver:
                restored, unused = view.read_sequence_kv_to_cpu(sid)
                assert unused.numel()==0
                assert torch.equal(restored.reshape(61,-1,1,576)[:,:length], expected)
                assert torch.equal(worker._trajectory_tokens(seq),trajectory)
                record = worker._trajectory_book.metadata(worker._trajectory_book.slot_for(uuid))
                assert record.prompt_length==prompt and record.decoded_length==decoded
                assert worker._pending_migrated_query_book[uuid]['restored']
                destination = torch.full((61,pages,64,1,576),float('nan'),dtype=torch.bfloat16,device='cuda:0')
                pointers = torch.tensor([[[destination[l,p].data_ptr() for p in range(pages)]] for l in range(61)],dtype=torch.int64)
                view.async_load_layer_paged_kv_to_device(torch.tensor([sid]),torch.tensor([pages]),pointers,None).wait()
                torch.cuda.synchronize()
                assert torch.equal(destination.reshape(61,-1,1,576)[:,:length].cpu(),expected)
                view.release_sequence_pages([sid])
                del destination, restored
            else:
                assert view.get_stats().num_used_pages==0, 'Source pages not released'
                del source
            worker._trajectory_book.release(worker._trajectory_book.slot_for(uuid))
            dist.barrier()
            assert view.get_stats().num_used_pages==0
            case={'tokens':length,'pages':pages,'sender':sender,'receiver':receiver,'layers':61,
                  'status':'passed','seconds_not_benchmark':time.monotonic()-start}
            result['cases'].append(case)
            print(json.dumps(case),flush=True)
            del expected
        result.update(status='passed', torch=torch.__version__, transport='actual worker Gloo CPU tensor send/recv',
                      worker_source_sha256=hashlib.sha256((root/'batchgen/batchgen_worker.py').read_bytes()).hexdigest(),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        view.shutdown()
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=(0,1),required=True)
    parser.add_argument('--init-address',required=True)
    parser.add_argument('--expected-hostname',required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    faulthandler.dump_traceback_later(45,repeat=True)
    report={'status':'failed','rank':args.rank,'hostname':socket.gethostname(),'cases':[],
            'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:run(args,report)
    except Exception:report['traceback']=traceback.format_exc()
    faulthandler.cancel_dump_traceback_later()
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
