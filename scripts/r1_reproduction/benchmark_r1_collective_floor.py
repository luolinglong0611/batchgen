#!/usr/bin/env python3
"""Balanced and controlled-late-arrival collective references, not wire time.

Run with the same two-node16-rank transport environment as the model. No
parallel-layout or NCCL setting is changed. No model computation is included.
"""
import argparse
import ast
import logging
import json
import os
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    import torch.distributed as dist
    from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    setup_file=Path(__file__).resolve().parents[1]/'repo/batchgen/server_worker_main_loop.py'
    # The staged job receives its repo location explicitly through PYTHONPATH.
    import batchgen
    setup_file=Path(batchgen.__file__).parent/'server_worker_main_loop.py'
    tree=ast.parse(setup_file.read_bytes());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_setup_nccl_env')
    namespace={'os':os,'logging':logging};exec(compile(ast.Module(body=[function],type_ignores=[]),str(setup_file),'exec'),namespace);namespace['_setup_nccl_env']()
    assert os.environ['NCCL_IB_GID_INDEX']=='3'and os.environ['NCCL_NVLS_ENABLE']=='0'and os.environ['NCCL_RAS_ENABLE']=='0'
    assert os.environ['NCCL_IB_HCA']=='='+','.join(f'mlx5_bond_{i}:1'for i in range(8))and os.environ['NCCL_BUFFSIZE']=='16777216'
    from batchgen.distributed.utils import StatelessProcessGroup
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==16
    metadata=StatelessProcessGroup.create(os.environ['MASTER_ADDR'],34371,rank,16)
    comm=PyNcclCommunicator(metadata,torch.device('cuda',local));assert comm.available
    state=comm.change_state(enable=True);state.__enter__()
    send=torch.full((32,7168),rank+1,device='cuda',dtype=torch.bfloat16)
    output=torch.empty((512,7168),device='cuda',dtype=torch.bfloat16)
    cases=[]
    for operation in ('all_gather','all_reduce'):
        def launch():
            if operation=='all_gather':comm.all_gather(output,send)
            else:comm.all_reduce(output)
        for delayed in (False,True):
            for _ in range(16):
                if operation=='all_reduce':output.fill_(rank+1)
                launch()
            torch.cuda.synchronize();metadata.barrier();torch.cuda.synchronize()
            pairs=[];wall=time.perf_counter()
            for _ in range(128):
                if operation=='all_reduce':output.fill_(rank+1)
                first,last=(torch.cuda.Event(enable_timing=True)for _ in range(2))
                first.record()
                if delayed and rank==0:time.sleep(.001)
                launch();last.record();pairs.append((first,last))
            torch.cuda.synchronize();wall=(time.perf_counter()-wall)*1000/128
            values=sorted(first.elapsed_time(last)for first,last in pairs)
            if operation=='all_gather':assert torch.all(output[:32]==1)and torch.all(output[-32:]==16)
            else:assert torch.all(output==136)
            cases.append({'operation':operation,'mode':'rank0_requested1ms_delay'if delayed else'balanced','iterations':128,'median_stream_ms':statistics.median(values),'p95_stream_ms':values[121],'max_stream_ms':max(values),'host_ms_per_iteration':wall,'send_bytes':send.numel()*2,'full_buffer_bytes':output.numel()*2})
    result={'rank':rank,'cases':cases}
    gathered=metadata.all_gather_obj(result);torch.cuda.synchronize()
    if local==0:
        args.output.write_text(json.dumps({'status':'all16_collective_values_and_balanced_delayed_reference_complete','ranks':gathered,'transport_environment':{k:v for k,v in os.environ.items()if k.startswith(('NCCL_','GLOO_','TORCH_NCCL_'))},'scope':'Standalone balanced/controlled-lateness intervals; includes enqueue gaps and collective execution. Not pure fabric time and not a causal decomposition of production.'},indent=2)+'\n')
    metadata.barrier();state.__exit__(None,None,None);comm.destroy()
    metadata.store.set(f'collective_done/{rank}',b'1')
    if rank==0:metadata.store.wait([f'collective_done/{r}'for r in range(16)])


if __name__=='__main__':main()
