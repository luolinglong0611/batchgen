#!/usr/bin/env python3
"""Actual CUDA graph output ownership and scoped Python GC teardown checks."""
import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time
import types
import weakref


def main():
    p=argparse.ArgumentParser()
    for name in ('scope','worker-method','worker-source-sha256','report'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    import torch
    torch.cuda.set_device(0);assert torch.cuda.get_device_capability()==(10,3)
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(1)
    os.environ.update(BATCHGEN_R1_SM103_FREEZE_DECODE_GC='1',BATCHGEN_R1_SM103_FAST_DIAGNOSTIC='1',BATCHGEN_R1_SM103_ALLOW_CAPTURE='0')
    spec=importlib.util.spec_from_file_location('gc_candidate',args.scope);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert gc.isenabled()and gc.get_freeze_count()==0
    class Model:
        def __init__(self):
            self.weight=torch.randn((128,128),device='cuda',dtype=torch.bfloat16)
            self.input=torch.randn_like(self.weight)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.input@self.weight
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=self.input@self.weight
            torch.cuda.current_stream().wait_stream(stream)
            self.cycle=self
        def forward(self,value):
            self.input.copy_(value);self.graph.replay();return self.output.clone()
    model=Model();original=model.forward;hidden=torch.randn_like(model.input)
    tensor_refs=[weakref.ref(x)for x in(model.weight,model.input,model.output)]
    model_ref=weakref.ref(model);m.install(model,0)
    expected=original(hidden);actual=model.forward(hidden);torch.cuda.synchronize();assert torch.equal(expected,actual)
    scope=model._r1_decode_gc_scope;assert scope.owns_freeze and gc.isenabled()
    retained=actual.clone()
    for _ in range(20):
        hidden.normal_();expected=original(hidden);actual=model.forward(hidden)
        torch.cuda.synchronize();assert torch.equal(expected,actual)
    assert not torch.equal(retained,actual)
    before_temporary=torch.cuda.memory_allocated()
    plain=torch.empty((262144,),device='cuda');plain_ref=weakref.ref(plain);del plain
    assert plain_ref()is None and torch.cuda.memory_allocated()==before_temporary
    class Node:pass
    young=Node();young.cycle=young;young.tensor=torch.empty((262144,),device='cuda')
    young_tensor=weakref.ref(young.tensor);young_node=weakref.ref(young);del young;gc.collect()
    assert young_tensor()is None and young_node()is None and gc.isenabled()
    del original,expected,actual,retained,hidden
    namespace={'torch':torch,'gc':gc,'sys':sys,'time':time,'logging':logging}
    method=Path(args.worker_method).read_bytes();exec(compile(method,args.worker_method,'exec'),namespace)
    class Worker:
        rank=0
        torch_device=torch.device('cuda',0)
        def _log_phase_memory_ledger(self,stage):pass
    worker=Worker();worker.model=model
    worker.parallel_manager=types.SimpleNamespace(model=model,persistent_phase_instances=False)
    worker.deep_free_model_memory=types.MethodType(namespace['deep_free_model_memory'],worker)
    allocated_before_release=torch.cuda.memory_allocated();del model
    worker.deep_free_model_memory();torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    assert model_ref()is None and all(ref()is None for ref in tensor_refs)
    assert scope.closed and scope.callback not in gc.callbacks and gc.get_freeze_count()==0 and gc.isenabled()
    report={'status':'actual_GPU_graph_parity_young_cycles_and_worker_unfreeze_release_passed',
            'scope_sha256':hashlib.sha256(Path(args.scope).read_bytes()).hexdigest(),
            'worker_method_sha256':hashlib.sha256(method).hexdigest(),'worker_source_sha256':args.worker_source_sha256,
            'dynamic_graph_replays_bitwise':21,'model_and_GPU_tensor_weakrefs_released':True,
            'new_GPU_cycles_collected_while_frozen':True,'acyclic_GPU_refcount_preserved':True,
            'automatic_GC_enabled':gc.isenabled(),'remaining_frozen_objects':gc.get_freeze_count(),
            'allocated_before_release':allocated_before_release,'allocated_after_release':torch.cuda.memory_allocated(),
            'scope':'Small synthetic CUDA graph lifecycle only; original FP8 attention/MoE kernels unchanged and keep their prior qualification. No full-model numerical or performance claim.'}
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
