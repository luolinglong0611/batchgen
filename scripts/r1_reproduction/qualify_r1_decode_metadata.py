#!/usr/bin/env python3
"""Paired original-weight R1 attention and metadata validation qualification."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    for key in ('repo', 'candidate', 'metadata', 'weights-root', 'report'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--negative', choices=('position', 'length', 'coverage', 'physical'))
    args = p.parse_args()
    sys.path.insert(0, str(args.repo));sys.path.insert(0, str(args.repo / 'scripts/r1_reproduction'))
    import torch
    import attention_layer_qualification as qa
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    torch.cuda.set_device(0);assert torch.cuda.get_device_capability() == (10, 3)
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(1)
    for name, value in [('BATCHGEN_R1_SM103_ASYNC_VALIDATION', '1'), ('BATCHGEN_R1_SM103_CACHE_ABSORB', '1'), ('BATCHGEN_R1_SM103_PROJECTION_GRAPHS', '1')]:
        os.environ[name] = value
    spec = importlib.util.spec_from_file_location('batchgen.attention.mla.r1_decode_metadata', args.metadata)
    metadata = importlib.util.module_from_spec(spec);spec.loader.exec_module(metadata);sys.modules[spec.name] = metadata
    spec = importlib.util.spec_from_file_location('batchgen.attention.mla.sm103_meta_candidate', args.candidate)
    candidate = importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    baseline = qa.isolated_backend()
    report = {'candidate_sha256': sha(args.candidate), 'metadata_sha256': sha(args.metadata), 'baseline_sha256': sha(args.repo/'batchgen/attention/mla/sm103_backend.py'), 'metadata_cases': [], 'layer_cases': []}
    if args.negative:
        lengths=torch.tensor([65], device='cuda', dtype=torch.int32)
        positions=torch.tensor([[64]], device='cuda', dtype=torch.int64)
        table=torch.tensor([[0, 1]], device='cuda', dtype=torch.int32)
        cache=torch.empty((2,64,1,576), device='cuda', dtype=torch.bfloat16)
        if args.negative=='position':positions.add_(1)
        elif args.negative=='length':lengths.fill_(0)
        elif args.negative=='coverage':table=table[:,:1]
        else:table[0,1]=9
        try:
            metadata.prepare(cache,table,lengths,positions,65);torch.cuda.synchronize()
        except RuntimeError as error:
            assert 'assert' in str(error).lower(),str(error)
            report.update(status='invalid_metadata_device_assert_rejected',case=args.negative)
            args.report.write_text(json.dumps(report,indent=2)+'\n');return
        raise AssertionError('Invalid metadata accepted')
    for width in (1,2,3,129,160):
        for batch in (1,32):
            cache=torch.empty((8,64,1,576),device='cuda',dtype=torch.bfloat16)
            lengths=torch.full((batch,),min(width*64,257),device='cuda',dtype=torch.int32)
            positions=(lengths.long()-1)[:,None]
            table=torch.full((batch,width*2),-77,device='cuda',dtype=torch.int32)[:,::2]
            used=(lengths+63)//64
            for row in range(batch):table[row,:int(used[row])]=torch.arange(int(used[row]),device='cuda')%8
            before=table.clone()
            expected=baseline._validated_block_table(cache,table,lengths,batch,torch.device('cuda:0'),async_validation=True)
            actual,write=metadata.prepare(cache,table,lengths,positions,int(lengths.max()))
            torch.cuda.synchronize();assert torch.equal(actual,expected)and torch.equal(write,positions[:,0].int())and torch.equal(table,before)
            report['metadata_cases'].append({'width':width,'batch':batch,'bitwise':True,'noncontiguous_table':True,'source_unchanged':True})
    metadata.prepare(cache,table,lengths,positions,257);torch.cuda.synchronize()
    cache_keys=set(metadata._prepare.device_caches[torch.cuda.current_device()][0])
    for bound in range(258,290):metadata.prepare(cache,table,lengths,positions,bound)
    torch.cuda.synchronize()
    assert set(metadata._prepare.device_caches[torch.cuda.current_device()][0])==cache_keys,'Dynamic max length recompiled the metadata kernel'
    report['dynamic_maxlen_compile_cache_unchanged']=True
    report['metadata_profiler']={}
    for name in ('old','fused'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(4):
                if name=='old':
                    baseline._validate_decode_lengths(lengths,positions,int(257),async_validation=True)
                    baseline._validated_block_table(cache,table,lengths,batch,torch.device('cuda:0'),async_validation=True)
                    positions[:,0].int().contiguous()
                else:metadata.prepare(cache,table,lengths,positions,257)
            torch.cuda.synchronize()
        gpu=[e for e in profiler.events()if 'CUDA'in str(e.device_type)]
        report['metadata_profiler'][name]={'GPU_events_in_4_calls':len(gpu),'names':sorted(set(e.name for e in gpu)),'GPU_count_complete':bool(gpu),'CPU_operator_counts':{e.key:e.count for e in profiler.key_averages()if e.key.startswith('aten::')}}
    load=SimpleNamespace(weights_root=args.weights_root,weights_manifest=args.repo/'scripts/r1_reproduction/r1_model_manifest.json',weights_shard=None,weights_sha256=None,layer_index=0,seed=20261025)
    weights,scales,norms=qa.load_weights(load,torch.device('cuda:0'),report)
    config=json.loads((args.weights_root/'config.json').read_bytes());qa.validate_config(config)
    layer=qa.build_layer(config,weights,norms,torch.device('cuda:0'));layer._r1_attention_implementation='fa4_flashinfer'
    for batch,length in ((1,2048),(32,2048),(32,8192),(32,10240)):
        manager=qa.QualificationPagedCache([length-1]*batch,1,torch.device('cuda:0'));manager.cache.normal_(mean=0,std=.1)
        hidden=torch.randn((batch,1,7168),device='cuda',dtype=torch.bfloat16)
        positions=torch.full((batch,1),length-1,device='cuda',dtype=torch.int64);lengths=torch.full((batch,),length,device='cuda',dtype=torch.int32)
        def call(module,fused):
            os.environ['BATCHGEN_R1_SM103_FUSED_METADATA']=str(fused)
            return module.mla_decoding_sm103_bf16_with_pagekv(layer,hidden,positions,lengths,length,scales,manager,0,None)
        old,oldkv=call(baseline,0);new,newkv=call(candidate,1);torch.cuda.synchronize()
        assert torch.equal(old,new)and torch.equal(oldkv,newkv)
        times={'old':[],'fused':[]}
        for _ in range(3):
            for name,module,fused in [('old',baseline,0),('fused',candidate,1)]:
                for _ in range(5):call(module,fused)
                torch.cuda.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();start=time.perf_counter()
                for _ in range(30):call(module,fused)
                end.record();end.synchronize();times[name].append({'host_ms':(time.perf_counter()-start)*1000/30,'stream_ms':begin.elapsed_time(end)/30})
        retained=new.clone();hidden.normal_();old,oldkv=call(baseline,0);new,newkv=call(candidate,1);torch.cuda.synchronize();assert torch.equal(old,new)and torch.equal(oldkv,newkv)
        report['layer_cases'].append({'batch':batch,'length':length,'output_and_KV_bitwise':True,'dynamic_input_bitwise':True,'timings':times})
    report['status']='original_weight_layer_and_fused_metadata_bitwise_passed'
    report['scope']='Paired finite-padding original layer0; historical strict attention/NaN-tail failures remain unresolved; no full-model performance claim.'
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':[{'batch':r['batch'],'length':r['length'],'median_ms':{k:statistics.median(t['host_ms']for t in v)for k,v in r['timings'].items()}}for r in report['layer_cases']]}))


if __name__=='__main__':main()
