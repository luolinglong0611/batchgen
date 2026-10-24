#!/usr/bin/env python3
"""Paired real-weight R1 decode-layer ablations; never full-model throughput."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--negative-case", choices=("physical_page", "table_coverage", "position", "max_length"))
    parser.add_argument("--graph-ablation", action="store_true")
    parser.add_argument("--projection-graphs", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 10 and not args.report.exists()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(args.candidate) == args.candidate_sha256
    assert sha(args.repo/"batchgen/attention/mla/sm103_backend.py") == args.baseline_sha256
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo/"scripts/r1_reproduction"))
    import torch
    import attention_layer_qualification as qualification
    assert torch.cuda.get_device_capability(0) == (10, 3)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    baseline = qualification.isolated_backend()
    spec = importlib.util.spec_from_file_location("batchgen.attention.mla.sm103_perf_candidate", args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    if args.negative_case:
        # A device assertion invalidates this process's CUDA context. Each
        # negative runs in its own owned process, never the model or other tests.
        try:
            lengths = torch.tensor([65],device=device,dtype=torch.int32)
            positions = torch.tensor([[64]],device=device,dtype=torch.int64)
            if args.negative_case in ('position','max_length'):
                candidate._validate_decode_lengths(lengths,positions+int(args.negative_case=='position'),
                    64 if args.negative_case=='max_length'else 65,async_validation=True)
            else:
                cache = torch.zeros((2,64,1,576),device=device,dtype=torch.bfloat16)
                table = torch.tensor([[0,9]if args.negative_case=='physical_page'else[0]],device=device,dtype=torch.int32)
                candidate._validated_block_table(cache,table,lengths,1,device,async_validation=True)
            torch.cuda.synchronize()
        except RuntimeError as error:
            assert 'device-side assert' in str(error) or 'Invalid R1' in str(error),str(error)
            with args.report.open('x')as stream:json.dump({'status':'expected_invalid_metadata_rejected','case':args.negative_case,'error':str(error)},stream)
            return
        raise AssertionError('Invalid metadata passed the optimized validator')
    result = {"schema": "r1-paired-decode-layer-performance-v1", "baseline_sha256": args.baseline_sha256,
              "candidate_sha256": args.candidate_sha256, "cases": [], "profiles": [],
              "scope": "Actual checkpoint layer0, same input/cache/scales/kernels and strict bitwise output/KV parity. Layer timings only, no full-model quality/throughput claim."}
    load = SimpleNamespace(weights_root=args.weights_root, weights_manifest=args.repo/"scripts/r1_reproduction/r1_model_manifest.json",
                           weights_shard=None, weights_sha256=None, layer_index=0, seed=20261025)
    weights, scales, norms = qualification.load_weights(load, device, result)
    config = json.loads((args.weights_root/"config.json").read_bytes())
    qualification.validate_config(config)
    layer = qualification.build_layer(config, weights, norms, device)
    layer._r1_attention_implementation = "fa4_flashinfer"
    flags = ("BATCHGEN_R1_SM103_ASYNC_VALIDATION", "BATCHGEN_R1_SM103_CACHE_ABSORB")
    variants = [("baseline", baseline, 0, 0), ("refactor_strict", candidate, 0, 0),
                ("async_validation", candidate, 1, 0), ("cached_absorb", candidate, 0, 1),
                ("both", candidate, 1, 1)]
    if args.projection_graphs: variants.append(("projection_graphs",candidate,1,1))
    for batch, length in ((1,2048),(16,2048),(32,2048),(1,8192),(16,8192),(32,8192)):
        available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        assert available > 32*1024**3
        manager = qualification.QualificationPagedCache([length-1]*batch, 1, device)
        # Finite physical padding matches the successful finite-padding control.
        # The separate known NaN-padding numerical failure is not reclassified.
        manager.cache.normal_(mean=0, std=0.1)
        hidden = torch.randn((batch,1,7168), device=device, dtype=torch.bfloat16)
        positions = torch.full((batch,1), length-1, device=device, dtype=torch.int64)
        lengths = torch.full((batch,), length, device=device, dtype=torch.int32)
        call = lambda module, bound=length: module.mla_decoding_sm103_bf16_with_pagekv(
            layer, hidden, positions, lengths, bound, scales, manager, 0, None)
        os.environ[flags[0]] = os.environ[flags[1]] = "0"
        expected, expected_kv = call(baseline)
        torch.cuda.synchronize()
        assert torch.isfinite(expected).all() and torch.isfinite(expected_kv).all()
        case = {"batch":batch,"length":length,"variants":{}}
        for name, module, asynchronous, cached in variants:
            os.environ[flags[0]], os.environ[flags[1]] = str(asynchronous), str(cached)
            os.environ['BATCHGEN_R1_SM103_PROJECTION_GRAPHS']=str(int(name=='projection_graphs'))
            actual, actual_kv = call(module)
            torch.cuda.synchronize()
            assert torch.equal(actual, expected) and torch.equal(actual_kv, expected_kv), (batch,length,name)
            for _ in range(5): call(module)
            torch.cuda.synchronize()
            times = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); wall = time.perf_counter()
                for _ in range(args.repeats): call(module)
                end.record(); end.synchronize()
                times.append({"wall_ms":(time.perf_counter()-wall)*1000/args.repeats,
                              "cuda_event_ms":start.elapsed_time(end)/args.repeats})
            case['variants'][name] = {"bitwise_output_and_KV":True,"timings":times,
                                      "median_wall_ms":statistics.median(t['wall_ms']for t in times)}
            if batch==16 and length==2048 and name in ('baseline','both','projection_graphs'):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    for _ in range(8): call(module)
                    torch.cuda.synchronize()
                events = [{"name":e.key,"count":e.count,"cpu_total_us":e.cpu_time_total,
                           "device_total_us":getattr(e,'device_time_total',0)}for e in profile.key_averages()]
                result['profiles'].append({"variant":name,"events":events})
                profile.export_chrome_trace(str(args.report.with_name(name+'-trace.json')))
        case['speedup'] = case['variants']['baseline']['median_wall_ms']/case['variants']['both']['median_wall_ms']
        if args.projection_graphs:
            retained,retained_kv=call(candidate)
            snapshot,snapshot_kv=retained.clone(),retained_kv.clone()
            for change in range(3):
                hidden.normal_();positions.fill_(length-1+change);lengths.fill_(length+change)
                if change:manager.table.copy_(manager.table.roll(1,dims=1))
                eager,eager_kv=call(baseline,length+change)
                actual,actual_kv=call(candidate,length+change)
                torch.cuda.synchronize()
                assert torch.equal(actual,eager)and torch.equal(actual_kv,eager_kv),(batch,length,'projection_dynamic_parity',change)
                assert torch.equal(retained,snapshot)and torch.equal(retained_kv,snapshot_kv),'Graph replay overwrote an exported output/KV allocation'
            case['projection_graph_checks']={'dynamic_input_position_max_length_and_page_table':'bitwise_equal',
                                             'retained_output_and_KV_unchanged':True,
                                             'stats':dict(layer._r1_sm103_projection_graph_stats)}
        if args.graph_ablation:
            os.environ['BATCHGEN_R1_SM103_ALLOW_CAPTURE']='1'
            os.environ[flags[0]]=os.environ[flags[1]]='1'
            bound=16384
            bound_output,bound_kv=call(candidate,bound)
            torch.cuda.synchronize()
            case['graph_bound_bitwise_parity']=torch.equal(bound_output,expected)and torch.equal(bound_kv,expected_kv)
            if case['graph_bound_bitwise_parity']:
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):call(candidate,bound)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph();capture_start=time.perf_counter()
                with torch.cuda.graph(graph,stream=stream):graph_output,graph_kv=call(candidate,bound)
                capture_wall=time.perf_counter()-capture_start
                for change in range(3):
                    positions.fill_(length-1+change);lengths.fill_(length+change)
                    hidden.normal_()
                    eager,eager_kv=call(baseline,length+change)
                    graph.replay();torch.cuda.synchronize()
                    assert torch.equal(eager,graph_output)and torch.equal(eager_kv,graph_kv),(batch,length,'graph_dynamic_parity',change)
                durations=[]
                for _ in range(3):
                    torch.cuda.synchronize();t=time.perf_counter()
                    for _ in range(args.repeats):graph.replay()
                    torch.cuda.synchronize();durations.append((time.perf_counter()-t)*1000/args.repeats)
                case['graph']={'median_wall_ms':statistics.median(durations),'capture_seconds':capture_wall,
                               'changed_input_and_position_bitwise_parity':True,'max_length_bound':bound}
                graph.reset()
            os.environ['BATCHGEN_R1_SM103_ALLOW_CAPTURE']='0'
        result['cases'].append(case)
        print(json.dumps({'event':'paired_case_complete','batch':batch,'length':length,'speedup':case['speedup'],
                          'wall_ms':{n:v['median_wall_ms']for n,v in case['variants'].items()}}),flush=True)
    assert getattr(layer,'_r1_sm103_absorbed_weight_cache',None), 'Cache was not exercised'
    # Normal in-place updates and phase-style .data replacement must invalidate.
    scale = scales['kv_b_proj.weight_scale_inv']
    a = candidate._decode_absorbed_weights(layer,scales,cache_enabled=True)
    assert candidate._decode_absorbed_weights(layer,scales,cache_enabled=True).data_ptr()==a.data_ptr()
    scale.mul_(1.01)
    b = candidate._decode_absorbed_weights(layer,scales,cache_enabled=True)
    assert b.data_ptr()!=a.data_ptr()
    original = layer.kv_b_proj.weight
    layer.kv_b_proj.weight = original.clone()
    c = candidate._decode_absorbed_weights(layer,scales,cache_enabled=True)
    assert c.data_ptr()!=b.data_ptr() and torch.equal(c,b)
    if args.projection_graphs:
        previous=layer._r1_sm103_projection_graph_entry
        eager,eager_kv=call(baseline,length+2)
        actual,actual_kv=call(candidate,length+2)
        torch.cuda.synchronize()
        assert layer._r1_sm103_projection_graph_entry is not previous
        assert torch.equal(eager,actual)and torch.equal(eager_kv,actual_kv),'Changed projection weights/scales reused a stale graph'
    torch.cuda.synchronize()
    result['cache_invalidation_checks'] = 'passed'
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['status'] = 'paired_bitwise_parity_and_ablations_complete'
    with args.report.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'report_sha256':sha(args.report)}),flush=True)


if __name__ == '__main__': main()
