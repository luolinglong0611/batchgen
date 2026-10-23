#!/usr/bin/env python3
"""Bounded FlashInfer0.5.2 TRTLLM primitive diagnostics, NOT a model benchmark.

Run ONE stage/one visible GPU at a time, only after the resource owner releases it.
Requires an external timeout/process-group supervisor. Does not install packages,
launch a service, create distributed groups, modify weights, or alter compiler env.
MLA retains the existing strict BF16 envelope (exit2 if violated). MoE records
errors against an explicitly unquantized-intermediate FP32 reference; no new
numerical acceptance threshold is invented for its different fused FP8 pipeline.
"""
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

H, I, E, TOPK, HEADS, PAGE = 7168, 128, 256, 8, 128, 64
ATOL, RTOL, MAX_FAILURE_FRACTION = 1e-5, 1.6e-2, 1e-4
EXPECTED = {'torch': '2.9.0+cu130', 'sglang': '0.5.5.post3',
            'flashinfer-python': '0.5.2', 'flashinfer-cubin': '0.5.2',
            'triton': '3.5.0+git4caa0328'}


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def metadata():
    return {name: md.version(name) for name in EXPECTED}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def source_record(fn):
    path = Path(inspect.getfile(fn))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'signature': str(inspect.signature(fn))}


def compare(actual, reference, name, report, output, strict_attention=False):
    a, r = actual.detach().float().cpu(), reference.detach().float().cpu()
    item = {'case': name, 'actual_shape': list(a.shape), 'reference_shape': list(r.shape),
            'output_dtype': str(actual.dtype)}
    report['cases'].append(item)
    torch.save({'actual': actual.detach().cpu(), 'reference_fp32': r}, output / (name + '.pt'))
    if a.shape != r.shape or actual.dtype != torch.bfloat16:
        item['status'] = 'failed_shape_or_dtype'
        raise AssertionError(name + ': BF16 output shape mismatch')
    item['finite'] = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    if not item['finite']:
        item['status'] = 'failed_nonfinite'
        raise AssertionError(name + ': nonfinite output/reference')
    err = a - r
    item.update(max_abs=float(err.abs().max()),
                relative_l2=float(err.norm() / r.norm().clamp_min(1e-30)))
    item['status'] = 'diagnostic_recorded_no_numerical_gate'
    if strict_attention:
        oracle = r.bfloat16().float()
        failures = int(((a-oracle).abs() > ATOL + RTOL*oracle.abs()).sum())
        fraction = failures / a.numel()
        item.update(strict_gate_passed=fraction < MAX_FAILURE_FRACTION,
                    failure_count=failures, failure_fraction=fraction,
                    gate={'atol': ATOL, 'rtol': RTOL, 'fraction_exclusive': MAX_FAILURE_FRACTION},
                    max_abs_bf16_oracle=float((a-oracle).abs().max()))
        item['status'] = 'strict_gate_passed' if item['strict_gate_passed'] else 'strict_gate_failed'
    emit(item)


def mla_reference(q, k, v, boundaries, scale):
    # Independent CPU FP32 oracle, head chunks bound temporary memory.
    q, k, v = q.float().cpu(), k.float().cpu(), v.float().cpu()
    out = torch.empty((len(q), HEADS, 128), dtype=torch.float32)
    for start, end in zip(boundaries, boundaries[1:]):
        mask = torch.ones(end-start, end-start, dtype=torch.bool).triu(1)
        for head in range(0, HEADS, 16):
            scores = (q[start:end, head:head+16].transpose(0, 1) @
                      k[start:end, head:head+16].transpose(0, 1).transpose(1, 2)) * scale
            p = scores.masked_fill(mask, -torch.inf).softmax(-1)
            out[start:end, head:head+16] = (p @ v[start:end, head:head+16].transpose(0, 1)).transpose(0, 1)
    return out


def run_mla(args, report):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as decode
    from flashinfer.prefill import trtllm_ragged_attention_deepseek as prefill
    report['functions'] = {'prefill': source_record(prefill), 'decode': source_record(decode)}
    # Real original R1 YaRN scaling, validated against pinned config below.
    scale = (1 + 0.1*math.log(40))**2 / math.sqrt(192)
    report['attention_scale'] = scale
    lengths, boundaries = [1, 63, 65], [0, 1, 64, 129]
    q = torch.randn(129, HEADS, 192, dtype=torch.bfloat16, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn(129, HEADS, 128, dtype=torch.bfloat16, device='cuda')
    workspace = torch.zeros(128*1024**2, dtype=torch.uint8, device='cuda')
    seq = torch.tensor(lengths, dtype=torch.int32, device='cuda')
    cu = torch.tensor(boundaries, dtype=torch.int32, device='cuda')
    actual = prefill(query=q, key=k, value=v, workspace_buffer=workspace,
        seq_lens=seq, max_q_len=65, max_kv_len=65, bmm1_scale=scale,
        bmm2_scale=1.0, o_sf_scale=1.0, batch_size=3, window_left=-1,
        cum_seq_lens_q=cu, cum_seq_lens_kv=cu, enable_pdl=False,
        is_causal=True, return_lse=False)
    torch.cuda.synchronize()
    compare(actual, mla_reference(q,k,v,boundaries,scale), 'mla_causal_prefill', report, args.output, True)
    del q,k,v,actual

    # Both zero and finite-poison tails; NaN tail behavior is a separate
    # diagnostic and cannot be silently hidden by clearing real runtime KV.
    query = torch.randn(3,1,HEADS,576,dtype=torch.bfloat16,device='cuda')
    histories = [torch.randn(n,576,dtype=torch.bfloat16,device='cuda') for n in lengths]
    table = torch.tensor([[8,2],[5,1],[7,3]], dtype=torch.int32, device='cuda')
    reference = torch.stack([((query[b,0].float().cpu() @ history.float().cpu().T)*scale).softmax(-1)
                             @ history[:,:512].float().cpu() for b,history in enumerate(histories)]).unsqueeze(1)
    report['decode_layout'] = {'query':[3,1,128,576], 'kv':[10,1,64,576],
                               'block_table':table.cpu().tolist(), 'lengths':lengths,
                               'padding_cases':['zero','finite_16'], 'nan_padding_tested':False}
    for label, poison in [('zero',0.0), ('finite_16',16.0)]:
        cache = torch.full((10,1,PAGE,576),poison,dtype=torch.bfloat16,device='cuda')
        for b,history in enumerate(histories):
            positions = torch.arange(len(history),device='cuda')
            cache[table[b,positions//PAGE].long(),0,positions%PAGE] = history
        workspace.zero_()
        actual = decode(query=query, kv_cache=cache, workspace_buffer=workspace,
            qk_nope_head_dim=128,kv_lora_rank=512,qk_rope_head_dim=64,
            block_tables=table,seq_lens=seq,max_seq_len=65,bmm1_scale=scale)
        torch.cuda.synchronize()
        compare(actual,reference,'mla_paged_decode_'+label,report,args.output,True)
        del cache,actual


def route_reference(logits, bias):
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    groups = choice.view(len(logits),8,32).topk(2,dim=-1).values.sum(-1).topk(4,dim=-1).indices
    mask = torch.zeros(len(logits),8,dtype=torch.bool).scatter_(1,groups,True).repeat_interleave(32,dim=1)
    ids = choice.masked_fill(~mask,-torch.inf).topk(TOPK,dim=-1).indices
    weights = scores.gather(1,ids)
    return ids,weights/weights.sum(-1,keepdim=True)*2.5


def dequant(weight, scale):
    return weight.float()*scale.float().repeat_interleave(128,0).repeat_interleave(128,1)[:weight.shape[0],:weight.shape[1]]


def run_moe(args, report):
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe as moe
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.layers.moe.utils import RoutingMethodType
    report['functions'] = {'moe':source_record(moe), 'activation_quantizer':source_record(per_token_group_quant_fp8)}
    assert int(RoutingMethodType.DeepSeekV3)==2
    w13=torch.empty(E,2*I,H,dtype=torch.float8_e4m3fn,device='cuda')
    w2=torch.empty(E,H,I,dtype=torch.float8_e4m3fn,device='cuda')
    # CPU per-expert generation avoids a full-size FP32 GPU temporary.
    cpu_gen=torch.Generator().manual_seed(args.seed+1)
    for e in range(E):
        w13[e].copy_((torch.randn(2*I,H,generator=cpu_gen)*3).to(torch.float8_e4m3fn))
        w2[e].copy_((torch.randn(H,I,generator=cpu_gen)*3).to(torch.float8_e4m3fn))
    s13=(0.0017+torch.rand(E,2,56,generator=cpu_gen)*0.0007).cuda()
    s2=(0.0071+torch.rand(E,56,1,generator=cpu_gen)*0.0013).cuda()
    original_scales=[s13.clone(),s2.clone()]
    if not all(t.dtype==torch.float32 and bool(((t.log2()-t.log2().round()).abs()>1e-4).any())
               for t in original_scales):
        raise AssertionError('Expected general FP32 scales, not power-of-two-only scales')
    scale_hash_before=[tensor_hash(t) for t in original_scales]
    x=(torch.randn(2,H,generator=cpu_gen)*0.2).bfloat16().cuda()
    xq,xs=per_token_group_quant_fp8(x,128)
    if xq.dtype!=torch.float8_e4m3fn or xs.dtype!=torch.float32:
        raise AssertionError('Unexpected SGLang activation quantizer format')
    xref=xq.float().cpu()*xs.float().cpu().repeat_interleave(128,dim=1)
    # Large routing margins ensure the selected groups/experts are unambiguous.
    logits=torch.full((2,E),-4.0)
    for row, groups in enumerate(([0,2,4,6],[1,3,5,7])):
        for group in groups:
            logits[row,group*32+1]=4.0+group*0.02
            logits[row,group*32+17]=3.0+group*0.02
    bias=torch.linspace(-0.01,0.01,E).bfloat16()
    ids,weights=route_reference(logits,bias)
    reference=torch.zeros(2,H)
    for row in range(2):
        for slot,e in enumerate(ids[row].tolist()):
            upgate=xref[row] @ dequant(w13[e].cpu(),s13[e].cpu()).T
            gate,up=upgate.chunk(2)
            intermediate=torch.nn.functional.silu(gate)*up
            reference[row] += (intermediate @ dequant(w2[e].cpu(),s2[e].cpu()).T)*weights[row,slot]
    report['moe_contract']={'E':E,'H':H,'I_per_TP_rank':I,'TP':16,'EP':1,'topk':TOPK,
        'w13':[E,2*I,H],'w2':[E,H,I],'weight_bytes':E*3*I*H,
        'scale_dtype':'float32','scale_block':[128,128],'power_of_two_only_scales':False,
        'routing_bias_dtype':'bfloat16 (matches SGLang correction_bias.to(x.dtype))',
        'routing_method':2,'reference_selected_experts':ids.tolist(),
        'reference':'CPU FP32 dequantized FP8 inputs/weights, FP32 SiLU/product/down and normalized sigmoid routing; excludes TRTLLM fused GEMM1/intermediate FP8 requantization and BF16 routing rounding',
        'scope':'One TP16 shard partial result; no TP allreduce,shared MLP,model or end-to-end equivalence',
        'numerical_gate':None}
    kwargs=dict(routing_logits=logits.cuda(),routing_bias=bias.cuda(),
        hidden_states=xq,hidden_states_scale=xs.t().contiguous(),
        gemm1_weights=w13,gemm1_weights_scale=s13,gemm2_weights=w2,gemm2_weights_scale=s2,
        num_experts=E,top_k=TOPK,n_group=8,topk_group=4,intermediate_size=I,
        local_expert_offset=0,local_num_experts=E,routed_scaling_factor=2.5,
        tile_tokens_dim=None,routing_method_type=RoutingMethodType.DeepSeekV3,use_shuffled_weight=False)
    actual=moe(**kwargs)
    torch.cuda.synchronize()
    report['scale_sha256_before']=scale_hash_before
    report['scale_sha256_after']=[tensor_hash(s13),tensor_hash(s2)]
    report['scales_bitwise_unchanged']=all(torch.equal(a,b) for a,b in zip(original_scales,[s13,s2]))
    if not report['scales_bitwise_unchanged']:
        raise AssertionError('FP32 scales changed')
    compare(actual,reference,'moe_tp16_partial',report,args.output)
    # Exact algebraic invariant, independent of fused intermediate quantization:
    # no linear biases => zero activations must produce a zero MoE partial sum.
    kwargs['hidden_states']=torch.zeros_like(xq)
    zero=moe(**kwargs)
    torch.cuda.synchronize()
    compare(zero,torch.zeros(2,H),'moe_zero_input',report,args.output)
    if torch.count_nonzero(zero).item()!=0:
        raise AssertionError('Zero-input MoE invariant failed')
    if not all(torch.equal(a,b) for a,b in zip(original_scales,[s13,s2])):
        raise AssertionError('FP32 scales changed after zero-input case')
    report['zero_input_exact']=True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['mla','moe'],required=True)
    ap.add_argument('--expected-hostname',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261023)
    ap.add_argument('--model-config',type=Path,default=Path('/mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8/config.json'))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'failed','stage':args.stage,'hostname':socket.gethostname(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases':[],'started_unix':time.time(),'GPU_qualification':False}
    code=1
    try:
        if not Path('/.dockerenv').exists() or socket.gethostname()!=args.expected_hostname:
            raise RuntimeError('Expected allocated container required')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible or ',' in visible or visible=='-1':
            raise RuntimeError('Expose exactly one assigned GPU')
        for key in ('TRITON_PTXAS_PATH','TRITON_PTXAS_BLACKWELL_PATH','DISABLE_MMA_V5'):
            if key in os.environ: raise RuntimeError('Baseline compiler override forbidden: '+key)
        for key in ('SGLANG_ENABLE_JIT_DEEPGEMM','SGLANG_ENABLE_FLASHINFER_FP8_GEMM','SGLANG_SUPPORT_CUTLASS_BLOCK_FP8'):
            if os.environ.get(key)!='0': raise RuntimeError('Required baseline policy: '+key+'=0')
        report['versions']=metadata()
        if report['versions']!=EXPECTED: raise RuntimeError('Baseline version mismatch')
        config_sha=hashlib.sha256(args.model_config.read_bytes()).hexdigest()
        if config_sha!='4422016f41c506339e0510a32812a63da554dc9febd5f1368e7240ded14e4a27':
            raise RuntimeError('R1 original config changed; scaling/shape assumptions require review')
        report['model_config_sha256']=config_sha
        global torch
        import torch
        import triton
        ptxas=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas-blackwell'
        ptxas_version=subprocess.check_output([str(ptxas),'--version'],text=True,timeout=30)
        if 'release 12.9' not in ptxas_version:
            raise RuntimeError('Expected original baseline Blackwell ptxas12.9')
        report['ptxas']={'path':str(ptxas),'sha256':hashlib.sha256(ptxas.read_bytes()).hexdigest(),
                         'version':ptxas_version}
        report['environment']={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_HOME',
            'LD_LIBRARY_PATH','TRITON_CACHE_DIR','SGLANG_ENABLE_JIT_DEEPGEMM',
            'SGLANG_ENABLE_FLASHINFER_FP8_GEMM','SGLANG_SUPPORT_CUTLASS_BLOCK_FP8')}
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(10,3):
            raise RuntimeError('Expected one SM103 GPU')
        torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        report['device']={'name':torch.cuda.get_device_name(),'capability':[10,3],
                          'visible':visible,'torch_allocator_cap_bytes':3*1024**3}
        (run_mla if args.stage=='mla' else run_moe)(args,report)
        strict_failed=any(c.get('strict_gate_passed') is False for c in report['cases'])
        report['status']='strict_attention_gate_failed' if strict_failed else 'diagnostics_completed_NOT_full_model_qualified'
        code=2 if strict_failed else 0
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
    finally:
        report['finished_unix']=time.time()
        if 'torch' in globals() and torch.cuda.is_initialized():
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        emit(report)
    return code


if __name__=='__main__':
    sys.exit(main())
