#!/usr/bin/env python3
"""One TP16 R1 synthetic shard through actual SGLang FP8 MoE apply method.

Requires an external one-GPU guard. Scope excludes distributed collectives and
full-model construction. The actual model's expert-construction call and optional
compat block are executed with a metadata factory (no full-model allocation).
Only distributed allocator metadata is stubbed for this single-GPU test; the
SGLang quantizer, apply method and FlashInfer native MoE are real.
"""
import argparse
import ast
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from patch_sglang_r1_routing import CONFIG_SHA, STOCK_SHA, BLOCK, ENV_NAME, patched_source, sha

H,I,E=7168,128,256


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def create_expert_metadata(source, config, layer, default, backend):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV2MoE')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    at=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id=='self' and t.attr=='experts' for t in n.targets))
    nodes=[init.body[at]]
    if isinstance(init.body[at+1],ast.If) and ENV_NAME in ast.unparse(init.body[at+1]):
        nodes.append(init.body[at+1])
    calls=[]
    def factory(**kwargs):
        calls.append(kwargs)
        layer.routing_method_type=default
        return layer
    owner=SimpleNamespace(num_fused_shared_experts=0,layer_id=3,routed_scaling_factor=2.5)
    ns={'self':owner,'config':config,'quant_config':None,'prefix':'model.layers.3.mlp',
        'get_moe_impl_class':lambda quant:factory,
        'get_global_server_args':lambda:SimpleNamespace(ep_num_redundant_experts=0),
        'add_prefix':lambda a,b:b+'.'+a,'os':os,'get_moe_runner_backend':lambda:backend}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 'installed-deepseek-expert-construction','exec'),ns)
    assert len(calls)==1
    assert calls[0]['num_experts']==E and calls[0]['intermediate_size']==2048
    return calls[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('stock','compat'),required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--stock-backup',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'mode':args.mode,'status':'failed','scope':__doc__,'cases':[]}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
        for name in ('SGLANG_ENABLE_JIT_DEEPGEMM','SGLANG_ENABLE_FLASHINFER_FP8_GEMM','SGLANG_SUPPORT_CUTLASS_BLOCK_FP8'):
            assert os.environ.get(name)=='0'
        assert os.environ.get(ENV_NAME)=='1'
        for name in ('TRITON_PTXAS_PATH','TRITON_PTXAS_BLACKWELL_PATH','DISABLE_MMA_V5'):
            assert name not in os.environ
        assert sha(args.config.read_bytes())==CONFIG_SHA
        config=SimpleNamespace(**json.loads(args.config.read_text()))
        import torch
        from sglang.srt.layers.quantization import fp8 as fp8module
        from sglang.srt.distributed.device_communicators import pynccl_allocator
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig
        from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
        from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
        site=Path(inspect.getfile(fp8module)).parents[4]
        model_file=site/'sglang/srt/models/deepseek_v2.py'
        source=model_file.read_bytes()
        expected=STOCK_SHA if args.mode=='stock' else sha(patched_source(args.stock_backup.read_bytes()))
        assert sha(source)==expected
        assert sha(Path(inspect.getfile(fp8module)).read_bytes())=='31faac58174b3ee8298504c140e00647f502db2205b7c0e6156a8c0a407595d2'
        versions={n:md.version(n) for n in ('torch','sglang','flashinfer-python','triton')}
        assert versions=={'torch':'2.9.0+cu130','sglang':'0.5.5.post3','flashinfer-python':'0.5.2','triton':'3.5.0+git4caa0328'}
        assert torch.cuda.get_device_capability()==(10,3)
        default=inspect.signature(FusedMoE.__init__).parameters['routing_method_type'].default
        assert default is None
        layer=SimpleNamespace(num_experts=E,moe_ep_rank=0,num_local_experts=E)
        call=create_expert_metadata(source.decode(),config,layer,default,MoeRunnerBackend.FLASHINFER_TRTLLM)
        report.update(versions=versions,model_source_sha256=sha(source),config_sha256=CONFIG_SHA,
            experts_constructor_kwargs=call,routing_method_type=layer.routing_method_type,
            FusedMoE_constructor_routing_default=default,apply_method_source_sha256=sha(Path(inspect.getfile(fp8module)).read_bytes()),
            real_dimensions={'H':H,'I_per_TP_rank':I,'E':E,'topk':8,'tokens':2,'TP':16,'EP':1},
            mocked_metadata_only=['get_tp_group returns None','is_allocation_symmetric returns False',
                                  'is_symmetric_memory_enabled returns False (no process-group allocator)',
                                  'expert factory records real model constructor kwargs, skips full-model allocation'])
        assert layer.routing_method_type==(None if args.mode=='stock' else RoutingMethodType.DeepSeekV3)
        gen=torch.Generator().manual_seed(7431)
        layer.w13_weight=torch.empty(E,2*I,H,dtype=torch.float8_e4m3fn,device='cuda')
        layer.w2_weight=torch.empty(E,H,I,dtype=torch.float8_e4m3fn,device='cuda')
        for e in range(E):
            layer.w13_weight[e].copy_((torch.randn(2*I,H,generator=gen)*3).to(torch.float8_e4m3fn))
            layer.w2_weight[e].copy_((torch.randn(H,I,generator=gen)*3).to(torch.float8_e4m3fn))
        layer.w13_weight_scale_inv=(.0017+torch.rand(E,2,56,generator=gen)*.0007).cuda()
        layer.w2_weight_scale_inv=(.0071+torch.rand(E,56,1,generator=gen)*.0013).cuda()
        tensors=[layer.w13_weight,layer.w2_weight,layer.w13_weight_scale_inv,layer.w2_weight_scale_inv]
        before=[tensor_sha(t) for t in tensors]
        x=(torch.randn(2,H,generator=gen)*.2).bfloat16().cuda()
        logits=torch.full((2,E),-4.,dtype=torch.bfloat16,device='cuda')
        for row,groups in enumerate(([0,2,4,6],[1,3,5,7])):
            for group in groups:
                logits[row,group*32+1]=4.+group*.02
                logits[row,group*32+17]=3.+group*.02
        bias=torch.linspace(-.01,.01,E,dtype=torch.bfloat16,device='cuda')
        topk=TopKConfig(top_k=8,use_grouped_topk=True,topk_group=4,num_expert_group=8,
                       renormalize=True,correction_bias=bias,routed_scaling_factor=2.5)
        dispatch=StandardDispatchOutput(hidden_states=x,topk_output=BypassedTopKOutput(x,logits,topk))
        method=fp8module.Fp8MoEMethod.__new__(fp8module.Fp8MoEMethod)
        method.quant_config=SimpleNamespace(weight_block_size=[128,128])
        method.moe_runner_config=SimpleNamespace(activation='silu',routed_scaling_factor=2.5)
        report['weight_scale_input_sha256_before']=before
        report['input_sha256']={'hidden':tensor_sha(x),'routing_logits':tensor_sha(logits),'routing_bias':tensor_sha(bias)}
        save()
        error=None;actual=None
        try:
            with patch.object(fp8module,'get_tp_group',return_value=None),patch.object(fp8module,'is_allocation_symmetric',return_value=False),patch.object(pynccl_allocator,'is_symmetric_memory_enabled',return_value=False):
                actual=method.apply_with_router_logits(layer,dispatch)
            torch.cuda.synchronize()
        except Exception as exc:
            error=str(exc);report['actual_apply_exception']=error;report['actual_apply_traceback']=traceback.format_exc()
        report['weight_scale_input_sha256_after']=[tensor_sha(t) for t in tensors]
        assert before==report['weight_scale_input_sha256_after']
        report['weights_and_scales_unchanged']=True
        if args.mode=='stock':
            assert error is not None and 'argument #17' in error and 'None' in error
            report['status']='stock_real_apply_argument17_None_failure_reproduced'
        else:
            if error is not None:raise RuntimeError(error)
            assert actual.shape==(2,H) and actual.dtype==torch.bfloat16 and bool(torch.isfinite(actual).all())
            aq,ascale=fp8module.per_token_group_quant_fp8(x,128)
            direct=trtllm_fp8_block_scale_moe(routing_logits=logits.float(),routing_bias=bias,
                hidden_states=aq,hidden_states_scale=ascale.t().contiguous(),gemm1_weights=layer.w13_weight,
                gemm1_weights_scale=layer.w13_weight_scale_inv,gemm2_weights=layer.w2_weight,
                gemm2_weights_scale=layer.w2_weight_scale_inv,num_experts=E,top_k=8,n_group=8,topk_group=4,
                intermediate_size=I,local_expert_offset=0,local_num_experts=E,routed_scaling_factor=2.5,
                tile_tokens_dim=None,routing_method_type=RoutingMethodType.DeepSeekV3,use_shuffled_weight=False)
            torch.cuda.synchronize()
            report['matches_explicit_DeepSeekV3_direct_call_exactly']=torch.equal(actual,direct)
            report['direct_max_abs']=float((actual.float()-direct.float()).abs().max())
            assert report['matches_explicit_DeepSeekV3_direct_call_exactly']
            torch.save({'sglang_apply':actual.cpu(),'explicit_route2':direct.cpu()},args.output/'outputs.pt')
            report['status']='compat_real_apply_finite_exact_explicit_route2'
        report['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except BaseException as exc:
        report['error']=repr(exc);report['traceback']=traceback.format_exc();raise
    finally:save()
    print(json.dumps(report,default=str),flush=True)


if __name__=='__main__':main()
