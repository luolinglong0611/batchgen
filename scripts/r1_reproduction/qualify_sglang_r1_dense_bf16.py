#!/usr/bin/env python3
"""Plan/CPU provenance/isolated original-weight dense qualification; no server.

No GPU imports in default plan or CPU provenance preparation. --execute uses
native SG TP1 MLP objects, the exact lazy installed helper, and BG main's real
DeepseekV3MLP.forward. This does not accept the full model, loaders or performance.
"""
import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import struct
import tempfile
import traceback
from types import SimpleNamespace


def helper():
    return importlib.import_module('scripts.r1_reproduction.sglang_r1_dense_bf16')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(16 * 1024 * 1024),b''):h.update(part)
    return h.hexdigest()


def frozen_manifest(path):
    if file_sha(path) != helper().MODEL_MANIFEST_SHA:
        raise ValueError('Original complete model manifest differs')
    records = json.loads(Path(path).read_text())
    return {r['Path']:r for r in records}


def checked_file(path, record):
    if path.stat().st_size != record['Size'] or file_sha(path) != record['Sha256']:
        raise ValueError('Original file SHA/size differs: ' + str(path))


def source_inventory(checkpoint, manifest_path):
    manifest = frozen_manifest(manifest_path)
    index_path = checkpoint/'model.safetensors.index.json'
    checked_file(index_path,manifest[index_path.name])
    mapping = json.loads(index_path.read_text())['weight_map']
    names = helper().expected_tensors()
    return manifest, {name:mapping[name] for name in names}


def original_provenance(checkpoint, manifest_path):
    """Independent safetensors headers/bytes; selected files full-SHA checked."""
    h = helper();manifest,mapping = source_inventory(checkpoint,manifest_path)
    result = {'schema':'r1-dense-fp8-provenance-v1','model_manifest_sha256':h.MODEL_MANIFEST_SHA,
              'model_weight_identity_sha256':h.WEIGHT_IDENTITY,
              'model_identity_scope':'reference to separately sealed complete model; this tool verifies selected original source files and 18 tensors only',
              'tensor_records':{},'source_files':{}}
    dtype_map={'F8_E4M3':('float8_e4m3fn',1),'F8_E4M3FN':('float8_e4m3fn',1),'F32':('float32',4)}
    for filename in sorted(set(mapping.values())):
        path=checkpoint/filename;checked_file(path,manifest[filename])
        with path.open('rb') as f:
            header_len=struct.unpack('<Q',f.read(8))[0]
            if header_len<=0 or header_len>100*1024*1024:raise ValueError('Invalid safetensors header size')
            header=json.loads(f.read(header_len));base=8+header_len
            for name in sorted(n for n,file in mapping.items() if file==filename):
                m=header[name];dtype,nbytes=dtype_map[m['dtype']];expected_dtype,shape=h.expected_tensors()[name]
                first,last=m['data_offsets']
                if (m['shape']!=shape or dtype!=expected_dtype or type(first)is not int or type(last)is not int
                        or first<0 or last-first!=math.prod(shape)*nbytes or base+last>path.stat().st_size):
                    raise ValueError('Actual original dense tensor header mismatch: '+name)
                f.seek(base+first);digest=hashlib.sha256();remaining=last-first
                while remaining:
                    part=f.read(min(16*1024*1024,remaining))
                    if not part:raise ValueError('Short original tensor read')
                    digest.update(part);remaining-=len(part)
                result['tensor_records'][name]={'dtype':dtype,'shape':shape,'raw_sha256':digest.hexdigest(),
                    'source_file':filename,'source_file_sha256':manifest[filename]['Sha256'],
                    'data_offsets':m['data_offsets'],'raw_bytes':last-first}
        checked_file(path,manifest[filename])
        result['source_files'][filename]={'sha256':manifest[filename]['Sha256'],'bytes':manifest[filename]['Size']}
    h.validate_provenance(result)
    return result


def server_defaults(module):
    cls=module.ServerArgs;value=cls.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:setattr(value,f.name,f.default)
        elif f.default_factory is not dataclasses.MISSING:setattr(value,f.name,f.default_factory())
    for k,v in {'tp_size':1,'dp_size':1,'ep_size':1,'pp_size':1,'enable_dp_attention':False,
                'moe_dense_tp_size':1,'enable_dp_lm_head':True,'moe_a2a_backend':'none',
                'moe_runner_backend':'triton','disable_cuda_graph':True,'disable_overlap_schedule':True,
                'disable_shared_experts_fusion':True,'enable_single_batch_overlap':False,
                'enable_two_batch_overlap':False,'enable_piecewise_cuda_graph':False,
                'enable_flashinfer_allreduce_fusion':False,'device':'cuda'}.items():setattr(value,k,v)
    return value


def cpu_dequant_proof(original, scale, derived, torch):
    """Independent per-128-row CPU formula against every derived BF16 value."""
    comparisons=0
    for first in range(0,original.shape[0],128):
        factors=scale[first//128].repeat_interleave(128)[:original.shape[1]]
        expected=(original[first:first+128].float()*factors[None,:]).to(torch.bfloat16)
        actual=derived[first:first+128].detach().cpu()
        if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
            raise ValueError('Derived BF16 weight differs from independent original CPU formula')
        comparisons+=expected.numel()
    return comparisons


def execute(args, output):
    h=helper()
    frozen = [(Path(__file__), args.worker_source_sha256), (Path(h.__file__), args.helper_source_sha256),
              (Path(__file__).with_name('sglang_r1_alignment_runtime.py'), args.runtime_source_sha256)]
    if any(not expected or file_sha(path) != expected for path, expected in frozen):
        raise RuntimeError('Explicit current worker/helper/runtime source SHA256 pins required')
    expected_env={'R1_DENSE_BF16_QUALIFICATION_GUARDED':'1','SGLANG_R1_DENSE_BF16':'1',
        'BATCHGEN_R1_SM103_FP8_COMPAT':'1','DISABLE_MMA_V5':'1','SGLANG_ENABLE_JIT_DEEPGEMM':'0',
        'SGLANG_ENABLE_FLASHINFER_FP8_GEMM':'0','SGLANG_R1_COMMON_FI_FA2':'0','R1_FULL_MODEL_TRACE':'0'}
    if any(os.environ.get(k)!=v for k,v in expected_env.items()):raise RuntimeError('Exact isolated dense qualification environment required')
    if socket.gethostname()!=args.expected_hostname:raise RuntimeError('Expected allocated hostname differs')
    if not args.provenance or not args.provenance_sha256:raise ValueError('Frozen original provenance required')
    os.environ[h.PROVENANCE_ENV]=str(args.provenance);os.environ[h.PROVENANCE_SHA_ENV]=args.provenance_sha256
    provenance,identity=h.load_provenance()
    # Revalidate source full-file pins and original byte inventory; no original is written.
    original=original_provenance(args.checkpoint_dir,args.model_manifest)
    for name,r in provenance['tensor_records'].items():
        if r['raw_sha256']!=original['tensor_records'][name]['raw_sha256']:raise ValueError('Original source tensor changed')
    if importlib.metadata.version('sglang')!='0.5.5.post3':raise RuntimeError('Frozen native SGLang version differs')
    torch=importlib.import_module('torch');torch.set_grad_enabled(False)
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one parent-selected visible GPU required')
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    if tuple(torch.cuda.get_device_capability(device))!=(10,3):raise RuntimeError('Actual SM103 required')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    sources={};server_mod=importlib.import_module('sglang.srt.server_args')
    sources['server_args']=h._source(server_mod.ServerArgs,'34dfebfe793e7e7409ee69a71c77d3550039e20ed71bff0e954e23c0042bdfdf')
    server=server_defaults(server_mod);server_mod.set_global_server_args_for_scheduler(server)
    moe=importlib.import_module('sglang.srt.layers.moe.utils');moe.initialize_moe_config(server)
    parallel=importlib.import_module('sglang.srt.distributed.parallel_state')
    sources['parallel_state']=h._source(parallel.init_distributed_environment,'274ba74d9a9ee5f4f182de0be11c30fbcc4862c146fd9f917c526b4619478fed')
    parallel.set_custom_all_reduce(False)
    rendezvous=output/'single-rank-rendezvous'
    parallel.init_distributed_environment(1,0,'file://'+str(rendezvous.resolve()),0,'nccl',timeout=180)
    parallel.initialize_model_parallel(1,1,1,backend='nccl',torch_compile=False)
    sg=importlib.import_module('sglang.srt.models.deepseek_v2')
    quant_module=importlib.import_module('sglang.srt.layers.quantization.fp8')
    sources['SG_model']=h._source(sg.DeepseekV2MLP,h.SG_MODEL_SHA)
    sources['SG_fp8']=h._source(quant_module.Fp8LinearMethod,h.SG_FP8_SHA)
    quant=quant_module.Fp8Config(is_checkpoint_fp8_serialized=True,activation_scheme='dynamic',weight_block_size=[128,128])
    runtime=importlib.import_module('scripts.r1_reproduction.sglang_r1_alignment_runtime')
    config=SimpleNamespace(model_type='deepseek_v3',architectures=['DeepseekV3ForCausalLM'],hidden_size=7168,
        n_routed_experts=256,n_shared_experts=1,num_experts_per_tok=8,moe_intermediate_size=2048,
        first_k_dense_replace=3,hidden_act='silu')
    runtime.install_fp8_linear_backend(config,quant)
    bg=importlib.import_module('batchgen.models.deepseek.deepseekv3.modeling_deepseek_v3')
    sources['BG_model']=h._source(bg.DeepseekV3MLP,'36efc5ce68346a713f3876e402b0a323d70a5ea6ccce60819f6ecb0f7f008d4d')
    dequant=importlib.import_module('batchgen_kernels.triton.fp8_quantize').deepseek_v3_dequantization
    sources['BG_dequant']=h._source(dequant,h.BG_DEQUANT_SHA)
    safe_open=importlib.import_module('safetensors').safe_open
    results=[]
    for layer in range(3):
        names={n:r for n,r in provenance['tensor_records'].items() if n.startswith(f'model.layers.{layer}.')}
        cpu={}
        for name,r in names.items():
            with safe_open(str(args.checkpoint_dir/r['source_file']),framework='pt',device='cpu') as f:cpu[name.rsplit('.mlp.',1)[1]]=f.get_tensor(name)
        with torch.device(device):
            model=sg.DeepseekV2MLP(7168,18432,'silu',quant_config=quant,prefix=f'model.layers.{layer}.mlp',tp_rank=0,tp_size=1)
        for part,linear in (('gate_up',model.gate_up_proj),('down',model.down_proj)):
            if part=='gate_up':
                linear.weight.copy_(torch.cat((cpu['gate_proj.weight'],cpu['up_proj.weight']),dim=0).to(device))
                linear.weight_scale_inv.copy_(torch.cat((cpu['gate_proj.weight_scale_inv'],cpu['up_proj.weight_scale_inv']),dim=0).to(device))
            else:
                linear.weight.copy_(cpu['down_proj.weight'].to(device));linear.weight_scale_inv.copy_(cpu['down_proj.weight_scale_inv'].to(device))
            linear.quant_method.process_weights_after_loading(linear)
        # First invocation follows the same installed lazy entry as production.
        model(torch.empty((0,7168),device=device,dtype=torch.bfloat16))
        with torch.device('meta'):
            reference=bg.DeepseekV3MLP(SimpleNamespace(hidden_size=7168,intermediate_size=18432,hidden_act='silu'))
        weight_checks=[]
        for part,derived in (('gate_proj',model.gate_up_proj.weight[:18432]),('up_proj',model.gate_up_proj.weight[18432:]),('down_proj',model.down_proj.weight)):
            count=cpu_dequant_proof(cpu[part+'.weight'],cpu[part+'.weight_scale_inv'],derived,torch)
            w=dequant(cpu[part+'.weight'].to(device),cpu[part+'.weight_scale_inv'].to(device))
            getattr(reference,part).weight=torch.nn.Parameter(w,requires_grad=False)
            if not torch.equal(w.view(torch.uint8),derived.view(torch.uint8)):raise ValueError('BG native dequant vs SG actual derived values differ')
            weight_checks.append({'projection':part,'all_elements_compared':count,'bitwise_equal':True})
        cases=[]
        for rows in (0,1,2,4,16,23,64,128):
            generator=torch.Generator(device=device);generator.manual_seed(20261026+rows)
            x=torch.randn((rows,7168),generator=generator,device=device,dtype=torch.bfloat16)
            actual=model(x);expected=reference(x);torch.cuda.synchronize()
            finite=bool(torch.isfinite(actual).all().item())
            exact=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
            maxabs=float((actual.float()-expected.float()).abs().max().item()) if rows else 0.0
            if not finite or not exact:raise ValueError(f'Native BG/SG dense contract differs layer={layer} rows={rows}; never retune thresholds')
            cases.append({'M':rows,'BF16_bitwise_equal':exact,'finite':finite,'maxabs':maxabs})
        results.append({'layer':layer,'weight_checks':weight_checks,'cases':cases,'actual_module_receipt':h.dense_backend_receipt()['modules'][-1]})
        (output/'progress.json').write_text(json.dumps({'completed_layers':len(results),'qualified':False}))
        del actual,expected,x,reference,model,cpu;torch.cuda.empty_cache()
    for filename,record in original['source_files'].items():
        checked_file(args.checkpoint_dir/filename,{'Sha256':record['sha256'],'Size':record['bytes']})
    h.load_provenance()
    for role,source in sources.items():
        if file_sha(source['path'])!=source['sha256']:
            raise RuntimeError('Recorded BG/SG dependency changed during qualification: '+role)
    if any(file_sha(path)!=expected for path,expected in frozen):
        raise RuntimeError('Execution source changed during qualification')
    return {'schema':'r1-original-dense-BF16-isolated-v1','status':'passed_original_dense_alignment_NOT_full_model_or_performance_qualified',
        'sources':sources,'execution_source_pins':{str(p):sha for p,sha in frozen},'torch_version':torch.__version__,'original_provenance':identity,'layers':results,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'memory_cap_bytes':8*1024**3,'threshold':'exact BF16 bytes for same original weights/native BG main math; independent CPU dequant all values',
        'scope':'real SG native TP1 MLPs, copied original serialized tensors; full SG checkpoint loader, distributed placement, all formal prefill shapes, FP64 MLP accuracy and full model NOT qualified',
        'formal_test_permitted':False,'full_model_qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True);parser.add_argument('--model-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--expected-hostname');parser.add_argument('--provenance',type=Path)
    parser.add_argument('--provenance-sha256');parser.add_argument('--worker-source-sha256');parser.add_argument('--helper-source-sha256');parser.add_argument('--runtime-source-sha256');mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-provenance',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not(args.prepare_provenance or args.execute):
        print(json.dumps({'schema':'r1-dense-BF16-plan-v1','execute':False,'source_tensor_count':18,'layers':[0,1,2],
            'cases_M':[0,1,2,4,16,23,64,128],'GPU_cap_bytes':8*1024**3,'persistent_bytes_increase_per_full_model_rank':1189085184,
            'requires_CPU_provenance_then_guarded_single_GPU_native_MLP':True,'formal_test_permitted':False}));return 0
    if not args.output or not args.expected_hostname or socket.gethostname()!=args.expected_hostname:raise ValueError('Explicit allocated host and fresh output required')
    if args.output.exists():raise FileExistsError(args.output)
    if args.prepare_provenance and (os.environ.get('R1_DENSE_PROVENANCE_CPU_ONLY')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise RuntimeError('CPU provenance requires explicit CPU-only environment/no GPU visibility')
    args.output.mkdir(parents=False)
    try:
        if args.prepare_provenance:
            result=original_provenance(args.checkpoint_dir,args.model_manifest);target=args.output/'dense-provenance.json'
        else:
            result=execute(args,args.output);target=args.output/'result.json'
        with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'path':str(target),'sha256':file_sha(target),'formal_test_permitted':False}));return 0
    except BaseException as error:
        with (args.output/'failed.json').open('x') as f:json.dump({'status':'failed_NOT_qualified','error':str(error),'traceback':traceback.format_exc(),'formal_test_permitted':False},f,indent=2)
        raise

if __name__=='__main__':raise SystemExit(main())
