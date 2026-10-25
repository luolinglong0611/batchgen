"""Paired real-checkpoint down-GEMM qualification; no model-quality claim."""
from pathlib import Path
import argparse,json,hashlib,sys,os,statistics,time,importlib.util

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--candidate-sha256',required=True);p.add_argument('--baseline-sha256',required=True);p.add_argument('--weights-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.report.exists();sha=lambda b:hashlib.sha256(b).hexdigest()
 assert sha(a.candidate.read_bytes())==a.candidate_sha256 and sha((a.repo/'batchgen/moe/fused_grouped_dequant_gemm.py').read_bytes())==a.baseline_sha256
 sys.path[:0]=[str(a.repo),str(a.repo/'scripts/r1_reproduction')]
 import torch
 from safetensors import safe_open
 from batchgen.models.deepseek.deepseekv3.kernel_dispatch import act_quant
 from convert_verified_shards import validate_manifest,download_ready,verify_source,INDEX_NAME,MANIFEST_SHA256,WEIGHT_REVISION
 assert torch.cuda.get_device_capability(0)==(10,3);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
 spec=importlib.util.spec_from_file_location('down_candidate',a.candidate);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 manifest=a.repo/'scripts/r1_reproduction/r1_model_manifest.json';entries=validate_manifest(manifest);root=a.weights_root;assert download_ready(root,entries[INDEX_NAME]);verify_source(root/INDEX_NAME,entries[INDEX_NAME]);index=json.loads((root/INDEX_NAME).read_bytes())
 selected={}
 for expert in range(16):
  for suffix in ('weight','weight_scale_inv'):
   name=f'model.layers.3.mlp.experts.{expert}.down_proj.{suffix}';filename=index['weight_map'][name];assert filename in entries;selected.setdefault(filename,[]).append(name)
 weights,scales={},{};sources=[]
 for filename,names in selected.items():
  assert download_ready(root,entries[filename]);verify_source(root/filename,entries[filename]);sources.append({'filename':filename,'sha256':entries[filename]['Sha256'],'bytes':entries[filename]['Size'],'tensors':names})
  with safe_open(str(root/filename),framework='pt',device='cpu')as sf:
   for name in names:
    eid=int(name.split('.')[5]);t=sf.get_tensor(name).to('cuda')
    if name.endswith('weight_scale_inv'):scales[eid]=t
    else:weights[eid]=t
 weight=[weights[i]for i in range(16)];scale=[scales[i]for i in range(16)]
 assert all(w.shape==(7168,2048)and w.dtype==torch.float8_e4m3fn for w in weight)
 assert all(s.shape==(56,16)and s.dtype==torch.float32 for s in scale)
 ptr=torch.tensor([w.data_ptr()for w in weight],device='cuda',dtype=torch.int64);sptr=torch.tensor([s.data_ptr()for s in scale],device='cuda',dtype=torch.int64)
 torch.manual_seed(20261025)
 cases=[('uniform_'+str(n),[n]*16,False)for n in (1,4,8,16,32,63,64,65,128)]
 cases += [('single_1',[0]*7+[1]+[0]*8,False),('single_65',[0]*7+[65]+[0]*8,False),('skew',[0,1,65,0,129,2,0,32,0,7,0,3,64,0,16,0],False),('empty',[0]*16,False),('inactive_tail',[16]*16,True),('no_active_poisoned_tail',[0]*16,True)]
 result={'schema':'r1-paired-MoE-down-real-checkpoint-v1','baseline_sha256':a.baseline_sha256,'candidate_sha256':a.candidate_sha256,'weight_source':{'root':str(root),'manifest_sha256':MANIFEST_SHA256,'revision':WEIGHT_REVISION,'layer':3,'experts':list(range(16)),'shards':sources},'cases':[],'scope':'Exact baseline/candidate down projection on checkpoint weights with deterministic synthetic quantized activations and routing. No full-model quality pass.'}
 result.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,device_capability=list(torch.cuda.get_device_capability(0)),device_name=torch.cuda.get_device_name(0),activation_source='Native R1 quantizer on deterministic normal BF16 inputs; optional poisoned unused tail')
 for label,counts,poison in cases:
  available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'));assert available>32*1024**3
  ids=[i for i,n in enumerate(counts)if n];ids=ids[::2]+ids[1::2];sizes=[counts[i]for i in ids];starts=[];total=0
  for n in sizes:starts.append(total);total+=n
  M=total+(19 if poison else 0);raw_lhs=torch.randn((M,2048),device='cuda',dtype=torch.bfloat16)
  if poison and M>total:raw_lhs[total:]=float('nan')
  lhs,lhs_scale=act_quant(raw_lhs)
  gm=torch.zeros(16,device='cuda',dtype=torch.int32);gi=gm.clone();gs=gm.clone()
  if ids:
   gm[:len(ids)]=torch.tensor(sizes,device='cuda',dtype=torch.int32);gi[:len(ids)]=torch.tensor(ids,device='cuda',dtype=torch.int32);gs[:len(ids)]=torch.tensor(starts,device='cuda',dtype=torch.int32)
  active=torch.tensor(len(ids),device='cuda',dtype=torch.int32)
  def call(parallel):
   os.environ['BATCHGEN_R1_SM103_PARALLEL_MOE_DOWN']=str(int(parallel))
   return module.fused_dequant_grouped_gemm_fp8_tma(lhs,lhs_scale,weight,ptr,scale,sptr,gm,gi,gs,active)
  expected=call(False);actual=call(True);torch.cuda.synchronize();assert torch.equal(expected,actual),(label,'not_bitwise_equal');assert torch.isfinite(actual).all(),(label,'nonfinite')
  if poison:assert torch.count_nonzero(actual[total:])==0
  for _ in range(3):call(False);call(True)
  torch.cuda.synchronize();timings={False:[],True:[]}
  for parallel in (False,True,True,False):
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();wall=time.perf_counter()
   for _ in range(20):call(parallel)
   end.record();end.synchronize();timings[parallel].append({'stream_ms':begin.elapsed_time(end)/20,'wall_ms':(time.perf_counter()-wall)*1000/20})
  row={'case':label,'counts':counts,'active_ids':ids,'rows':M,'valid_rows':total,'bitwise_equal':True,'finite':True,'inactive_tail_zero':bool(poison),'baseline':timings[False],'parallel':timings[True]};row['stream_speedup']=statistics.median(x['stream_ms']for x in timings[False])/statistics.median(x['stream_ms']for x in timings[True]);result['cases'].append(row);print(json.dumps({'case':label,'bitwise_equal':True,'speedup':row['stream_speedup']}),flush=True)
 result['peak_allocated_bytes']=torch.cuda.max_memory_allocated();result['status']='paired_bitwise_parity_and_ABBA_timings_complete';a.report.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
