"""R1 real-weight expert tile ablations with exact old-kernel controls."""
from pathlib import Path
import argparse,json,hashlib,sys,os,statistics,time,importlib.util

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--down-candidate',type=Path,required=True);p.add_argument('--candidate-sha256',required=True);p.add_argument('--weights-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.report.exists();assert hashlib.sha256(a.down_candidate.read_bytes()).hexdigest()==a.candidate_sha256
 sys.path[:0]=[str(a.repo),str(a.repo/'scripts/r1_reproduction')]
 import torch
 from safetensors import safe_open
 from batchgen.models.deepseek.deepseekv3.kernel_dispatch import act_quant
 from convert_verified_shards import validate_manifest,download_ready,verify_source,INDEX_NAME,MANIFEST_SHA256,WEIGHT_REVISION
 from batchgen.gemm.w8a8_grouped_gemm_stage_1 import fused_fp8_moe_stage_1_tma
 assert torch.cuda.get_device_capability(0)==(10,3);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
 spec=importlib.util.spec_from_file_location('down_candidate',a.down_candidate);down_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(down_module)
 root=a.weights_root;entries=validate_manifest(a.repo/'scripts/r1_reproduction/r1_model_manifest.json');assert download_ready(root,entries[INDEX_NAME]);verify_source(root/INDEX_NAME,entries[INDEX_NAME]);index=json.loads((root/INDEX_NAME).read_bytes());selected={};tensors={}
 for eid in range(16):
  for part in ('gate_proj','up_proj','down_proj'):
   for suffix in ('weight','weight_scale_inv'):
    name=f'model.layers.3.mlp.experts.{eid}.{part}.{suffix}';selected.setdefault(index['weight_map'][name],[]).append(name)
 sources=[]
 for filename,names in selected.items():
  assert download_ready(root,entries[filename]);verify_source(root/filename,entries[filename]);sources.append({'filename':filename,'sha256':entries[filename]['Sha256'],'bytes':entries[filename]['Size'],'tensors':names})
  with safe_open(str(root/filename),framework='pt',device='cpu')as sf:
   for name in names:tensors[name]=sf.get_tensor(name).to('cuda')
 def weights(part,suffix):return [tensors[f'model.layers.3.mlp.experts.{eid}.{part}.{suffix}']for eid in range(16)]
 def ptr(t):return torch.tensor([v.data_ptr()for v in t],device='cuda',dtype=torch.int64)
 gate,up,down=[weights(part,'weight')for part in ('gate_proj','up_proj','down_proj')];gs,us,ds=[weights(part,'weight_scale_inv')for part in ('gate_proj','up_proj','down_proj')];gp,uptr,dp,gsp,usp,dsp=[ptr(x)for x in (gate,up,down,gs,us,ds)]
 torch.manual_seed(20261025);cases=[('uniform_'+str(n),[n]*16)for n in (1,8,16,32,64,65)]+[('skew',[0,1,65,0,129,2,0,32,0,7,0,3,64,0,16,0]),('single',[0]*7+[65]+[0]*8),('empty',[0]*16)]
 result={'schema':'r1-real-expert-GEMM-tile-ablation-v1','candidate_sha256':a.candidate_sha256,'weight_source':{'root':str(root),'layer':3,'experts':list(range(16)),'manifest_sha256':MANIFEST_SHA256,'revision':WEIGHT_REVISION,'shards':sources},'cases':[],'scope':'Tile-only and expert-axis ablations on real checkpoint expert weights with deterministic synthetic FP8 inputs/scales. Bitwise failures remain failures, never approximated away.'}
 result.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,device_capability=list(torch.cuda.get_device_capability(0)),device_name=torch.cuda.get_device_name(0),activation_source='Native R1 quantizer on BF16 normal activations; down input is quantized baseline real-weight gate/up output')
 for label,counts in cases:
  ids=[i for i,n in enumerate(counts)if n];ids=ids[::2]+ids[1::2];sizes=[counts[i]for i in ids];starts=[];M=0
  for n in sizes:starts.append(M);M+=n
  gm=torch.zeros(16,device='cuda',dtype=torch.int32);gi=gm.clone();start=gm.clone();active=torch.tensor(len(ids),device='cuda',dtype=torch.int32)
  if ids:
   gm[:len(ids)]=torch.tensor(sizes,device='cuda',dtype=torch.int32);gi[:len(ids)]=torch.tensor(ids,device='cuda',dtype=torch.int32);start[:len(ids)]=torch.tensor(starts,device='cuda',dtype=torch.int32)
  x,xs=act_quant(torch.randn((M,7168),device='cuda',dtype=torch.bfloat16))
  def first(tile):return fused_fp8_moe_stage_1_tma(x,xs,gate,gp,up,uptr,gs,gsp,us,usp,gm,gi,start,active,16,gate_gemm_block_size=tile)
  def last(tile,parallel):
   os.environ['BATCHGEN_R1_SM103_PARALLEL_MOE_DOWN']=str(int(parallel))
   return down_module.fused_dequant_grouped_gemm_fp8_tma(y,ys,down,dp,ds,dsp,gm,gi,start,active,gemm_block_size=tile)
  ref_first=first((64,32,128));y,ys=act_quant(ref_first)
  ref_last=last((64,32,128),False);torch.cuda.synchronize();row={'case':label,'counts':counts,'variants':[]}
  serial=[]
  for _ in range(3):
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();wall=time.perf_counter()
   for _ in range(20):last((64,32,128),False)
   end.record();end.synchronize();serial.append({'stream_ms':begin.elapsed_time(end)/20,'wall_ms':(time.perf_counter()-wall)*1000/20})
  row['down_serial_control']=serial
  for tile in [(64,32,128),(32,32,128),(16,32,128),(32,64,128),(16,64,128)]:
   for name,fn,reference in [('gate_up',lambda:first(tile),ref_first),('down_parallel',lambda:last(tile,True),ref_last)]:
    item={'stage':name,'tile':list(tile)}
    try:
     actual=fn();torch.cuda.synchronize();item['bitwise_equal']=bool(torch.equal(actual,reference));item['mismatched_elements']=int(torch.count_nonzero(actual!=reference));item['max_abs_difference']=float((actual.float()-reference.float()).abs().max())if M else 0.
     if item['bitwise_equal']:
      for _ in range(3):fn()
      torch.cuda.synchronize();times=[]
      for _ in range(3):
       begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();wall=time.perf_counter()
       for _ in range(20):fn()
       end.record();end.synchronize();times.append({'stream_ms':begin.elapsed_time(end)/20,'wall_ms':(time.perf_counter()-wall)*1000/20})
      item['times']=times;item['median_stream_ms']=statistics.median(t['stream_ms']for t in times);item['median_wall_ms']=statistics.median(t['wall_ms']for t in times)
    except Exception as error:
     # A compile rejection is recorded; do not select this configuration.
     item['status']='rejected';item['error']=str(error)[-1500:];torch.cuda.synchronize()
    row['variants'].append(item);print(json.dumps({'case':label,**{k:v for k,v in item.items()if k not in ['times','error']}}),flush=True)
  result['cases'].append(row)
 result['status']='all_tile_ablations_collected_NOT_blanket_pass';result['peak_allocated_bytes']=torch.cuda.max_memory_allocated();a.report.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
