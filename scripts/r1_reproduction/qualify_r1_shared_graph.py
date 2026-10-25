"""Real shared-expert graph parity, dynamic sources/streams and paired timings."""
from pathlib import Path
import argparse,json,hashlib,sys,torch,statistics,time,importlib.util,types

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--candidate-sha256',required=True);p.add_argument('--weights-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.report.exists();assert hashlib.sha256(a.candidate.read_bytes()).hexdigest()==a.candidate_sha256
 sys.path[:0]=[str(a.repo),str(a.repo/'scripts/r1_reproduction')]
 from safetensors import safe_open
 from convert_verified_shards import validate_manifest,download_ready,verify_source,INDEX_NAME,MANIFEST_SHA256,WEIGHT_REVISION
 from batchgen.models.deepseek.deepseekv3.modeling_deepseek_v3 import DeepseekV3MLP
 from batchgen.models.deepseek.deepseekv3.wrappers import DeepSeekExpertWrapper
 assert torch.cuda.get_device_capability(0)==(10,3);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
 spec=importlib.util.spec_from_file_location('shared_candidate',a.candidate);candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
 root=a.weights_root;entries=validate_manifest(a.repo/'scripts/r1_reproduction/r1_model_manifest.json');assert download_ready(root,entries[INDEX_NAME]);verify_source(root/INDEX_NAME,entries[INDEX_NAME]);index=json.loads((root/INDEX_NAME).read_bytes());names=[f'model.layers.3.mlp.shared_experts.{part}.{suffix}'for part in ['gate_proj','up_proj','down_proj']for suffix in ['weight','weight_scale_inv']];selected={}
 for name in names:selected.setdefault(index['weight_map'][name],[]).append(name)
 tensors={};shards=[]
 for filename,names in selected.items():
  assert download_ready(root,entries[filename]);verify_source(root/filename,entries[filename]);shards.append({'filename':filename,'sha256':entries[filename]['Sha256'],'bytes':entries[filename]['Size'],'tensors':names})
  with safe_open(str(root/filename),framework='pt',device='cpu')as sf:
   for name in names:tensors[name.split('shared_experts.')[1]]=sf.get_tensor(name).to('cuda')
 config=types.SimpleNamespace(hidden_size=7168,intermediate_size=2048,hidden_act='silu',batchgen_skeleton=True)
 model=DeepseekV3MLP(config)
 for name in ['gate_proj','up_proj','down_proj']:getattr(model,name).weight=torch.nn.Parameter(tensors[name+'.weight'],requires_grad=False)
 scales={k:v for k,v in tensors.items()if k.endswith('weight_scale_inv')}
 engine=types.SimpleNamespace(Module_Batching_Config=types.SimpleNamespace(expert_decoding_batch_size_upper_bound=128),Basic_Config=types.SimpleNamespace(device_torch=torch.device('cuda',0)))
 wrapper=DeepSeekExpertWrapper(model,3,-1,None,engine,config,True,scales);wrapper.phase='decode';wrapper._register_fp8_weights();original_down=wrapper.fp8_down.clone()
 # Match the production forward's cached-FP8 binding before invoking the helper.
 def bind():
  for name,weight in [('gate_proj',wrapper.fp8_gate),('up_proj',wrapper.fp8_up),('down_proj',wrapper.fp8_down)]:getattr(model,name).weight.data=weight
 def eager(x):bind();return wrapper.micro_batch_forward(x,'expert')
 def graph(x):bind();return candidate.forward(wrapper,x)
 result={'schema':'r1-real-shared-expert-graph-qualification-v1','candidate_sha256':a.candidate_sha256,'weight_source':{'root':str(root),'layer':3,'manifest_sha256':MANIFEST_SHA256,'revision':WEIGHT_REVISION,'shards':shards},'cases':[],'scope':'Unchanged shared expert math/quantization on checkpoint weights, deterministic random BF16 inputs. Graph parity is not full-model quality.'}
 result.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,device_capability=list(torch.cuda.get_device_capability(0)),device_name=torch.cuda.get_device_name(0))
 torch.manual_seed(20261025)
 with torch.inference_mode():
  for batch in (0,1,16,32,17,129):
   x=torch.randn((batch,7168),device='cuda',dtype=torch.bfloat16);expected=eager(x);actual=graph(x);torch.cuda.synchronize();assert torch.equal(expected,actual),(batch,'initial')
   retained=actual;snapshot=actual.clone()
   for change in range(3):
    x.normal_();expected=eager(x);actual=graph(x);torch.cuda.synchronize();assert torch.equal(expected,actual),(batch,'dynamic',change);assert torch.equal(retained,snapshot),(batch,'retained_output_overwritten')
   for _ in range(3):eager(x);graph(x)
   torch.cuda.synchronize();times={'eager':[],'graph':[]}
   for name in ('eager','graph','graph','eager'):
    fn=eager if name=='eager'else graph;begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();wall=time.perf_counter()
    for _ in range(30):fn(x)
    end.record();end.synchronize();times[name].append({'stream_ms':begin.elapsed_time(end)/30,'wall_ms':(time.perf_counter()-wall)*1000/30})
   result['cases'].append({'batch':batch,'bitwise_equal':True,'dynamic_inputs':True,'retained_output_unchanged':True,'timings':times,'wall_speedup':statistics.median(r['wall_ms']for r in times['eager'])/statistics.median(r['wall_ms']for r in times['graph'])})
  x=torch.randn((32,7168),device='cuda',dtype=torch.bfloat16);graph(x);before=wrapper._r1_shared_graph_captures
  # In-place scale mutation, in-place module-parameter mutation, and source
  # replacement must invalidate the graph even when addresses are reused.
  for mutation in ['scale_inplace','parameter_inplace','cached_weight_replacement']:
   if mutation=='scale_inplace':scales['down_proj.weight_scale_inv'].mul_(1.0001)
   elif mutation=='parameter_inplace':model.down_proj.weight.copy_(torch.zeros_like(model.down_proj.weight))
   else:
    with torch.inference_mode(False):wrapper.fp8_down=original_down.clone()
   expected=eager(x);actual=graph(x);torch.cuda.synchronize();assert torch.equal(expected,actual),mutation;assert wrapper._r1_shared_graph_captures>before,mutation;before=wrapper._r1_shared_graph_captures
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   expected=eager(x);actual=graph(x)
  stream.synchronize();assert torch.equal(expected,actual)
  assert len(wrapper._r1_shared_graphs)==2
  before=wrapper._r1_shared_graph_captures
  scales['down_proj.weight_scale_inv']=scales['down_proj.weight_scale_inv'].clone()
  expected=eager(x);actual=graph(x);torch.cuda.synchronize();assert torch.equal(expected,actual)
  assert wrapper._r1_shared_graph_captures==before, 'Unversioned scale must use eager fallback'
 result['source_invalidation']='passed';result['per_stream_isolation']='passed';result['status']='paired_bitwise_parity_ABBA_and_invalidation_complete';result['peak_allocated_bytes']=torch.cuda.max_memory_allocated();a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'speedups':{c['batch']:c['wall_speedup']for c in result['cases']}}))
if __name__=='__main__':main()
