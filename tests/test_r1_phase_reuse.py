"""Single resident R1 model phase/lifetime contracts on real CPU tensors.

The actual MoE phase dispatcher and resident prefill forward execute; CUDA/EP
boundaries are replaced. Whole-model CUDA/Host lifecycle still needs GPU runs.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PSM = ROOT/'batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py'
WORKER = ROOT/'batchgen/batchgen_worker.py'


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


phase = load('r1_phase_cpu_contract',PSM.with_name('r1_phase_reuse.py'))
resident = load('r1_phase_resident_cpu_contract',PSM.with_name('resident_prefill.py'))


def isolated(path, classname, names, **globals):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)and n.name==classname)
    body=[copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name))for name in names]
    tree=ast.Module(body=[ast.ClassDef(name='Candidate',bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[])
    ns=dict(globals);exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns);return ns['Candidate']


class Entry:
    def __init__(self):
        self.weight=torch.ones(2,2);self.scale=torch.ones(2,1,dtype=torch.int32);self.fresh=True
    def valid(self):return self.fresh


class MiniMoE(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared_experts=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():self.shared_experts.weight.copy_(2*torch.eye(2))
        self.prefill_calls=self.decode_calls=0
        def prefill(x):self.prefill_calls+=1;return 3*x
        def decode(x):self.decode_calls+=1;return 5*x
        self._r1_resident_prefill_adapter=prefill;self.r1_deepep_adapter=decode
    def forward(self,x):return self.r1_deepep_adapter(x)+self.shared_experts(x)


def model():
    layers=[NS(self_attn=NS(module=NS(_r1_native_gemm_weights=[Entry()])),mlp=NS())for _ in range(3)]
    layers += [NS(self_attn=NS(module=NS(_r1_native_gemm_weights=[Entry()])),mlp=MiniMoE())for _ in range(2)]
    return NS(config=NS(phase='decode'),model=NS(config=NS(phase='decode'),layers=layers))


def state(m=None):
    return phase.R1SharedPhaseState(m or model(),first_moe_layer=3,capacity=192,generation=7,
                                    prefill_impl=resident.resident_moe_forward)


def manager(monkeypatch, m=None):
    monkeypatch.setenv(phase.ENVIRONMENT,'1');monkeypatch.setenv('BATCHGEN_R1_ORDINARY_GEMM','deep_gemm')
    cls=isolated(PSM,'DeepseekV3ParallelStrategyManager',[
        'configure_prefill','configure_decoding','r1_phase_model_available','_activate_r1_phase_model',
        'release_r1_phase_model','set_comm'],r1_phase_reuse=phase,os=os,logging=NS(info=lambda*a,**k:None))
    obj=cls();obj.world_size=8;obj.rank=0
    obj.engine_config=NS(EP_Config=NS(enable_offloading=False),Basic_Config=NS(attn_mode=3),
                         Module_Batching_Config=NS(r1_deepep_ll=True))
    obj._r1_native_moe=lambda:True;obj.prefill_uses_resident_ep=lambda:True
    obj.model=m or model();obj.loaded_model_config=NS(phase='decode')
    obj._r1_shared_phase_state=state(obj.model);obj._r1_shared_phase_state.activate('decode');obj._r1_phase_model_builds=7
    obj.pool_calls=[];obj._r1_deepep_pool=NS(begin_phase=lambda:obj.pool_calls.append('begin'))
    return obj


def test_one_model_four_transitions_keep_tensors_and_native_banks(monkeypatch):
    obj=manager(monkeypatch);m=obj.model
    entries=[l.self_attn.module._r1_native_gemm_weights[0]for l in m.model.layers]
    identities=[(id(e),e.weight.data_ptr(),e.scale.data_ptr(),e.weight._version,e.scale._version)for e in entries]
    x=torch.ones(4,1,2)
    for target in ('prefill','decode','prefill','decode'):
        actual,copy_tasks=(obj.configure_prefill()if target=='prefill'else obj.configure_decoding(padding_bsz=180,comm=object()))
        assert actual is m and copy_tasks=={'attn':[],'routed_expert':[],'shared_expert':[]}
        assert m.config.phase==m.model.config.phase==target
        assert m._r1_weight_generation==7 and m._r1_weight_storage_immutable
        for layer in m.model.layers[3:]:
            expected=5*x if target=='prefill'else 7*x
            torch.testing.assert_close(layer.mlp(x),expected)
    assert [(id(e),e.weight.data_ptr(),e.scale.data_ptr(),e.weight._version,e.scale._version)for e in entries]==identities
    assert obj.pool_calls==['begin','begin']
    assert all(l.mlp.prefill_calls==2 and l.mlp.decode_calls==2 for l in m.model.layers[3:])


def test_first_prefill_delegates_to_one_decode_capable_build(monkeypatch):
    obj=manager(monkeypatch);obj._r1_shared_phase_state=None;calls=[];m=obj.model
    def build(**kwargs):
        calls.append(kwargs);obj._r1_shared_phase_state=state(m);return m,{}
    obj.configure_decoding=build;obj.set_comm('worker-comm')
    assert obj.configure_prefill()[0]is m
    assert obj.configure_prefill()[0]is m
    assert len(calls)==1 and calls[0]['comm']=='worker-comm'


def test_capacity_and_stale_sources_fail_before_forward(monkeypatch):
    obj=manager(monkeypatch);obj.configure_prefill()
    with pytest.raises(ValueError,match='capacity'):obj.configure_decoding(padding_bsz=193)
    assert obj._r1_shared_phase_state.phase=='prefill'
    obj.model.model.layers[0].self_attn.module._r1_native_gemm_weights[0].fresh=False
    with pytest.raises(RuntimeError,match='stale'):obj.configure_decoding(padding_bsz=180)
    assert obj._r1_shared_phase_state.phase=='prefill'


def test_release_drops_only_cache_and_revokes_immutable_marker(monkeypatch):
    obj=manager(monkeypatch);m=obj.model;s=obj._r1_shared_phase_state
    obj.release_r1_phase_model()
    assert obj.model is None and obj._r1_shared_phase_state is None and not m._r1_weight_storage_immutable
    assert s.model is None and s.layers==()
    with pytest.raises(RuntimeError,match='Released'):s.activate('decode')


def test_profile_gate_default_and_incompatible_modes(monkeypatch):
    obj=manager(monkeypatch)
    monkeypatch.delenv(phase.ENVIRONMENT);assert not phase.enabled_for(NS())
    monkeypatch.setenv(phase.ENVIRONMENT,'1')
    obj.world_size=16
    with pytest.raises(ValueError,match='single EP8'):phase.enabled_for(obj)
    obj.world_size=8;obj.persistent_phase_instances=True
    with pytest.raises(ValueError,match='dual persistent'):phase.enabled_for(obj)


def test_missing_adapter_does_not_partially_replace_moe_forward():
    m=model();moe=m.model.layers[-1].mlp;del moe.r1_deepep_adapter
    first=m.model.layers[3].mlp
    with pytest.raises(RuntimeError,match='both native'):state(m)
    assert first.forward.__func__ is MiniMoE.forward


def test_worker_handoff_drains_host_and_graph_before_kv_release(monkeypatch):
    calls=[];m=model();m._r1_weight_generation=7
    fake_graph=ModuleType('batchgen.attention.mla.r1_decode_cuda_graph')
    fake_graph.release_attention_graph=lambda w:calls.append('attention_graph')
    monkeypatch.setitem(sys.modules,fake_graph.__name__,fake_graph)
    cls=isolated(WORKER,'BatchGenWorker',['_r1_phase_model_reuse_active','_retain_r1_phase_model_for_switch',
                    '_finish_decode_interval','_unregister_fp8_weights'],
        torch=NS(cuda=NS(synchronize=lambda d:calls.append('GPU_drain'))),
        AttnWrapperBase=NS(retire_pending_prefill_offloads=lambda**k:calls.append('prefill_Host_drain')),
        logging=NS(info=lambda*a,**k:None))
    obj=cls();obj.model=m;obj.rank=0;obj.torch_device='cpu';obj.args=NS(persistent_phase_instances=False)
    obj.parallel_manager=NS(model=m,r1_phase_model_available=lambda:True)
    obj._pending_kv_append_tasks=['pending'];obj._wait_pending_kv_append_tasks=lambda:calls.append('decode_Host_drain')
    obj._r1_whole_decode_state=NS(record=object(),invalidate=lambda**k:calls.append('whole_graph'))
    assert obj._retain_r1_phase_model_for_switch()
    assert calls==['decode_Host_drain','prefill_Host_drain','whole_graph']+['attention_graph']*5+['GPU_drain']
    # The ordinary banks and registered aliases survive interval/prefill cleanup.
    obj._persistent_phase_enabled=lambda:False;obj._decode_instance_is_reusable=lambda:False
    obj.deep_free_model_memory=lambda:pytest.fail('Resident interval freed model')
    obj._finish_decode_interval();obj._unregister_fp8_weights()
    assert obj.model is m and len(m.model.layers[0].self_attn.module._r1_native_gemm_weights)==1


def test_true_worker_teardown_releases_shared_owner_after_old_graph(monkeypatch):
    pm=manager(monkeypatch);m=pm.model;events=[]
    release=pm.release_r1_phase_model
    def tracked_release():events.append('release_shared_owner');release()
    pm.release_r1_phase_model=tracked_release
    cls=isolated(WORKER,'BatchGenWorker',['deep_free_model_memory'],
        torch=NS(cuda=NS(is_available=lambda:False)),
        time=NS(perf_counter=lambda:0.0),sys=NS(modules={}),gc=NS(collect=lambda:None),
        logging=NS(info=lambda*a,**k:None))
    obj=cls();obj.model=m;obj.parallel_manager=pm;obj.rank=0;obj._cuda_graph_adapter=None
    obj._r1_whole_decode_state=NS(record=object(),invalidate=lambda**k:events.append('old_graph_invalidated'))
    obj._log_phase_memory_ledger=lambda stage:events.append(stage)
    obj.deep_free_model_memory()
    assert obj.model is None and pm.model is None and pm._r1_shared_phase_state is None
    assert not m._r1_weight_storage_immutable
    assert events[:2]==['old_graph_invalidated','release_shared_owner']
