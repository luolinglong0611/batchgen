"""CPU launch contracts; fused arithmetic is verified by the CUDA probe."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'batchgen/attention/mla/r1_prefill_rope.py'
spec=importlib.util.spec_from_file_location('r1_prefill_rope_cpu_contract',SOURCE)
backend=importlib.util.module_from_spec(spec)
before=set(sys.modules)
spec.loader.exec_module(backend)
new_imports=set(sys.modules)-before


class PrefillRoPEContracts(unittest.TestCase):
    def test_import_is_lazy_and_default_remains_disabled(self):
        self.assertFalse({'torch','triton'} & new_imports)
        with patch.dict(os.environ,{},clear=True):self.assertFalse(backend.enabled())
        with patch.dict(os.environ,{backend.ENV:'1'}):self.assertTrue(backend.enabled())
        with patch.dict(os.environ,{backend.ENV:'bad'}),self.assertRaises(ValueError):backend.enabled()

    def test_only_prepacked_function_selects_fusion_and_keeps_attention_call(self):
        source=ROOT/'batchgen/attention/mla/fa3_backend.py'
        module=ast.parse(source.read_text())
        owners=[]
        for fn in module.body:
            if isinstance(fn,ast.FunctionDef) and any(isinstance(node,ast.Attribute) and node.attr=='prepare_query_and_offload' for node in ast.walk(fn)):
                owners.append(fn)
        self.assertEqual([fn.name for fn in owners],['mla_prefill_flashattention3_w8a16_deepgemm_prepacked'])
        calls=[n for n in ast.walk(owners[0]) if isinstance(n,ast.Call)]
        # The explicit selector now assigns the legacy or CUTLASS callable to
        # one local alias; count its actual invocation instead of assuming a
        # direct legacy function name. Only this R1 prepacked path may opt in.
        selectors=[(fn,node) for fn in module.body if isinstance(fn,ast.FunctionDef)
                   for node in ast.walk(fn) if isinstance(node,ast.Assign)
                   and isinstance(node.value,ast.IfExp)
                   and isinstance(node.value.body,ast.Attribute)
                   and isinstance(node.value.body.value,ast.Name)
                   and node.value.body.value.id=='r1_blackwell_prefill'
                   and node.value.body.attr=='flash_attn_varlen_func']
        self.assertEqual([fn.name for fn,_ in selectors],[owners[0].name])
        selector=selectors[0][1]
        self.assertEqual(ast.dump(selector.value.test),ast.dump(ast.Name(id='_cutlass_prefill',ctx=ast.Load())))
        self.assertEqual(ast.dump(selector.value.orelse),ast.dump(ast.Name(id='flash_attn_varlen_func',ctx=ast.Load())))
        alias=selector.targets[0].id
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id==alias for n in calls),1)
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='materialize_key_value' for n in calls))
        # The legacy interleaved oracle remains the explicit disabled branch.
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='rotary_pos_emb_interleaved_native' for n in calls),2)

    def test_real_attention_tail_preserves_default_dispatch_call_and_host_kv(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch unavailable')
        package_name='r1_prefill_rope_dispatch_contract'
        package=ModuleType(package_name)
        package.__path__=[str(ROOT/'batchgen/attention/mla')]
        sys.modules[package_name]=package
        native_spec=importlib.util.spec_from_file_location(package_name+'.r1_blackwell_prefill',
            ROOT/'batchgen/attention/mla/r1_blackwell_prefill.py')
        selector_backend=importlib.util.module_from_spec(native_spec)
        sys.modules[native_spec.name]=selector_backend
        native_spec.loader.exec_module(selector_backend)

        source=ROOT/'batchgen/attention/mla/fa3_backend.py'
        fn=next(node for node in ast.parse(source.read_text()).body
                if isinstance(node,ast.FunctionDef)
                and node.name=='mla_prefill_flashattention3_w8a16_deepgemm_prepacked')
        start=next(i for i,node in enumerate(fn.body)
                   if isinstance(node,ast.Assign) and isinstance(node.value,ast.IfExp)
                   and isinstance(node.value.body,ast.Attribute)
                   and node.value.body.attr=='flash_attn_varlen_func')
        args=['self','query_states','key_states','value_states','cu_seqlens',
              'max_seqlen','total_tokens','offload_kv','_cutlass_prefill','_gemm','weight_scale']
        tail=ast.FunctionDef(name='actual_attention_tail',
            args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=name) for name in args],
                               kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=fn.body[start:],decorator_list=[])

        for setting,expected in ((None,'legacy'),('cutlass-sm100','cutlass')):
            with self.subTest(selector=setting):
                calls=[]
                attention_values=torch.full((3,2,128),0.25,dtype=torch.bfloat16)
                def attention(name):
                    def invoke(*args,**kwargs):
                        calls.append((name,args,kwargs))
                        return attention_values.clone()
                    return invoke
                namespace={'flash_attn_varlen_func':attention('legacy'),
                           'r1_blackwell_prefill':SimpleNamespace(flash_attn_varlen_func=attention('cutlass'))}
                code=ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[]))
                exec(compile(code,str(source),'exec'),namespace)
                q=torch.randn((3,2,192),dtype=torch.bfloat16)
                k=torch.randn_like(q)
                v=torch.randn((3,2,128),dtype=torch.bfloat16)
                cu=torch.tensor([0,3],dtype=torch.int32)
                host_kv=torch.arange(3*576,dtype=torch.float32).view(3,576)
                original=tuple(x.clone() for x in (q,k,v,host_kv))
                weight,scale=object(),object()
                owner=SimpleNamespace(num_heads=2,v_head_dim=128,softmax_scale=0.1352337788608801,
                    o_proj=SimpleNamespace(weight=SimpleNamespace(data=weight)))
                projection_calls=[]
                def gemm(w,s,x):
                    self.assertIs(w,weight);self.assertIs(s,scale)
                    projection_calls.append(x)
                    return x+1
                with patch.dict(os.environ,{},clear=True):
                    if setting is not None:os.environ[selector_backend.ENV]=setting
                    chosen=selector_backend.enabled()
                    self.assertEqual(chosen,expected=='cutlass')
                    result,returned_kv=namespace['actual_attention_tail'](
                        owner,q,k,v,cu,3,3,host_kv,chosen,gemm,{'o_proj.weight_scale_inv':scale})
                self.assertEqual(len(calls),1)
                name,positional,keywords=calls[0]
                self.assertEqual(name,expected)
                for actual,original_tensor in zip(positional,(q,k,v)):self.assertIs(actual,original_tensor)
                self.assertEqual(keywords,{'cu_seqlens_q':cu,'cu_seqlens_k':cu,
                    'max_seqlen_q':3,'max_seqlen_k':3,'softmax_scale':owner.softmax_scale,'causal':True})
                self.assertEqual(len(projection_calls),1)
                self.assertTrue(torch.equal(result,attention_values.view(3,256)+1))
                self.assertIs(returned_kv,host_kv)
                self.assertTrue(all(torch.equal(actual,before) for actual,before in zip((q,k,v,host_kv),original)))

    def test_invalid_cpu_inputs_fail_before_triton_is_imported(self):
        try:import torch
        except ImportError:self.skipTest('CPU Torch unavailable')
        q=torch.empty((3,2,192),dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'CUDA query'):
            backend.prepare_query_and_offload(q,None,None,None,None,None)
        with self.assertRaisesRegex(ValueError,'CUDA'):
            backend.materialize_key_value(torch.empty((3,2,256),dtype=torch.bfloat16),None)
        self.assertNotIn('r1_prefill_rope_kernels',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
