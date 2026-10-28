"""CPU launch contracts; fused arithmetic is verified by the CUDA probe."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
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
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='flash_attn_varlen_func' for n in calls),1)
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='materialize_key_value' for n in calls))
        # The legacy interleaved oracle remains the explicit disabled branch.
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='rotary_pos_emb_interleaved_native' for n in calls),2)

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
