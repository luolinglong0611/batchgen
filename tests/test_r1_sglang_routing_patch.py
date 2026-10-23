import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/patch_sglang_r1_routing.py'
spec=importlib.util.spec_from_file_location('routing_patch',PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class R1RoutingPatch(unittest.TestCase):
    def config(self):
        return types.SimpleNamespace(model_type='deepseek_v3',architectures=['DeepseekV3ForCausalLM'],
            scoring_func='sigmoid',topk_method='noaux_tc',n_group=8,topk_group=4,
            n_routed_experts=256,num_experts_per_tok=8,hidden_size=7168,moe_intermediate_size=2048)
    def apply(self, value, enabled=True, config=None, backend=True):
        layer=types.SimpleNamespace(experts=types.SimpleNamespace(routing_method_type=value))
        fake=types.ModuleType('sglang.srt.layers.moe.utils')
        fake.RoutingMethodType=types.SimpleNamespace(DeepSeekV3=2)
        import textwrap
        code=compile(textwrap.dedent(mod.BLOCK),'actual-inserted-block','exec')
        with patch.dict(os.environ,{mod.ENV_NAME:'1' if enabled else '0'}),patch.dict(sys.modules,{'sglang.srt.layers.moe.utils':fake}):
            exec(code,{'os':os,'config':config or self.config(),'self':layer,
                       'get_moe_runner_backend':lambda:types.SimpleNamespace(is_flashinfer_trtllm=lambda:backend)})
        return layer.experts.routing_method_type
    def test_none_is_set_only_with_opt_in(self):
        self.assertEqual(self.apply(None),2)
        self.assertIsNone(self.apply(None,enabled=False))
    def test_non_none_zero_and_other_enums_preserved(self):
        for value in (0,1,2,3,4,5,6):self.assertEqual(self.apply(value),value)
    def test_other_models_and_backend_unchanged(self):
        for field,value in [('model_type','qwen3_moe'),('architectures',['Other']),('scoring_func','softmax'),
                            ('topk_method','greedy'),('n_group',1),('topk_group',1),('n_routed_experts',128),
                            ('num_experts_per_tok',4),('hidden_size',2048),('moe_intermediate_size',4096)]:
            config=self.config();setattr(config,field,value)
            with self.subTest(field=field):self.assertIsNone(self.apply(None,config=config))
        self.assertIsNone(self.apply(None,backend=False))
    def test_unknown_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):mod.patched_source(b'unknown')
    def test_insertion_preserves_source_and_is_after_experts_before_topk(self):
        source=('class Test:\n    def __init__(self):\n        self.experts = factory(\n'+mod.ANCHOR+'\n            x=1,\n        )\n').encode()
        with patch.object(mod,'STOCK_SHA',mod.sha(source)):
            patched=mod.patched_source(source)
        self.assertEqual(patched.decode().replace(mod.BLOCK,''),source.decode())
        ast.parse(patched)
        self.assertLess(patched.index(b'self.experts ='),patched.index(b'BATCHGEN_SGLANG'))
        self.assertLess(patched.index(b'BATCHGEN_SGLANG'),patched.index(b'self.topk ='))
if __name__=='__main__':unittest.main()
