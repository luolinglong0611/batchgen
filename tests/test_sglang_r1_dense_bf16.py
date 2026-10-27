"""CPU-only contracts; actual BF16 numerical qualification is a separate GPU job."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load():
    spec = importlib.util.spec_from_file_location('dense_candidate_test', ROOT / 'scripts/r1_reproduction/sglang_r1_dense_bf16.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def provenance(m):
    return {'schema':'r1-dense-fp8-provenance-v1','model_manifest_sha256':m.MODEL_MANIFEST_SHA,
            'model_weight_identity_sha256':m.WEIGHT_IDENTITY,'tensor_records':{
                n:{'dtype':d,'shape':s,'raw_sha256':'a'*64} for n,(d,s) in m.expected_tensors().items()}}

def args(**updates):
    v={'hidden_size':7168,'intermediate_size':18432,'hidden_act':'silu',
       'tp_size':1,'tp_rank':0,'prefix':'model.layers.0.mlp'};v.update(updates);return v

class DenseContracts(unittest.TestCase):
    def setUp(self): self.m=load()

    def test_disabled_import_and_install_are_side_effect_free(self):
        with patch.dict(os.environ,{self.m.ENV:'0'}),patch.object(self.m.importlib,'import_module',side_effect=AssertionError('no Torch/SG')):
            self.assertFalse(self.m.enabled());r=self.m.install()
        self.assertEqual(r['status'],'unverified');self.assertEqual(r['modules'],[])
        self.assertFalse(r['formal_test_permitted'])
        with patch.dict(os.environ,{self.m.ENV:'true'}),self.assertRaises(ValueError):self.m.enabled()

    def test_dense_roles_are_explicit_and_shared_excluded(self):
        for i in range(3):self.assertEqual(self.m.dense_layer_id(args(prefix=f'model.layers.{i}.mlp')),i)
        self.assertIsNone(self.m.dense_layer_id(args(intermediate_size=2048,prefix='model.layers.3.mlp.shared_experts')))
        for updates in ({'prefix':''},{'prefix':'model.layers.3.mlp'},{'prefix':'other.layers.0.mlp'},
                        {'tp_size':16},{'tp_size':True},{'tp_rank':None},{'hidden_size':8192},{'hidden_act':'gelu'}):
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.m.dense_layer_id(args(**updates))

    def test_provenance_requires_all18_original_tensors_and_exact_identity(self):
        good=provenance(self.m);self.assertEqual(len(self.m.validate_provenance(good)['tensor_records']),18)
        for key in ('model_manifest_sha256','model_weight_identity_sha256','schema'):
            bad=json.loads(json.dumps(good));bad.pop(key)
            with self.assertRaises(ValueError):self.m.validate_provenance(bad)
        bad=json.loads(json.dumps(good));bad['tensor_records'].pop(next(iter(bad['tensor_records'])))
        with self.assertRaises(ValueError):self.m.validate_provenance(bad)
        for field,value in [('dtype','bfloat16'),('shape',[1,2]),('raw_sha256','planned')]:
            bad=json.loads(json.dumps(good));bad['tensor_records'][next(iter(bad['tensor_records']))][field]=value
            with self.assertRaises(ValueError):self.m.validate_provenance(bad)

    def test_provenance_file_is_hash_bound_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proof.json';p.write_text(json.dumps(provenance(self.m)));digest=hashlib.sha256(p.read_bytes()).hexdigest()
            env={self.m.PROVENANCE_ENV:str(p),self.m.PROVENANCE_SHA_ENV:digest}
            with patch.dict(os.environ,env),patch.object(self.m.importlib,'import_module',side_effect=AssertionError('no GPU')):
                self.assertEqual(self.m.load_provenance()[1]['sha256'],digest)
                p.write_text(p.read_text()+' ')
                with self.assertRaisesRegex(RuntimeError,'changed'):self.m.load_provenance()

    def test_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'source mismatch'):self.m._source(load,'0'*64)

    def test_installed_but_uncalled_stays_unverified_and_graph_guard_atomic(self):
        class MLP:
            def __init__(self,hidden_size,intermediate_size,hidden_act,prefix='',tp_rank=None,tp_size=None):self.original=True
            def forward(self,x,*args):return 'original'
        before=(MLP.__init__,MLP.forward)
        sg=SimpleNamespace(DeepseekV2MLP=MLP,get_global_server_args=lambda:SimpleNamespace(disable_cuda_graph=False,enable_piecewise_cuda_graph=False))
        fp8=SimpleNamespace(Fp8LinearMethod=object)
        with patch.dict(os.environ,{self.m.ENV:'1'}),patch.object(self.m,'load_provenance',return_value=(provenance(self.m),{})),patch.object(self.m,'_source',return_value={}),patch.object(self.m.importlib,'import_module',side_effect=lambda n:sg if n.endswith('deepseek_v2') else fp8):
            with self.assertRaisesRegex(RuntimeError,'graphs OFF'):self.m.install()
            self.assertEqual((MLP.__init__,MLP.forward),before)
            sg.get_global_server_args=lambda:SimpleNamespace(disable_cuda_graph=True,enable_piecewise_cuda_graph=False)
            self.m.install();self.assertEqual(self.m.dense_backend_receipt()['status'],'unverified')
            shared=MLP(**args(intermediate_size=2048,prefix='model.layers.3.mlp.shared_experts'))
            self.assertEqual(shared.forward('x'),'original')
            dense=MLP(**args());self.assertEqual(dense._r1_dense_bf16_layer_id,0)
            record={'completed_forward_calls':0}
            with patch.object(self.m,'align_dense_mlp',return_value=record),patch.object(self.m,'ordered_dense_forward',return_value='bf16'),patch.object(self.m,'_metadata',return_value={}):
                self.assertEqual(dense.forward('x'),'bf16');self.assertEqual(record['completed_forward_calls'],1)
                with self.assertRaisesRegex(RuntimeError,'fusion'):dense.forward('x',should_allreduce_fusion=True)
            self.m.rollback_unexecuted_installation();self.assertEqual((MLP.__init__,MLP.forward),before)

    def test_receipt_is_defensive_copy_and_not_quality_pass(self):
        self.m._INSTALLATION={'status':'installed'};receipt=self.m.dense_backend_receipt();receipt['installation']['status']='forged'
        self.assertEqual(self.m._INSTALLATION['status'],'installed');self.assertEqual(receipt['actual_converted_layer_ids'],[])

    def test_old_fp8_method_cannot_consume_derived_bf16(self):
        reject=self.m._RejectFp8AfterDenseDerivation()
        with self.assertRaisesRegex(RuntimeError,'FP8 bypass'):reject.apply(None)
        with self.assertRaisesRegex(RuntimeError,'complete native FP8'):reject.process_weights_after_loading(None)

    def test_production_math_evaluation_order_has_bf16_materializations(self):
        events=[];device=object()
        class T:
            dtype='bf16';shape=(2,7168)
            def __init__(self,name):self.name=name;self.device=device
            def __getitem__(self,key):return T(self.name+('gate' if key.stop==18432 else 'up'))
            def __mul__(self,other):events.append('mul');return T('intermediate')
        x=T('x');gu=T('gu');dw=T('down')
        def linear(x,w):events.append('linear:'+w.name);return T('output')
        def silu(x):events.append('silu');return T('silu')
        torch=SimpleNamespace(bfloat16='bf16');functional=SimpleNamespace(linear=linear,silu=silu)
        module=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=gu),down_proj=SimpleNamespace(weight=dw))
        with patch.object(self.m.importlib,'import_module',side_effect=lambda n:torch if n=='torch' else functional):
            self.m.ordered_dense_forward(module,x)
            self.assertEqual(events,['linear:gugate','silu','linear:guup','mul','linear:down'])
            x.dtype='fp8'
            with self.assertRaises(ValueError):self.m.ordered_dense_forward(module,x)

    def test_runtime_has_optin_and_transaction_rollback_hooks(self):
        text=(ROOT/'scripts/r1_reproduction/sglang_r1_alignment_runtime.py').read_text()
        self.assertIn("os.environ.get('SGLANG_R1_DENSE_BF16', '0')",text)
        self.assertIn('dense.rollback_unexecuted_installation()',text)
        self.assertIn('dense_backend_receipt()',text)


    def test_actual_original_value_mismatch_and_dtype_fail_before_conversion(self):
        m=self.m;proof=provenance(m);name=next(iter(proof['tensor_records']))
        class Tensor:
            dtype='torch.float8_e4m3fn';shape=(18432,7168);device=SimpleNamespace(type='cuda')
            def is_contiguous(self):return True
        t=Tensor()
        with patch.object(m,'tensor_raw_sha256',return_value='b'*64),self.assertRaisesRegex(ValueError,'values differ'):
            m._verify(t,name,proof,object())
        t.dtype='torch.bfloat16'
        with patch.object(m,'tensor_raw_sha256',side_effect=AssertionError('hash must not run')),self.assertRaisesRegex(ValueError,'contract differs'):
            m._verify(t,name,proof,object())

    def test_native_conversion_is_atomic_and_shared_scale_formula_is_source_bound(self):
        import contextlib
        m=self.m;device=SimpleNamespace(type='cuda');old_method=object()
        class Tensor:
            def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype;self.device=device
            def __getitem__(self,key):
                first=key.start or 0;last=key.stop if key.stop is not None else self.shape[0]
                return Tensor((last-first,self.shape[1]),self.dtype)
            def stride(self):return (self.shape[1],1)
        class Linear:
            tp_size=1;bias=None
            def __init__(self,shape,scales):self.weight=Tensor(shape,'torch.float8_e4m3fn');self.weight_scale_inv=Tensor(scales,'torch.float32');self.quant_method=old_method
            def __setattr__(self,name,value):
                if name=='weight' and getattr(self,'reject_next_bf16',False) and value.dtype=='torch.bfloat16':
                    self.reject_next_bf16=False;raise RuntimeError('injected assignment failure')
                object.__setattr__(self,name,value)
        class Module:
            tp_size=1
            def __init__(self):self.gate_up_proj=Linear((36864,7168),(288,56));self.down_proj=Linear((7168,18432),(56,144))
        torch=SimpleNamespace(bfloat16='torch.bfloat16',cuda=SimpleNamespace(get_device_capability=lambda d:(10,3),is_current_stream_capturing=lambda:False),
            inference_mode=contextlib.nullcontext,cat=lambda ts,dim:Tensor((sum(t.shape[0] for t in ts),ts[0].shape[1]),'torch.bfloat16'),
            nn=SimpleNamespace(Parameter=lambda t,requires_grad:t))
        dequant=SimpleNamespace(deepseek_v3_dequantization=lambda w,s:Tensor(w.shape,'torch.bfloat16'))
        module=Module();old=(module.gate_up_proj.weight,module.down_proj.weight);module.down_proj.reject_next_bf16=True
        patches=[patch.object(m.importlib,'import_module',side_effect=lambda n:torch if n=='torch' else dequant),
                 patch.object(m,'_verify',side_effect=lambda t,n,p,th:{'name':n,'raw_values_verified_against_original':True}),
                 patch.object(m,'_source',return_value={'sha256':m.BG_DEQUANT_SHA})]
        with patches[0],patches[1],patches[2]:
            with self.assertRaisesRegex(RuntimeError,'assignment failure'):m.align_dense_mlp(module,0,provenance(m),{})
            self.assertIs(module.gate_up_proj.weight,old[0]);self.assertIs(module.down_proj.weight,old[1]);self.assertEqual(m.dense_backend_receipt()['modules'],[])
            record=m.align_dense_mlp(module,0,provenance(m),{'sha256':'a'*64})
            self.assertEqual(len(record['original_tensor_proofs']),6);self.assertEqual(record['completed_forward_calls'],0)
            self.assertEqual(record['dequant_formula'],m.FORMULA);self.assertFalse(m.dense_backend_receipt()['formal_test_permitted'])
            with self.assertRaisesRegex(RuntimeError,'after model weights'):m.rollback_unexecuted_installation()

if __name__=='__main__':unittest.main()
