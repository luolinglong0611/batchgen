"""CPU/stdlib contracts and mocked-HTTP collection; never load a model."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/full_model_correctness.py'
spec = importlib.util.spec_from_file_location('r1_full_correctness', SOURCE)
tool = importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture():
    return {'schema': 'r1-fixed-correctness-v1', 'conditions': copy.deepcopy(tool.CONDITIONS),
            'generation': {'max_new_tokens':128,'temperature':0.0,'top_p':1.0,'ignore_eos':False,'repeats':2,'request_concurrency':1},
            'tokenizer_files': {'tokenizer.json': {'sha256':'a'*64}, 'tokenizer_config.json':{'sha256':'b'*64}},
            'cases': [{'id':n,'user_text':u,'teacher_text':t,'prompt':u,'input_ids':[0,11,12], 'teacher_ids':[20,21]}
                      for n,u,t in tool.FIXED_CASES]}


def deployment(engine, fixed):
    return {**copy.deepcopy(tool.CONDITIONS), 'backend':engine,
            'server_revision':tool.SGLANG_COMMIT if engine=='sglang' else 'local-commit', 'model_id':'DeepSeek-R1',
            'attention_backend':'audited-eager', 'activation_quantization':'per-token-block128',
            'launch_command_sha256':'c'*64,'weight_verification_evidence':'verification.json',
            'tp':8,'ep':8,'nodes':1,'gpu_count':8,
            'tokenizer_sha256':{n:v['sha256'] for n,v in fixed['tokenizer_files'].items()}}


class ProtocolTests(unittest.TestCase):
    def test_fixed_inputs_and_precision_cannot_silently_change(self):
        tool.validate_fixture(fixture())
        for mutate in (lambda f:f['conditions'].update(kv_dtype='fp8'),
                       lambda f:f['generation'].update(temperature=.1),
                       lambda f:f['cases'].reverse()):
            f=fixture();mutate(f)
            with self.assertRaises(ValueError):tool.validate_fixture(f)

    def test_deployment_requires_matching_assets_and_disabled_optimizations(self):
        f=fixture();d=deployment('sglang',f);tool.validate_deployment(d,'sglang',f)
        for key,value in [('prefix_cache',True),('protocol_seed',0),('server_revision','different'),('launch_command_sha256','unknown')]:
            bad=copy.deepcopy(d);bad[key]=value
            with self.assertRaises(ValueError):tool.validate_deployment(bad,'sglang',f)

    def test_greedy_payload_and_teacher_ids_are_fixed(self):
        case=fixture()['cases'][0]
        generation=tool.sglang_payload(case);teacher=tool.sglang_payload(case,True)
        self.assertEqual(generation['sampling_params']['sampling_seed'],20261023)
        self.assertEqual(generation['sampling_params']['temperature'],0)
        self.assertEqual(teacher['input_ids'],[0,11,12,20,21])
        self.assertEqual(teacher['sampling_params']['max_new_tokens'],0)
        self.assertEqual(teacher['logprob_start_len'],0)

    def test_teacher_alignment_and_nll(self):
        case=fixture()['cases'][0]
        for rows in ([[None,0],[-1,11],[-2,12],[-3,20],[-5,21]],
                     [[-1,11],[-2,12],[-3,20],[-5,21]]):
            out=tool.teacher_metrics({'meta_info':{'input_token_logprobs':rows}},case)
            self.assertEqual(out['mean_negative_log_likelihood'],4)
            self.assertEqual(out['teacher_token_logprobs'],[-3,-5])
            self.assertFalse(out['cross_engine_comparison_available'])

    def test_teacher_bad_alignment_and_nonfinite_fail(self):
        case=fixture()['cases'][0]
        for rows in ([[-1,11],[-2,12],[-3,21],[-5,20]],
                     [[-1,11],[-2,12],[float('nan'),20],[-5,21]],
                     [[-3,20],[-5,21]]):
            with self.assertRaises(ValueError):tool.teacher_metrics({'meta_info':{'input_token_logprobs':rows}},case)
        self.assertEqual(tool.teacher_metrics({},case)['status'],'unavailable')

    def test_token_metrics_do_not_confuse_prefix_with_equality(self):
        self.assertEqual(tool.sequence_metrics([1,2],[1,2])['first_divergence_index'],None)
        self.assertEqual(tool.sequence_metrics([1,2],[1,2,3])['first_divergence_index'],2)
        self.assertEqual(tool.sequence_metrics([1,2],[1,3])['edit_distance'],1)
        self.assertEqual(tool.sequence_metrics([],[1])['edit_distance'],1)

    def test_reports_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json';tool.save_new(p,{})
            with self.assertRaises(FileExistsError):tool.save_new(p,{'changed':True})

    def test_mocked_collection_two_repeats_both_backends_no_quality_pass(self):
        for engine in ('batchgen','sglang'):
            with self.subTest(engine=engine),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixed=fixture();d=deployment(engine,fixed)
                tool.save_new(root/'fixture.json',fixed);tool.save_new(root/'deployment.json',d)
                calls=[]
                class Transport:
                    def __init__(self,*args):pass
                    def call(self,method,path,payload=None,content_type=None):
                        calls.append((method,path))
                        if path=='/v1/models':return {'data':[{'id':'DeepSeek-R1'}]}
                        body=json.loads(payload)
                        if body['sampling_params']['max_new_tokens']==0:
                            return {'meta_info':{'input_token_logprobs':[[None,0],[-1,11],[-2,12],[-3,20],[-5,21]]}}
                        return {'text':'answer','output_ids':[42], 'meta_info':{'prompt_tokens':3,'completion_tokens':1,'finish_reason':{'type':'length'}}}
                def run_phase(cfg,phase,items,directory,*unused):
                    self.assertEqual(len(items),1)
                    self.assertNotIn('seed',items[0]['body']) # unsupported BatchGen field
                    row={'success':True,'output_text':'answer','finish_reason':'length',
                         'usage':{'prompt_tokens':3,'completion_tokens':1}}
                    (directory/(phase+'.requests.jsonl')).write_text(json.dumps(row)+'\n')
                    return {'all_requests_valid':True}
                fake=SimpleNamespace(base_url=lambda url:url,Transport=Transport,run_phase=run_phase)
                args=SimpleNamespace(fixture=root/'fixture.json',deployment=root/'deployment.json',backend=engine,
                                     endpoint='http://unused',output=root/'result',api_key_env=None,timeout=30)
                with patch.dict(sys.modules,{'batch_benchmark_client':fake}):self.assertEqual(tool.run(args),0)
                report=json.loads((root/'result/run.json').read_text())
                self.assertEqual(report['status'],'diagnostic_collected_not_quality_pass')
                self.assertEqual(len(report['records']),12)
                self.assertEqual(report['records'][0]['native_tokens_available'],engine=='sglang')
                if engine=='sglang':self.assertEqual(len(calls),19) # discovery + 12 generation + 6 teacher

    def test_offline_compare_labels_retokenization_and_never_returns_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=fixture()
            for name in f['tokenizer_files']:
                (root/name).write_text(name)
                f['tokenizer_files'][name]['sha256']=tool.sha(root/name)
            tool.save_new(root/'fixture.json',f)
            for engine in ('batchgen','sglang'):
                records=[]
                for repeat in range(2):
                    for case in f['cases']:
                        records.append({'id':case['id'],'repeat':repeat,'text':'same text',
                                        'native_output_ids':None if engine=='batchgen' else [999],
                                        'usage':{'completion_tokens':1},'finish_reason':'stop'})
                tool.save_new(root/(engine+'.json'),{'backend':engine,'fixture_sha256':tool.sha(root/'fixture.json'),
                    'status':'diagnostic_collected_not_quality_pass','deployment':deployment(engine,f),'records':records})
            class Tokenizer:
                def encode(self,text,**kwargs):return [ord(c) for c in text]
            fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer()))
            args=SimpleNamespace(fixture=root/'fixture.json',batchgen_run=root/'batchgen.json',sglang_run=root/'sglang.json',
                                 tokenizer=root,output=root/'comparison.json')
            with patch.dict(sys.modules,{'transformers':fake}):tool.compare(args)
            report=json.loads(args.output.read_text())
            self.assertEqual(report['status'],'diagnostic_comparison_only')
            self.assertEqual(report['exact_text_matches'],12)
            self.assertEqual(report['prior_attention_matrix_v4']['failed'],8)
            self.assertIsNone(report['full_logits_metrics'])
            self.assertIsNone(report['cross_engine'][0]['native_cross_engine_token_comparison'])
            self.assertFalse(report['cross_engine'][0]['sglang_native_vs_retokenized']['exact_match'])

    def test_missing_model_stops_before_any_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=fixture();tool.save_new(root/'f.json',f);tool.save_new(root/'d.json',deployment('sglang',f))
            class Transport:
                def __init__(self,*args):pass
                def call(self,method,path,*args):
                    self_test.assertEqual(path,'/v1/models');return {'data':[]}
            self_test=self
            args=SimpleNamespace(fixture=root/'f.json',deployment=root/'d.json',backend='sglang',endpoint='http://unused',
                                 output=root/'out',api_key_env=None,timeout=30)
            with patch.dict(sys.modules,{'batch_benchmark_client':SimpleNamespace(base_url=lambda x:x,Transport=Transport)}):
                self.assertEqual(tool.run(args),2)
            report=json.loads((root/'out/run.json').read_text());self.assertEqual(report['records'],[])


if __name__=='__main__':unittest.main()
