import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
M=load('dense_qual_plan_test','scripts/r1_reproduction/qualify_sglang_r1_dense_bf16.py')
H=load('dense_qual_helper_test','scripts/r1_reproduction/sglang_r1_dense_bf16.py')

class QualificationPlan(unittest.TestCase):
    def test_default_does_not_import_helper_torch_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out'
            with patch.object(M,'helper',side_effect=AssertionError('no dependency imports')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(M.main(['--checkpoint-dir','missing','--model-manifest','missing','--output',str(out)]),0)
            self.assertFalse(out.exists())

    def test_independent_raw_header_provenance_checks_full_file_sha_and_ignores_stale_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);name='model.layers.0.mlp.gate_proj.weight';raw=b'12345678'
            header=json.dumps({name:{'dtype':'F8_E4M3','shape':[2,4],'data_offsets':[0,len(raw)]}}).encode()
            shard=root/'one.safetensors';shard.write_bytes(struct.pack('<Q',len(header))+header+raw)
            index=root/'model.safetensors.index.json';index.write_text(json.dumps({'metadata':{'total_size':99999999},'weight_map':{name:shard.name}}))
            manifest=root/'manifest.json';manifest.write_text(json.dumps([{'Path':p.name,'Size':p.stat().st_size,'Sha256':M.file_sha(p)} for p in (shard,index)]))
            with patch.object(M,'helper',return_value=H),patch.object(H,'MODEL_MANIFEST_SHA',M.file_sha(manifest)),patch.object(H,'expected_tensors',return_value={name:('float8_e4m3fn',[2,4])}):
                result=M.original_provenance(root,manifest)
                self.assertEqual(result['tensor_records'][name]['raw_bytes'],8)
                self.assertEqual(result['tensor_records'][name]['raw_sha256'],hashlib.sha256(raw).hexdigest())
                shard.write_bytes(shard.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError,'SHA/size'):M.original_provenance(root,manifest)

    def test_execution_is_guarded_before_torch_and_requires_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            argv=['--checkpoint-dir','missing','--model-manifest','missing','--output',str(out),'--expected-hostname',socket.gethostname(),'--execute']
            with patch.dict(os.environ,{'R1_DENSE_BF16_QUALIFICATION_GUARDED':'0'}),self.assertRaisesRegex(RuntimeError,'source SHA256'):
                M.main(argv)
            self.assertTrue((out/'failed.json').exists())
            with self.assertRaises(FileExistsError):M.main(argv)

if __name__=='__main__':unittest.main()
