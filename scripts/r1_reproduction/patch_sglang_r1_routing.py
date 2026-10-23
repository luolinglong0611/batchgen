#!/usr/bin/env python3
"""Audited, opt-in R1-only routing compatibility patch for SGLang0.5.5.post3.

Default prints the exact diff. --apply backs up the stock file and writes a
receipt before atomically replacing ONE model source file. This is a compat
baseline, never stock. Each spawned model worker imports the patched file.
"""
import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

STOCK_SHA = '620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49'
CONFIG_SHA = '4422016f41c506339e0510a32812a63da554dc9febd5f1368e7240ded14e4a27'
MANIFEST_SHA = 'd9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253'
ENV_NAME = 'BATCHGEN_SGLANG_R1_ROUTING_COMPAT'
RELATIVE_SOURCE = 'sglang/srt/models/deepseek_v2.py'
ANCHOR = '''            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK('''
BLOCK = '''        # R1 reproduction compatibility: 0.5.5.post3 leaves this optional
        # attribute None, but FlashInfer0.5.2 native ABI requires an integer.
        # Explicit opt-in and original R1 grouped-sigmoid routing only.
        if (
            os.environ.get("BATCHGEN_SGLANG_R1_ROUTING_COMPAT") == "1"
            and getattr(config, "model_type", None) == "deepseek_v3"
            and getattr(config, "architectures", None) == ["DeepseekV3ForCausalLM"]
            and getattr(config, "scoring_func", None) == "sigmoid"
            and getattr(config, "topk_method", None) == "noaux_tc"
            and getattr(config, "n_group", None) == 8
            and getattr(config, "topk_group", None) == 4
            and getattr(config, "n_routed_experts", None) == 256
            and getattr(config, "num_experts_per_tok", None) == 8
            and getattr(config, "hidden_size", None) == 7168
            and getattr(config, "moe_intermediate_size", None) == 2048
            and get_moe_runner_backend().is_flashinfer_trtllm()
            and getattr(self.experts, "routing_method_type", None) is None
        ):
            from sglang.srt.layers.moe.utils import RoutingMethodType

            self.experts.routing_method_type = RoutingMethodType.DeepSeekV3

'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patched_source(data):
    if sha(data) != STOCK_SHA:
        raise ValueError('Stock source SHA mismatch; do not patch unknown/already patched source')
    text = data.decode()
    if text.count(ANCHOR) != 1:
        raise ValueError('Expected exactly one DeepseekV2MoE expert initializer')
    result = text.replace(ANCHOR, ANCHOR.replace('        self.topk = TopK(', BLOCK+'        self.topk = TopK('))
    compile(result, RELATIVE_SOURCE, 'exec')
    return result.encode()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--env', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--receipt-dir', type=Path, required=True)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    site = args.env.resolve()/'lib/python3.11/site-packages'
    versions = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions(path=[str(site)])}
    if versions.get('sglang') != '0.5.5.post3' or versions.get('flashinfer-python') != '0.5.2':
        raise RuntimeError('Only the pinned SGLang0.5.5.post3/FlashInfer0.5.2 environment is supported')
    if sha(args.config.read_bytes()) != CONFIG_SHA or sha(args.manifest.read_bytes()) != MANIFEST_SHA:
        raise RuntimeError('Original R1 config/manifest SHA mismatch')
    selected = [x for x in json.loads(args.manifest.read_text()) if x['Path']=='config.json']
    if len(selected)!=1 or selected[0]['Sha256']!=CONFIG_SHA:
        raise RuntimeError('Manifest/config pin mismatch')
    source = site/RELATIVE_SOURCE
    if source.is_symlink():
        raise RuntimeError('Refuse symlink source')
    old = source.read_bytes()
    new = patched_source(old)
    diff = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                      fromfile=RELATIVE_SOURCE+'.stock', tofile=RELATIVE_SOURCE+'.r1-compat'))
    print(diff)
    if not args.apply:
        return
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    backup = args.receipt_dir/'deepseek_v2.py.stock'
    backup.write_bytes(old)
    (args.receipt_dir/'patch.diff').write_text(diff)
    receipt = {'status':'prepared','identity':'sglang0.5.5.post3+r1-routing-None-compat',
               'source_path':str(source),'backup_path':str(backup),
               'stock_sha256':sha(old),'patched_sha256':sha(new),
               'config_sha256':CONFIG_SHA,'manifest_sha256':MANIFEST_SHA,
               'patch_tool_sha256':sha(Path(__file__).read_bytes()),'activation_env':{ENV_NAME:'1'},
               'scope':'DeepseekV2MoE original R1 grouped sigmoid config, flashinfer_trtllm, existing routing None only; non-None including enum0 preserved'}
    record = args.receipt_dir/'receipt.json'
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    if source.read_bytes()!=old or backup.read_bytes()!=old:
        raise RuntimeError('Source changed while preparing backup')
    with tempfile.NamedTemporaryFile(dir=source.parent, prefix='.r1-routing-', delete=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(new);tmp.flush();os.fsync(tmp.fileno())
    try:
        os.chmod(temp_path, source.stat().st_mode)
        os.replace(temp_path, source)
    finally:
        if temp_path.exists():temp_path.unlink()
    if sha(source.read_bytes()) != sha(new):
        raise RuntimeError('Post-patch source SHA mismatch')
    receipt['status']='applied_NOT_GPU_qualified'
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
