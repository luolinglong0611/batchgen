#!/usr/bin/env python3
"""Prepare an isolated SG local-shared model candidate; default is plan-only.

--execute writes a candidate and its exact diff to a new exclusive directory.
It never replaces installed SG, imports GPU libraries, starts a model, or uses
the network. Deploying and qualifying this candidate is a separate step.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


STOCK_SHA = '620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49'
RUNTIME = Path(__file__).with_name('sglang_r1_alignment_runtime.py')
ATTENTION_RUNTIME = Path(__file__).with_name('sglang_r1_attention_compat.py')
IMPORTS = '''from scripts.r1_reproduction.sglang_r1_alignment_runtime import (
    local_shared_enabled as _r1_local_shared_enabled,
    require_shared_contract as _r1_require_shared_contract,
    capture_local_shared_input as _r1_capture_local_shared_input,
    finish_local_shared as _r1_finish_local_shared,
    install_fp8_linear_backend as _r1_install_fp8_linear_backend,
)

'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_method(text, class_name, method_name, old, new):
    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise ValueError('Missing/ambiguous SG class: ' + class_name)
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == method_name]
    if len(methods) != 1:
        raise ValueError('Missing/ambiguous SG method: ' + class_name + '.' + method_name)
    node = methods[0]
    lines = text.splitlines(keepends=True)
    part = ''.join(lines[node.lineno - 1:node.end_lineno])
    if part.count(old) != 1:
        raise ValueError('SG patch anchor mismatch: ' + class_name + '.' + method_name)
    lines[node.lineno - 1:node.end_lineno] = [part.replace(old, new)]
    return ''.join(lines)


def patched_source(data):
    if sha(data) != STOCK_SHA:
        raise ValueError('Pinned stock SG source SHA mismatch; unknown/already patched source refused')
    text = data.decode('utf-8')
    if text.count('class DeepseekV2MoE(nn.Module):') != 1:
        raise ValueError('SG MoE class anchor mismatch')
    text = text.replace('class DeepseekV2MoE(nn.Module):', IMPORTS + 'class DeepseekV2MoE(nn.Module):')

    old = '        self.moe_ep_size = get_moe_expert_parallel_world_size()\n'
    new = old + '''        self._r1_local_shared_tp1 = _r1_local_shared_enabled()
        if self._r1_local_shared_tp1:
            _r1_require_shared_contract(
                config, quant_config, get_global_server_args(),
                tp_size=self.tp_size, ep_size=self.moe_ep_size,
                attention_tp_size=get_attention_tp_size(),
                dp_attention=is_dp_attention_enabled(),
                a2a_none=get_moe_a2a_backend().is_none(),
                runner_triton=get_moe_runner_backend().is_triton(),
                deepgemm_enabled=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM,
            )
'''
    text = replace_method(text, 'DeepseekV2MoE', '__init__', old, new)
    old = '''                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
'''
    new = '''                    dict(tp_rank=0, tp_size=1)
                    if self._r1_local_shared_tp1 or get_moe_a2a_backend().is_deepep()
'''
    text = replace_method(text, 'DeepseekV2MoE', '__init__', old, new)
    old = '        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()'
    text = replace_method(text, 'DeepseekV2MoE', '__init__', old, old + '''
        if self._r1_local_shared_tp1:
            self._fuse_shared_experts_inside_sbo = False
''')
    old = '        if not self._enable_a2a_moe:\n'
    text = replace_method(text, 'DeepseekV2MoE', 'forward', old, '''        if self._r1_local_shared_tp1:
            # The shared branch is local and added after routed reduction/scatter.
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter,
                gemm_output_zero_allocator,
            )
''' + old)
    old = '        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):\n'
    text = replace_method(text, 'DeepseekV2MoE', '_forward_shared_experts', old, '''        if self._r1_local_shared_tp1:
            return None
''' + old)
    old = '            allow_reduce_scatter=True,\n'
    text = replace_method(text, 'DeepseekV2DecoderLayer', '__init__', old, '''            allow_reduce_scatter=not getattr(self.mlp, "_r1_local_shared_tp1", False),
''')
    old = '''        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
'''
    text = replace_method(text, 'DeepseekV2DecoderLayer', 'forward', old, old + '''
        r1_local_shared_input = _r1_capture_local_shared_input(
            self.mlp, hidden_states, residual, forward_batch
        )
''')
    old = '''        if isinstance(self.mlp, DeepseekV2MLP):
'''
    text = replace_method(text, 'DeepseekV2DecoderLayer', 'forward', old, '''        if r1_local_shared_input is not None and (
            should_allreduce_fusion or use_reduce_scatter
        ):
            raise RuntimeError("R1 local shared candidate requires routed all-reduce then local scatter")

''' + old)
    old = '        return hidden_states, residual\n'
    text = replace_method(text, 'DeepseekV2DecoderLayer', 'forward', old, '''        hidden_states = _r1_finish_local_shared(
            self.mlp, r1_local_shared_input, hidden_states
        )
''' + old)
    old = '        self.determine_num_fused_shared_experts()\n'
    text = replace_method(text, 'DeepseekV2ForCausalLM', '__init__', old, '''        if _r1_local_shared_enabled():
            _r1_install_fp8_linear_backend(config, quant_config)
''' + old)
    old = '        total_num_layers = self.end_layer - self.start_layer\n'
    text = replace_method(text, 'DeepseekV2Model', 'forward', old, '''        if _r1_local_shared_enabled() and getattr(forward_batch, "can_run_tbo", False):
            raise RuntimeError("R1 local shared candidate rejects TBO execution")
''' + old)
    compile(text, 'deepseek_v2.py.r1-local-shared-candidate', 'exec')
    return text.encode('utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    source = args.source.resolve(strict=True)
    original = source.read_bytes()
    candidate = patched_source(original)
    diff = ''.join(difflib.unified_diff(original.decode().splitlines(True),
                                      candidate.decode().splitlines(True),
                                      fromfile='deepseek_v2.py.stock',
                                      tofile='deepseek_v2.py.r1-local-shared-candidate'))
    receipt = {'status': 'plan_only', 'source': str(source),
               'source_sha256': sha(original), 'candidate_sha256': sha(candidate),
               'runtime_helper_sha256': sha(RUNTIME.read_bytes()),
               'attention_helper_sha256': sha(ATTENTION_RUNTIME.read_bytes()),
               'generator_sha256': sha(Path(__file__).read_bytes()),
               'activation_environment': {'SGLANG_R1_LOCAL_SHARED_TP1': '1',
                   'SGLANG_R1_COMMON_FI_FA2': '1',
                   'BATCHGEN_R1_SM103_FP8_COMPAT': '1', 'DISABLE_MMA_V5': '1',
                   'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
                   'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '0'},
               'gpu_operations': False, 'installed_source_changed': False,
               'model_qualified': False, 'formal_test_permitted': False,
               'diff': diff}
    if args.execute:
        if args.output_dir is None or not args.output_dir.is_absolute():
            ap.error('--execute requires an absolute --output-dir')
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / 'deepseek_v2.py.stock').write_bytes(original)
        (output / 'deepseek_v2.py').write_bytes(candidate)
        (output / RUNTIME.name).write_bytes(RUNTIME.read_bytes())
        (output / ATTENTION_RUNTIME.name).write_bytes(ATTENTION_RUNTIME.read_bytes())
        (output / 'candidate.diff').write_text(diff)
        receipt.update(status='prepared_NOT_installed_NOT_GPU_qualified', output_dir=str(output))
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
