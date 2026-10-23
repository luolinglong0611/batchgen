#!/usr/bin/env python3
"""Import actual R1 modules; optionally qualify only the small routing gate.

Default mode never constructs DeepseekV3ForCausalLM. --skeleton-only constructs
its empty-storage serving skeleton, without loading weights or forwarding. The
optional GPU check constructs only the 256 x 7168 MoEGate and compares native CUDA
routing against MoEGate.forward's independent PyTorch routing implementation.
This does not qualify attention, MoE expert GEMMs, offload, or model outputs.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import math
from pathlib import Path
import socket
import resource
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ('batchgen.attention.mla.fa3_backend',
          'batchgen.attention.mla.flashmla_backend',
          'flash_attn_interface', 'flash_mla', 'deep_gemm')


def skeleton_check(modeling, config_path, device):
    import torch
    config_data = json.loads(config_path.read_text())
    required = {'hidden_size': 7168, 'num_hidden_layers': 61, 'n_routed_experts': 256,
                'num_experts_per_tok': 8, 'first_k_dense_replace': 3,
                'q_lora_rank': 1536, 'kv_lora_rank': 512, 'num_attention_heads': 128,
                'qk_rope_head_dim': 64, 'qk_nope_head_dim': 128, 'v_head_dim': 128}
    for key, value in required.items():
        if config_data.get(key) != value:
            raise ValueError(f'Expected R1 {key}={value}, got {config_data.get(key)}')
    configuration = importlib.import_module('batchgen.models.deepseek.deepseekv3.configuration_deepseek_v3')
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError('Skeleton qualification expects an allocated SM103 device for real RoPE buffers')
    torch.cuda.set_device(device)
    results = []
    for layers in (4, 61):
        config = configuration.DeepseekV3Config(**config_data)
        config.num_hidden_layers = layers
        config.batchgen_skeleton = True
        config.phase = 'prefill'
        config.ep_size = 1
        config._attn_implementation = 'eager'
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        model = modeling.DeepseekV3ForCausalLM(config, comm=None)
        parameter_bytes = {'router': 0, 'norm': 0, 'other': 0}
        empty_count, logical_elements = 0, 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                shapes = getattr(module, '_batchgen_parameter_shapes', None)
                if not shapes:
                    raise AssertionError(f'Large module is missing skeleton shape metadata: {name}')
                for local_name, parameter in module.named_parameters(recurse=False):
                    if parameter.is_meta or parameter.numel() != 0 or parameter.untyped_storage().nbytes() != 0:
                        raise AssertionError(f'Large weight allocated: {name}.{local_name}')
                    empty_count += 1
                    logical_elements += math.prod(shapes[local_name])
        for name, parameter in model.named_parameters():
            if parameter.is_meta:
                raise AssertionError(f'Residual meta tensor: {name}')
            category = 'router' if '.gate.' in name else 'norm' if 'norm' in name else 'other'
            parameter_bytes[category] += parameter.numel() * parameter.element_size()
        storage_bytes, seen = {}, set()
        rope_logical_bytes, rope_buffers = 0, 0
        for name, buffer in model.named_buffers(remove_duplicate=False):
            storage = buffer.untyped_storage()
            key = (str(buffer.device), storage.data_ptr())
            if key not in seen:
                storage_bytes[str(buffer.device)] = storage_bytes.get(str(buffer.device), 0) + storage.nbytes()
                seen.add(key)
            if '.rotary_emb.' in name:
                rope_buffers += 1
                rope_logical_bytes += buffer.numel() * buffer.element_size()
        if layers == 61 and logical_elements < 600_000_000_000:
            raise AssertionError('The full R1 skeleton does not describe the expected model scale')
        actual_bytes = sum(parameter_bytes.values()) + sum(storage_bytes.values())
        if actual_bytes >= 10 * 1024**3:
            raise AssertionError(f'Skeleton exceeded the 10 GiB qualification bound: {actual_bytes}')
        torch.cuda.synchronize(device)
        record = {'layers': layers, 'empty_large_parameters': empty_count,
                  'logical_large_parameter_elements': logical_elements,
                  'resident_parameter_bytes': parameter_bytes,
                  'unique_buffer_storage_bytes_by_device': storage_bytes,
                  'rope_buffer_references': rope_buffers,
                  'rope_logical_bytes_including_shared_references': rope_logical_bytes,
                  'cuda_allocated_before_bytes': allocated_before,
                  'cuda_allocated_after_bytes': torch.cuda.memory_allocated(device),
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'seconds': time.monotonic() - start,
                  'phase': 'prefill', 'ep_size': 1, 'attention_class': 'eager',
                  'weights_loaded': False, 'forward_executed': False}
        results.append(record)
        print(json.dumps({'skeleton_constructed': record}), flush=True)
        del parameter, buffer, module, model
        gc.collect()
    return {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'cases': results}


def gate_check(modeling, config_path, device):
    import torch
    config_data = json.loads(config_path.read_text())
    required = {'hidden_size': 7168, 'n_routed_experts': 256,
                'num_experts_per_tok': 8, 'n_group': 8, 'topk_group': 4,
                'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
                'norm_topk_prob': True}
    for key, value in required.items():
        if config_data.get(key) != value:
            raise ValueError(f'Expected original R1 {key}={value}, got {config_data.get(key)}')
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError('GPU qualification is restricted to the allocated SM103 device')
    configuration = importlib.import_module('batchgen.models.deepseek.deepseekv3.configuration_deepseek_v3')
    config = configuration.DeepseekV3Config(**config_data)
    torch.manual_seed(20261023)
    torch.cuda.manual_seed_all(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    gate = modeling.MoEGate(config).eval().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        gate.weight.normal_(mean=0.0, std=0.01)
        gate.e_score_correction_bias.copy_(torch.linspace(-0.15, 0.15, 256, device=device))
    cases = []
    for batch, length in ((1, 1), (7, 1), (3, 11)):
        hidden = torch.randn(batch, length, 7168, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            reference_ids, reference_weights = gate(hidden)
            actual_ids, actual_weights = gate.moe_gate_forward_hybrid(hidden)
        if tuple(actual_ids.shape) != (batch * length, 8) or tuple(actual_weights.shape) != (batch * length, 8):
            raise AssertionError(f'Incorrect routed expert count: {actual_ids.shape}, {actual_weights.shape}')
        reference_ids, order = reference_ids.sort(dim=-1)
        reference_weights = reference_weights.gather(-1, order)
        actual_ids, order = actual_ids.sort(dim=-1)
        actual_weights = actual_weights.gather(-1, order)
        torch.testing.assert_close(actual_ids.long(), reference_ids.long(), rtol=0, atol=0)
        if not torch.isfinite(actual_weights).all().item():
            raise AssertionError('Native routing produced nonfinite weights')
        if not (actual_ids[:, 1:] != actual_ids[:, :-1]).all().item():
            raise AssertionError('Native routing selected a duplicated expert')
        # FP32 sigmoid scores and normalized FP32 weights: keep a strict gate.
        torch.testing.assert_close(actual_weights, reference_weights, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_weights.sum(-1),
                                   torch.full((batch * length,), config.routed_scaling_factor, device=device),
                                   rtol=1e-5, atol=1e-6)
        torch.cuda.synchronize(device)
        cases.append({'batch': batch, 'sequence_length': length, 'topk': 8,
                      'max_abs_weight_error': float((actual_weights - reference_weights).abs().max().item()),
                      'indices_equal': True})
    return {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'device': torch.cuda.get_device_name(device), 'capability': [10, 3],
            'gate_parameter_count': sum(p.numel() for p in gate.parameters()),
            'weight_rtol': 1e-5, 'weight_atol': 1e-6, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-hostname')
    parser.add_argument('--worker', action='store_true', help='Also import the real batchgen_worker module')
    parser.add_argument('--check-gate', action='store_true')
    parser.add_argument('--skeleton-only', action='store_true',
                        help='Construct 4-layer then 61-layer empty serving skeletons; no model forward or loading')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if (args.check_gate or args.skeleton_only) and not args.config:
        parser.error('--check-gate / --skeleton-only requires the real R1 --config')
    if args.check_gate and args.skeleton_only:
        parser.error('--skeleton-only cannot run --check-gate: no forward is allowed in skeleton mode')
    report = {'status': 'failed', 'scope': 'Actual module import; optional isolated gate or empty model construction',
              'hostname': socket.gethostname(), 'source_root': str(ROOT), 'imports': {},
              'environment': {name: os.environ.get(name) for name in (
                  'TORCH_CUDA_CHANNEL', 'BATCHGEN_R1_SM103_EXPERIMENTAL',
                  'DISABLE_MMA_V5', 'TRITON_PTXAS_PATH')}}
    start = time.monotonic()
    try:
        if not Path('/.dockerenv').exists():
            raise RuntimeError('Run inside the allocated container')
        if args.expected_hostname and args.expected_hostname != report['hostname']:
            raise RuntimeError('Unexpected container hostname')
        sys.path.insert(0, str(ROOT))
        import libucx
        libucx.load_library()
        import torch
        report['torch'] = torch.__version__
        report['cuda'] = torch.version.cuda
        report['transformers'] = importlib.metadata.version('transformers')
        modules = ['batchgen.models.deepseek.deepseekv3.modeling_deepseek_v3',
                   'batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager',
                   'batchgen.models.deepseek.deepseekv3.deepseekv3_initializer']
        if args.worker:
            modules.append('batchgen.batchgen_worker')
        for name in modules:
            module = importlib.import_module(name)
            report['imports'][name] = module.__file__
        modeling = sys.modules[modules[0]]
        if args.check_gate:
            report['gate'] = gate_check(modeling, args.config, torch.device(args.device))
        if args.skeleton_only:
            report['skeleton'] = skeleton_check(modeling, args.config, torch.device(args.device))
        imported_legacy = [name for name in LEGACY if name in sys.modules]
        report['legacy_modules_loaded'] = imported_legacy
        if imported_legacy:
            raise RuntimeError(f'Unexpected legacy runtime imports: {imported_legacy}')
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    finally:
        report['seconds'] = time.monotonic() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
