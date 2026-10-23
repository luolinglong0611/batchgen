#!/usr/bin/env python3
"""R1 SM103 synthetic MoE/GEMM qualification with a predeclared v1 gate.

Run inside an allocated container with DISABLE_MMA_V5=1, for example:
  python moe_gemm_qualification.py --expected-hostname HOST --report REPORT.json

Real H7168/I2048, original E4M3+FP32 block128 scale format, 256-expert/top8
router and eight distinct expert weights in a sparse 16-local-expert fixture.
No checkpoint, scheduler, host offload, EP collective, model-quality, attention,
or speed qualification. Native grouped TMA kernels are tested unchanged.
Prefill has BF16 gate/up, SiLU, and product boundaries; decode's fused epilogue
keeps gate/up/SiLU/product FP32 until one BF16 cast. Separate oracles are used.
The fixed MoE gate does NOT supersede the existing attention gate. Full actual,
reference and error tensors are retained next to the JSON report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py'
HIDDEN, INTERMEDIATE, TOPK, LOCAL_EXPERTS = 7168, 2048, 8, 16
ACTIVE = [0, 2, 4, 6, 8, 10, 12, 14]
GATES = {
    'stage': {'relative_l2_max': 0.005, 'max_abs_over_reference_rms_max': 0.05},
    'composition': {'relative_l2_max': 0.02, 'max_abs_over_reference_rms_max': 0.15, 'cosine_min': 0.9998},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def accepts(metrics, kind):
    gate = GATES[kind]
    return (metrics['finite'] and metrics['relative_l2'] <= gate['relative_l2_max']
            and metrics['max_abs_over_reference_rms'] <= gate['max_abs_over_reference_rms_max']
            and metrics['cosine'] >= gate.get('cosine_min', -1))


def compare(actual, expected, name, result, tensors, kind='stage', dtype=None):
    dtype = torch.bfloat16 if dtype is None else dtype
    record = {'case': name, 'gate': kind, 'shape': list(expected.shape), 'status': 'failed'}
    tensors[name] = {'actual': actual.cpu(), 'reference': expected.cpu()}
    if actual.shape != expected.shape or actual.dtype != dtype:
        record['error'] = f'Expected {expected.shape}/{dtype}, got {actual.shape}/{actual.dtype}'
    else:
        a, b = actual.float(), expected.float()
        error = a - b
        tensors[name] = {'actual': actual.cpu(), 'reference': expected.cpu(), 'error': error.cpu()}
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        record['finite'] = finite
        if finite:
            rms = float(b.square().mean().sqrt())
            diff_norm = float(torch.linalg.vector_norm(error))
            ref_norm = float(torch.linalg.vector_norm(b))
            norm_product = float(torch.linalg.vector_norm(a)) * ref_norm
            cosine = float((a * b).sum()) / norm_product if norm_product else float(diff_norm == 0)
            denom = max(rms, 1e-30)
            record.update(reference_rms=rms, max_abs=float(error.abs().max()),
                          mean_abs=float(error.abs().mean()),
                          p99_abs=float(torch.quantile(error.abs().flatten(), 0.99)),
                          relative_l2=diff_norm / max(ref_norm, 1e-30),
                          max_abs_over_reference_rms=float(error.abs().max()) / denom,
                          cosine=cosine)
            record['status'] = 'passed' if accepts(record, kind) else 'failed'
    result['cases'].append(record)
    emit(record)


def quantize_reference(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = (blocks.abs().amax(-1).clamp_min(2**-16) * (1 / 448)).clamp_min(1e-12)
    values = (blocks / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.reshape_as(x), scales


def gemm_reference(x, weight, scales):
    """IEEE FP32 per-K128 dots and FP32 original scales; independent quantizer."""
    quantized, activation_scales = quantize_reference(x)
    out = torch.zeros((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    rows = torch.arange(weight.shape[0], device=x.device) // 128
    for block, start in enumerate(range(0, x.shape[1], 128)):
        dot = quantized[:, start:start+128].float() @ weight[:, start:start+128].float().T
        out += dot * (activation_scales[:, block, None] * scales[rows, block][None, :])
    return out


def expert_reference(x, expert, fused):
    up = gemm_reference(x, *expert['up'])
    gate = gemm_reference(x, *expert['gate'])
    if fused:
        intermediate = ((gate / (1 + torch.exp(-gate))) * up).bfloat16()
    else:
        up, gate = up.bfloat16(), gate.bfloat16()
        activated = torch.nn.functional.silu(gate.float()).bfloat16()
        intermediate = (activated.float() * up.float()).bfloat16()
    return intermediate, gemm_reference(intermediate, *expert['down']).bfloat16()


def namespaces():
    # Import actual low-level source modules without constructing the engine.
    for name in ('batchgen', 'batchgen.attention', 'batchgen.attention.mla', 'batchgen.gemm', 'batchgen.moe'):
        if name not in sys.modules:
            mod = ModuleType(name)
            mod.__path__ = [str(ROOT.joinpath(*name.split('.')))]
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
            sys.modules[name] = mod


def extracted_model_functions(backend, gate_function):
    """Compile selected unchanged production definitions, avoiding full-model imports."""
    import triton
    import triton.language as tl
    tree = ast.parse(MODEL.read_text())
    names = {'moe_fp32_accum_kernel_v2', 'moe_fp32_accum_triton_v2',
             'scatter_weight_reduce_optimized_kernel', 'build_inverse_mapping', 'scatter_weight_reduce_optimized'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for cls, method, alias in [('DeepseekV3MLP', 'deepgemm_forward', 'expert_forward'),
                               ('MoEGate', 'forward', 'router_prefill'),
                               ('MoEGate', 'moe_gate_forward_hybrid', 'router_decode')]:
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        method_node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
        method_node.name = alias
        nodes.append(method_node)
    module = ModuleType('r1_moe_qualification_production')
    module.__file__ = str(MODEL)
    module.__dict__.update(torch=torch, triton=triton, tl=tl, F=torch.nn.functional,
                           w8a16_gemm=backend.w8a16_gemm, moe_fused_gate=gate_function)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL), 'exec', dont_inherit=True), module.__dict__)
    return module


def make_expert(device):
    expert = {}
    for name, rows, cols in [('gate', INTERMEDIATE, HIDDEN), ('up', INTERMEDIATE, HIDDEN), ('down', HIDDEN, INTERMEDIATE)]:
        # Arbitrary non-power-of-two FP32 scales: no UE8M0 conversion.
        weight = (torch.randn((rows, cols), device=device) * 32).clamp(-448, 448).to(torch.float8_e4m3fn)
        scales = (0.75 + torch.rand((rows // 128, cols // 128), device=device)) / (32 * math.sqrt(cols))
        expert[name] = (weight, scales.contiguous())
    return expert


def expert_fixture(expert):
    module = SimpleNamespace(act_fn=torch.nn.SiLU())
    scales = {}
    for name, (weight, scale) in expert.items():
        setattr(module, f'{name}_proj', SimpleNamespace(weight=weight))
        scales[f'{name}_proj.weight_scale_inv'] = scale
    return module, scales


def routing_reference(x, weight, bias):
    scores = torch.sigmoid(x.float() @ weight.float().T)
    choices = scores + bias
    group_scores = choices.view(x.shape[0], 8, 32).sort(descending=True).values[..., :2].sum(-1)
    groups = group_scores.argsort(descending=True)[:, :4]
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    restricted = choices.masked_fill(~mask.repeat_interleave(32, dim=1), -float('inf'))
    indices = restricted.argsort(descending=True)[:, :8]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 2.5
    return indices, weights


def run_router(model, device, result, tensors):
    x = torch.randn((65, HIDDEN), device=device, dtype=torch.bfloat16)
    weight = (torch.randn((256, HIDDEN), device=device) / math.sqrt(HIDDEN)).bfloat16()
    bias = torch.randn(256, device=device) * 0.05
    fixture = SimpleNamespace(weight=weight, e_score_correction_bias=bias, n_group=8, topk_group=4,
                              n_routed_experts=256, top_k=8, routed_scaling_factor=2.5,
                              scoring_func='sigmoid', topk_method='noaux_tc', norm_topk_prob=True, training=False)
    expected_ids, expected_weights = routing_reference(x, weight, bias)
    for name in ('router_prefill', 'router_decode'):
        try:
            ids, weights = getattr(model, name)(fixture, x[:, None])
            order = ids.argsort(-1)
            expected_order = expected_ids.argsort(-1)
            match = ids.shape == expected_ids.shape and torch.equal(ids.gather(1, order), expected_ids.gather(1, expected_order))
            close = match and weights.dtype == torch.float32 and bool(torch.isclose(
                weights.gather(1, order), expected_weights.gather(1, expected_order), atol=1e-6, rtol=1e-5).all())
            record = {'case': name, 'status': 'passed' if close else 'failed', 'expert_sets_identical': bool(match),
                      'weight_atol': 1e-6, 'weight_rtol': 1e-5, 'actual_shape': list(ids.shape)}
            tensors[name] = {'ids': ids.cpu(), 'weights': weights.cpu(), 'reference_ids': expected_ids.cpu(),
                             'reference_weights': expected_weights.cpu()}
        except Exception as exc:
            record = {'case': name, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        result['cases'].append(record)
        emit(record)


def run(args, result, tensors):
    global torch
    import torch
    if not Path('/.dockerenv').exists() or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Run only inside the explicitly allocated container')
    if os.environ.get('DISABLE_MMA_V5') != '1':
        raise RuntimeError('Set DISABLE_MMA_V5=1 before process launch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.device) != (10, 3):
        raise RuntimeError('SM103 required')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    namespaces()
    backend = importlib.import_module('batchgen.attention.mla.sm103_backend')
    stage1 = importlib.import_module('batchgen.gemm.w8a8_grouped_gemm_stage_1')
    stage2 = importlib.import_module('batchgen.moe.fused_grouped_dequant_gemm')
    mgn = importlib.import_module('batchgen_kernels.common.mgn')
    # Bind-check before native routing launch; the old positional ABI passed
    # 256 as topk. Reject that call safely, never launch a malformed CUDA kernel.
    def checked_gate(*positional, **kwargs):
        bound = inspect.signature(mgn.moe_fused_gate).bind(*positional, **kwargs)
        if bound.arguments.get('topk') != 8:
            raise ValueError(f"Production routing ABI passes topk={bound.arguments.get('topk')}, expected 8")
        return mgn.moe_fused_gate(*positional, **kwargs)
    model = extracted_model_functions(backend, checked_gate)
    result['source_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in [
        MODEL, Path(__file__).resolve(), ROOT / 'batchgen/attention/mla/sm103_backend.py',
        ROOT / 'batchgen/gemm/w8a8.py', ROOT / 'batchgen/gemm/w8a8_grouped_gemm_stage_1.py',
        ROOT / 'batchgen/moe/fused_grouped_dequant_gemm.py', ROOT / 'batchgen_kernels/common/mgn.py',
        ROOT / 'batchgen_kernels/triton/fp8_quantize.py']}
    result['torch_version'] = torch.__version__
    run_router(model, device, result, tensors)
    # Eight independent experts plus one shared; inactive local slots alias an
    # allocated expert but must never execute. Total raw FP8 weights ~378 MiB.
    experts = [make_expert(device) for _ in ACTIVE]
    shared = make_expert(device)
    slots = [experts[0] for _ in range(LOCAL_EXPERTS)]
    for index, expert in zip(ACTIVE, experts):
        slots[index] = expert
    for rows in (1, 65, 129):
        for name, width in [('gate', HIDDEN), ('down', INTERMEDIATE)]:
            x = torch.randn((rows, width), device=device, dtype=torch.bfloat16)
            if rows > 1:
                x[0].zero_()  # epsilon scale / exact zero output case
                x[1, :128] *= 16  # per-block dynamic-range difference
            weight, scales = experts[0][name]
            actual = backend.w8a16_gemm(weight, scales, x)
            expected = gemm_reference(x, weight, scales).bfloat16()
            compare(actual, expected, f'gemm_{name}_m{rows}', result, tensors)
    for tokens in (1, 65):
        emit({'event': 'top8_start', 'tokens': tokens})
        x = torch.randn((tokens, HIDDEN), device=device, dtype=torch.bfloat16)
        ids = torch.tensor(ACTIVE, device=device)[None].expand(tokens, -1)
        routing_weights = torch.rand((tokens, 8), device=device)
        routing_weights = (routing_weights / routing_weights.sum(-1, keepdim=True) * 2.5).contiguous()
        sorting = ids.flatten().argsort()
        grouped_x = x[sorting // 8].contiguous()
        groups = torch.full((8,), tokens, dtype=torch.int32, device=device)
        group_ids = torch.tensor(ACTIVE, dtype=torch.int32, device=device)
        starts = torch.arange(8, device=device, dtype=torch.int32) * tokens
        active_count = torch.tensor([8], device=device, dtype=torch.int32)
        lists = {kind: [entry[kind][0] for entry in slots] for kind in ('gate', 'up', 'down')}
        scales = {kind: [entry[kind][1] for entry in slots] for kind in lists}
        ptrs = lambda values: torch.tensor([value.data_ptr() for value in values], dtype=torch.int64, device=device)
        pointer_lists = {kind: ptrs(values) for kind, values in lists.items()}
        scale_pointers = {kind: ptrs(values) for kind, values in scales.items()}
        prefill_outputs, prefill_refs, decode_refs, decode_intermediates = [], [], [], []
        for slot, expert in enumerate(experts):
            expert_x = grouped_x[slot*tokens:(slot+1)*tokens]
            fixture, expert_scales = expert_fixture(expert)
            actual = model.expert_forward(fixture, expert_x, expert_scales)
            _, prefill_expected = expert_reference(expert_x, expert, fused=False)
            intermediate_expected, decode_expected = expert_reference(expert_x, expert, fused=True)
            compare(actual, prefill_expected, f'prefill_expert{ACTIVE[slot]}_m{tokens}', result, tensors, 'composition')
            prefill_outputs.append(actual)
            prefill_refs.append(prefill_expected)
            decode_refs.append(decode_expected)
            decode_intermediates.append(intermediate_expected)
        prefill_outputs, prefill_refs = torch.cat(prefill_outputs), torch.cat(prefill_refs)
        decode_refs, decode_intermediates = torch.cat(decode_refs), torch.cat(decode_intermediates)
        q, qs = backend._quantize(grouped_x)
        intermediate = stage1.fused_fp8_moe_stage_1_tma(
            q, qs, lists['gate'], pointer_lists['gate'], lists['up'], pointer_lists['up'],
            scales['gate'], scale_pointers['gate'], scales['up'], scale_pointers['up'],
            groups, group_ids, starts, active_count, LOCAL_EXPERTS)
        compare(intermediate, decode_intermediates, f'decode_fused_stage1_m{tokens}', result, tensors)
        # Isolated stage2 gets independent oracle inputs; chained stage2 gets DUT inputs.
        rq, rs = backend._quantize(decode_intermediates)
        isolated_down = stage2.fused_dequant_grouped_gemm_fp8_tma(
            rq, rs, lists['down'], pointer_lists['down'], scales['down'], scale_pointers['down'],
            groups, group_ids, starts, active_count)
        compare(isolated_down, decode_refs, f'decode_isolated_stage2_m{tokens}', result, tensors)
        iq, ins = backend._quantize(intermediate)
        decode_outputs = stage2.fused_dequant_grouped_gemm_fp8_tma(
            iq, ins, lists['down'], pointer_lists['down'], scales['down'], scale_pointers['down'],
            groups, group_ids, starts, active_count)
        compare(decode_outputs, decode_refs, f'decode_experts_chained_m{tokens}', result, tensors, 'composition')
        inverse = sorting.argsort()
        def weighted_reference(outputs):
            return (outputs[inverse].view(tokens, 8, HIDDEN).float() * routing_weights[..., None]).sum(1).bfloat16()
        pref = model.moe_fp32_accum_triton_v2(prefill_outputs, sorting, routing_weights)
        dec = model.scatter_weight_reduce_optimized(decode_outputs, sorting // 8, sorting % 8,
                                                    routing_weights, tokens, 8).bfloat16()
        fixture, expert_scales = expert_fixture(shared)
        shared_actual = model.expert_forward(fixture, x, expert_scales)
        _, shared_expected = expert_reference(x, shared, fused=False)
        compare(shared_actual, shared_expected, f'shared_expert_m{tokens}', result, tensors, 'composition')
        compare(pref + shared_actual, weighted_reference(prefill_refs) + shared_expected,
                f'prefill_top8_plus_shared_m{tokens}', result, tensors, 'composition')
        compare(dec + shared_actual, weighted_reference(decode_refs) + shared_expected,
                f'decode_top8_plus_shared_m{tokens}', result, tensors, 'composition')
        # Different algorithms: retain raw drift, never gate one against the other.
        tensors[f'prefill_vs_decode_oracle_m{tokens}'] = {
            'prefill_reference': weighted_reference(prefill_refs).cpu(),
            'decode_reference': weighted_reference(decode_refs).cpu(),
            'error': (weighted_reference(prefill_refs).float() - weighted_reference(decode_refs).float()).cpu()}
    torch.cuda.synchronize(device)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    result['status'] = 'passed' if result['cases'] and all(c['status'] == 'passed' for c in result['cases']) else 'failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    tensors_path = args.report.with_suffix('.tensors.pt')
    if args.report.exists() or tensors_path.exists():
        parser.error('Use fresh output paths; previous failures must be retained')
    result = {'status': 'failed', 'gate_version': 'r1_moe_smoke_v1', 'gates': GATES, 'cases': [],
              'hostname': socket.gethostname(), 'seed': args.seed, 'device': args.device,
              'environment': {name: os.environ.get(name) for name in ('DISABLE_MMA_V5', 'TRITON_PTXAS_PATH')},
              'scope': 'synthetic single-GPU operators; excludes EP collective, native token dispatch/compaction, model quality, attention and performance',
              'grouping_fixture': 'Independent sorted top8 assignments; does not qualify the production dispatch allocator',
              'precision_contract': {'prefill': 'FP8 GEMMs -> BF16 gate/up -> BF16 SiLU -> BF16 product -> FP8 requantization -> BF16 down',
                                     'decode': 'FP8 grouped GEMMs -> FP32 gate/up/SiLU/product -> BF16 product -> FP8 requantization -> BF16 down',
                                     'reduction': 'FP32 local weighted sum -> BF16 -> BF16 shared-expert add; distributed BF16 all-reduce excluded'},
              'shapes': {'hidden': HIDDEN, 'intermediate': INTERMEDIATE, 'router_experts': 256, 'topk': TOPK,
                         'local_expert_slots': LOCAL_EXPERTS, 'active_experts': ACTIVE, 'shared_experts': 1},
              'weights': 'synthetic FP8 E4M3 plus arbitrary FP32 block128 scales; no UE8M0 conversion',
              'gate_rationale': 'Stage budget isolates BF16 rounding; composition budget includes nonlinear and requantization amplification. RMS-normalization avoids near-zero relative-error singularities. Not a quality guarantee.',
              'raw_tensors': str(tensors_path)}
    tensors = {}
    start = time.monotonic()
    try:
        import torch
        with torch.inference_mode():
            run(args, result, tensors)
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        result['elapsed_seconds_not_benchmark'] = time.monotonic() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if tensors:
            torch.save(tensors, tensors_path)
        else:
            result['raw_tensors'] = None
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        emit(result)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
