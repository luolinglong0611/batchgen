#!/usr/bin/env python3
"""Bounded RoPE layout/MLA handoff fixture, not whole-model qualification.

Plan-only by default; root runs --execute under its owned single-GPU monitor
and separately under compute-sanitizer. No model server is started here.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
from types import ModuleType

PROMPTS = (63, 64, 65, 2048, 8192, 16381)
STEPS = 3
HEADS, LATENT, ROPE, PAGE = 128, 512, 64, 64
SOURCES = ('kernel', 'legacy', 'rotary', 'model', 'config', 'backend', 'metadata')
DIAGNOSTIC_VERSION = 'native-fp32-repair-v3'
V1_TOOL_SHA256 = '4924c1aa1f6bfb137b4a5a6cdf9a73bbefcf9f0ec85c6c269d16e0b332b03711'
V1_FAILED_REPORT_SHA256 = '4bd9a0d66b19e73984825eb6a8612c2ac3ae08b9a6f6f5b3ce3a90050be3cd72'
V2_TOOL_SHA256 = '01079a2d19ecaa719ec556844d379a70cbd41722fc072501ab64c061ba4b4e38'
V2_FAILED_REPORT_SHA256 = '700c5ea5841a0c3b346cc881951da3ea8d57496b9fcc1fde117c8aa903d19568'
LEGACY_SOURCE_SHA256 = 'fc0810fca79e10c4d4faaaee30733d0e78b889bf9f40f046ce7e56e1e8e30f13'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_contract(args):
    result = {}
    for name in SOURCES:
        path = getattr(args, name + '_source').resolve()
        if sha(path) != getattr(args, name + '_source_sha256'):
            raise ValueError('Original pinned source mismatch: ' + name)
        result[name] = {'path': str(path), 'sha256': sha(path)}
    if result['legacy']['sha256'] != LEGACY_SOURCE_SHA256:
        raise ValueError('Need the preserved pre-FP32-repair legacy source, not a moving reference')
    if args.metadata_source.parent.resolve() != args.backend_source.parent.resolve() or args.metadata_source.name != 'common_sdpa_backend.py':
        raise ValueError('The actual FlashInfer shim and metadata helper must be siblings')
    config = json.loads(args.config_source.read_text())
    geometry = {'num_attention_heads': HEADS, 'kv_lora_rank': LATENT, 'qk_rope_head_dim': ROPE,
                'qk_nope_head_dim': 128, 'v_head_dim': 128}
    if any(config.get(k) != v for k, v in geometry.items()) or config.get('rope_scaling', {}).get('type') != 'yarn':
        raise ValueError('Expected original R1 geometry and explicit YaRN config')
    return result, config


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def yarn_class(path):
    """Execute the exact pinned RoPE classes only; do not initialize a model."""
    import torch
    names = {'DeepseekV3RotaryEmbedding', 'DeepseekV3YarnRotaryEmbedding', 'yarn_find_correction_dim',
             'yarn_find_correction_range', 'yarn_get_mscale', 'yarn_linear_ramp_mask'}
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError('Pinned model no longer contains the reviewed YaRN definitions')
    namespace = {'torch': torch, 'nn': torch.nn, 'math': math, 'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['DeepseekV3YarnRotaryEmbedding']


def reference_rotation(raw, cosine, sine):
    """Independent CPU FP64 pair formula, rounded once to the input dtype."""
    import torch
    x = raw.cpu().double().reshape(*raw.shape[:-1], ROPE // 2, 2)
    c, s = cosine.cpu().double()[..., :ROPE // 2], sine.cpu().double()[..., :ROPE // 2]
    while c.ndim < x.ndim - 1:
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    even = x[..., 0] * c - x[..., 1] * s
    odd = x[..., 0] * s + x[..., 1] * c
    return torch.stack((even, odd), dim=-1).flatten(-2).to(raw.dtype)


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def max_ulps(actual, reference):
    import torch
    a, b = actual.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or a.shape != b.shape:
        raise ValueError('ULP comparison requires same BF16 shape')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite values in a finite fixture')
    def ordered(t):
        bits = t.view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)
    return int((ordered(a) - ordered(b)).abs().max())


def component_metrics(actual, reference, *, raw=None, cosine=None, sine=None):
    """Read diagnostic values only; retain the original one-ULP gate elsewhere."""
    import torch
    a, b = actual.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
    maximum = max_ulps(a, b)
    abits, bbits = a.view(torch.int16).int(), b.view(torch.int16).int()
    ordered_a = torch.where(abits < 0, -32768 - abits, abits)
    ordered_b = torch.where(bbits < 0, -32768 - bbits, bbits)
    ulps = (ordered_a - ordered_b).abs()
    error = (a.double() - b.double()).abs()
    flat_index = int(ulps.reshape(-1).argmax())
    remainder, indices = flat_index, []
    for size in reversed(a.shape):
        indices.append(remainder % size); remainder //= size
    indices.reverse()
    index = tuple(indices)
    worst = {'index': indices, 'actual': float(a[index]), 'reference': float(b[index]),
             'actual_bf16_bits': int(abits[index]) & 0xffff, 'reference_bf16_bits': int(bbits[index]) & 0xffff,
             'absolute_error': float(error[index]), 'BF16_ULPs': int(ulps[index])}
    if raw is not None:
        source = raw.detach().cpu().double()
        pair = indices[-1] // 2
        prefix = tuple(indices[:-1])
        even, odd = float(source[prefix + (2 * pair,)]), float(source[prefix + (2 * pair + 1,)])
        c, s = float(cosine.cpu()[pair]), float(sine.cpu()[pair])
        worst['pair_inputs'] = {'pair': pair, 'output_parity': indices[-1] % 2, 'raw_even': even, 'raw_odd': odd,
                               'cos_first': c, 'sin_first': s,
                               'cos_second': float(cosine.cpu()[pair + ROPE // 2]),
                               'sin_second': float(sine.cpu()[pair + ROPE // 2]),
                               'fp64_even_cos': even * c, 'fp64_odd_sin': odd * s,
                               'fp64_odd_cos': odd * c, 'fp64_even_sin': even * s,
                               'fp64_unrounded_even': even * c - odd * s,
                               'fp64_unrounded_odd': odd * c + even * s}
    return {'elements': a.numel(), 'max_BF16_ULPs': maximum, 'mismatching_elements': int((abits != bbits).sum()),
            'over_one_ULP_elements': int((ulps > 1).sum()), 'max_absolute_error': float(error.max()),
            'actual_sha256': tensor_sha(a), 'reference_sha256': tensor_sha(b), 'worst_by_ULP': worst}


def staged_rotation_models(raw, cosine, sine):
    """Diagnostic hypotheses, not accepted or substituted numerical references.

    All products start from the same BF16 input values. Explicit BF16 casts
    expose possible intermediate rounding; no model is inferred from a match
    alone and none changes the fresh FP64 acceptance reference.
    """
    import torch
    source = raw.cpu().float().reshape(*raw.shape[:-1], ROPE // 2, 2)
    c, s = cosine.cpu().float()[:ROPE // 2], sine.cpu().float()[:ROPE // 2]
    even, odd = source[..., 0], source[..., 1]
    a, b = even * c, -odd * s
    d, e = odd * c, even * s
    rounded = lambda value: value.bfloat16().float()
    native = lambda first, second: torch.stack((first, second), dim=-1).flatten(-2).bfloat16()
    return {'FP32_products_and_sum': native(a + b, d + e),
            'both_products_rounded_BF16_before_sum': native(rounded(a) + rounded(b), rounded(d) + rounded(e)),
            'first_product_rounded_BF16_then_other_product_plus_sum': native(rounded(a) + b, rounded(d) + e),
            'second_product_rounded_BF16_then_other_product_plus_sum': native(a + rounded(b), d + rounded(e))}


def diagnostic_components(rotary, q_fixed, kv_fixed, raw_q, raw_k, position, cos, sin,
                          oracle_q, fresh_kv, q_legacy_native, kv_legacy_native):
    import torch
    pos = torch.tensor([[position]], device='cuda', dtype=torch.int64)
    # This is the actual pinned main helper on independent copies of original
    # inputs, on both CPU and GPU; never feed candidate output into its oracle.
    def main_helper(raw, gpu):
        value = raw.cuda() if gpu else raw.cpu()
        c, s, p = (cos, sin, pos) if gpu else (cos.cpu(), sin.cpu(), pos.cpu())
        return rotary.rotary_pos_emb_interleaved_native(value.view(1, 1, -1, ROPE), c, s, p, 2).view_as(value)
    raw_k_rope = raw_k.cpu().reshape(-1)[LATENT:]
    q_cpu, q_gpu = main_helper(raw_q, False), main_helper(raw_q, True)
    k_cpu, k_gpu = main_helper(raw_k_rope, False), main_helper(raw_k_rope, True)
    q_observed, k_observed = q_fixed.view(HEADS, ROPE), kv_fixed.reshape(-1)[LATENT:]
    c, s = cos[position].cpu(), sin[position].cpu()
    q_kwargs, k_kwargs = {'raw': raw_q, 'cosine': c, 'sine': s}, {'raw': raw_k_rope, 'cosine': c, 'sine': s}
    result = {'Q_vs_independent_FP64': component_metrics(q_observed, oracle_q, **q_kwargs),
              'K_rope_vs_fresh_main_helper': component_metrics(k_observed, fresh_kv[LATENT:], **k_kwargs),
              'latent_vs_independent_FP64_RMS': component_metrics(kv_fixed.reshape(-1)[:LATENT], fresh_kv[:LATENT]),
              'Q_vs_actual_main_Torch_CPU': component_metrics(q_observed, q_cpu, **q_kwargs),
              'Q_vs_actual_main_Torch_GPU': component_metrics(q_observed, q_gpu, **q_kwargs),
              'K_rope_vs_actual_main_Torch_CPU': component_metrics(k_observed, k_cpu, **k_kwargs),
              'K_rope_vs_actual_main_Torch_GPU': component_metrics(k_observed, k_gpu, **k_kwargs),
              'main_Torch_Q_CPU_vs_GPU': component_metrics(q_cpu, q_gpu),
              'main_Torch_K_CPU_vs_GPU': component_metrics(k_cpu, k_gpu),
              'legacy_inverse_permuted_Q_vs_independent_FP64': component_metrics(q_legacy_native.view(HEADS, ROPE), oracle_q),
              'legacy_inverse_permuted_K_rope_vs_fresh': component_metrics(kv_legacy_native.reshape(-1)[LATENT:], fresh_kv[LATENT:]),
              'rounding_hypotheses_NOT_acceptance_references': {'Q': {}, 'K_rope': {}}}
    for label, raw, observed in (('Q', raw_q, q_observed), ('K_rope', raw_k_rope, k_observed)):
        for name, predicted in staged_rotation_models(raw, c, s).items():
            result['rounding_hypotheses_NOT_acceptance_references'][label][name] = component_metrics(observed, predicted)
    return result


def raw_fixture(length):
    import torch
    position = torch.arange(length)[:, None]
    feature = torch.arange(LATENT)[None, :]
    # Exactly half ones and half twos: constant norm, positive outputs and
    # position-dependent values make the wrong attention weights observable.
    latent = torch.where(((feature // 128 + (position % 11 < 5).int()) % 2) == 0, 1.0, 2.0).bfloat16()
    rope = (((position * 3 + torch.arange(ROPE)[None, :]) % 11) - 5).float().div(2).bfloat16()
    return torch.cat((latent, rope), dim=-1)


def raw_query(position):
    import torch
    return (((torch.arange(HEADS)[:, None] * 3 + torch.arange(ROPE)[None, :] * 2 + position) % 13) - 6).float().div(2).bfloat16()


def cpu_attention(query, history, scale):
    """Full same-prefix CPU FP64 scores/softmax/value reference."""
    import torch
    q, kv = query.cpu().double(), history.cpu().double()
    scores = torch.matmul(q, kv[:, LATENT:].T) * scale
    probability = torch.softmax(scores, dim=-1)
    return probability @ kv[:, :LATENT], scores


def execute(args, report, config):
    import torch
    import importlib.metadata
    if os.environ.get('R1_ROPE_HANDOFF_GUARDED') != '1' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Require the declared host and owned GPU monitor marker')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (10, 3):
        raise ValueError('Require exactly one assigned SM103 GPU')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory), 0)
    report['runtime'] = {name: importlib.metadata.version(name) for name in ('torch', 'triton', 'flashinfer-python')}
    kernel = load(args.kernel_source, 'r1_exact_rope_handoff_kernel')
    legacy_kernel = load(args.legacy_source, 'r1_frozen_rope_handoff_legacy_kernel')
    rotary = load(args.rotary_source, 'r1_exact_rope_handoff_rotation')
    package = ModuleType('r1_rope_handoff_fi')
    package.__path__ = [str(args.backend_source.resolve().parent)]
    sys.modules[package.__name__] = package
    backend = load(args.backend_source, package.__name__ + '.common_flashinfer_backend')
    if Path(sys.modules[package.__name__ + '.common_sdpa_backend'].__file__).resolve() != args.metadata_source.resolve():
        raise ValueError('Actual metadata import differs from pinned helper')
    rope_config = config['rope_scaling']
    klass = yarn_class(args.model_source)
    max_length = max(PROMPTS) + STEPS
    yarn = klass(ROPE, max_position_embeddings=max_length, base=config['rope_theta'], device=torch.device('cuda:0'),
                 scaling_factor=rope_config['factor'], original_max_position_embeddings=rope_config['original_max_position_embeddings'],
                 beta_fast=rope_config['beta_fast'], beta_slow=rope_config['beta_slow'],
                 mscale=rope_config['mscale'], mscale_all_dim=rope_config['mscale_all_dim'])
    cos, sin = yarn(torch.empty(1, 1, 1, ROPE, device='cuda', dtype=torch.bfloat16), seq_len=max_length)
    cos_cpu, sin_cpu = cos.cpu(), sin.cpu()
    scale = (192 ** -0.5) * (0.1 * rope_config['mscale_all_dim'] * math.log(rope_config['factor']) + 1.0) ** 2
    norm_weight = torch.ones(LATENT, device='cuda', dtype=torch.bfloat16)
    failures, rows = [], []
    legacy_output_failures = 0
    report['cases'] = rows
    for prompt in PROMPTS:
        length = prompt + STEPS
        raw = raw_fixture(length)
        normalized = (raw[:, :LATENT].double() / (raw[:, :LATENT].double().square().mean(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        # Fresh prefill uses the exact model helper, independently of the
        # candidate fused decode and its post-permutation.
        fresh_k = rotary.rotary_pos_emb_interleaved_native(raw[:, LATENT:].view(1, length, 1, ROPE),
                   cos_cpu, sin_cpu, torch.arange(length).view(1, length), 2).view(length, ROPE)
        oracle_k = reference_rotation(raw[:, LATENT:], cos_cpu[:length], sin_cpu[:length])
        if not torch.equal(fresh_k, oracle_k):
            failures.append('fresh native RoPE differs from independent pair formula')
        fresh_history = torch.cat((normalized, fresh_k), dim=-1)
        pages = (length + PAGE - 1) // PAGE
        # Reversed physical pages exercise real indirection, with NaN tails.
        physical = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device='cuda')
        table = physical.view(1, -1)
        cache = torch.full((pages, PAGE, 1, LATENT + ROPE), float('nan'), device='cuda', dtype=torch.bfloat16)
        old_cache = cache.clone()
        positions = torch.arange(prompt, device='cuda')
        cache[physical[positions // PAGE].long(), positions % PAGE, 0] = fresh_history[:prompt].cuda()
        old_cache.copy_(cache)
        for step in range(STEPS):
            position = prompt + step
            raw_q = raw_query(position)
            oracle_q = reference_rotation(raw_q, cos_cpu[position], sin_cpu[position])
            q_new = raw_q.cuda().view(1, HEADS, 1, ROPE).clone()
            q_old = q_new.clone()
            q_baseline = q_new.clone()
            raw_k = raw[position].cuda().view(1, 1, LATENT + ROPE)
            raw_k_sha = tensor_sha(raw_k)
            position_tensor = torch.tensor([[position]], dtype=torch.int64, device='cuda')
            legacy = kernel.fused_rmsnorm_rope_with_q(raw_k, q_old, cos, sin, position_tensor, norm_weight, LATENT, ROPE)
            baseline = legacy_kernel.fused_rmsnorm_rope_with_q(raw_k, q_baseline, cos, sin, position_tensor, norm_weight, LATENT, ROPE)
            fixed = kernel.fused_rmsnorm_rope_with_q(raw_k, q_new, cos, sin, position_tensor, norm_weight, LATENT, ROPE,
                                                   output_layout='native_interleaved')
            # V3 changes Q/K arithmetic deliberately to match fresh prefill.
            # Preserve the old layout-only comparison as evidence, but require
            # byte equality only for the unchanged default and latent RMS.
            def unpermute(x):
                return torch.stack((x[..., :ROPE // 2], x[..., ROPE // 2:]), -1).flatten(-2)
            layout_only_bitwise = (torch.equal(fixed[..., :LATENT], legacy[..., :LATENT])
                       and torch.equal(fixed[..., LATENT:], unpermute(legacy[..., LATENT:]))
                       and torch.equal(q_new, unpermute(q_old)) and tensor_sha(raw_k) == raw_k_sha)
            legacy_unchanged = torch.equal(legacy, baseline) and torch.equal(q_old, q_baseline)
            latent_unchanged = torch.equal(fixed[..., :LATENT], legacy[..., :LATENT])
            input_unchanged = tensor_sha(raw_k) == raw_k_sha
            if not legacy_unchanged or not latent_unchanged or not input_unchanged:
                failures.append('legacy default, latent RMS or input bytes changed')
            diagnostic = diagnostic_components(rotary, q_new, fixed, raw_q, raw_k, position, cos, sin,
                                               oracle_q, fresh_history[position], unpermute(q_old),
                                               torch.cat((legacy[..., :LATENT], unpermute(legacy[..., LATENT:])), dim=-1))
            if (diagnostic['Q_vs_actual_main_Torch_GPU']['mismatching_elements']
                    or diagnostic['K_rope_vs_actual_main_Torch_GPU']['mismatching_elements']):
                failures.append('native FP32 RoPE differs bitwise from actual main helper')
            if max_ulps(q_new.view(HEADS, ROPE), oracle_q) > 1 or max_ulps(fixed.view(LATENT + ROPE), fresh_history[position]) > 1:
                failures.append('native Q/K exceeds one BF16 ULP from fresh independent reference')
            page = physical[position // PAGE]
            cache[page, position % PAGE, 0] = fixed.view(-1)
            old_cache[page, position % PAGE, 0] = legacy.view(-1)
            inclusive = torch.tensor([position + 1], device='cuda', dtype=torch.int32)
            def attend(q_rope, history_cache):
                query = torch.zeros((1, 1, HEADS, LATENT + ROPE), device='cuda', dtype=torch.bfloat16)
                query[0, 0, :, LATENT:] = q_rope.view(HEADS, ROPE)
                metadata, splits = backend.get_mla_metadata(inclusive, HEADS, 1)
                output, _ = backend.flash_mla_with_kvcache(query, history_cache, table, inclusive, LATENT,
                                  metadata, splits, scale, True)
                return output[0, 0]
            before = tensor_sha(cache)
            actual, wrong = attend(q_new, cache), attend(q_old, old_cache)
            expected, expected_scores = cpu_attention(oracle_q, fresh_history[:position + 1], scale)
            ulps = max_ulps(actual, expected.bfloat16())
            if ulps > 1 or tensor_sha(cache) != before:
                failures.append('fixed MLA differs from same-prefix FP64 reference or mutated KV')
            wrong_ulps = max_ulps(wrong, expected.bfloat16())
            legacy_output_failures += int(wrong_ulps > 1)
            logical = torch.arange(position + 1, device='cuda')
            old_history = old_cache[physical[logical // PAGE].long(), logical % PAGE, 0].cpu()
            _, wrong_scores = cpu_attention(q_old.view(HEADS, ROPE), old_history, scale)
            score_error = float((wrong_scores - expected_scores).abs().max())
            if score_error <= 1e-3:
                failures.append('negative-control mixed-layout attention scores were not reproduced')
            entry = {'prompt': prompt, 'step': step, 'position': position, 'inclusive_length': position + 1,
                     'layout_only_bitwise_vs_legacy_INFORMATIONAL_old_goal': layout_only_bitwise,
                     'legacy_default_bitwise_unchanged': legacy_unchanged,
                     'latent_bitwise_unchanged_vs_legacy': latent_unchanged,
                     'raw_input_bytes_unchanged': input_unchanged,
                     'max_BF16_ULPs_vs_FP64': ulps,
                     'legacy_wrong_layout_max_BF16_ULPs': wrong_ulps, 'cache_sha_unchanged': before,
                     'legacy_wrong_layout_max_score_error': score_error,
                     'component_diagnostics': diagnostic,
                     'output_sha256': tensor_sha(actual), 'full_logits_or_model_qualified': False}
            if prompt <= 65:
                # Same-prefix fresh packed prefill through the actual common
                # FI fa2 API. Only the final row is compared; Q_nope=0 means
                # no projection/absorption discrepancy contaminates the test.
                n = position + 1
                q = torch.zeros(n, HEADS, 192, device='cuda', dtype=torch.bfloat16)
                q[-1, :, 128:] = oracle_q.cuda()
                k = torch.zeros_like(q); k[:, :, 128:] = fresh_k[:n, None, :].cuda()
                v = fresh_history[:n, None, :128].expand(n, HEADS, 128).contiguous().cuda()
                cu = torch.tensor([0, n], device='cuda', dtype=torch.int32)
                fresh_output = backend.flash_attn_varlen_func(q, k, v, cu, cu, n, n, softmax_scale=scale, causal=True)[-1]
                fresh_ulps = max_ulps(fresh_output, expected[:, :128].bfloat16())
                entry['fresh_packed_prefill_max_BF16_ULPs_vs_FP64'] = fresh_ulps
                if fresh_ulps > 1 or max_ulps(actual[:, :128], fresh_output) > 1:
                    failures.append('incremental/fresh packed prefill mismatch')
            rows.append(entry)
        # Valid NaN must remain valid NaN; neither permutation nor attention
        # may cleanse it. Tail NaNs above must not affect finite outputs.
        nan_raw = raw[-1].cuda().view(1, 1, -1).clone(); nan_raw[0, 0, LATENT] = float('nan')
        nan_q = raw_query(length - 1).cuda().view(1, HEADS, 1, ROPE).clone()
        nan_processed = kernel.fused_rmsnorm_rope_with_q(nan_raw, nan_q, cos, sin,
             torch.tensor([[length - 1]], device='cuda'), norm_weight, LATENT, ROPE, output_layout='native_interleaved')
        if not bool(torch.isnan(nan_processed[..., LATENT]).all()):
            failures.append('valid RoPE NaN was sanitized')
        cache[physical[0], 0, 0, LATENT] = float('nan')
        nan_before = tensor_sha(cache)
        contaminated = attend(q_new, cache)
        if bool(torch.isfinite(contaminated).all()) or tensor_sha(cache) != nan_before:
            failures.append('valid cached NaN was hidden or cache was mutated')
    if not legacy_output_failures:
        failures.append('legacy output negative control never exceeded the fixed envelope')
    torch.cuda.synchronize()
    source_contract(args)
    report.update(status='passed_isolated_rope_handoff_NOT_full_model' if not failures else 'failed_isolated_rope_handoff',
                  failures=failures, legacy_output_cases_outside_envelope=legacy_output_failures,
                  max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument('--' + name + '-source', type=Path, required=True)
        parser.add_argument('--' + name + '-source-sha256', required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    sources, config = source_contract(args)
    report = {'schema': 'r1-native-rope-handoff-fixture-v3', 'status': 'plan_only_NOT_executed',
              'diagnostic_version': DIAGNOSTIC_VERSION, 'prior_v1_tool_sha256': V1_TOOL_SHA256,
              'prior_v1_failed_report_sha256': V1_FAILED_REPORT_SHA256,
              'prior_v2_tool_sha256': V2_TOOL_SHA256, 'prior_v2_failed_report_sha256': V2_FAILED_REPORT_SHA256,
              'original_one_ULP_acceptance_unchanged': True,
              'proof_goal_changed_from_layout_only': True,
              'legacy_default_and_latent_bitwise_required': True,
              'native_QK_main_FP32_bitwise_required': True,
              'tool_sha256': sha(__file__), 'sources': sources, 'prompts': list(PROMPTS), 'incremental_steps': STEPS,
              'positions': 'prompt + step; inclusive cache length = position + 1',
              'changed_proof_goal': 'native Q/K must match actual main FP32 helper; default split_half and latent RMS must remain bitwise unchanged; old layout-only comparison remains informational',
              'reference_envelope': 'one BF16 ULP from independent same-prefix CPU FP64 attention and pair rotation',
              'negative_control': 'every mixed-layout case must alter scores; at least one legacy output must exceed the fixed envelope',
              'full_model_qualified': False, 'formal_performance_permitted': False,
              'required_monitor': {'max_seconds': 900, 'GPU_bytes': 8 * 1024**3, 'RSS_bytes': 8 * 1024**3}}
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        started = time.time_ns()
        try:
            execute(args, report, config)
        except Exception as error:
            report.update(status='failed_isolated_rope_handoff', error=type(error).__name__ + ': ' + str(error))
        report.update(started_unix_ns=started, finished_unix_ns=time.time_ns())
        with args.output.open('x') as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report if not args.execute else {'status': report['status'], 'output': str(args.output), 'sha256': sha(args.output)}))
    if args.execute and report['status'] != 'passed_isolated_rope_handoff_NOT_full_model':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
