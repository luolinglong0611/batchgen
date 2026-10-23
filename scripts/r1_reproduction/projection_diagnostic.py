#!/usr/bin/env python3
"""Observe an unchanged R1 layer run, with independent and same-input shadows.

All original backend/reference return values and the original gate are preserved.
FP32/FP64 shadows are diagnostics only and are never fed into either main chain.
Requires the allocated SM103 container and DISABLE_MMA_V5=1. Use the same seed,
checkpoint shard and at least three decode steps as the failing qualification.
Full boundary tensors and primary failure coordinates are retained beside JSON.
"""
import argparse
import json
from pathlib import Path
import traceback
import torch
import attention_layer_qualification as layer_test


def install_diagnostics(backend, result, tensors):
    restore = []
    hooks = []
    state = {'ref_phase': 'prefill', 'dut_phase': 'prefill', 'ref_step': -1, 'dut_step': -1,
             'prefill_call': 0, 'label': 'initial', 'pending_quant': None}
    reference = {}
    original_einsum = torch.einsum

    def replace(obj, name, value):
        restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def metrics(actual, expected, label):
        label = f'{state["label"]}/{label}'
        comparison_dtype = torch.float64 if torch.float64 in (actual.dtype, expected.dtype) else torch.float32
        a, b = actual.to(comparison_dtype), expected.to(comparison_dtype)
        entry = {'label': label, 'actual_shape': list(a.shape), 'reference_shape': list(b.shape),
                 'actual_dtype': str(actual.dtype), 'reference_dtype': str(expected.dtype)}
        tensors[label] = {'actual': actual.detach().cpu(), 'reference': expected.detach().cpu()}
        if a.shape != b.shape:
            entry['shape_match'] = False
        elif not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            entry['finite'] = False
        else:
            error = a - b
            tensors[label]['error'] = error.cpu()
            a64, b64 = a.double().flatten(), b.double().flatten()
            cosine = (a64 @ b64) / (a64.norm() * b64.norm()).clamp_min(1e-300)
            entry.update(shape_match=True, finite=True, equal_fraction=float((a == b).float().mean()),
                         unequal_count=int((a != b).sum()),
                         relative_l2=float(error.norm() / b.norm().clamp_min(1e-30)),
                         max_abs=float(error.abs().max()), reference_rms=float(b.square().mean().sqrt()),
                         cosine_similarity=float(cosine.clamp(-1, 1)))
        result.setdefault('operation_diagnostics', []).append(entry)
        print(json.dumps(entry, allow_nan=False), flush=True)

    def current_reference():
        return reference[state['dut_phase']]

    def quantized_dot(quantized, weight, scales):
        values, activation_scales = quantized
        output = torch.zeros((values.shape[0], weight.shape[0]), device=values.device, dtype=torch.float32)
        rows = torch.arange(weight.shape[0], device=values.device) // 128
        for block, start in enumerate(range(0, values.shape[1], 128)):
            dot = values[:, start:start+128].float() @ weight[:, start:start+128].float().T
            output = output + dot * (activation_scales[:, block, None] * scales[rows, block][None])
        return output.bfloat16()

    ref_prefill, ref_decode = layer_test.ReferenceLayer.prefill, layer_test.ReferenceLayer.decode
    ref_projection, ref_qkv = layer_test.ReferenceLayer.projection, layer_test.ReferenceLayer.qkv
    ref_absorb = layer_test.ReferenceLayer.absorb_weight

    def ref_p(self, name, x):
        output = ref_projection(self, name, x)
        reference[state['ref_phase']].setdefault('projections', {})[name] = (x.detach().clone(), output.detach().clone())
        return output

    def ref_q(self, hidden, positions):
        output = ref_qkv(self, hidden, positions)
        reference[state['ref_phase']]['qkv'] = tuple(value.detach().clone() for value in output)
        return output

    def ref_a(self):
        output = ref_absorb(self)
        reference[state['ref_phase']]['absorb'] = output.detach().clone()
        return output

    def ref_pre(self, *args, **kwargs):
        state['ref_phase'] = 'prefill'
        reference['prefill'] = {}
        return ref_prefill(self, *args, **kwargs)

    def ref_dec(self, hidden, positions, histories):
        state['ref_step'] += 1
        state['ref_phase'] = f'decode_{state["ref_step"]}'
        reference[state['ref_phase']] = {}
        output = ref_decode(self, hidden, positions, histories)
        reference[state['ref_phase']]['histories'] = [history.detach().clone() for history in histories]
        return output

    replace(layer_test.ReferenceLayer, 'projection', ref_p)
    replace(layer_test.ReferenceLayer, 'qkv', ref_q)
    replace(layer_test.ReferenceLayer, 'absorb_weight', ref_a)
    replace(layer_test.ReferenceLayer, 'prefill', ref_pre)
    replace(layer_test.ReferenceLayer, 'decode', ref_dec)

    original_pre = backend.mla_prefill_sm103_w8a16_prepacked
    original_dec = backend.mla_decoding_sm103_bf16_with_pagekv

    def pre(*args, **kwargs):
        state['dut_phase'] = 'prefill'
        state['prefill_call'] += 1
        state['label'] = f'prefill_call_{state["prefill_call"]}'
        return original_pre(*args, **kwargs)

    def dec(*args, **kwargs):
        state['dut_step'] += 1
        state['dut_phase'] = f'decode_{state["dut_step"]}'
        state['label'] = state['dut_phase']
        return original_dec(*args, **kwargs)

    replace(backend, 'mla_prefill_sm103_w8a16_prepacked', pre)
    replace(backend, 'mla_decoding_sm103_bf16_with_pagekv', dec)
    original_quant, original_p, original_d = backend._quantize, backend._project_prefill, backend._project_decode

    def quant(x):
        output = original_quant(x)
        expected = layer_test.reference_quantize(x)
        state['pending_quant'] = x.detach().clone()
        # Projection hooks label cross-chain quantization; this same-input
        # check alone cannot establish that the main chains received equal x.
        state['quant_count'] = state.get('quant_count', 0) + 1
        metrics(output[0], expected[0], f'quant_{state["quant_count"]}_same_input_values')
        metrics(output[1], expected[1], f'quant_{state["quant_count"]}_same_input_scales')
        return output

    def project_prefill(module, name, x, scales):
        output = original_p(module, name, x, scales)
        ref_x, ref_out = current_reference()['projections'][name]
        metrics(x, ref_x, f'{name}_independent_input_bf16')
        quantized, q_scales = layer_test.reference_quantize(x)
        ref_quantized, ref_scales = layer_test.reference_quantize(ref_x)
        metrics(quantized, ref_quantized, f'{name}_independent_input_fp8')
        metrics(q_scales, ref_scales, f'{name}_independent_input_scales')
        metrics(output, ref_out.bfloat16(), f'{name}_independent_output_bf16')
        expected = layer_test.reference_gemm(x, getattr(module, name).weight, scales[name+'.weight_scale_inv']).bfloat16()
        metrics(output, expected, f'{name}_same_input_gemm')
        return output

    def project_decode(module, name, x, scales):
        output = original_d(module, name, x, scales)
        ref_x, ref_out = current_reference()['projections'][name]
        expected_quantized = layer_test.reference_quantize(ref_x)
        metrics(state['pending_quant'], ref_x, f'{name}_independent_input_bf16')
        metrics(x[0], expected_quantized[0], f'{name}_independent_input_fp8')
        metrics(x[1], expected_quantized[1], f'{name}_independent_input_scales')
        metrics(output, ref_out.bfloat16(), f'{name}_independent_output_bf16')
        metrics(output, quantized_dot(x, getattr(module, name).weight, scales[name+'.weight_scale_inv']),
                f'{name}_same_input_gemm')
        return output

    replace(backend, '_quantize', quant)
    replace(backend, '_project_prefill', project_prefill)
    replace(backend, '_project_decode', project_decode)

    original_build = layer_test.build_layer

    def build(*args, **kwargs):
        layer = original_build(*args, **kwargs)
        for name in ('q_a_layernorm', 'kv_a_layernorm'):
            def norm_hook(module, inputs, output, name=name):
                x = inputs[0]
                expected = layer_test.reference_norm(x, module.weight, module.variance_epsilon)
                values = x.double()
                fp64 = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
                        * module.weight.double()).bfloat16()
                metrics(output, expected, f'{name}_same_input_fp32_shadow')
                metrics(output, fp64, f'{name}_same_input_fp64_shadow')
                ref = current_reference()
                expected_chain = ref['projections']['q_b_proj'][0] if name == 'q_a_layernorm' else ref['qkv'][2]
                metrics(output.reshape_as(expected_chain), expected_chain, f'{name}_independent_output_bf16')
            hooks.append(getattr(layer, name).register_forward_hook(norm_hook))
        return layer
    replace(layer_test, 'build_layer', build)

    def einsum(equation, *operands):
        output = original_einsum(equation, *operands)
        if equation not in ('bhd,hdc->bhc', 'bhc,hdc->bhd', 'bqhc,hdc->bqhd'):
            return output
        if all(value.dtype == torch.float32 for value in operands):
            # Reference's actual operations; observe, do not substitute shadows.
            ref = reference[state['ref_phase']]
            if equation == 'bhd,hdc->bhc':
                ref['latent_query'] = output.bfloat16().detach().clone()
            elif equation == 'bhc,hdc->bhd':
                ref['attended'] = operands[0].bfloat16().detach().clone()
            return output
        if not all(value.dtype == torch.bfloat16 for value in operands):
            return output
        label = 'query_absorb' if equation == 'bhd,hdc->bhc' else 'output_absorb'
        fp32 = original_einsum(equation, *(value.float() for value in operands))
        fp64 = original_einsum(equation, *(value.double() for value in operands))
        metrics(output, fp32.bfloat16(), f'{label}_same_input_fp32_shadow')
        metrics(output, fp64.bfloat16(), f'{label}_same_input_fp64_shadow')
        metrics(fp32, fp64, f'{label}_fp32_vs_fp64_shadow')
        ref = current_reference()
        if label == 'query_absorb':
            metrics(operands[0], ref['qkv'][0], 'query_absorb_independent_input')
            metrics(output, ref['latent_query'], 'query_absorb_independent_output')
            metrics(operands[1], ref['absorb'][:, :128], 'absorb_weight_exact')
        else:
            metrics(operands[0].squeeze(1), ref['attended'], 'output_absorb_independent_input')
            metrics(output.reshape_as(ref['projections']['o_proj'][0]), ref['projections']['o_proj'][0],
                    'output_absorb_independent_output')
        return output
    replace(torch, 'einsum', einsum)

    original_attend_prefill = backend._attend_prefill

    def attend_prefill(module, queries, keys, values, cu_seqlens, max_seqlen):
        ref = current_reference()
        q_nope, q_pe, _, k_pe = ref['qkv']
        expanded = ref['projections']['kv_b_proj'][1].bfloat16().view(-1, 128, 256)
        ref_keys, ref_values = expanded.split((128, 128), -1)
        ref_queries = torch.cat((q_nope, q_pe), -1)
        ref_keys = torch.cat((ref_keys, k_pe.expand(-1, 128, -1)), -1)
        metrics(queries, ref_queries, 'prefill_attention_independent_query')
        metrics(keys, ref_keys, 'prefill_attention_independent_key')
        metrics(values, ref_values, 'prefill_attention_independent_value')
        output = original_attend_prefill(module, queries, keys, values, cu_seqlens, max_seqlen)
        metrics(output.reshape_as(ref['projections']['o_proj'][0]),
                ref['projections']['o_proj'][0], 'prefill_attention_independent_output')
        same_input = torch.empty_like(values)
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries, boundaries[1:]):
            if end-start > 256:
                same_input[start:end] = layer_test.reference_causal_attention_chunked(
                    queries[start:end], keys[start:end], values[start:end], module.softmax_scale)
            else:
                q, k, v = [tensor[start:end].float().transpose(0, 1) for tensor in (queries, keys, values)]
                causal = torch.ones(end-start, end-start, dtype=torch.bool, device=queries.device).triu(1)
                probabilities = ((q @ k.transpose(-1, -2)) * module.softmax_scale).masked_fill(causal, -torch.inf).softmax(-1)
                same_input[start:end] = (probabilities @ v).transpose(0, 1).bfloat16()
        metrics(output, same_input, 'prefill_attention_same_input_fp32_shadow')
        return output

    replace(backend, '_attend_prefill', attend_prefill)

    original_attend = backend._attend_decode

    def attend(module, query, cache, table, lengths, max_seqlen):
        ref = current_reference()
        expected_query = torch.cat((ref['latent_query'], ref['qkv'][1]), -1).unsqueeze(1)
        metrics(query, expected_query, 'attention_independent_query')
        actual_histories = []
        for row, length in enumerate(lengths.tolist()):
            positions = torch.arange(length, device=cache.device)
            actual_history = cache[table[row, positions // 64].long(), positions % 64, 0]
            metrics(actual_history, ref['histories'][row], f'valid_cache_row_{row}')
            actual_histories.append(actual_history)
        output = original_attend(module, query, cache, table, lengths, max_seqlen)
        metrics(output.squeeze(1), ref['attended'], 'attention_independent_output')
        same_input = []
        for row, history in enumerate(actual_histories):
            keys = history.float()
            probabilities = ((query[row, 0].float() @ keys.T) * module.softmax_scale).softmax(-1)
            same_input.append((probabilities @ keys[:, :512]).bfloat16())
        metrics(output.squeeze(1), torch.stack(same_input), 'attention_same_input_fp32_shadow')
        return output
    replace(backend, '_attend_decode', attend)

    original_check = layer_test.check

    def check(actual, expected, record, reports):
        try:
            return original_check(actual, expected, record, reports)
        finally:
            if actual.shape == expected.shape and actual.dtype == torch.bfloat16:
                oracle = expected.bfloat16().float()
                error = actual.float() - oracle
                tolerance = layer_test.ATOL + layer_test.RTOL * oracle.abs()
                indices = (error.abs() > tolerance).nonzero()
                tensors['primary/'+record['case']] = {'actual': actual.cpu(), 'fp32_reference': expected.cpu(),
                                                       'bf16_oracle': oracle.cpu(), 'error': error.cpu(),
                                                       'failure_indices': indices.cpu()}
                a, b = actual.float().flatten(), expected.float().flatten()
                a64, b64 = a.double(), b.double()
                result.setdefault('primary_error_geometry', []).append({
                    'case': record['case'], 'relative_l2_to_fp32': float((a-b).norm() / b.norm().clamp_min(1e-30)),
                    'cosine_to_fp32': float(((a64 @ b64) / (a64.norm() * b64.norm()).clamp_min(1e-300)).clamp(-1, 1)),
                    'reference_rms': float(b.square().mean().sqrt()),
                    'relative_l2_to_bf16': float(error.norm() / oracle.norm().clamp_min(1e-30))})
                details = []
                for index in indices[:256].tolist():
                    key = tuple(index)
                    details.append({'index': index, 'actual': float(actual[key]), 'fp32_reference': float(expected[key]),
                                    'bf16_oracle': float(oracle[key]), 'error': float(error[key]),
                                    'tolerance': float(tolerance[key])})
                result.setdefault('primary_failure_details', []).append({
                    'case': record['case'], 'failure_count': len(indices), 'first_256_failures': details})
    replace(layer_test, 'check', check)

    def cleanup():
        for hook in hooks:
            hook.remove()
        for obj, name, original in reversed(restore):
            setattr(obj, name, original)
    return cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--weights-shard', type=Path)
    parser.add_argument('--weights-sha256')
    parser.add_argument('--layer-index', type=int, default=0)
    parser.add_argument('--decode-steps', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--profile', choices=('small', 'standard'), default='small')
    args = parser.parse_args()
    if args.decode_steps < 3:
        parser.error('At least three decode steps are required to reproduce decode1 and check decode2')
    tensor_path = args.report.with_suffix('.tensors.pt')
    if args.report.exists() or tensor_path.exists():
        parser.error('Use fresh report/tensor paths to preserve prior failures')
    args.attention_implementation = 'triton_precision'
    result = {'cases': [], 'status': 'failed', 'scope': __doc__, 'raw_tensors': str(tensor_path),
              'diagnostic_source_sha256': layer_test.sha256_file(__file__),
              'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    tensors = {}
    backend = layer_test.isolated_backend()
    cleanup = install_diagnostics(backend, result, tensors)
    try:
        with torch.inference_mode():
            layer_test.run(args, result)
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        cleanup()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if tensors:
            torch.save(tensors, tensor_path)
        else:
            result['raw_tensors'] = None
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'layer_status': result['status'], 'operation_count': len(result.get('operation_diagnostics', []))}), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
