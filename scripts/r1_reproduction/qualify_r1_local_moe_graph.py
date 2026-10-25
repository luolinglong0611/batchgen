"""Real-weight parity and synchronization checks for R1 fixed-capacity MoE.

No model-service requests or collectives. Compare the exact preceding dynamic
slice path with GPU counts and graph replay, including changing route metadata.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import types
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'candidate_model', 'candidate_graph', 'weights_root', 'report'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--graph-sha256', required=True)
    a = parser.parse_args()
    assert not a.report.exists()
    assert hashlib.sha256(a.candidate_model.read_bytes()).hexdigest() == a.model_sha256
    assert hashlib.sha256(a.candidate_graph.read_bytes()).hexdigest() == a.graph_sha256
    sys.path[:0] = [str(a.repo), str(a.repo / 'scripts/r1_reproduction')]
    import torch
    from safetensors import safe_open
    from convert_verified_shards import validate_manifest, download_ready, verify_source, INDEX_NAME, MANIFEST_SHA256, WEIGHT_REVISION
    assert torch.cuda.get_device_capability(0) == (10, 3)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    graph = load('batchgen.moe.r1_local_graph', a.candidate_graph)
    model = load('batchgen.models.deepseek.deepseekv3.modeling_r1_candidate', a.candidate_model)
    root = a.weights_root
    entries = validate_manifest(a.repo / 'scripts/r1_reproduction/r1_model_manifest.json')
    assert download_ready(root, entries[INDEX_NAME])
    verify_source(root / INDEX_NAME, entries[INDEX_NAME])
    index = json.loads((root / INDEX_NAME).read_bytes())
    selected = {}
    names = [f'model.layers.3.mlp.experts.{e}.{part}.{suffix}'
             for e in range(16) for part in ('gate_proj', 'up_proj', 'down_proj')
             for suffix in ('weight', 'weight_scale_inv')]
    names += ['model.layers.3.mlp.gate.weight', 'model.layers.3.mlp.gate.e_score_correction_bias']
    for name in names:
        selected.setdefault(index['weight_map'][name], []).append(name)
    tensors, shards = {}, []
    for filename, names in selected.items():
        assert download_ready(root, entries[filename])
        verify_source(root / filename, entries[filename])
        shards.append({'filename': filename, 'sha256': entries[filename]['Sha256'], 'tensors': names})
        with safe_open(str(root / filename), framework='pt', device='cpu') as sf:
            for name in names:
                tensors[name] = sf.get_tensor(name).to('cuda')
    config = types.SimpleNamespace(hidden_size=7168, moe_intermediate_size=2048,
                                  intermediate_size=2048, hidden_act='silu', batchgen_skeleton=True,
                                  num_experts_per_tok=8, n_routed_experts=256,
                                  n_shared_experts=1, routed_scaling_factor=2.5,
                                  scoring_func='sigmoid', topk_method='noaux_tc',
                                  n_group=8, topk_group=4, norm_topk_prob=True)
    router = model.MoEGate(config).cuda()
    for name in ('weight', 'e_score_correction_bias'):
        setattr(router, name, torch.nn.Parameter(tensors['model.layers.3.mlp.gate.' + name], requires_grad=False))

    class ControlledRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = router
            self.mode = 'native'

        def moe_gate_forward_hybrid(self, hidden):
            if self.mode == 'native':
                return self.native.moe_gate_forward_hybrid(hidden)
            n = hidden.shape[0]
            ids = torch.arange(8, device=hidden.device, dtype=torch.int32).expand(n, 8)
            if self.mode == 'empty':
                ids = ids + 32
            # Weights are finite and unchanged across reference/candidate.
            weights = torch.full((n, 8), 2.5 / 8, device=hidden.device, dtype=torch.float32)
            return ids.contiguous(), weights

    moe = types.SimpleNamespace(rank=0, world_size=16, routed_expert_start_idx=0,
                                routed_expert_end_idx=16, num_experts_per_tok=8,
                                experts_per_rank=16, gate=ControlledRouter())
    def weights(part, suffix):
        return [tensors[f'model.layers.3.mlp.experts.{e}.{part}.{suffix}'] for e in range(16)]
    for name, part, suffix in [('gate_list', 'gate_proj', 'weight'), ('up_list', 'up_proj', 'weight'),
                               ('down_list', 'down_proj', 'weight'), ('gate_scale_list', 'gate_proj', 'weight_scale_inv'),
                               ('up_scale_list', 'up_proj', 'weight_scale_inv'), ('down_scale_list', 'down_proj', 'weight_scale_inv')]:
        setattr(moe, name, weights(part, suffix))
        pointer_name = name.replace('_list', '_ptrs_ptr')
        setattr(moe, pointer_name, torch.tensor([t.data_ptr() for t in getattr(moe, name)], device='cuda', dtype=torch.int64))
    moe._r1_local_moe_compute = types.MethodType(model.DeepseekV3MoE_Decoding_FP8._r1_local_moe_compute, moe)
    moe.grouped_dequant_moe_fp8 = types.MethodType(model.DeepseekV3MoE_Decoding_FP8.grouped_dequant_moe_fp8, moe)

    def shape(n):
        moe.num_tokens_per_rank = n // 16
        moe.token_idx = torch.arange(n, device='cuda', dtype=torch.int32).repeat_interleave(8)
        moe.topk_pos = torch.arange(8, device='cuda', dtype=torch.int32).repeat(n)

    def reference(x):
        ids, weights = moe.gate.moe_gate_forward_hybrid(x.unsqueeze(1))
        rows, eids, tokens, positions, counts, offsets = model.fused_moe_token_dispatch(
            x, ids.to(torch.int32), moe.token_idx, moe.topk_pos, 0, 16)
        output = moe.grouped_dequant_moe_fp8(rows, eids, counts, offsets)
        return model.scatter_weight_reduce_optimized(output, tokens, positions, weights, x.shape[0], 8).to(torch.bfloat16)

    result = {'schema': 'r1-real-local-MoE-GPU-counts-graph-qualification-v1',
              'model_sha256': a.model_sha256, 'graph_sha256': a.graph_sha256,
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'weight_source': {'layer': 3, 'experts': list(range(16)), 'manifest_sha256': MANIFEST_SHA256,
                                'revision': WEIGHT_REVISION, 'shards': shards},
              'cases': [], 'scope': 'Operator/local-graph qualification only; no full-model quality qualification.'}
    torch.manual_seed(20261025)
    with torch.inference_mode():
        for n, mode in [(16, 'native'), (256, 'native'), (512, 'native'), (1024, 'native'),
                        (512, 'skew'), (512, 'empty')]:
            shape(n); moe.gate.mode = mode
            # Changing mode changes Python control flow, so invalidate explicitly.
            moe._r1_local_graphs = {}
            x = torch.randn((n, 7168), device='cuda', dtype=torch.bfloat16)
            expected = reference(x); eager = moe._r1_local_moe_compute(x); actual = graph.forward(moe, x)
            torch.cuda.synchronize()
            assert torch.equal(expected, eager), (n, mode, 'GPU count parity')
            assert torch.equal(expected, actual), (n, mode, 'graph parity')
            assert torch.isfinite(actual).all(), (n, mode, 'finite')
            assert getattr(moe, '_r1_local_graph_captures', 0) > 0, 'Graph was not actually captured'
            retained, snapshot = actual, actual.clone()
            for _ in range(3):
                x.normal_(); expected = reference(x); actual = graph.forward(moe, x)
                torch.cuda.synchronize()
                assert torch.equal(expected, actual), (n, mode, 'dynamic inputs/routes')
                assert torch.equal(retained, snapshot), 'retained result overwritten'
            if mode == 'empty':
                assert torch.count_nonzero(actual) == 0
            timing = {k: [] for k in ('dynamic_slice', 'GPU_counts', 'graph')}
            functions = {'dynamic_slice': reference, 'GPU_counts': moe._r1_local_moe_compute,
                         'graph': lambda x: graph.forward(moe, x)}
            for label in ('dynamic_slice', 'GPU_counts', 'graph', 'graph', 'GPU_counts', 'dynamic_slice'):
                fn = functions[label]
                for _ in range(3): fn(x)
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); wall = time.perf_counter()
                for _ in range(30): fn(x)
                end.record(); end.synchronize()
                timing[label].append({'stream_ms': begin.elapsed_time(end) / 30,
                                      'wall_ms': (time.perf_counter() - wall) * 1000 / 30})
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                for _ in range(3): graph.forward(moe, x)
            scalar = sum(e.count for e in prof.key_averages() if e.key == 'aten::_local_scalar_dense')
            assert scalar == 0
            result['cases'].append({'global_tokens': n, 'routing': mode, 'bitwise_equal': True,
                                    'retained_output_unchanged': True, 'dynamic_inputs': True,
                                    'graph_scalar_readbacks_in_3_replays': scalar, 'timings': timing})
            print(json.dumps({'case': [n, mode], 'status': 'bitwise_and_scalar_check_passed',
                              'median_wall_ms': {k: statistics.median(t['wall_ms'] for t in v) for k, v in timing.items()}}), flush=True)

        shape(512); moe.gate.mode = 'native'; moe._r1_local_graphs = {}
        x = torch.randn((512, 7168), device='cuda', dtype=torch.bfloat16)
        graph.forward(moe, x)
        for name in ('scale_mutation', 'weight_mutation', 'router_mutation', 'source_replacement'):
            before = moe._r1_local_graph_captures
            if name == 'scale_mutation': moe.down_scale_list[0].mul_(1.0001)
            elif name == 'weight_mutation': moe.down_list[0].zero_()
            elif name == 'router_mutation': router.e_score_correction_bias.add_(.01)
            else:
                with torch.inference_mode(False): replacement = moe.down_list[0].clone()
                moe.down_list[0] = replacement
                moe.down_ptrs_ptr[0] = replacement.data_ptr()
            expected = reference(x); actual = graph.forward(moe, x); torch.cuda.synchronize()
            assert torch.equal(expected, actual), name
            assert moe._r1_local_graph_captures > before, name
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = reference(x); actual = graph.forward(moe, x)
        stream.synchronize(); assert torch.equal(expected, actual)
        assert len(moe._r1_local_graphs) == 2
        # Poison invalid dispatch metadata; no invalid address may reach mapping.
        tokens = torch.tensor([1, 3, -2147483648, 2147483647], device='cuda', dtype=torch.int32)
        positions = torch.tensor([2, 7, 2147483647, -2147483648], device='cuda', dtype=torch.int32)
        mapping = graph.inverse_mapping(tokens, positions, torch.tensor(2, device='cuda', dtype=torch.int32), 4, 8)
        assert mapping[1, 2] == 0 and mapping[3, 7] == 1 and torch.count_nonzero(mapping >= 0) == 2
        result['source_invalidation'] = result['per_stream_isolation'] = result['poisoned_metadata_tail'] = 'passed'
        native_dispatch = model.fused_moe_token_dispatch
        def poisoned_dispatch(*args):
            outputs = native_dispatch(*args)
            tail = torch.arange(outputs[0].shape[0], device='cuda') >= outputs[-1][-1]
            outputs[0].masked_fill_(tail[:, None], float('nan'))
            outputs[2].masked_fill_(tail, 2147483647)
            outputs[3].masked_fill_(tail, -2147483648)
            return outputs
        model.fused_moe_token_dispatch = poisoned_dispatch
        moe._r1_local_graphs = {}
        expected = reference(x); actual = graph.forward(moe, x); torch.cuda.synchronize()
        assert torch.equal(expected, actual) and torch.isfinite(actual).all()
        model.fused_moe_token_dispatch = native_dispatch
        result['poisoned_dispatched_activation_and_index_tails'] = 'passed'
        before = moe._r1_local_graph_captures
        moe.down_scale_list[1] = moe.down_scale_list[1].clone()
        moe.down_scale_ptrs_ptr[1] = moe.down_scale_list[1].data_ptr()
        expected = reference(x); actual = graph.forward(moe, x); torch.cuda.synchronize()
        assert torch.equal(expected, actual)
        assert moe._r1_local_graph_captures == before
        assert moe._r1_local_graph_unversioned_fallbacks >= 1
        result['unversioned_source_eager_fallback'] = 'passed'

    # Measure actual old/new constructor work, preserving all local parameter contracts.
    skeleton = []
    for rank in (0, 15):
        contracts, times = {}, {}
        for flag in ('0', '1'):
            os.environ['BATCHGEN_R1_SM103_LOCAL_EXPERT_SKELETON'] = flag
            with patch.object(model.dist, 'is_initialized', return_value=True), patch.object(model.dist, 'get_rank', return_value=rank), patch.object(model.dist, 'get_world_size', return_value=16):
                durations = []
                for _ in range(3):
                    begin = time.perf_counter(); instance = model.DeepseekV3MoE_Decoding_FP8(copy.copy(config), None)
                    durations.append(time.perf_counter() - begin)
                    local = {e: getattr(instance.experts[e].gate_proj, '_batchgen_parameter_shapes') for e in range(rank * 16, (rank + 1) * 16)}
                    count = sum(isinstance(e, model.DeepseekV3MLP) for e in instance.experts)
                    assert count == (256 if flag == '0' else 16)
                    del instance
                contracts[flag] = local; times[flag] = durations
        assert contracts['0'] == contracts['1']
        skeleton.append({'rank': rank, 'identical_local_parameter_shapes': True, 'old_seconds': times['0'], 'new_seconds': times['1']})
    result['local_expert_skeleton'] = skeleton
    result['status'] = 'actual_real_weight_parity_zero_graph_scalar_reads_and_invalidation_passed'
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['graph_captures'] = moe._r1_local_graph_captures
    result['graph_replays'] = moe._r1_local_graph_replays
    a.report.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
