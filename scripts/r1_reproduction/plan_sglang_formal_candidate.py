#!/usr/bin/env python3
"""Emit a reviewable R1 SGLang candidate; never launch, inspect GPUs or use HTTP.

Only local, SHA-pinned source snapshots are read. This is deliberately separate
from execute_sglang_diagnostic.py: its preparation receipts cannot authorize
these larger-capacity or two-replica configurations.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = REPO / '.repro-private/baseline-recovery/formal-source-audit'
RECEIPT_SHA = '7732753365d15c53297f88a6752bbb36bd1a9a9ac57c4d39a0a7a167f802ae1b'
DIAGNOSTIC_SHA = '86f4af26428fe98a2aafd43dceb53d655225ad1a6d818569aaca05a525bc102f'
ROUTING_PINS_SHA = '019e1524ad03c45455620fa425cb707054baa866ffeb6d77bc6936d1c34dff5f'
SOURCE_PINS = {
    'sglang/srt/server_args.py': '34dfebfe793e7e7409ee69a71c77d3550039e20ed71bff0e954e23c0042bdfdf',
    'sglang/srt/managers/tp_worker.py': 'bb33eaf8231a1ea26b86ccb67034ad49d6f673998979c6bd5c949311eaea8691',
    'sglang/srt/managers/data_parallel_controller.py': 'e7eea6ddc9f6825782c1e8c9c3be9af92e5cfd196a894cf39ad55aef1a23cbba',
    'sglang/srt/model_executor/model_runner.py': 'c16d0d2b133e0e1efd7d338034a7908e662bbadab1dcef418418c08c28910c45',
    'sglang/srt/model_executor/cuda_graph_runner.py': 'b6435c9c63252bb7c8e97a6aae601e69859c573c01c5d45bf737cf78a3e1074b',
    'sglang/srt/layers/attention/trtllm_mla_backend.py': 'ffe156f30a14228a648f36eeecc43810392cb62ec748c24c8d7f2b14fdcbc65e',
    'launch-server-help.stdout': '6796ce283ff97871da166af089028cf77c461e8ab2af266e9cb85d3afb9ccde0',
}
# Previously observed allocated-container addresses; no arbitrary remote targets.
# The coordinator must recheck identity and availability before any execution.
CONTAINER_IPS = ('10.66.9.163', '10.66.9.166')
HTTP_PORT = 30000
DIST_PORT = 33155
RUNNING_CHOICES = (16, 32, 64, 128)
CHUNK_CHOICES = (2048, 4096, 8192)
FRACTION_CHOICES = ('0.85', '0.90')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_sha(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f'Pinned local source changed or missing: {path}')


def load_inputs(audit=DEFAULT_AUDIT):
    """Verify local provenance, then import only the pinned pure-plan provider."""
    audit = Path(audit)
    receipt_path = audit / 'COMBINED_SOURCE_RECEIPT.json'
    require_sha(receipt_path, RECEIPT_SHA)
    receipt = json.loads(receipt_path.read_text())
    for name, expected in SOURCE_PINS.items():
        require_sha(audit / name, expected)
        if receipt['files'][name]['sha256'] != expected:
            raise ValueError(f'Pinned source receipt disagrees: {name}')
    diagnostic_path = Path(__file__).with_name('launch_sglang_diagnostic.py')
    routing_path = Path(__file__).with_name('sglang_diagnostic_source_pins.json')
    require_sha(diagnostic_path, DIAGNOSTIC_SHA)
    require_sha(routing_path, ROUTING_PINS_SHA)
    spec = importlib.util.spec_from_file_location('r1_frozen_sg_diagnostic', diagnostic_path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    return diagnostic, (audit / 'launch-server-help.stdout').read_text()


def replace_value(command, flag, value):
    if command.count(flag) != 1:
        raise ValueError(f'Expected one source-plan flag: {flag}')
    command[command.index(flag) + 1] = str(value)


def build_candidate(diagnostic, help_text, topology, per_dp_running,
                    per_dp_chunk, memory_fraction, execution_mode='eager',
                    cuda_graph_max_bs=None):
    if topology not in ('A', 'B'):
        raise ValueError('Topology must be A or B')
    if type(per_dp_running) is not int or per_dp_running not in RUNNING_CHOICES:
        raise ValueError('Unsupported per-DP-rank running-request target')
    if type(per_dp_chunk) is not int or per_dp_chunk not in CHUNK_CHOICES:
        raise ValueError('Unsupported per-DP-rank prefill chunk')
    if memory_fraction not in FRACTION_CHOICES:
        raise ValueError('Memory fraction must be the explicit string 0.85 or 0.90')
    if execution_mode not in ('eager', 'graph'):
        raise ValueError('Execution mode must be eager or graph')
    capture_sizes = []
    if execution_mode == 'graph':
        if type(cuda_graph_max_bs) is not int or not 1 <= cuda_graph_max_bs <= per_dp_running:
            raise ValueError('Graph maximum must be a positive integer <= per-DP running target')
        # Pinned server_args.py:849-877, non-spec default capture buckets.
        capture_sizes = [n for n in [1, 2, 4, 8, 12] + list(range(16, 129, 8))
                         if n <= cuda_graph_max_bs]
        if cuda_graph_max_bs not in capture_sizes:
            raise ValueError('Graph maximum must be an actual default capture bucket')
    elif cuda_graph_max_bs is not None:
        raise ValueError('Eager mode must not specify a CUDA graph maximum')
    dp = 16 if topology == 'A' else 8
    services = 1 if topology == 'A' else 2
    mode_id = 'eager' if execution_mode == 'eager' else f'graph-bs{cuda_graph_max_bs}'
    candidate_id = f'{topology}-c{per_dp_running}-m{memory_fraction.replace(".", "")}-k{per_dp_chunk}-{mode_id}'
    nodes = []
    for physical_node in (0, 1):
        # Empty inherited mapping: no credentials or caller-specific environment
        # are serialized. Execution must replace its environment with this map.
        env, _, command = diagnostic.plan(diagnostic.ROOT, physical_node, {})
        env.update(NCCL_RAS_ENABLE='0', HOME='/home/admin')
        endpoint_node = 0 if topology == 'A' else physical_node
        changes = {
            '--tp-size': dp, '--dp-size': dp, '--ep-size': 1,
            '--nnodes': 2 if topology == 'A' else 1,
            '--node-rank': physical_node if topology == 'A' else 0,
            '--dist-init-addr': f'{CONTAINER_IPS[endpoint_node]}:{DIST_PORT}',
            '--chunked-prefill-size': per_dp_chunk * dp,
            '--max-running-requests': per_dp_running * dp,
            '--mem-fraction-static': memory_fraction,
            # trtllm_mla rejects explicit bf16/bfloat16 in this pinned release.
            # auto is required; effective torch.bfloat16 is an admission check.
            '--kv-cache-dtype': 'auto',
        }
        for flag, value in changes.items():
            replace_value(command, flag, value)
        limit_index = command.index('--max-total-tokens')
        del command[limit_index:limit_index + 2]
        if execution_mode == 'graph':
            if command.count('--disable-cuda-graph') != 1:
                raise ValueError('Expected the original diagnostic eager flag exactly once')
            command.remove('--disable-cuda-graph')
            command.extend(['--cuda-graph-max-bs', str(cuda_graph_max_bs)])
        diagnostic.check_help(command, help_text)
        review_argv = ['env', '-i'] + [f'{key}={value}' for key, value in sorted(env.items())] + command
        nodes.append({
            'physical_node': physical_node,
            'expected_container_hostname': diagnostic.HOSTS[physical_node],
            'expected_container_ip': CONTAINER_IPS[physical_node],
            'service_index': 0 if topology == 'A' else physical_node,
            'local_gpu_ordinals': list(range(8)),
            'environment_policy': 'replace_with_exact_allowlisted_environment',
            'environment': env, 'argv': command,
            'review_only_shell_command': shlex.join(review_argv),
            'routing_compat': diagnostic.routing_compat_plan(diagnostic.ROOT, physical_node),
            'effective_per_dp_rank_running_requests_target': per_dp_running,
            'effective_per_dp_rank_prefill_chunk_tokens': per_dp_chunk,
            'actual_running_request_limit': None,
            'request_pool_slots_per_rank_requested': per_dp_running * dp,
            'request_pool_note': 'Pinned runner receives undivided CLI max-running; A/B metadata costs differ.',
            'actual_request_pool_slots_per_rank': None,
            'actual_kv_tokens_per_rank': None,
            'reserved_known_ports_to_recheck': [HTTP_PORT, *range(DIST_PORT, DIST_PORT + 6), DIST_PORT + 13],
        })
    return {
        'schema': 'r1-sglang-formal-candidate-plan-v1',
        'status': 'PLAN_ONLY_NOT_ADMITTED_NOT_EXECUTED',
        'candidate_id': candidate_id,
        'baseline_identity': diagnostic.BASELINE_IDENTITY,
        'launch_authorized': False, 'model_quality_qualified': False,
        'resource_capacity_qualified': False, 'performance_qualified': False,
        'topology': topology, 'total_gpus': 16, 'physical_nodes': 2,
        'independent_model_services': services,
        'gpus_per_service': dp,
        'total_dp_ranks': 16,
        'cluster_running_requests_target': per_dp_running * 16,
        'cluster_prefill_chunk_tokens_target': per_dp_chunk * 16,
        'cuda_graph_policy': {
            'mode': execution_mode, 'max_bs_per_dp_rank': cuda_graph_max_bs,
            'capture_batch_sizes_expected': capture_sizes,
            'dp_decode_bucket_selection': 'smallest captured bucket >= maximum local DP decode batch',
            'padding_mode_if_enabled': 'MAX_LEN',
            'max_gathered_moe_tokens_per_service_expected': cuda_graph_max_bs * dp if capture_sizes else None,
            'prefill_graph_enabled': False, 'actual_capture_replay_qualified': False,
            'memory_admission': 'Explicit mem_fraction skips automatic graph reserve; measure graph_mem_usage and free memory.',
        },
        'kv_policy': {
            'cli_dtype': 'auto', 'required_effective_dtype_every_rank': 'torch.bfloat16',
            'max_total_tokens_argument': None,
            'allocation': 'profiled after weight load; no diagnostic 32768-token cap',
            'memory_fraction_static': memory_fraction,
            'fraction_meaning': '(weights + KV pool) / GPU capacity, not KV-only fraction',
            'mla_bytes_per_token_per_rank': 61 * (512 + 64) * 2,
            'actual_capacity_known': False,
            'request_target_is_guaranteed_capacity': False,
        },
        'fixed_runtime': {
            'original_fp8_weights': str(diagnostic.WEIGHTS),
            'weight_manifest_sha256': diagnostic.MANIFEST_SHA,
            'weight_scales': 'original F32 block128; no UE8M0 conversion selected',
            'dtype': 'bfloat16', 'attention': 'trtllm_mla',
            'moe': 'flashinfer_trtllm', 'expert_parallel_size': 1,
            'cuda_graphs': execution_mode == 'graph', 'radix_cache': False, 'speculative_decoding': False,
            'shared_experts_fusion': False,
            'effective_schedule_conservativeness_from_defaults': 0.3,
            'max_prefill_tokens_from_defaults': 16384,
            'piecewise_cuda_graphs_from_defaults': False,
            'torch_compile_from_defaults': False,
            'expected_package_versions_not_reinspected': diagnostic.VERSIONS,
            'nccl_ras_enable': '0',
            'nccl_ras_reason': 'same allocated hardware stability adaptation; no library replacement',
            'nccl_binary_version_same_as_batchgen': None,
            'nccl_binary_version_comparison': 'unknown; independently inspect loaded libraries before admission',
        },
        'nodes': nodes,
        'client_contract': {
            'endpoint_urls': [f'http://{CONTAINER_IPS[i]}:{HTTP_PORT}/v1'
                              for i in range(services)],
            'frozen_workload_changed': False,
            'formal_requests_per_profile_cluster': 6000,
            'formal_requests_per_service': 6000 // services,
            'formal_submission': 'one async-burst client, deterministic round-robin across service endpoints',
            'formal_ignore_eos': True,
            'bct': 'one cluster clock: first network attempt to last complete response body',
            'combine_replica_bct_by_adding_or_averaging': False,
            'minimum_warm_repetitions': 3,
        },
        'local_source_provenance': {
            'candidate_planner_sha256': sha(__file__),
            'combined_source_receipt_sha256': RECEIPT_SHA,
            'checked_semantic_source_sha256': SOURCE_PINS,
            'diagnostic_plan_sha256': DIAGNOSTIC_SHA,
            'routing_pins_sha256': ROUTING_PINS_SHA,
            'live_container_source_verified_now': False,
        },
        'required_before_execution': [
            'New parent-coordinated admission; diagnostic executor receipts do not authorize this plan.',
            'Both allocated containers and all 16 GPUs released; node identity, ports, disk and memory checked.',
            'Current original weight receipts, routing patch, runtime sources and loaded library identities checked.',
            'Transport and selected kernels qualified for the exact candidate; NCCL RAS0 alone is not model qualification.',
            'Record effective torch.bfloat16 KV and profiled KV tokens on every rank.',
            'Measure actual request-pool cap, host/GPU peaks and sustained concurrency; auto KV is not an OOM guarantee.',
            'Resolve model-quality admission, including prior short/empty-EOS diagnostics; successful HTTP is insufficient.',
            'Explicit lifecycle supervision and owned-process cleanup on both nodes; no direct execution of review strings.',
            'Tune on separate subsets, freeze selected configuration, then run unchanged 6000-request profiles.',
            *(['Qualify capture and actual replay on all ranks, including uneven/idle/padded DP batches and prefill-to-decode transitions; eager results do not qualify graph mode.'] if capture_sizes else []),
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology', required=True, choices=('A', 'B'))
    parser.add_argument('--per-dp-running', required=True, type=int, choices=RUNNING_CHOICES)
    parser.add_argument('--per-dp-chunk', required=True, type=int, choices=CHUNK_CHOICES)
    parser.add_argument('--memory-fraction', required=True, choices=FRACTION_CHOICES)
    parser.add_argument('--execution-mode', required=True, choices=('eager', 'graph'))
    parser.add_argument('--cuda-graph-max-bs', type=int, help='Graph only: actual capture bucket <= per-DP running target')
    parser.add_argument('--source-audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output', type=Path, help='Create a new local JSON file; never overwrite')
    args = parser.parse_args(argv)
    diagnostic, help_text = load_inputs(args.source_audit)
    candidate = build_candidate(diagnostic, help_text, args.topology,
                                args.per_dp_running, args.per_dp_chunk, args.memory_fraction,
                                args.execution_mode, args.cuda_graph_max_bs)
    encoded = json.dumps(candidate, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
