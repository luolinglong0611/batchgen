#!/usr/bin/env python3
"""Bounded 16-GPU R1 native routed-MoE EP qualification, no full model.

Uses actual MoEGate, fused dispatch, grouped TMA FP8 experts, weighted scatter,
and PyNccl all-gather/BF16 all-reduce. __new__ injects one synthetic layer's
16 distinct local experts per rank, preserving H7168/I2048/256 experts/top8.
Excludes shared expert, attention, scheduler, expert loading/PSM and host DMA.
The unchanged predeclared MoE composition gate is used, not an attention gate.
Run this node-local supervisor on BOTH allocated nodes only when GPUs are free.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCOPE = __doc__
HOSTS = ('serverless-010066009163.wm226', 'serverless-010066009166.wm226')
CASES = (
    ('uniform_cross_nodes', [1] * 16, [2, 14, 66, 78, 130, 142, 194, 206]),
    ('uneven_empty_ranks', [0, 1, 2, 3] * 4, [34, 46, 98, 110, 162, 174, 226, 238]),
    ('odd_expert_ranks', [2, 0, 3, 1] * 4, [18, 30, 82, 94, 146, 158, 210, 222]),
    ('remaining_expert_ranks', [3, 1, 0, 2] * 4, [50, 62, 114, 126, 178, 190, 242, 254]),
    ('all_empty', [0] * 16, [0, 1, 64, 65, 128, 129, 192, 193]),
)
GATE = {'relative_l2_max': 0.02, 'max_abs_over_reference_rms_max': 0.15, 'cosine_min': 0.9998}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    import torch
    import torch.distributed as dist
    import moe_gemm_qualification as oracle
    oracle.torch = torch
    if oracle.GATES['composition'] != GATE:
        raise RuntimeError('Existing predeclared MoE gate changed; review explicitly')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 16:
        raise RuntimeError('Requires exactly 16 ranks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError('Requires actual SM103')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    sys.path.insert(0, str(ROOT))
    report = {'status': 'failed', 'scope': SCOPE, 'rank': rank, 'hostname': socket.gethostname(),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gate': GATE, 'cases': [],
              'env': {k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'GLOO_', 'BATCHGEN_', 'DISABLE_MMA'))}}
    path = args.output / f'rank{rank:02d}.json'
    tensors, comm = {}, None
    started = time.monotonic()
    try:
        import libucx
        libucx.load_library()
        from batchgen.models.deepseek.deepseekv3 import modeling_deepseek_v3 as model
        from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
        from batchgen.distributed.utils import StatelessProcessGroup
        report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(oracle.__file__), Path(model.__file__),
                                            ROOT / 'batchgen/distributed/device_communicators/pynccl.py',
                                            ROOT / 'batchgen/gemm/w8a8_grouped_gemm_stage_1.py',
                                            ROOT / 'batchgen/moe/fused_grouped_dequant_gemm.py')}
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
        # Production StatelessProcessGroup/PyNccl, with an explicitly bounded
        # store (create() uses PyTorch's much longer default store timeout).
        store = dist.TCPStore(args.master, args.comm_port, 16, rank == 0,
                              timeout=timedelta(seconds=180))
        group = StatelessProcessGroup(rank=rank, world_size=16, store=store)
        comm = PyNcclCommunicator(group, device)
        if not comm.available:
            raise RuntimeError('Actual PyNccl is unavailable; no torch adapter fallback')
        report['torch_nccl_version'] = list(torch.cuda.nccl.version())
        report['pynccl_version'] = comm.nccl.ncclGetVersion()
        report['pynccl_library'] = comm.nccl.lib._name
        major, minor, patch = torch.cuda.nccl.version()
        if report['pynccl_version'] != f'{major}.{minor}.{patch}':
            raise RuntimeError('PyNccl library version differs from Torch NCCL')
        moe = model.DeepseekV3MoE_Decoding_FP8.__new__(model.DeepseekV3MoE_Decoding_FP8)
        torch.nn.Module.__init__(moe)
        moe.config = SimpleNamespace(hidden_size=7168)
        moe.rank, moe.world_size, moe.device, moe.comm = rank, 16, device, comm
        moe.num_experts_per_tok, moe.experts_per_rank, moe.total_experts = 8, 16, 256
        moe.routed_expert_start_idx, moe.routed_expert_end_idx = rank * 16, (rank + 1) * 16
        moe.enable_ep_offloading = False
        moe.num_tokens_per_rank = None
        gate = model.MoEGate.__new__(model.MoEGate)
        torch.nn.Module.__init__(gate)
        for key, value in dict(n_group=8, topk_group=4, n_routed_experts=256, top_k=8,
                               routed_scaling_factor=2.5, scoring_func='sigmoid',
                               topk_method='noaux_tc', norm_topk_prob=True).items():
            setattr(gate, key, value)
        gen = torch.Generator(device=device).manual_seed(20261023)
        gate.weight = torch.nn.Parameter((torch.randn((256, 7168), generator=gen, device=device) * 0.001).bfloat16(), requires_grad=False)
        gate.e_score_correction_bias = torch.nn.Parameter(torch.zeros(256, device=device), requires_grad=False)
        gate.eval()
        moe.gate = gate
        torch.manual_seed(20261023 + rank)
        experts = [oracle.make_expert(device) for _ in range(16)]
        moe.experts = [None] * 256
        for index, expert in enumerate(experts):
            moe.experts[rank * 16 + index] = SimpleNamespace(
                fp8_gate=expert['gate'][0], fp8_up=expert['up'][0], fp8_down=expert['down'][0],
                weight_dequant_scale={kind + '_proj.weight_scale_inv': expert[kind][1] for kind in expert})
        moe.init(3)  # Actual production list/scale pointer setup.
        for case_number, (name, counts, selected) in enumerate(CASES):
            padding = max(max(counts), 1)
            moe.set_num_tokens_per_rank(padding)
            gate.e_score_correction_bias.zero_()
            gate.e_score_correction_bias[selected] = 2.0
            gen.manual_seed(20261024 + case_number)
            global_x = torch.randn((16, padding, 7168), generator=gen, device=device).bfloat16()
            for peer, count in enumerate(counts):
                global_x[peer, count:].zero_()
            x = global_x[rank, :counts[rank]].contiguous()
            global_flat = global_x.reshape(-1, 7168)
            ids, weights = oracle.routing_reference(global_flat, gate.weight, gate.e_score_correction_bias)
            actual_ids, actual_weights = gate.moe_gate_forward_hybrid(global_flat[:, None])
            order, actual_order = ids.argsort(-1), actual_ids.argsort(-1)
            router_ok = torch.equal(ids.gather(1, order), actual_ids.gather(1, actual_order)) and bool(torch.isclose(
                weights.gather(1, order), actual_weights.gather(1, actual_order), atol=1e-6, rtol=1e-5).all())
            if not router_ok or set(ids[0].tolist()) != set(selected):
                raise RuntimeError(f'{name}: actual native router differs from independent reference')
            partial = torch.zeros_like(global_flat, dtype=torch.float32)
            active_local = []
            for local_id, expert in enumerate(experts):
                token_ids, topk_positions = torch.where(ids == rank * 16 + local_id)
                if token_ids.numel():
                    active_local.append(local_id)
                    _, out = oracle.expert_reference(global_flat[token_ids], expert, fused=True)
                    partial.index_add_(0, token_ids, out.float() * weights[token_ids, topk_positions, None])
            # Production rounds each rank's weighted partial to BF16 before
            # NCCL BF16 reduction. Reference adds these rounded partials in FP32
            # to avoid inheriting the tested BF16 reduction order/error.
            expected = partial.bfloat16().float()
            dist.all_reduce(expected)
            expected = expected.bfloat16().view(16, padding, 7168)[rank, :counts[rank]]
            dist.barrier()
            actual = moe.moe_infer_allgather_allreduce_bf16_acc(x)
            torch.cuda.synchronize(device)
            if counts[rank]:
                oracle.compare(actual, expected, name, report, tensors, kind='composition')
            else:
                report['cases'].append({'case': name, 'status': 'passed' if actual.shape == (0, 7168)
                                        and actual.dtype == torch.bfloat16 else 'failed', 'empty_local_rank': True})
                tensors[name] = {'actual': actual.cpu(), 'reference': expected.cpu()}
            report['cases'][-1].update(token_counts=counts, padding=padding, router_exact=True,
                                       active_local_experts=active_local, selected_experts=selected,
                                       cross_node_expert_routes=any(
                                           count and peer // 8 != expert // 128
                                           for peer, count in enumerate(counts) for expert in selected))
            # A mismatched rank must not cause peers to skip future collectives.
            dist.barrier()
        report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['cases']) else 'failed'
        passed = torch.tensor([int(report['status'] == 'passed')], device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        report['all_16_ranks_passed'] = bool(passed.item())
        if not report['all_16_ranks_passed']:
            report['status'] = 'failed'
    except Exception:
        report['traceback'] = traceback.format_exc()
    finally:
        report['seconds_not_benchmark'] = time.monotonic() - started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        save(path, report)
        torch.save(tensors, path.with_suffix('.pt'))
        print(json.dumps(report), flush=True)
        # The outer supervisor bounds hangs in native NCCL cleanup as well.
        if comm is not None:
            comm.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--master', default='10.66.9.163')
    parser.add_argument('--port', type=int, default=35155)
    parser.add_argument('--comm-port', type=int, default=35156)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or socket.gethostname() != HOSTS[args.node_rank]:
        parser.error('Run only in the matching allocated container')
    if args.worker:
        return worker(args)
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True).stdout.strip()
    if busy:
        raise RuntimeError('GPU processes already exist; do not overlap another qualification')
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', BATCHGEN_ENABLE_ALL_TO_ALL='0',
               BATCHGEN_R1_SM103_EXPERIMENTAL='1', BATCHGEN_R1_SM103_ATTENTION='triton_precision',
               BATCHGEN_W8A16_DEQUANT='0', TORCH_CUDA_CHANNEL='cu130', DISABLE_MMA_V5='1',
               NCCL_SOCKET_IFNAME='eth0', GLOO_SOCKET_IFNAME='eth0', NCCL_IB_GID_INDEX='3',
               NCCL_IB_HCA='=' + ','.join(f'mlx5_bond_{i}:1' for i in range(8)),
               PYTHONUNBUFFERED='1')
    if not Path(env.get('TRITON_PTXAS_PATH', '/missing')).is_file():
        raise RuntimeError('Require private CUDA13.0 TRITON_PTXAS_PATH')
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    processes = []
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    code = 1
    try:
        # Do not wrap torchrun: elastic workers may create their own sessions.
        # Each tracked direct child instead owns exactly one cleanup PGID.
        for local_rank in range(8):
            child_env = dict(env, RANK=str(args.node_rank * 8 + local_rank),
                             LOCAL_RANK=str(local_rank), WORLD_SIZE='16',
                             MASTER_ADDR=args.master, MASTER_PORT=str(args.port))
            processes.append(subprocess.Popen(command, env=child_env, cwd=ROOT, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A local worker failed; terminate this launch\'s remaining workers')
            if time.monotonic() >= deadline:
                raise TimeoutError('Worker deadline exceeded')
            time.sleep(0.25)
        code = 0 if all(p.returncode == 0 for p in processes) else 1
    except (KeyboardInterrupt, TimeoutError):
        code = 124
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        # Only the eight PGIDs created by this supervisor; never pkill.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
            time.sleep(0.1)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    save(args.output / 'supervisor.json', {'exit_code': code, 'scope': SCOPE, 'command': command,
                                          'gate': GATE, 'timeout_seconds': args.timeout,
                                          'worker_pids': [p.pid for p in processes],
                                          'worker_exit_codes': [p.returncode for p in processes]})
    return code


if __name__ == '__main__':
    sys.exit(main())
