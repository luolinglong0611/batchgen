"""REAL eight-rank FI MegaMoE qualification; default is a CPU-only plan.

No fake FusedMoE/communication objects. Full R1 EP8/H7168/I2048/localE32/top8
weights are FP8 with ORIGINAL FP32 block128 scales. The deliberately sparse
random fixture allows a compact independent GEMM oracle without materializing
all expert banks in FP32. It does not qualify full-model accuracy or TPS.

Root owns GPU execution:
  torchrun --standalone --nproc_per_node=8 smoke_r1_mxfp8_megamoe_gpu.py \
    --execute --capacity 8192 --rows 180 --output /absolute/new-result.json
"""
from __future__ import annotations

import argparse
import ast
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

D = Path(__file__).resolve().parent
H, I, E, LOCAL_E, TOPK = 7168, 2048, 256, 32, 8
ACTIVE_WIDTH = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_contract():
    pins = json.loads((D / "PINS.json").read_text())
    if pins["upstream_commit"] != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise RuntimeError("Wrong SGLang source identity")
    for item in pins["files"]:
        if sha(D / "candidate" / item["path"]) != item["candidate_sha256"]:
            raise RuntimeError("Candidate source changed: " + item["path"])
    refs = json.loads((D / "REFERENCE_PINS.json").read_text())
    for relative, expected in refs.items():
        if sha(D / relative) != expected:
            raise RuntimeError("Numerical/source reference changed: " + relative)
    # Check genuine v0.5.22 FusedMoE member names. No fake object is constructed.
    tree = ast.parse((D / "fused_moe_layer.upstream.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    attrs = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == "self"}
    required = {"hidden_size", "num_experts", "num_local_experts", "moe_ep_size",
                "top_k", "intermediate_size_per_partition"}
    if not required <= attrs:
        raise RuntimeError("Real FusedMoE member contract changed")
    return {"bridge_pins": pins, "original_numerical_sources": refs,
            "real_FusedMoE_attributes_source_verified": sorted(required)}


def plan(args):
    if args.rows != 180 or args.capacity < 180:
        raise ValueError("This qualification fixes real decode rows180 and capacity>=180")
    return {"schema": "R1_MXFP8_MegaMoE_operator_v1", "executed": False,
        "world_size": 8, "required_sm": [10, 3], "H": H, "I": I,
        "global_experts": E, "local_experts": LOCAL_E, "top_k": TOPK,
        "capacity": args.capacity, "local_rows": [0] + [args.rows] * 7,
        "fixed_expert_ids": list(range(31, E, LOCAL_E)),
        "source_scales": "original_FP32_block128_non_power_of_two",
        "math": "official_FI_MXFP8_MegaMoE_no_rewritten_communication",
        "reference": "unchanged_C1_numeric_functions_plus_official_FI_MX32_and_hw_SwiGLU",
        "weight_fixture": "full_geometry_sparse_random_32x32_blocks_last_local_expert",
        "oracle_tolerances": {"atol": 0.01, "rtol": 0.01,
            "source": "official_FI_moe_mxfp8_glu/mega_runner.py:487-502"},
        "capture": "ordinary_full_layer_CUDA_graph_after_all_rank_eager_warmup",
        "graph_replays": 3, "full_model_accuracy_or_performance_qualified": False,
        "sources": source_contract()}


def execute(args, receipt):
    import torch
    import torch.distributed as dist
    import flashinfer
    from flashinfer.moe_ep import (BootstrapConfig, FleetParams, MegaConfig,
        MoEEpMegaLayer, MoEEpTensors, MoEWeightPack, Mxfp8CutedslMegaMoeConfig,
        preprocess_mxfp8_cutedsl_mega_weights)

    torch.set_grad_enabled(False)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise RuntimeError("Requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("This operator qualification is restricted to actual SM103")
    dist.init_process_group("nccl", timeout=timedelta(seconds=900))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=900))
    fi_root = Path(flashinfer.__file__).resolve().parent
    fi_src = fi_root / "moe_ep/kernel_src/sm100/cutedsl_megamoe/src"
    refs = receipt["sources"]["original_numerical_sources"]
    for installed, original in ((fi_src / "common/host_utils.py", "reference/fi_host_utils.upstream.py"),
        (fi_src / "moe_nvfp4_swapab/runner_common.py", "reference/fi_runner_common.upstream.py")):
        if sha(installed) != refs[original]:
            raise RuntimeError("Installed FI numerical source differs; re-audit before GPU: " + str(installed))
    oracle = load_local("r1_mxfp8_oracle", D / "oracle_functions.py")
    old = oracle.c1_functions(torch)
    fi = oracle.fi_functions(torch)
    bridge = load_local("r1_fp8_megamoe_bridge_operator", D / "candidate/python/sglang/srt/r1_fp8_megamoe_bridge.py")
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device=device).manual_seed(280000 + rank)
    setup_started = time.monotonic()
    weights = []
    scales = []
    effective = []
    c1_checks = []
    for kind, n, k in (("w13", 2 * I, H), ("w2", H, I)):
        weight = torch.empty((LOCAL_E, n, k), device=device, dtype=torch.float8_e4m3fn)
        weight.view(torch.uint8).zero_()
        original_scales = torch.full((LOCAL_E, n // 128, k // 128),
            1.113 if kind == "w13" else 1.371, device=device, dtype=torch.float32)
        # Only the last local expert is active. Every rank owns one distinct
        # random expert; all eight receive every nonempty source token.
        weight[31, :32, :32].copy_((torch.randn((32, 32), generator=generator,
            device=device) * 0.25).to(torch.float8_e4m3fn))
        if kind == "w13":
            weight[31, I:I + 32, :32].copy_((torch.randn((32, 32),
                generator=generator, device=device) * 0.25).to(torch.float8_e4m3fn))
        scale = torch.empty((LOCAL_E, n, k // 32), device=device, dtype=torch.uint8)
        for expert in range(LOCAL_E):
            q, sf = bridge.c1_expert_to_mxfp8(weight[expert], original_scales[expert])
            if expert == 31:
                # Execute the actual original C1 source functions before its
                # packed-scale layout step, which cannot change weight values.
                bf16 = old["block_quant_dequant"](weight[expert], original_scales[expert],
                    [128, 128], torch.bfloat16)
                old_q, old_sf = old["quant_weight_ue8m0"](bf16, [128, 128])
                equal_q = torch.equal(q.view(torch.uint8), old_q.view(torch.uint8))
                old_expanded = old_sf.repeat_interleave(128, -2).repeat_interleave(4, -1)
                equal_scales = torch.equal(torch.pow(2.0, sf.int() - 127), old_expanded)
                c1_checks.append({"projection": kind, "original_C1_FP8_bytes_equal": equal_q,
                    "per32_scales_equal_expanded_C1": equal_scales})
                if not equal_q or not equal_scales:
                    raise AssertionError("Actual original C1 preparation disagrees with bridge")
                del bf16, old_q, old_sf, old_expanded
            weight[expert].copy_(q)
            scale[expert].copy_(sf)
            del q, sf
        row_sf = torch.pow(2.0, scale[31, :32, :1].int() - 127)
        first = weight[31, :32, :32].float() * row_sf
        if kind == "w13":
            up_sf = torch.pow(2.0, scale[31, I:I + 32, :1].int() - 127)
            effective.extend((first, weight[31, I:I + 32, :32].float() * up_sf))
        else:
            effective.append(first)
        weights.append(weight)
        scales.append(scale)
        del original_scales
    local_ref = torch.stack(effective).contiguous()
    all_ref = [torch.empty_like(local_ref) for _ in range(8)]
    dist.all_gather(all_ref, local_ref)
    transformed = preprocess_mxfp8_cutedsl_mega_weights(
        MoEWeightPack(w13=weights[0], w2=weights[1], w13_scale=scales[0], w2_scale=scales[1]),
        intermediate_size=I, hidden_size=H, kind="mxfp8_e4m3",
        gate_up_clamp=None, activation_clamp=None)
    del weights, scales, effective, local_ref
    mega = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=8, rank=rank, device=local_rank,
            process_group=dist.group.WORLD),
        fleet_params=FleetParams(num_experts=E, max_tokens_per_rank=args.capacity,
            token_hidden_size=H), weights=None,
        backend=MegaConfig(megakernel=Mxfp8CutedslMegaMoeConfig(
            intermediate_size=I, top_k=TOPK, kind="mxfp8_e4m3",
            gate_up_clamp=None, enable_in_kernel_fc2_reduce=False),
            preprocess_weights=False, transformed_weights=transformed))
    rows = 0 if rank == 0 else args.rows
    x = torch.zeros((rows, H), dtype=torch.bfloat16, device=device)
    ids = torch.tensor(list(range(31, E, LOCAL_E)), dtype=torch.int64, device=device).repeat(rows, 1)
    topk_weights = torch.full((rows, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device)
    t = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=topk_weights)
    counter = torch.zeros((), dtype=torch.int64, device=device)

    def change_input(step):
        if rows:
            small = torch.randn((rows, 32), device=device, generator=generator) * (0.5 + 0.25 * step)
            x[:, :32].copy_(small.to(torch.bfloat16))

    def reference():
        output = torch.zeros_like(x)
        if not rows:
            return output
        q, sf = fi["mxfp8_quantize_per_block_32"](x[:, :32].float(), torch.float8_e4m3fn)
        deq = q.float() * torch.pow(2.0, sf.view(torch.uint8).float() - 127).repeat_interleave(32, -1)
        experts = []
        for gate, up, down in all_ref:
            g = deq @ gate.T
            u = deq @ up.T
            mid = fi["_swiglu_pair_hw_match_cuda"](g, u) * (1.0 / TOPK)
            qm, sm = fi["mxfp8_quantize_per_block_32"](mid, torch.float8_e4m3fn)
            md = qm.float() * torch.pow(2.0, sm.view(torch.uint8).float() - 127).repeat_interleave(32, -1)
            experts.append((md @ down.T).to(torch.bfloat16))
        output[:, :32] = torch.stack(experts).sum(dim=0)
        if float(output[:, :32].abs().max()) < 0.01:
            raise AssertionError("Nontrivial fixture produced too-small reference")
        return output

    cases = []

    def check(result, label):
        expected = reference()
        torch.cuda.synchronize()
        if tuple(result.shape) != (rows, H):
            raise AssertionError("Native output shape mismatch")
        actual = result.clone()
        error = (actual.float() - expected.float()).abs()
        row = {"case": label, "rows": rows,
            "max_abs": float(error.max()) if rows else 0.0,
            "nrmse": float(torch.linalg.vector_norm(error) /
                torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)) if rows else 0.0,
            "zero_output_tail_exact": bool(torch.count_nonzero(actual[:, 32:]) == 0),
            "reference_atol": 0.01, "reference_rtol": 0.01}
        cases.append(row)
        # Match the upstream test tolerance; never increase it on failure.
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        if not row["zero_output_tail_exact"]:
            raise AssertionError("Unrouted/zero weight output tail was written incorrectly")

    change_input(0)
    dist.barrier(group=control)
    mega.warmup(t)
    check(mega.forward(t, return_workspace_view=mega.supports_output_view), "eager")
    dist.barrier(group=control)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        graph_output = mega.forward(t, return_workspace_view=mega.supports_output_view)
        counter.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # All captures must retire before any rank advances the collective epoch.
    dist.barrier(group=control)
    counter.zero_()
    for step in (1, 2, 3):
        change_input(step)
        dist.barrier(group=control)
        graph.replay()
        check(graph_output, "changed_input_graph_replay_" + str(step))
    if int(counter.item()) != 3:
        raise AssertionError("CUDA device graph replay counter is not three")
    torch.cuda.synchronize()
    rank_result = {"rank": rank, "passed": True, "device": torch.cuda.get_device_name(),
        "sm": list(torch.cuda.get_device_capability()), "capacity": args.capacity,
        "rows": rows, "total_active_rows": 7 * args.rows,
        "active_expert_per_rank": 31, "C1_comparison": c1_checks,
        "cases": cases, "graph_device_counter": int(counter.item()),
        "operator_setup_and_checks_seconds": time.monotonic() - setup_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "workspace_pooling": "native_backend_fixed_knobs_source_eligible_not_a_58layer_runtime_proof",
        "torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "unknown")}
    all_results = [None] * 8
    dist.all_gather_object(all_results, rank_result, group=control)
    # Graph resources must retire before the collective native workspace teardown.
    del graph_output, graph, stream, t, counter, transformed
    torch.cuda.synchronize()
    dist.barrier(group=control)
    mega.destroy()
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    dist.destroy_process_group()
    if rank == 0:
        receipt.update(executed=True, passed=True, rank_results=all_results,
            status="operator_only_NOT_full_model_or_performance_qualification")
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"passed": True, "output": args.output, "ranks": 8,
            "graph_replays_per_rank": 3}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rows", type=int, default=180)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--output", default="r1-mxfp8-megamoe-operator.json")
    args = parser.parse_args()
    receipt = plan(args)
    if not args.execute:
        print(json.dumps(receipt, indent=2))
        return
    try:
        execute(args, receipt)
    except BaseException:
        receipt.update(executed=True, passed=False, traceback=traceback.format_exc(),
            rank=os.environ.get("RANK"))
        path = Path(args.output + ".rank" + os.environ.get("RANK", "unknown") + ".failure.json")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
