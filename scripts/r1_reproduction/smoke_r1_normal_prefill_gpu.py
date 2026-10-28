"""Plan by default; root alone may torchrun this8-rank normal-EP routing probe.

Uses the production normal exchange/scatter/gather with an independent
expert-id multiplier in place of GEMMs. It qualifies routing/zero-row/buffer
reuse only, not DeepGEMM expert arithmetic or a full model performance result.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback


def load_source(root):
    p = root / "batchgen/models/deepseek/deepseekv3/resident_prefill.py"
    spec = importlib.util.spec_from_file_location("R1_normal_prefill_operator_candidate", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def execute(args):
    import torch
    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("Explicit8-rank torchrun is required")
    local = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    source, source_pin = load_source(args.source_root.resolve())
    started = time.monotonic()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    result = {"schema": "R1-normal-prefill-routing-operator-v1", "rank": rank,
              "source": source_pin, "GPU": torch.cuda.get_device_name(device),
              "capability": list(torch.cuda.get_device_capability(device)),
              "scope": "normal_FP8_scatter_gather_routing_NOT_DeepGEMM_or_fullmodel_qualification"}
    try:
        exchange = source.R1NormalPrefillExchange(dist.group.WORLD)
        class IdentityExpert:
            def contiguous(self, fp8, packed_scale, m_indices):
                # Independent decode of four UE8M0 exponent bytes per int32.
                # Source values are dyadic powers exactly representable after
                # FP8 quantization, so CPU source values remain the reference.
                assert packed_scale.dtype == torch.int32 and packed_scale.stride(0) == 1
                shifts = torch.arange(4, device=device, dtype=torch.int32) * 8
                exponent = ((packed_scale.unsqueeze(-1) >> shifts) & 255).flatten(1)
                scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)
                values = fp8.float() * scale.repeat_interleave(128, dim=1)
                gid = rank * 32 + m_indices.to(torch.int64)
                multiplier = 1 + ((gid // 7) % 2)
                return (values * multiplier[:, None]).to(torch.bfloat16)
        rounds = []
        for step in range(4):
            base = [0, 1, 2, 3, 0, 2, 1, 4]
            rows = base[(rank - step) % 8]
            values = torch.tensor([-.5, -.25, -.125, -.0625, .0625, .125, .25, .5])
            offsets = torch.arange(rows)[:, None] * 3 + torch.arange(7168)[None, :] + rank + step
            x_cpu = values[offsets % 8].to(torch.bfloat16)
            ids_cpu = (torch.arange(rows)[:, None] * 13 + rank * 31 + step * 5 + torch.arange(8)[None, :] * 17) % 256
            weights_cpu = torch.full((rows, 8), .125, dtype=torch.float32)
            coeff = ((1 + ((ids_cpu // 7) % 2)).float() * weights_cpu).sum(1)
            expected = (x_cpu.float() * coeff[:, None]).to(torch.bfloat16)
            ids, weights = ids_cpu.to(device), weights_cpu.to(device)
            class Gate:
                def __call__(self, hidden):
                    assert hidden.shape == (1, rows, 7168)
                    return ids, weights
            moe = type("ProbeMoE", (), {"gate": Gate()})()
            adapter = source.R1ResidentPrefillAdapter(moe, exchange, IdentityExpert(), layer_idx=3)
            actual = adapter(x_cpu.to(device))
            torch.cuda.synchronize(device)
            actual_cpu = actual.cpu()
            mismatch = int(torch.count_nonzero(actual_cpu != expected))
            error = 0. if not rows else float((actual_cpu.float() - expected.float()).abs().max())
            rounds.append({"round": step, "local_rows": rows, "mismatched_elements": mismatch,
                           "max_abs_error": error, "exact_CPU_reference": mismatch == 0})
            if mismatch:
                raise AssertionError(f"Independent CPU routing reference differs: round={step}, rank={rank}, maxabs={error}")
        assert any(row["local_rows"] == 0 for row in rounds)
        assert exchange.dispatch_submissions == exchange.combine_submissions == 4
        result.update(status="normal_routing_CPU_reference_exact_four_rounds", rounds=rounds,
            normal_dispatch_submissions=exchange.dispatch_submissions,
            normal_combine_submissions=exchange.combine_submissions,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), wall_seconds=time.monotonic() - started)
        gathered = [None] * 8
        dist.all_gather_object(gathered, result)
        (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        if rank == 0:
            summary = {"status": "all8_normal_FP8_routing_permutation_zero_rows_reuse_passed_NOT_model_qualified",
                       "ranks": gathered, "all_rank_zero_rows_exercised": True,
                       "expert_math": "independent_global_expert_id_multiplier_not_GEMM",
                       "rounds_per_rank": 4, "no_performance_claim": True}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        dist.barrier()
        return result
    except BaseException:
        result.update(status="failed_NOT_qualified", traceback=traceback.format_exc())
        if args.output.is_dir():
            (args.output / f"rank{rank}-failure.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    finally:
        # The normal exchange uses explicit process lifetime, so no collective
        # DeepEP destructor can run during out-of-order interpreter teardown.
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({"status": "CPU_plan_only_NOT_GPU_executed", "ranks": 8,
            "rounds": 4, "normal_EP_only": True, "all_ranks_zero_row_once": True,
            "independent_reference": "dyadic input times original global-expert multiplier and weights on CPU",
            "source_root": str(args.source_root), "output": str(args.output)}))
        return
    print(json.dumps(execute(args)))


if __name__ == "__main__":
    main()
