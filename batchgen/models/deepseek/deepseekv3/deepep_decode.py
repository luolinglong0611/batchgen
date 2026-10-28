"""PRIVATE CANDIDATE: R1 BF16 DeepEP-v1 LL transport, existing FP8 expert math.

Not GPU-qualified. Does not select an attention backend or disable CUDA Graph.
The CUDA Graph caller must retain this pool and update valid_rows in place.
One pool/exchange is shared by sequential MoE layers in one rank. Concurrent
layers/microbatches require distinct pools and an explicit lifetime protocol.
"""
import os

import torch

from batchgen.moe.deepep_ll import get_low_latency_exchange, require_deepep


class R1DeepEPDecodePool:
    def __init__(self, group, *, max_tokens_per_rank, hidden, intermediate,
                 num_experts, topk, device):
        self.capacity = int(max_tokens_per_rank)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.world_size = int(group.size())
        if self.capacity <= 0 or self.world_size <= 1:
            raise ValueError("R1 DeepEP LL requires a positive capacity and EP > 1")
        require_deepep(hidden=self.hidden, topk=self.topk,
                      num_experts=self.num_experts, num_ranks=self.world_size,
                      max_tokens_per_rank=self.capacity, use_fp8=False)
        self.num_local_experts = self.num_experts // self.world_size
        if self.world_size * self.capacity % 4:
            raise ValueError("DeepEP-v1 LL row capacity must be a multiple of 4")
        qp_depth = int(os.environ.get("NVSHMEM_QP_DEPTH", "1024"))
        if qp_depth < 2 * (self.capacity + 1):
            raise RuntimeError("NVSHMEM_QP_DEPTH must be >= 2*(LL capacity+1)")
        self.exchange = get_low_latency_exchange(
            group, max_tokens_per_rank=self.capacity, hidden=self.hidden,
            num_experts=self.num_experts, num_local_experts=self.num_local_experts)
        self.stride = self.world_size * self.capacity
        e, m, h, n = (self.num_local_experts, self.stride, self.hidden,
                      self.intermediate)
        self.x = torch.zeros((self.capacity, h), dtype=torch.bfloat16, device=device)
        self.topk_idx = torch.full((self.capacity, self.topk), -1,
                                   dtype=torch.int64, device=device)
        self.topk_weights = torch.zeros((self.capacity, self.topk),
                                       dtype=torch.float32, device=device)
        self.valid_rows = torch.tensor([self.capacity], dtype=torch.int32, device=device)
        self.row_ids = torch.arange(self.capacity, dtype=torch.int32, device=device)
        self.x_fp8 = torch.empty((e, m, h), dtype=torch.float8_e4m3fn, device=device)
        self.x_scale = torch.empty((e, m, h // 128), dtype=torch.float32, device=device)
        self.intermediate_bf16 = torch.empty((e, m, n), dtype=torch.bfloat16, device=device)
        self.intermediate_fp8 = torch.empty((e, m, n), dtype=torch.float8_e4m3fn, device=device)
        self.intermediate_scale = torch.empty((e, m, n // 128), dtype=torch.float32, device=device)
        self.expert_out = torch.empty((e, m, h), dtype=torch.bfloat16, device=device)
        self.local_out = torch.empty((self.capacity, h), dtype=torch.bfloat16, device=device)
        # Device counters survive CUDA Graph replay. Columns: dispatch calls,
        # combine calls, live local rows, received local-expert rows. Logging
        # these as Python counters would only count graph capture, not replay.
        self.layer_counters = torch.zeros((61, 4), dtype=torch.int64, device=device)
        self.phase_configurations = 0
        # False is the eager integration. The graph caller explicitly opts in
        # after wiring valid_rows to its page/admission boundary metadata.
        self.external_valid_rows = False

    def begin_phase(self):
        """After the caller drained old graphs/work, before attaching new layers."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Do not configure an R1 LL phase during capture")
        self.phase_configurations += 1
        self.external_valid_rows = False

    def trace_snapshot(self):
        """Explicit D2H sync: call outside the timed loop after rank agreement."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot snapshot R1 LL counters during capture")
        return {
            "transport": "deep_ep_v1_low_latency_bf16",
            "capacity": self.capacity, "ep": self.world_size,
            "phase_configurations": self.phase_configurations,
            "columns": ["dispatch_calls", "combine_calls", "live_local_rows",
                        "received_expert_rows"],
            "layers": self.layer_counters.detach().cpu().tolist(),
        }

    def set_valid_rows(self, count):
        """Call before graph replay. A CUDA int32 [1] is copied without D2H."""
        if isinstance(count, int):
            if not 0 <= count <= self.capacity:
                raise ValueError("valid_rows is outside the fixed LL capacity")
            self.valid_rows.fill_(count)
        else:
            if (count.shape != self.valid_rows.shape or
                    count.dtype != self.valid_rows.dtype or
                    count.device != self.valid_rows.device):
                raise ValueError("valid_rows must be a device int32 [1] tensor")
            self.valid_rows.copy_(count)


class R1DeepEPDecodeAdapter:
    """Called once after all local attention subbatches have been assembled."""
    def __init__(self, moe, pool, *, layer_idx):
        if getattr(moe, "enable_ep_offloading", False):
            raise RuntimeError("R1 LL candidate requires GPU-resident local experts")
        for name in ("gate_list", "up_list", "down_list", "gate_ptrs_ptr",
                     "up_ptrs_ptr", "down_ptrs_ptr", "gate_scale_list",
                     "up_scale_list", "down_scale_list", "gate_scale_ptrs_ptr",
                     "up_scale_ptrs_ptr", "down_scale_ptrs_ptr"):
            if getattr(moe, name, None) is None:
                raise RuntimeError("R1 LL missing resident expert metadata: " + name)
        self.moe = moe
        self.pool = pool
        self.layer_idx = int(layer_idx)
        if not 3 <= self.layer_idx < pool.layer_counters.shape[0]:
            raise ValueError("R1 routed-MoE layer index must be in [3,61)")
        # Lazy import after modeling_deepseek_v3 has finished importing.
        from .modeling_deepseek_v3 import act_quant_kernel_3d_sparse
        from .grouped_gemm_kernel import (
            fused_fp8_moe_stage_1_tma_wrapper,
            fused_dequant_grouped_gemm_fp8_tma_wrapper,
        )
        self.quant_kernel = act_quant_kernel_3d_sparse
        self.s1 = fused_fp8_moe_stage_1_tma_wrapper
        self.s3 = fused_dequant_grouped_gemm_fp8_tma_wrapper

    def quantize_into(self, x, counts, output, scales):
        e, m, h = x.shape
        self.quant_kernel[(e, h // 128)](
            x, output, scales, counts, e, m, h,
            *x.stride(), *output.stride(), *scales.stride(),
            eps=1e-12, fp8_max=448.0, BLOCK_SIZE=128)

    def __call__(self, x):
        p, moe = self.pool, self.moe
        rows, hidden = x.shape
        if rows > p.capacity or hidden != p.hidden or x.dtype != torch.bfloat16:
            raise ValueError("R1 LL requires BF16 [rows<=fixed_capacity, hidden]")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not p.external_valid_rows:
            raise RuntimeError("Wire external valid_rows before capturing R1 LL")
        if not p.external_valid_rows:
            p.set_valid_rows(rows)
        # Static capacity is deliberate: every EP rank calls the same LL
        # geometry, including ranks with zero live rows. Invalid rows do not
        # select experts, so stale receive bytes cannot reach combine output.
        p.x.zero_()
        p.x[:rows].copy_(x)
        idx, weights = moe.gate.moe_gate_forward_hybrid(p.x[:, None, :])
        valid = (p.row_ids < p.valid_rows).unsqueeze(1)
        p.topk_idx.copy_(torch.where(valid, idx, -1))
        p.topk_weights.copy_(torch.where(valid, weights, 0.0))
        recv_x, counts, handle = p.exchange.dispatch(p.x, p.topk_idx)
        p.layer_counters[self.layer_idx, 0].add_(1)
        p.layer_counters[self.layer_idx, 2].add_(p.valid_rows[0])
        p.layer_counters[self.layer_idx, 3].add_(counts.sum(dtype=torch.int64))
        self.quantize_into(recv_x, counts, p.x_fp8, p.x_scale)
        self.s1(
            p.x_fp8, p.x_scale, moe.gate_list, moe.gate_ptrs_ptr,
            moe.up_list, moe.up_ptrs_ptr, moe.gate_scale_list,
            moe.gate_scale_ptrs_ptr, moe.up_scale_list, moe.up_scale_ptrs_ptr,
            counts, p.num_local_experts, out=p.intermediate_bf16)
        self.quantize_into(p.intermediate_bf16, counts,
                           p.intermediate_fp8, p.intermediate_scale)
        self.s3(
            p.intermediate_fp8, p.intermediate_scale,
            moe.down_list, moe.down_ptrs_ptr,
            moe.down_scale_list, moe.down_scale_ptrs_ptr,
            counts, p.num_local_experts, out=p.expert_out)
        p.exchange.combine(p.expert_out, p.topk_idx, p.topk_weights,
                           handle, out=p.local_out)
        p.layer_counters[self.layer_idx, 1].add_(1)
        # The enclosing DeepSeek MoE.forward adds its existing shared expert.
        # This view is consumed before another sequential layer reuses p.
        return p.local_out[:rows]
