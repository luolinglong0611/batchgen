"""Opt-in R1 resident-EP prefill using DeepEP normal and DeepGEMM FP8.

This path changes the prefill execution strategy. It is not the legacy
replicated/Host-streamed prefill, and never partitions a prefill through LL.
Imports of GPU packages are deferred until the explicitly selected path runs.
"""
import importlib
import inspect
import logging
import os


def moe_backend():
    value = os.environ.get("BATCHGEN_R1_MOE_BACKEND", "legacy")
    if value not in ("legacy", "deep_gemm"):
        raise ValueError("BATCHGEN_R1_MOE_BACKEND must be legacy or deep_gemm")
    return value


def resident_prefill_requested():
    value = os.environ.get("BATCHGEN_R1_RESIDENT_EP_PREFILL", "0")
    if value not in ("0", "1"):
        raise ValueError("BATCHGEN_R1_RESIDENT_EP_PREFILL must be 0 or 1")
    if value == "1" and moe_backend() != "deep_gemm":
        raise ValueError("Resident R1 prefill requires BATCHGEN_R1_MOE_BACKEND=deep_gemm")
    return value == "1"


def resident_prefill_token_cap():
    value = os.environ.get("BATCHGEN_R1_PREFILL_TOKENS_PER_RANK", "8192")
    if not value.isascii() or not value.isdecimal() or not 128 <= int(value) <= 131072 or int(value) % 128:
        raise ValueError("BATCHGEN_R1_PREFILL_TOKENS_PER_RANK must be a multiple of128 in [128,131072]")
    return int(value)


def local_expert_ids(rank, world_size, num_experts=256):
    if world_size != 8 or num_experts != 256 or not 0 <= rank < world_size:
        raise ValueError("R1 resident prefill requires EP8 and256 experts")
    count = num_experts // world_size
    return range(rank * count, (rank + 1) * count)


class R1NormalPrefillExchange:
    """One process-lifetime intranode normal buffer, shared by routed layers."""
    def __init__(self, group, *, hidden=7168, num_experts=256):
        if group.size() != 8 or hidden != 7168 or num_experts != 256:
            raise ValueError("R1 normal prefill requires EP8/hidden7168/256 experts")
        deep_ep = importlib.import_module("deep_ep")
        buffer_type = deep_ep.Buffer
        for name, keys in (("dispatch", ("expert_alignment", "topk_weights", "num_tokens_per_expert")),
                           ("combine", ("handle", "previous_event"))):
            params = inspect.signature(getattr(buffer_type, name)).parameters
            if not set(keys) <= set(params):
                raise RuntimeError("Installed DeepEP lacks the reviewed normal " + name + " API")
        self.hidden, self.num_experts = hidden, num_experts
        self.world_size = group.size()
        from .resident_prefill_tuning import requested_normal_sms, normal_config
        self.requested_sms = requested_normal_sms()
        device_sms = None
        if any(self.requested_sms.values()):
            import torch
            device_sms = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        self.dispatch_config = normal_config(buffer_type, getattr(deep_ep, "Config", None),
            role="dispatch", requested_sms=self.requested_sms["dispatch"],
            world_size=self.world_size, device_sm_count=device_sms)
        self.combine_config = normal_config(buffer_type, getattr(deep_ep, "Config", None),
            role="combine", requested_sms=self.requested_sms["combine"],
            world_size=self.world_size, device_sm_count=device_sms)
        nvl = max(c.get_nvl_buffer_size_hint(hidden * 2, self.world_size)
                  for c in (self.dispatch_config, self.combine_config))
        rdma = max(c.get_rdma_buffer_size_hint(hidden * 2, self.world_size)
                   for c in (self.dispatch_config, self.combine_config))
        # Explicit lifetime avoids an uncoordinated collective destructor at
        # interpreter exit, as in the existing R1 LL exchange. No model or
        # expert weights are retained by this process-lifetime communication pool.
        self.buffer = buffer_type(group, num_nvl_bytes=nvl, num_rdma_bytes=rdma,
                                  low_latency_mode=False, explicitly_destroy=True)
        self.buffer_type = buffer_type
        self.nvl_bytes, self.rdma_bytes = nvl, rdma
        self.dispatch_submissions = self.combine_submissions = 0
        logging.info("[R1_PREFILL_NORMAL_CONFIG] dispatch_sms_override=%d combine_sms_override=%d "
                     "active_GPU_sms=%s NVL_bytes=%d RDMA_bytes=%d global_Buffer_num_sms_unchanged=True",
                     self.requested_sms["dispatch"], self.requested_sms["combine"],
                     device_sms, nvl, rdma)

    def dispatch(self, quantized, topk_ids, topk_weights):
        previous = self.buffer_type.capture()
        rank_counts, rdma_counts, expert_counts, in_rank, previous = self.buffer.get_dispatch_layout(
            topk_ids, self.num_experts, previous_event=previous,
            async_finish=True, allocate_on_comm_stream=True)
        received = self.buffer.dispatch(
            quantized, topk_idx=topk_ids, topk_weights=topk_weights,
            num_tokens_per_rank=rank_counts, num_tokens_per_rdma_rank=rdma_counts,
            is_token_in_rank=in_rank, num_tokens_per_expert=expert_counts,
            expert_alignment=128, config=self.dispatch_config,
            previous_event=previous, async_finish=True, allocate_on_comm_stream=True)
        received[-1].current_stream_wait()
        self.dispatch_submissions += 1
        return received[:-1]

    def combine(self, hidden, handle):
        result, _, event = self.buffer.combine(
            hidden, handle, config=self.combine_config,
            previous_event=self.buffer_type.capture(), async_finish=True,
            allocate_on_comm_stream=True)
        event.current_stream_wait()
        self.combine_submissions += 1
        return result


class R1ResidentPrefillAdapter:
    """Normal FP8 dispatch → SG permutation → contiguous experts → combine.

    DeepEP normal returns local expert IDs and an aligned count per local
    expert. SG's unchanged scatter/gather applies routing weights exactly once;
    DeepEP combine then adds the already weighted expert-rank contributions.
    """
    def __init__(self, moe, exchange, expert_math, *, layer_idx, runtime=None):
        if runtime is None:
            import torch
            kernels = importlib.import_module("sglang.srt.layers.moe.ep_moe.kernels")
            quant = importlib.import_module("sglang.srt.layers.quantization.fp8_kernel")
            runtime = (torch, kernels.ep_scatter, kernels.ep_gather,
                       quant.sglang_per_token_group_quant_fp8)
        self.torch, self.scatter, self.gather, self.quantize = runtime
        self.moe, self.exchange, self.expert_math = moe, exchange, expert_math
        self.layer_idx = int(layer_idx)
        self.calls = 0

    def __call__(self, x):
        torch = self.torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Variable-size normal prefill is outside decode CUDA Graph capture")
        if x.ndim != 2 or x.shape[1] != 7168 or x.dtype != torch.bfloat16:
            raise ValueError("R1 normal prefill requires BF16 [tokens,7168]")
        if x.shape[0]:
            ids, weights = self.moe.gate(x.unsqueeze(0))
            ids, weights = ids.to(torch.int64), weights.to(torch.float32)
            quantized = self.quantize(x, 128, column_major_scales=True,
                                     scale_tma_aligned=True, scale_ue8m0=True)
        else:
            ids = torch.empty((0, 8), dtype=torch.int64, device=x.device)
            weights = torch.empty((0, 8), dtype=torch.float32, device=x.device)
            quantized = (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                         torch.empty((14, 0), dtype=torch.int32, device=x.device).t())
        (recv_x, recv_scale), recv_ids, recv_weights, counts, handle = self.exchange.dispatch(
            quantized, ids, weights)
        if len(counts) != 32 or any(type(n) is not int or n < 0 or n % 128 for n in counts):
            raise RuntimeError("Normal prefill requires32 local expert counts aligned to128")
        total = sum(counts)
        if total:
            sorted_x = torch.empty((total, 7168), dtype=recv_x.dtype, device=x.device)
            sorted_scale = torch.zeros((14, total), dtype=torch.int32, device=x.device).t()
            m_indices = torch.empty(total, dtype=torch.int32, device=x.device)
            output_index = torch.empty_like(recv_ids)
            counts_cpu = torch.tensor(counts, dtype=torch.int32, device="cpu", pin_memory=True)
            counts_gpu = counts_cpu.to(x.device, non_blocking=True)
            expert_start = torch.empty_like(counts_gpu)
            self.scatter(recv_x, recv_scale, recv_ids, counts_gpu, expert_start,
                         sorted_x, sorted_scale, m_indices, output_index, scale_ue8m0=True)
            expert_output = self.expert_math.contiguous(sorted_x, sorted_scale, m_indices)
            gathered = torch.empty((recv_x.shape[0], 7168), dtype=torch.bfloat16, device=x.device)
            self.gather(expert_output, recv_ids, recv_weights, output_index, gathered)
        else:
            gathered = torch.zeros((recv_x.shape[0], 7168), dtype=torch.bfloat16, device=x.device)
        result = self.exchange.combine(gathered, handle)
        self.calls += 1
        if self.calls == 1:
            logging.info("[R1_PREFILL_EP] layer=%d normal_dispatch=FP8_UE8M0 local_rows=%d "
                         "aligned_expert_rows=%d math=DeepGEMM_contiguous no_weight_H2D_in_forward=True",
                         self.layer_idx, x.shape[0], total)
        return result


def resident_moe_forward(moe, hidden_states):
    """Preserve existing shared-expert math and the model's input shape."""
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    out = moe._r1_resident_prefill_adapter(x)
    if x.shape[0] and getattr(moe, "shared_experts", None) is not None:
        out = out + moe.shared_experts(x)
    return out.view(shape)
