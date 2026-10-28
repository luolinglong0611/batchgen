"""Opt-in R1 segmented decode graphs with explicit replay-time Host KV staging.

The module imports no accelerator packages until its runtime entry points run.
This first integration graphs attention, including native page64 KV writes;
the enclosing decoder still invokes its MoE once after all attention slices.
It does not claim whole-model capture. No Host operation occurs in capture.

FlashInfer 0.5.2 MLA planning runs outside capture at every model step. The
complete plan_info vector is part of the graph identity: changing scheduler
layout or launch geometry forces recapture, never reuse of a stale plan.
Only one graph per attention wrapper is retained. Metadata and FI workspace
are shared across layers, used serially on the owning CUDA current stream.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import math
import os
import time


ENVIRONMENT = "BATCHGEN_R1_DECODE_CUDA_GRAPH"
_FORWARD = ContextVar("r1_decode_graph_forward", default=None)
_WHOLE_MODEL = ContextVar("r1_whole_model_attention_capture", default=None)
_CAPTURE_INPUTS = ContextVar("r1_decode_graph_external_inputs", default=None)
_INDEX_KERNEL = None


def fast_metadata_requested():
    value = os.environ.get("BATCHGEN_R1_DECODE_FAST_METADATA", "0")
    if value not in ("0", "1"):
        raise ValueError("BATCHGEN_R1_DECODE_FAST_METADATA must be 0 or 1")
    return value == "1"


def _expand_token_page_indices(page_table, indptr, lengths, output,
                               TABLE_WIDTH: tl.constexpr, ROW_START: tl.constexpr,
                               BLOCK: tl.constexpr):
    # Compiled lazily: importing the metadata planner stays Torch/Triton-free.
    row = tl.program_id(0)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = token < tl.load(lengths + row)
    physical = tl.load(page_table + (row + ROW_START) * TABLE_WIDTH + token // 64,
                       mask=valid, other=0)
    start = tl.load(indptr + row)
    tl.store(output + start + token, physical * 64 + token % 64, mask=valid)


def _index_kernel():
    global _INDEX_KERNEL
    if _INDEX_KERNEL is None:
        import triton
        import triton.language
        globals()["tl"] = triton.language
        _INDEX_KERNEL = triton.jit(_expand_token_page_indices)
    return _INDEX_KERNEL


def attention_source_tensors(wrapper):
    """Actual source tensors, rather than fresh Parameter.data wrappers."""
    module = wrapper.module
    rows = []
    for child in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                  "o_proj", "q_a_layernorm", "kv_a_layernorm"):
        rows.append((child + ".weight", getattr(module, child).weight))
    rows.extend(("scale:" + name, value)
                for name, value in sorted(wrapper.weight_dequant_scale.items()))
    for index, item in enumerate(getattr(module, "_r1_native_gemm_weights", ())):
        rows.extend((("prepared_ordinary_weight:" + str(index), item.weight),
                     ("prepared_ordinary_scale:" + str(index), item.scale)))
    return tuple(rows)


def _source_signature(name, tensor, *, require_version):
    try:
        version = tensor._version
    except RuntimeError as exc:
        if require_version:
            raise RuntimeError(
                "R1 graph requires versioned resident weight/scale tensors; "
                "reload under inference_mode(False) before capture: " + name
            ) from exc
        version = None
    return (name, id(tensor), tensor.data_ptr(), tensor.untyped_storage().data_ptr(),
            version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device)


def attention_source_signature(wrapper):
    """Cheap CPU metadata key, also used by the whole-decode graph owner.

    Normal in-place writes and storage/data rebinds invalidate a graph.
    Mutations through .data/external writers that bypass version counters
    remain prohibited; load/clear hooks must invalidate before such writes.
    RoPE buffers are constant inputs: keep any version that exists and always
    check their storage identity, including a cache regenerated at a new cap.
    """
    result = [_source_signature(name, tensor, require_version=True)
              for name, tensor in attention_source_tensors(wrapper)]
    rotary = wrapper.module.rotary_emb
    for name in ("cos_cached", "sin_cached"):
        value = getattr(rotary, name, None)
        if value is not None:
            result.append(_source_signature("rotary:" + name, value, require_version=False))
    return tuple(result)


def _retain_capture_inputs(*tensors):
    """Retain storage aliases for external inputs baked into a graph.

    CUDA Graph owns allocations created inside capture, but is not a strong
    owner of arbitrary warmup/cache inputs. Detach aliases are essential when
    a Parameter's .data is rebound or a stream-specific BF16 cache is replaced.
    Warmup alone must not retain a second, unused copy of every MLA matrix.
    """
    collector = _CAPTURE_INPUTS.get()
    if collector is None:
        return
    import torch
    if not torch.cuda.is_current_stream_capturing():
        return
    for tensor in tensors:
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        if key not in collector:
            collector[key] = tensor.detach()


def requested():
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENVIRONMENT} must be explicitly 0 or 1")
    return value == "1"


def validate_profile(*, model_identifier, enabled, disabled):
    """CPU-only launch guard. Existing eager defaults are unchanged."""
    if not requested():
        return False
    if ("deepseek-r1" not in str(model_identifier).lower()
            or os.environ.get("BATCHGEN_R1_SM103_FP8_COMPAT") != "1"
            or os.environ.get("BATCHGEN_R1_SM103_MLA_COMPAT") != "flashinfer-fa2"
            or os.environ.get("BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP") != "1"):
        raise ValueError("R1 decode graphs require the explicit R1 SM103 resident-EP FA2 profile")
    if enabled is not True or disabled is not False:
        raise ValueError("R1 decode graphs require explicit --enable-cuda-graph")
    return True


@dataclass(frozen=True)
class MetadataPlan:
    gids: tuple
    lengths: tuple
    pages: tuple
    page_table_width: int
    kv_indptr: tuple
    gpu_page_count: int
    reservation_table: object = None

    @property
    def rows(self):
        return len(self.gids)

    @property
    def positions(self):
        return tuple(length - 1 for length in self.lengths)

    def arrays(self):
        """Vectorized page64-to-tokenpage1 expansion, no GPU metadata reads."""
        import numpy as np
        table = np.full((self.rows, self.page_table_width), -1, dtype=np.int32)
        indices = np.empty(self.kv_indptr[-1], dtype=np.int32)
        offsets = np.arange(64, dtype=np.int32)
        for row, (length, pages) in enumerate(zip(self.lengths, self.pages)):
            page_ids = np.asarray(pages, dtype=np.int32)
            table[row, :len(pages)] = page_ids
            expanded = page_ids[:, None] * np.int32(64) + offsets
            indices[self.kv_indptr[row]:self.kv_indptr[row + 1]] = expanded.reshape(-1)[:length]
        return {"page_table": table, "kv_indices": indices,
                "kv_indptr": np.asarray(self.kv_indptr, dtype=np.int32),
                "lengths": np.asarray(self.lengths, dtype=np.int32),
                "positions": np.asarray(self.positions, dtype=np.int64).reshape(self.rows, 1),
                "query_indptr": np.arange(self.rows + 1, dtype=np.int32)}

    def small_arrays(self):
        """CPU scheduler inputs only; token-page IDs are generated on GPU."""
        import numpy as np
        table = self.reservation_table
        if table is None:
            table = np.full((self.rows, self.page_table_width), -1, dtype=np.int32)
            for row, pages in enumerate(self.pages):
                table[row, :len(pages)] = pages
        lengths = np.asarray(self.lengths, dtype=np.int32)
        return {"page_table": table, "kv_indptr": np.asarray(self.kv_indptr, dtype=np.int32),
                "lengths": lengths, "positions": (lengths.astype(np.int64) - 1).reshape(self.rows, 1)}


class MetadataPlanCache:
    """Reuse validated reservations until the manager's mapping epoch changes.

    All allocator APIs update rebuild_version. Normal CPU page-tensor changes
    are additionally detected by identity/pointer/size/version; unversioned
    page tensors still require the allocator epoch protocol. Lengths and row
    order are validated on every use, including the final partial page.
    """
    def __init__(self):
        self.key = None
        self.reservations = None
        self.table = None
        self.hits = self.rebuilds = 0

    def prepare(self, manager, gids, lengths, *, context_cap, gpu_page_count):
        gids, lengths = tuple(gids), tuple(lengths)
        if len(gids) != len(lengths) or len(set(gids)) != len(gids):
            raise ValueError("R1 graph gids must be unique and align with cache lengths")
        binding = []
        for gid in gids:
            pages = manager._sequences[gid].pages
            try:
                version = pages._version
            except RuntimeError:
                version = None
            binding.append((id(pages), pages.data_ptr(), tuple(pages.shape), version))
        key = (id(manager), id(manager._gpu_page_table_manager), manager.get_page_table_version(),
               gids, context_cap, gpu_page_count, tuple(binding))
        if key != self.key:
            import numpy as np
            pages_by_gid = {gid: tuple(int(x) for x in manager._sequences[gid].pages.tolist()) for gid in gids}
            # Validate all usable reservations once, including future pages.
            capacities = tuple(min(len(pages_by_gid[gid]) * 64, context_cap) for gid in gids)
            checked = plan_metadata(gids, capacities, pages_by_gid,
                                    context_cap=context_cap, gpu_page_count=gpu_page_count)
            table = np.full((len(gids), checked.page_table_width), -1, dtype=np.int32)
            for row, pages in enumerate(checked.pages):
                table[row, :len(pages)] = pages
            self.key, self.reservations, self.table = key, checked.pages, table
            self.rebuilds += 1
        else:
            self.hits += 1
        selected, indptr = [], [0]
        for length, pages in zip(lengths, self.reservations):
            if type(length) is not int or not 1 <= length <= context_cap or length > len(pages) * 64:
                raise ValueError("R1 graph row length is invalid or exceeds validated GPU reservation")
            selected.append(pages[:math.ceil(length / 64)])
            indptr.append(indptr[-1] + length)
        return MetadataPlan(gids, lengths, tuple(selected), math.ceil(context_cap / 64),
                            tuple(indptr), gpu_page_count, self.table)


def plan_metadata(gids, lengths, pages_by_gid, *, context_cap, gpu_page_count):
    """Validate the production CPU sequence/page metadata before graph replay."""
    if (type(context_cap) is not int or context_cap <= 0 or context_cap >= 2**31
            or type(gpu_page_count) is not int or gpu_page_count <= 0
            or gpu_page_count > (2**31 - 1) // 64):
        raise ValueError("R1 graph metadata requires positive int32-safe context/page capacities")
    gids, lengths = tuple(gids), tuple(lengths)
    if len(gids) != len(lengths) or len(set(gids)) != len(gids):
        raise ValueError("R1 graph gids must be unique and align with cache lengths")
    if len(gids) * context_cap >= 2**31:
        raise ValueError("R1 graph fixed index-buffer capacity exceeds int32 metadata")
    rows, indptr = [], [0]
    for gid, length in zip(gids, lengths):
        if type(gid) is not int or type(length) is not int or not 1 <= length <= context_cap:
            raise ValueError("R1 graph row id/length is invalid or exceeds context capacity")
        needed = math.ceil(length / 64)
        pages = tuple(pages_by_gid[gid])
        if len(pages) < needed:
            raise ValueError("R1 graph cache length exceeds the real GPU reservation")
        # Only valid history pages are exposed to attention. Future reserved
        # pages are not needed for this token's write or its attention.
        pages = pages[:needed]
        if any(type(page) is not int or not 0 <= page < gpu_page_count for page in pages):
            raise ValueError("R1 graph page ids are outside the real GPU allocator")
        rows.append(pages)
        indptr.append(indptr[-1] + length)
    if indptr[-1] >= 2**31:
        raise ValueError("R1 graph total valid tokens exceed int32 metadata")
    return MetadataPlan(gids, lengths, tuple(rows), math.ceil(context_cap / 64),
                        tuple(indptr), gpu_page_count)


def slice_ranges(rows, micro_batch_size):
    if type(rows) is not int or rows < 0 or type(micro_batch_size) is not int or micro_batch_size < 0:
        raise ValueError("R1 graph rows/microbatch must be nonnegative integers")
    width = micro_batch_size or max(rows, 1)
    return tuple((start, min(start + width, rows)) for start in range(0, rows, width))


class _AttentionPlan:
    """One fixed-row FI wrapper shared by all physical layers of a slice."""
    def __init__(self, rows, context_cap, heads, scale, device):
        import torch
        from .common_flashinfer_backend import _flashinfer, WORKSPACE_BYTES
        flashinfer = _flashinfer()
        self.rows, self.context_cap = rows, context_cap
        self.workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.q_indptr = torch.arange(rows + 1, dtype=torch.int32, device=device)
        self.kv_indptr = torch.empty(rows + 1, dtype=torch.int32, device=device)
        self.kv_indices = torch.empty(rows * context_cap, dtype=torch.int32, device=device)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.wrapper = flashinfer.BatchMLAPagedAttentionWrapper(
            self.workspace, use_cuda_graph=True, qo_indptr=self.q_indptr,
            kv_indptr=self.kv_indptr, kv_indices=self.kv_indices,
            kv_len_arr=self.lengths, backend="fa2",
        )
        if self.wrapper._backend != "fa2" or self.wrapper._use_cuda_graph is not True:
            raise RuntimeError("R1 graph requires explicit FlashInfer 0.5.2 FA2 graph buffers")
        self.heads, self.scale = heads, scale
        self.signature = None
        self.cpu_keepalive = None
        self.fast_initialized = False

    def prepare(self, arrays, start, end):
        import numpy as np
        import torch
        begin, stop = int(arrays["kv_indptr"][start]), int(arrays["kv_indptr"][end])
        # CPU tensors are kept through the next preparation and all intervening
        # layers. FI owns the pinned scheduler staging workspace itself.
        q = torch.arange(self.rows + 1, dtype=torch.int32)
        indptr = torch.from_numpy(np.array(arrays["kv_indptr"][start:end + 1] - begin, copy=True))
        # These are immutable per-forward CPU arrays. from_numpy retains the
        # slice and its backing allocation; cpu_keepalive keeps that storage
        # alive through FI's asynchronous staging and all 61 layers. Avoid a
        # second 6--8 MiB copy of the token-page indices on every decode step.
        indices = torch.from_numpy(arrays["kv_indices"][begin:stop])
        lengths = torch.from_numpy(arrays["lengths"][start:end])
        self.cpu_keepalive = (q, indptr, indices, lengths)
        self.wrapper.plan(q, indptr, indices, lengths, self.heads, 512, 64, 1,
                          False, self.scale, torch.bfloat16, torch.bfloat16)
        # FI 0.5.2 bakes these offsets and grid dimensions into run(). A graph
        # may survive changing lengths ONLY while this entire vector agrees.
        self.signature = tuple(int(value) for value in self.wrapper._plan_info)

    def run(self, query, blocked_k):
        query = query[:, 0]
        cache = blocked_k.view(-1, 1, 576)
        output = self.wrapper.run(query[..., :512], query[..., 512:],
                                  cache[..., :512], cache[..., 512:], return_lse=False)
        return output.unsqueeze(1)

    def prepare_fast(self, arrays, start, end, page_table):
        """FI0.5.2 native scheduler + GPU page expansion, with full plan key."""
        import numpy as np
        import torch
        if not self.fast_initialized:
            from importlib.metadata import version
            if version("flashinfer-python") != "0.5.2":
                raise RuntimeError("Fast R1 metadata requires the qualified FlashInfer0.5.2 API")
        begin = int(arrays["kv_indptr"][start])
        q = torch.arange(self.rows + 1, dtype=torch.int32)
        indptr = torch.from_numpy(np.asarray(arrays["kv_indptr"][start:end + 1] - begin, dtype=np.int32))
        lengths = torch.from_numpy(arrays["lengths"][start:end])
        self.cpu_keepalive = (q, indptr, lengths, arrays)
        if not self.fast_initialized:
            # Device indices alias the wrapper's capture-stable destination;
            # plan() needs only indptr/lengths for its CPU scheduler. The GPU
            # expansion below fills the valid prefix before attention runs.
            self.wrapper.plan(q, indptr, self.kv_indices, lengths, self.heads, 512, 64, 1,
                              False, self.scale, torch.bfloat16, torch.bfloat16)
            self.fast_initialized = True
        else:
            self.kv_indptr.copy_(indptr, non_blocking=True)
            self.lengths.copy_(lengths, non_blocking=True)
            self.wrapper._plan_info = self.wrapper._cached_module.plan(
                self.wrapper._float_workspace_buffer, self.wrapper._int_workspace_buffer,
                self.wrapper._pin_memory_int_workspace_buffer, q, indptr, lengths,
                self.heads, 512, False)
        self.signature = tuple(int(value) for value in self.wrapper._plan_info)
        _index_kernel()[(self.rows, math.ceil(self.context_cap / 256))](
            page_table, self.kv_indptr, self.lengths, self.kv_indices,
            TABLE_WIDTH=page_table.shape[1], ROW_START=start, BLOCK=256)


class PreparedBatch:
    """Persistent device addresses, staged once before the model forward."""
    def __init__(self, rows, micro_batch_size, context_cap, heads, scale, device):
        import torch
        self.rows, self.micro_batch_size = rows, micro_batch_size
        self.context_cap, self.heads, self.scale = context_cap, heads, scale
        self.device = device
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.positions = torch.empty((rows, 1), dtype=torch.int64, device=device)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.page_table = torch.empty((rows, math.ceil(context_cap / 64)), dtype=torch.int32, device=device)
        self.slot_indices = torch.arange(rows, dtype=torch.int32, device=device)
        self.ranges = slice_ranges(rows, micro_batch_size)
        self.plans = tuple(_AttentionPlan(end - start, context_cap, heads, scale, device)
                           for start, end in self.ranges)
        self.cpu_keepalive = None
        self.plan_identity = None
        self.gids = ()
        self.forward_ordinal = 0
        self.source_reservation_table = None
        self.fast_metadata = fast_metadata_requested()
        self.last_prepare_timing_ms = {}

    def prepare(self, plan):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("R1 graph metadata planning/staging must run outside capture")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("R1 graph workspace cannot change its serial owning CUDA stream")
        if plan.rows != self.rows:
            raise ValueError("R1 graph prepared-batch rows changed")
        started = time.perf_counter()
        arrays = plan.small_arrays() if self.fast_metadata else plan.arrays()
        arrays_done = time.perf_counter()
        staged = {key: torch.from_numpy(value) for key, value in arrays.items()
                  if key in ("positions", "lengths", "page_table")}
        self.cpu_keepalive = staged
        self.positions.copy_(staged["positions"], non_blocking=True)
        self.lengths.copy_(staged["lengths"], non_blocking=True)
        table_changed = not self.fast_metadata or self.source_reservation_table is not arrays["page_table"]
        if table_changed:
            self.page_table.copy_(staged["page_table"], non_blocking=True)
            self.source_reservation_table = arrays["page_table"]
        staged_done = time.perf_counter()
        for item, (start, end) in zip(self.plans, self.ranges):
            if self.fast_metadata:
                item.prepare_fast(arrays, start, end, self.page_table)
            else:
                item.prepare(arrays, start, end)
        plans_done = time.perf_counter()
        self.plan_identity = tuple(item.signature for item in self.plans)
        self.gids = plan.gids
        self.forward_ordinal += 1
        self.last_prepare_timing_ms = {"metadata_CPU_arrays": (arrays_done - started) * 1000,
            "metadata_small_staging": (staged_done - arrays_done) * 1000,
            "metadata_FI_plan_and_indices_submit": (plans_done - staged_done) * 1000,
            "metadata_table_refreshed": int(table_changed)}


class R1DecodeGraphPool:
    """Model-owned metadata, with no references to model/weights or GPU manager."""
    def __init__(self, context_cap, heads, scale, device):
        self.context_cap, self.heads, self.scale, self.device = context_cap, heads, scale, device
        self.prepared_batch = None
        self.metadata_cache = MetadataPlanCache()

    def prepare(self, plan, micro_batch_size):
        batch = self.prepared_batch
        if batch is None or (batch.rows, batch.micro_batch_size) != (plan.rows, micro_batch_size):
            batch = PreparedBatch(plan.rows, micro_batch_size, self.context_cap,
                                  self.heads, self.scale, self.device)
            self.prepared_batch = batch
        batch.prepare(plan)
        return batch


@contextmanager
def forward_scope(model, manager, gids, lengths, *, context_cap, micro_batch_size):
    """Prepare metadata outside capture; callable also by a whole-model adapter.

    A future whole-model graph can invoke compute_attention() directly for
    each layer, capture LL MoE with external_valid_rows=True, and stage every
    layer's returned KV after replay. Replan/capture identity remains explicit.
    """
    if _FORWARD.get() is not None:
        raise RuntimeError("Nested R1 graph forward scopes are unsupported")
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("R1 graph model entry must prepare outside capture")
    if list(gids) != list(manager._gpu_page_table_manager.slot_to_seq_id):
        raise ValueError("R1 graph row order differs from the active GPU page table")
    if not gids:
        token = _FORWARD.set((None, manager))
        try:
            yield None
        finally:
            _FORWARD.reset(token)
        return
    first = model.model.layers[0].self_attn
    first = getattr(first, "module", first)
    pool = getattr(model, "_r1_decode_cuda_graph_pool", None)
    geometry = (context_cap, first.num_heads, first.softmax_scale, manager.device)
    if pool is None:
        pool = R1DecodeGraphPool(*geometry)
        model._r1_decode_cuda_graph_pool = pool
    elif (pool.context_cap, pool.heads, pool.scale, pool.device) != geometry:
        raise ValueError("R1 graph model geometry/device changed without releasing graphs")
    plan_started = time.perf_counter()
    if fast_metadata_requested():
        plan = pool.metadata_cache.prepare(manager, gids, lengths, context_cap=context_cap,
                                           gpu_page_count=int(manager.config.num_pages))
    else:
        pages = {int(gid): tuple(int(page) for page in manager._sequences[gid].pages.tolist()) for gid in gids}
        plan = plan_metadata(gids, lengths, pages, context_cap=context_cap,
                             gpu_page_count=int(manager.config.num_pages))
    plan_done = time.perf_counter()
    batch = pool.prepare(plan, micro_batch_size)
    batch.last_prepare_timing_ms["metadata_CPU_validation"] = (plan_done - plan_started) * 1000
    token = _FORWARD.set((batch, manager))
    try:
        yield batch
    finally:
        _FORWARD.reset(token)


def compute_attention(wrapper, hidden_states, batch, blocked_k):
    """Capture-safe existing R1 FP8/BF16 attention math, with static metadata.

    No Host callback, CPU tensor read, FI plan, allocator mutation, collective,
    or actual-batch padding occurs here. Valid row order is retained.
    """
    import torch
    from . import flashmla_backend as native
    from batchgen.kv_cache.gpu_paged_kv_manager import run_paged_kv_token_update_fused
    module, weights = wrapper.module, wrapper.weight_dequant_scale
    outputs, kv_rows = [], []
    for (start, end), item in zip(batch.ranges, batch.plans):
        x = hidden_states[start:end].squeeze(1)
        rows = end - start
        x_fp8, x_scale = native.act_quant(x)
        q = native.w8a8_deepgemm(x_fp8, x_scale, module.q_a_proj.weight,
                                weights["q_a_proj.weight_scale_inv"])
        compressed = native.w8a8_deepgemm(x_fp8, x_scale, module.kv_a_proj_with_mqa.weight,
                                         weights["kv_a_proj_with_mqa.weight_scale_inv"]).view(rows, 1, -1)
        q = module.q_a_layernorm(q)
        q, q_scale = native.act_quant(q)
        q = native.w8a8_deepgemm(q, q_scale, module.q_b_proj.weight,
                                weights["q_b_proj.weight_scale_inv"])
        q = q.view(rows, 1, module.num_heads, module.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.contiguous()
        cos, sin = module.rotary_emb(q_pe, seq_len=batch.context_cap)
        _retain_capture_inputs(cos, sin)
        positions = batch.positions[start:end]
        kv = native.fused_rmsnorm_rope_with_q(
            compressed, q_pe, cos, sin, positions, module.kv_a_layernorm.weight,
            module.kv_lora_rank, module.qk_rope_head_dim,
            output_layout=native._r1_paged_rope_output_layout(module),
        ).view(rows, 1, 1, 576)
        run_paged_kv_token_update_fused(
            k_cache=blocked_k, k_tokens=kv.view(rows, -1),
            page_table=batch.page_table, slot_indices=batch.slot_indices[start:end],
            token_indices=positions[:, 0].to(torch.int32), page_size_tokens=64,
            v_cache=None, v_tokens=None,
        )
        expanded = native.get_mla_decode_weight(module, weights, native.deepseek_v3_dequantization)
        _retain_capture_inputs(expanded, *(value for _, value in attention_source_tensors(wrapper)))
        expanded = expanded.view(module.num_heads, -1, module.kv_lora_rank)
        q_absorb = expanded[:, :module.qk_nope_head_dim, :]
        out_absorb = expanded[:, module.qk_nope_head_dim:, :]
        query = torch.empty(rows, module.num_heads, 1, 576, dtype=blocked_k.dtype, device=blocked_k.device)
        query[..., :512] = torch.einsum("bhd,hdc->bhc", q_nope.squeeze(2), q_absorb).view(rows, module.num_heads, 1, 512)
        query[..., 512:] = q_pe
        attended = item.run(query.view(rows, 1, module.num_heads, 576), blocked_k)
        output = torch.einsum("bqhc,hdc->bhqd", attended, out_absorb)
        output = output.transpose(1, 2).contiguous().reshape(rows, module.num_heads * module.v_head_dim)
        output, output_scale = native.act_quant(output)
        output = native.w8a8_deepgemm(output, output_scale, module.o_proj.weight,
                                     weights["o_proj.weight_scale_inv"])
        outputs.append(output.view(rows, 1, -1))
        kv_rows.append(kv)
    if len(outputs) == 1:
        # The one-slice profile already owns contiguous output/KV tensors.
        # Avoid copying both again merely to concatenate a single element.
        return outputs[0], kv_rows[0]
    return torch.cat(outputs, dim=0), torch.cat(kv_rows, dim=0)


class _LayerGraph:
    def __init__(self, wrapper, hidden_states, batch, blocked_k, signature):
        import torch
        self.batch = batch
        self.capture_inputs = {}
        self.static_hidden = torch.empty_like(hidden_states)
        self.static_hidden.copy_(hidden_states)
        # All module caches/JIT kernels initialize before entering capture.
        stream = torch.cuda.Stream(device=hidden_states.device)
        current = torch.cuda.current_stream(hidden_states.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                compute_attention(wrapper, self.static_hidden, batch, blocked_k)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        keepalive = _CAPTURE_INPUTS.set(self.capture_inputs)
        try:
            with torch.cuda.graph(self.graph, stream=stream):
                self.output, self.kv = compute_attention(wrapper, self.static_hidden, batch, blocked_k)
        finally:
            _CAPTURE_INPUTS.reset(keepalive)
        # Warmup may have built a larger RoPE cache. Record the signature only
        # after it exists, and keep the exact captured input storage alive.
        self.signature = signature[:-1] + (attention_source_signature(wrapper),)
        current.wait_stream(stream)
        self.replays = 0

    def replay(self, hidden_states):
        self.static_hidden.copy_(hidden_states)
        self.graph.replay()
        self.replays += 1
        return self.output, self.kv


def release_attention_graph(wrapper):
    # Called before any weight .data replacement, .to(), or clear_weights().
    owner = getattr(wrapper, "_r1_whole_decode_owner", None)
    if owner is not None:
        owner.invalidate(reason="attention_weights_reloaded")
    wrapper.__dict__.pop("_r1_decode_attention_graph", None)


class WholeModelKVOutputs:
    """Static KV outputs for a separately owned whole-decode graph adapter.

    The adapter must replay its graph before calling stage_after_replay, on
    the same stream as the replay. An ordinal prevents duplicate Host appends.
    This class does not create/replay the whole model or configure DeepEP.
    """
    def __init__(self, expected_layers, capture_stream):
        self.expected_layers = tuple(expected_layers)
        if len(set(self.expected_layers)) != len(self.expected_layers):
            raise ValueError("Whole-model expected layers must be unique")
        self.capture_stream = capture_stream
        self.kv_by_layer = {}
        self.capture_inputs = {}
        self.last_staged_replay = 0

    def record(self, layer, kv):
        if layer not in self.expected_layers:
            raise ValueError("Unexpected layer in R1 whole-model attention capture")
        self.kv_by_layer[layer] = kv

    def stage_after_replay(self, callback, *, replay_ordinal):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Whole-model Host KV staging must run after capture/replay")
        if type(replay_ordinal) is not int or replay_ordinal <= self.last_staged_replay:
            raise ValueError("A whole-model graph replay may stage Host KV only once")
        if set(self.kv_by_layer) != set(self.expected_layers):
            raise ValueError("Whole-model graph did not expose every required layer's KV")
        # Caller owns the replay order and source completion stream. Existing
        # native D2H records its producer event on this issuing current stream.
        for layer in self.expected_layers:
            callback(layer, self.kv_by_layer[layer].clone(), None)
        self.last_staged_replay = replay_ordinal


@contextmanager
def whole_model_attention_scope(outputs):
    """Let a whole-model adapter capture existing wrapper dispatch directly.

    Enter forward_scope outside capture first, then warmup/capture the model
    under this scope on outputs.capture_stream. LL must be configured with
    external_valid_rows=True; its per-replay counts and FA2 metadata updates
    remain outside capture. This scope suppresses all attention Host callbacks.
    """
    if _FORWARD.get() is None or _WHOLE_MODEL.get() is not None:
        raise RuntimeError("Whole-model attention capture requires one prepared forward scope")
    token = _WHOLE_MODEL.set(outputs)
    inputs_token = _CAPTURE_INPUTS.set(outputs.capture_inputs)
    try:
        yield outputs
    finally:
        _CAPTURE_INPUTS.reset(inputs_token)
        _WHOLE_MODEL.reset(token)


def _compare_eager_attention(wrapper, hidden_states, batch, manager, output, kv):
    """Explicit diagnostic only: original backend, no second Host append.

    This intentionally synchronizes for validation and is excluded from any
    performance measurement. The same last-token KV is written again at the
    same slot; no context or sequence state advances during this comparison.
    """
    import torch
    from batchgen.models.wrappers.attention import AttnWrapperBase
    from .common_flashinfer_backend import decode_forward_scope
    from .decode_position_validation import mla_decode_position_scope
    from .decode_microbatch_scope import decode_attention_microbatch_scope
    positions, lengths = AttnWrapperBase.position_ids, AttnWrapperBase.cache_seqlens
    expected_output, expected_kv = [], []
    with mla_decode_position_scope(positions), decode_forward_scope(manager):
        for start, end in batch.ranges:
            if len(batch.ranges) > 1:
                with decode_attention_microbatch_scope(manager, positions, lengths, start, end) as (p, c):
                    y, k = wrapper.module.decoding_attn_mode_3_bf16(
                        hidden_states[start:end], p, c, AttnWrapperBase.max_seqlen,
                        wrapper.weight_dequant_scale, manager, wrapper.layer_idx, (start, end))
            else:
                y, k = wrapper.module.decoding_attn_mode_3_bf16(
                    hidden_states, positions, lengths, AttnWrapperBase.max_seqlen,
                    wrapper.weight_dequant_scale, manager, wrapper.layer_idx, None)
            expected_output.append(y)
            expected_kv.append(k)
    reference = torch.cat(expected_output, dim=0)
    reference_kv = torch.cat(expected_kv, dim=0)
    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
    if not torch.equal(kv, reference_kv):
        raise RuntimeError("R1 graph/current native eager last-token KV differs")
    error = (output.float() - reference.float()).abs().max().item()
    logging.info("[R1_ATTN_GRAPH_COMPARE] layer=%s rows=%s output_max_abs=%.9g KV=bitwise_equal native_eager=pass diagnostic_sync=true",
                 wrapper.layer_idx, batch.rows, error)


def run_attention_graph(wrapper, hidden_states):
    import torch
    from batchgen.models.wrappers.attention import AttnWrapperBase
    state = _FORWARD.get()
    if state is None:
        raise RuntimeError("R1 graph wrapper ran without its prepared model forward scope")
    batch, manager = state
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("R1 attention graph requires [rows,1,hidden] inputs")
    if hidden_states.shape[0] == 0:
        if batch is not None:
            raise ValueError("R1 empty graph rows differ from the prepared batch")
        return hidden_states, None, None
    if batch is None or hidden_states.shape[0] != batch.rows:
        raise ValueError("R1 graph rows differ from the actual prepared local batch")
    if wrapper.decode_attn_micro_batch_size != batch.micro_batch_size:
        raise ValueError("R1 attention microbatch configuration changed within the model")
    blocked_k, _, _ = manager.get_layer_kv_with_page_table(wrapper.layer_idx)
    if tuple(blocked_k.shape[1:]) != (64, 1, 576) or blocked_k.dtype != torch.bfloat16:
        raise ValueError("R1 attention graph requires native page64 MLA576 BF16 cache")
    whole = _WHOLE_MODEL.get()
    if whole is not None:
        if torch.cuda.current_stream(hidden_states.device).cuda_stream != whole.capture_stream:
            raise RuntimeError("R1 whole-model attention capture used an unexpected CUDA stream")
        output, kv = compute_attention(wrapper, hidden_states, batch, blocked_k)
        whole.record(wrapper.layer_idx, kv)
        return output, None, None
    if (torch.cuda.is_current_stream_capturing()
            or torch.cuda.current_stream(hidden_states.device).cuda_stream != batch.stream):
        raise RuntimeError("R1 attention segmented replay requires its owning current CUDA stream")
    geometry = (id(batch), batch.plan_identity, blocked_k.data_ptr(), tuple(blocked_k.shape),
                tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device)
    signature = geometry + (attention_source_signature(wrapper),)
    record = getattr(wrapper, "_r1_decode_attention_graph", None)
    if record is None or record.signature != signature:
        previous_replays = record.replays if record is not None else 0
        release_attention_graph(wrapper)
        record = _LayerGraph(wrapper, hidden_states, batch, blocked_k, signature)
        wrapper._r1_decode_attention_graph = record
        logging.info("[R1_ATTN_GRAPH_CAPTURE] layer=%s rows=%s slices=%s prior_replays=%s plan_info=%s",
                     wrapper.layer_idx, batch.rows, [end-start for start, end in batch.ranges],
                     previous_replays, batch.plan_identity)
    output, kv = record.replay(hidden_states)
    if record.replays == 1 and os.environ.get("BATCHGEN_R1_GRAPH_COMPARE", "0") == "1":
        _compare_eager_attention(wrapper, hidden_states, batch, manager, output, kv)
    if record.replays == 1 or record.replays % 64 == 0:
        logging.info("[R1_ATTN_GRAPH_REPLAY] layer=%s rows=%s slices=%s replays=%s Host_callback=post_replay MoE=outside_attention_graph",
                     wrapper.layer_idx, batch.rows, [end-start for start, end in batch.ranges], record.replays)
    # Static graph KV storage is overwritten on the next replay. Clone on the
    # same current stream before handing it to existing asynchronous Host D2H.
    # Captures/warmups never append: exactly one append/layer/real decode step.
    callback = AttnWrapperBase.kv_append_callback
    if callback is not None:
        callback(wrapper.layer_idx, kv.clone(), None)
    return output, None, None
