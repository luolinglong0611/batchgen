"""Fixed-capacity R1 routed compute; collectives remain outside the graph.

Only GPU metadata determines which dispatched rows are valid. Uninitialized
dispatch tails must never become inverse-mapping indices or GEMM inputs.
"""
import logging
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_valid_prefix(tokens, positions, valid_count, mapping,
                          CAPACITY: tl.constexpr, TOKENS: tl.constexpr,
                          TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < CAPACITY) & (row < tl.load(valid_count))
    token = tl.load(tokens + row, valid, other=-1)
    position = tl.load(positions + row, valid, other=-1)
    valid &= (token >= 0) & (token < TOKENS) & (position >= 0) & (position < TOPK)
    tl.store(mapping + token * TOPK + position, row.to(tl.int64), valid)


def inverse_mapping(tokens, positions, valid_count, num_tokens, topk):
    mapping = torch.full((num_tokens, topk), -1, dtype=torch.int64, device=tokens.device)
    if tokens.numel():
        _inverse_valid_prefix[(triton.cdiv(tokens.numel(), 256),)](
            tokens, positions, valid_count, mapping, tokens.numel(), num_tokens, topk, 256,
        )
    return mapping


def _sources(moe):
    tensors = list(moe.gate.parameters())
    # Some router corrections are buffers rather than Parameters.
    tensors.extend(moe.gate.buffers())
    for name in ('gate_list', 'up_list', 'down_list', 'gate_scale_list',
                 'up_scale_list', 'down_scale_list'):
        tensors.extend(getattr(moe, name))
    for name in ('gate_ptrs_ptr', 'up_ptrs_ptr', 'down_ptrs_ptr',
                 'gate_scale_ptrs_ptr', 'up_scale_ptrs_ptr', 'down_scale_ptrs_ptr'):
        tensors.append(getattr(moe, name))
    # The pinned native fused dispatch derives indices itself and never reads
    # token_idx/topk_pos. Those compatibility arguments may be inference tensors;
    # they are neither graph sources nor a reason to fall back to eager.
    return tensors


def forward(moe, hidden):
    """One bounded shape/source entry per caller stream; retain output ownership."""
    compute = moe._r1_local_moe_compute
    if hidden.numel() == 0 or torch.cuda.is_current_stream_capturing():
        return compute(hidden)
    tensors = _sources(moe)
    try:
        # Registered weight layouts are fixed for this decode instance. Rebinding
        # a Tensor/storage or an ordinary tracked in-place mutation invalidates
        # the entry. Avoid constructing >100 layout/device tuples per layer and
        # step; a new decode instance owns a new cache after a phase rebuild.
        sources = tuple((id(t), t.data_ptr(), t._version) for t in tensors)
    except RuntimeError:
        moe._r1_local_graph_unversioned_fallbacks = getattr(moe, '_r1_local_graph_unversioned_fallbacks', 0) + 1
        return compute(hidden)
    current = torch.cuda.current_stream(hidden.device)
    gate_config = tuple((name, getattr(moe.gate, name, None)) for name in
                        ('n_group', 'topk_group', 'n_routed_experts', 'top_k',
                         'routed_scaling_factor', 'norm_topk_prob', 'mode'))
    key = (tuple(hidden.shape), tuple(hidden.stride()), hidden.dtype, hidden.device,
           moe.num_tokens_per_rank, moe.routed_expert_start_idx, moe.routed_expert_end_idx,
           moe.num_experts_per_tok, gate_config, sources)
    stream_id = (hidden.device, current.cuda_stream)
    entries = getattr(moe, '_r1_local_graphs', None)
    if entries is None:
        entries = {}; moe._r1_local_graphs = entries
    entry = entries.get(stream_id)
    if entry is None or entry.key != key:
        entry = SimpleNamespace(key=key, aliases=[t.detach() for t in tensors], hidden=torch.empty_like(hidden))
        entry.hidden.copy_(hidden)
        capture = torch.cuda.Stream(device=hidden.device)
        capture.wait_stream(current)
        with torch.cuda.stream(capture):
            for _ in range(3):
                compute(entry.hidden)
        current.wait_stream(capture)
        entry.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(entry.graph, stream=capture):
            entry.output = compute(entry.hidden)
        current.wait_stream(capture)
        entries[stream_id] = entry
        moe._r1_local_graph_captures = getattr(moe, '_r1_local_graph_captures', 0) + 1
        logging.info('[R1_LOCAL_MOE_GRAPH] rank=%d global_tokens=%d captures=%d caller_stream=%d',
                     moe.rank, hidden.shape[0], moe._r1_local_graph_captures, current.cuda_stream)
    entry.hidden.copy_(hidden)
    entry.graph.replay()
    moe._r1_local_graph_replays = getattr(moe, '_r1_local_graph_replays', 0) + 1
    return entry.output.clone()
