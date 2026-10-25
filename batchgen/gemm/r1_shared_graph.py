"""Bounded per-stream CUDA graph for the unchanged resident R1 shared expert.

No collectives, routing, quantization changes, or prefill offload are captured.
The caller binds cached FP8 parameters before entering this helper.
"""
from types import SimpleNamespace
import logging
import torch


def forward(wrapper, hidden):
    compute = lambda x: wrapper.micro_batch_forward(x, 'expert')
    if hidden.numel() == 0 or torch.cuda.is_current_stream_capturing():
        return compute(hidden)
    parameters = [getattr(wrapper.module, name).weight for name in ('gate_proj','up_proj','down_proj')]
    tensors = parameters + [wrapper.fp8_gate,wrapper.fp8_up,wrapper.fp8_down]
    tensors += [wrapper.weight_dequant_scale[k]for k in sorted(wrapper.weight_dequant_scale)]
    try:
        sources = tuple((t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.dtype,t.device)for t in tensors)
    except RuntimeError:
        # Inference-created unversioned weights cannot prove cache validity.
        wrapper._r1_shared_graph_unversioned_fallbacks = getattr(wrapper, '_r1_shared_graph_unversioned_fallbacks', 0) + 1
        return compute(hidden)
    current = torch.cuda.current_stream(hidden.device)
    cap = wrapper.get_batch_size('expert')
    key = (tuple(hidden.shape),tuple(hidden.stride()),hidden.dtype,hidden.device,current.cuda_stream,
           cap,wrapper.phase,sources)
    entries = getattr(wrapper,'_r1_shared_graphs',None)
    if entries is None:
        entries={};wrapper._r1_shared_graphs=entries
    stream_id = (hidden.device,current.cuda_stream)
    entry = entries.get(stream_id)
    if entry is None or entry.key != key:
        entry = SimpleNamespace(key=key,aliases=[t.detach()for t in tensors],hidden=torch.empty_like(hidden))
        entry.hidden.copy_(hidden)
        capture = torch.cuda.Stream(device=hidden.device)
        capture.wait_stream(current)
        with torch.cuda.stream(capture):
            for _ in range(3):compute(entry.hidden)
        current.wait_stream(capture)
        entry.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(entry.graph,stream=capture):entry.output=compute(entry.hidden)
        current.wait_stream(capture)
        # Keep at most one shape/source version on each caller stream.
        entries[stream_id]=entry
        wrapper._r1_shared_graph_captures=getattr(wrapper,'_r1_shared_graph_captures',0)+1
        if wrapper.layer_idx == 3:
            logging.info('[R1_SHARED_GRAPH] rank=%d layer=3 batch=%d captures=%d caller_stream=%d',
                         wrapper.get_rank_safe(), hidden.shape[0], wrapper._r1_shared_graph_captures, current.cuda_stream)
    entry.hidden.copy_(hidden)
    entry.graph.replay()
    wrapper._r1_shared_graph_replays = getattr(wrapper, '_r1_shared_graph_replays', 0) + 1
    # Never expose reusable graph storage to a later layer or async consumer.
    return entry.output.clone()
