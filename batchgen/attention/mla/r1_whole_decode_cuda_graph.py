"""PRIVATE CANDIDATE: coordinated R1 whole-decode CUDA Graph with DeepEP-v1 LL.

Never silently falls back. Segmented attention remains a separately selected
debug path. Metadata planning, CPU coordination, Host appends and sampling stay
outside capture. No GPU qualification is implied by this source file.
"""
from contextlib import ExitStack, contextmanager
from datetime import timedelta
import json
import logging
import os
import time
from types import SimpleNamespace


ENVIRONMENT = "BATCHGEN_R1_WHOLE_DECODE_CUDA_GRAPH"
COMPARE_ENVIRONMENT = "BATCHGEN_R1_WHOLE_GRAPH_COMPARE"
_MISSING_BINDING = object()


@contextmanager
def _trace_range(enabled, name):
    if enabled:
        import torch
        torch.cuda.nvtx.range_push("BatchGen.decode." + name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def comparison_requested():
    value = os.environ.get(COMPARE_ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(COMPARE_ENVIRONMENT + " must be explicitly 0 or 1")
    return value == "1"


def _compare_tensor(graph_value, eager_value):
    """Same-engine diagnostic on CPU; never called in performance mode."""
    import torch
    if graph_value.shape != eager_value.shape:
        raise ValueError("Graph/eager diagnostic shape differs")
    graph, eager = graph_value.detach().float().cpu(), eager_value.detach().float().cpu()
    if not graph.numel():
        return {"empty": True}
    if not bool(torch.isfinite(graph).all() and torch.isfinite(eager).all()):
        raise ValueError("Graph/eager diagnostic has non-finite values")
    delta = graph - eager
    rms_reference = float(eager.square().mean().sqrt())
    rmse = float(delta.square().mean().sqrt())
    nrmse = rmse / max(rms_reference, 1e-12)
    max_abs = float(delta.abs().max())
    torch.testing.assert_close(graph, eager, rtol=0.01, atol=0.02)
    if nrmse > 0.005:
        raise ValueError("Graph/eager diagnostic NRMSE exceeds 0.5%: " + str(nrmse))
    return {"rms_eager": rms_reference, "rmse": rmse, "nrmse": nrmse,
            "max_abs": max_abs, "rtol": 0.01, "atol": 0.02, "nrmse_limit": 0.005}


def requested():
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(ENVIRONMENT + " must be explicitly 0 or 1")
    return value == "1"


def validate_profile(*, model_identifier, enabled, disabled, deepep):
    if not requested():
        return False
    from .r1_decode_cuda_graph import requested as attention_requested
    from .r1_decode_cuda_graph import validate_profile as attention_profile
    if not attention_requested() or deepep is not True:
        raise ValueError("R1 whole decode requires the R1 attention graph profile and --enable-deepep")
    attention_profile(model_identifier=model_identifier, enabled=enabled, disabled=disabled)
    return True


class WholeDecodeCoordinator:
    """CPU-only control collectives; every rank takes the same capture branch."""
    def __init__(self, group, *, world_size, rank, dist_api=None):
        import torch
        import torch.distributed as dist
        self.group, self.world_size, self.rank = group, world_size, rank
        self.dist = dist if dist_api is None else dist_api
        # Gloo's backend thread writes this tensor outside the caller's
        # inference-mode context, so it must be a normal, versioned CPU tensor.
        with torch.inference_mode(False):
            self.flags = torch.zeros(2, dtype=torch.int32, device="cpu")

    def agree(self, stage, *, changed=False, error=None):
        self.flags[0] = int(bool(changed))
        self.flags[1] = int(error is not None)
        self.dist.all_reduce(self.flags, op=self.dist.ReduceOp.MAX, group=self.group)
        if int(self.flags[1]):
            errors = [None] * self.world_size
            self.dist.all_gather_object(errors,
                {"rank": self.rank, "stage": stage,
                 "error": None if error is None else str(error)}, group=self.group)
            details = [entry for entry in errors if entry["error"] is not None]
            raise RuntimeError("R1 whole-decode collective abort: " + repr(details))
        return bool(int(self.flags[0]))


def _tensor_signature(tensor, *, version=False):
    if tensor is None:
        return None
    identity = (int(tensor.data_ptr()), tuple(tensor.shape), tuple(tensor.stride()),
                str(tensor.dtype), str(tensor.device))
    if version:
        try:
            value = tensor._version
        except RuntimeError:
            value = None
        identity += (value,)
    return identity


def _weight_signature(model):
    """Check source pointers without traversing all placeholder expert modules."""
    result = []
    def projections(owner):
        module = getattr(owner, "module", owner)
        for item in getattr(module, "_r1_native_gemm_weights", ()):
            result.extend(_tensor_signature(value, version=True)
                          for value in (item.weight, item.scale))
        for key in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(module, key, None)
            if projection is not None:
                result.append(_tensor_signature(projection.weight, version=True))
        result.extend((key, _tensor_signature(value, version=True))
                      for key, value in sorted(getattr(owner, "weight_dequant_scale", {}).items()))
    for layer in model.model.layers:
        wrapper = layer.self_attn
        attn = getattr(wrapper, "module", wrapper)
        result.append(_attention_signature(wrapper))
        cache_entry = getattr(attn, "_mla_decode_weight_cache_entry", None)
        result.append(_tensor_signature(None if cache_entry is None else cache_entry[-1]))
        mlp = layer.mlp
        gate = getattr(mlp, "gate", None)
        for name in ("weight", "e_score_correction_bias"):
            value = getattr(gate, name, None)
            if value is not None:
                result.append(_tensor_signature(value, version=True))
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, norm_name, None)
            if norm is not None:
                result.append(_tensor_signature(norm.weight, version=True))
        if getattr(mlp, "r1_deepep_adapter", None) is not None:
            adapter = mlp.r1_deepep_adapter
            native_weights = getattr(adapter, "graph_weight_tensors", None)
            if callable(native_weights):
                weights = tuple(native_weights())
                if not weights or any(value is None for value in weights):
                    raise RuntimeError("Native R1 MoE graph requires its actual weight tensors")
                result.append((type(adapter).__name__, tuple(
                    _tensor_signature(value, version=True) for value in weights)))
            else:
                for name in ("gate_ptrs_ptr", "up_ptrs_ptr", "down_ptrs_ptr",
                             "gate_scale_ptrs_ptr", "up_scale_ptrs_ptr", "down_scale_ptrs_ptr"):
                    result.append(_tensor_signature(getattr(mlp, name), version=True))
            projections(mlp.shared_experts)
        else:
            projections(mlp)
    for owner in (getattr(model.model, "embed_tokens", None),
                  getattr(model.model, "norm", None)):
        if owner is not None:
            result.append(_tensor_signature(owner.weight, version=True))
    result.append(_tensor_signature(model.lm_head.weight, version=True))
    return tuple(result)


def _attention_signature(wrapper):
    from .r1_decode_cuda_graph import attention_source_signature
    return attention_source_signature(wrapper)


def fast_weight_guard_requested():
    value = os.environ.get("BATCHGEN_R1_DECODE_FAST_WEIGHT_GUARD", "0")
    if value not in ("0", "1"):
        raise ValueError("BATCHGEN_R1_DECODE_FAST_WEIGHT_GUARD must be 0 or 1")
    return value == "1"


def _guard_tensor_key(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    # Retain pointer/layout checks as well as normal inplace version guards:
    # a .data rebind can bypass the version counter. Tensor data writes via
    # external aliases still require the owner's existing invalidation hook.
    return (tensor.data_ptr(), version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor.storage_offset())


class CapturedWeightSignature:
    """Flat binding/version/layout guards over the captured weight owners.

    Stable steps reuse the complete signature, avoiding repeated model-tree
    traversal, scale sorting, storage wrappers and dtype/device formatting.
    Every live Tensor is still checked each step; module/Parameter/scale/bank
    replacement and structural changes force a fresh scan and graph capture.
    """
    def __init__(self, model):
        self.revision = 0
        self.refreshes = 0
        self._capture(model)

    def _capture(self, model):
        self.model = model
        self.bindings, self.sequences, self.mappings, self.native_apis = [], [], [], []
        tensors, dictionaries = {}, set()
        def mapping(value):
            if id(value) not in dictionaries:
                dictionaries.add(id(value));self.mappings.append((value, len(value)))
        def attr(owner, key, default=None):
            if owner is None:
                return default
            storage = vars(owner)
            value = getattr(owner, key, default)
            if key in storage:
                self.bindings.append((storage, key, value))
            else:
                found = False
                for child in ("_parameters", "_buffers", "_modules"):
                    values = storage.get(child)
                    if isinstance(values, dict) and key in values:
                        # Guard dictionary replacement as well as entry writes.
                        self.bindings.append((storage, child, values))
                        mapping(values)
                        self.bindings.append((values, key, value));found = True;break
                if not found:
                    # Optional missing attributes must remain missing. Normal
                    # nn.Module registrations occur in one of these mappings.
                    self.bindings.append((storage, key, _MISSING_BINDING))
                    for child in ("_parameters", "_buffers", "_modules"):
                        values = storage.get(child)
                        if isinstance(values, dict):
                            self.bindings.append((storage, child, values));mapping(values)
            return value
        def tensor(value):
            if value is not None and id(value) not in tensors:
                tensors[id(value)] = (value, _guard_tensor_key(value))
        def source_scales(owner):
            values = attr(owner, "weight_dequant_scale", {})
            mapping(values)
            for key, value in values.items():
                self.bindings.append((values, key, value));tensor(value)
        def prepared(owner):
            bank = attr(owner, "_r1_native_gemm_weights", ())
            self.sequences.append((bank, tuple(bank)))
            for item in bank:
                tensor(attr(item, "weight"));tensor(attr(item, "scale"))
        def projections(owner):
            module = attr(owner, "module", owner)
            prepared(module)
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = attr(module, name)
                if projection is not None:
                    tensor(attr(projection, "weight"))
            source_scales(owner)
        inner = attr(model, "model")
        layers = attr(inner, "layers")
        self.sequences.append((layers, tuple(layers)))
        attr(model, "_r1_weight_generation")
        for layer in layers:
            wrapper = attr(layer, "self_attn")
            attn = attr(wrapper, "module", wrapper)
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                         "q_a_layernorm", "kv_a_layernorm"):
                projection = attr(attn, name)
                if projection is not None:
                    tensor(attr(projection, "weight"))
            source_scales(wrapper);prepared(attn)
            rotary = attr(attn, "rotary_emb")
            for name in ("cos_cached", "sin_cached"):
                tensor(attr(rotary, name))
            entry = attr(attn, "_mla_decode_weight_cache_entry")
            tensor(None if entry is None else entry[-1])
            for name in ("input_layernorm", "post_attention_layernorm"):
                norm = attr(layer, name)
                if norm is not None:
                    tensor(attr(norm, "weight"))
            mlp = attr(layer, "mlp")
            gate = attr(mlp, "gate")
            if gate is not None:
                for name in ("weight", "e_score_correction_bias"):
                    tensor(attr(gate, name))
            adapter = attr(mlp, "r1_deepep_adapter")
            if adapter is not None:
                native = getattr(adapter, "graph_weight_tensors", None)
                if callable(native):
                    values = tuple(native())
                    if not values or any(value is None for value in values):
                        raise RuntimeError("Native R1 MoE graph requires actual weight tensors")
                    self.native_apis.append((adapter, values))
                    for value in values:
                        tensor(value)
                else:
                    for name in ("gate_ptrs_ptr", "up_ptrs_ptr", "down_ptrs_ptr",
                                 "gate_scale_ptrs_ptr", "up_scale_ptrs_ptr", "down_scale_ptrs_ptr"):
                        tensor(attr(mlp, name))
                projections(attr(mlp, "shared_experts"))
            else:
                projections(mlp)
        for name in ("embed_tokens", "norm"):
            owner = attr(inner, name)
            if owner is not None:
                tensor(attr(owner, "weight"))
        tensor(attr(attr(model, "lm_head"), "weight"))
        self.tensors = tuple(tensors.values())
        self.full_signature = _weight_signature(model)
        self.refreshes += 1
        self.key = (id(self), self.revision, self.full_signature)

    def signature(self, model):
        changed = model is not self.model
        if not changed:
            changed = any(values.get(key, _MISSING_BINDING) is not expected for values, key, expected in self.bindings)
        if not changed:
            changed = any(len(values) != size for values, size in self.mappings)
        if not changed:
            changed = any(len(seq) != len(expected) or any(a is not b for a, b in zip(seq, expected))
                          for seq, expected in self.sequences)
        if not changed:
            for adapter, values in self.native_apis:
                current = tuple(adapter.graph_weight_tensors())
                if len(current) != len(values) or any(a is not b for a, b in zip(current, values)):
                    changed = True
                    break
        if not changed:
            changed = any(_guard_tensor_key(value) != expected for value, expected in self.tensors)
        if changed:
            self.revision += 1
            self._capture(model)
        return self.key


def _signature(worker, model, manager, batch, pool, input_ids, invalidations, weight_guard=None):
    import torch
    # An initial zero-row rank may have no active GPU page table at all.
    # Its ghost graph executes only LL expert work and captures no KV input.
    caches = (() if batch is None else tuple(
        _tensor_signature(manager.get_layer_kv_with_page_table(i)[0])
        for i in range(len(model.model.layers))))
    metadata = None
    if batch is not None:
        metadata = (id(batch), batch.rows, batch.micro_batch_size, batch.context_cap,
            batch.plan_identity, batch.stream,
            tuple(_tensor_signature(getattr(batch, key))
                  for key in ("positions", "lengths", "page_table", "slot_indices")),
            tuple((tuple(_tensor_signature(getattr(plan, key))
                        for key in ("workspace", "q_indptr", "kv_indptr", "kv_indices", "lengths")),
                   tuple((name, _tensor_signature(value))
                         for name, value in sorted(vars(plan.wrapper).items())
                         if isinstance(value, torch.Tensor) and value.device.type == "cuda"))
                  for plan in batch.plans))
    return (id(model), id(manager), id(manager._gpu_page_table_manager),
            id(getattr(manager, "_k_cache", None)),
            getattr(manager, "_r1_graph_generation", None),
            id(pool), pool.capacity, pool.phase_configurations, invalidations,
            tuple(input_ids.shape), str(input_ids.dtype), str(input_ids.device),
            metadata, caches, (_weight_signature(model) if weight_guard is None else weight_guard.signature(model)))


def _wait_event(event, *, timeout=120):
    deadline = time.monotonic() + timeout
    while not event.query():
        if time.monotonic() > deadline:
            raise TimeoutError("R1 whole-decode CUDA event did not complete")
        time.sleep(0.005)


def _stream_complete(stream):
    import torch
    event = torch.cuda.Event()
    event.record(stream)
    _wait_event(event)


def _prewarm_local_kernels(model, batch, manager, pool, *, stream):
    """Initialize local-only JIT/caches before any peer enters LL warmup."""
    import torch
    from .r1_decode_cuda_graph import compute_attention
    rows = 0 if batch is None else batch.rows
    device = pool.x.device
    if rows:
        hidden = torch.ones((rows, 1, pool.hidden), dtype=torch.bfloat16, device=device)
        for layer_idx, layer in enumerate(model.model.layers):
            blocked_k = manager.get_layer_kv_with_page_table(layer_idx)[0]
            compute_attention(layer.self_attn, hidden, batch, blocked_k)
            layer.input_layernorm(hidden)
            layer.post_attention_layernorm(hidden)
            if layer_idx < 3:
                layer.mlp(hidden)
            else:
                layer.mlp.shared_experts(hidden.view(rows, pool.hidden))
        model.model.norm(hidden)
        model.lm_head(hidden)
    counts = torch.ones(pool.num_local_experts, dtype=torch.int32, device=device)
    pool.expert_out.fill_(1)
    for layer in model.model.layers[3:]:
        a, moe = layer.mlp.r1_deepep_adapter, layer.mlp
        # Gate specialization always sees the fixed LL capacity, even for a
        # local zero-row rank. None of these calls communicate or append KV.
        moe.gate.moe_gate_forward_hybrid(pool.x[:, None, :])
        native_warmup = getattr(a, "warmup_local_only", None)
        if callable(native_warmup):
            native_warmup(rows=rows)
            continue
        a.quantize_into(pool.expert_out, counts, pool.x_fp8, pool.x_scale)
        a.s1(pool.x_fp8, pool.x_scale, moe.gate_list, moe.gate_ptrs_ptr,
             moe.up_list, moe.up_ptrs_ptr, moe.gate_scale_list, moe.gate_scale_ptrs_ptr,
             moe.up_scale_list, moe.up_scale_ptrs_ptr, counts, pool.num_local_experts,
             out=pool.intermediate_bf16)
        a.quantize_into(pool.intermediate_bf16, counts,
                        pool.intermediate_fp8, pool.intermediate_scale)
        a.s3(pool.intermediate_fp8, pool.intermediate_scale,
             moe.down_list, moe.down_ptrs_ptr, moe.down_scale_list, moe.down_scale_ptrs_ptr,
             counts, pool.num_local_experts, out=pool.expert_out)
    _stream_complete(stream)


class _WholeRecord:
    def __init__(self, state, worker, model, manager, batch, pool, input_ids):
        import torch
        from .r1_decode_cuda_graph import WholeModelKVOutputs
        self.owner, self.model, self.manager, self.batch, self.pool = state, model, manager, batch, pool
        self.static_ids = torch.empty_like(input_ids)
        self.static_ids.copy_(input_ids)
        self.stream = torch.cuda.Stream(device=input_ids.device)
        self.stream.wait_stream(torch.cuda.current_stream(input_ids.device))
        self.outputs = WholeModelKVOutputs(
            () if batch is None else range(len(model.model.layers)), self.stream.cuda_stream)
        self.empty_hidden = torch.empty((0, pool.hidden), dtype=torch.bfloat16, device=input_ids.device)
        self.empty_output = SimpleNamespace(logits=torch.empty(
            (0, 1, model.config.vocab_size), dtype=torch.float32, device=input_ids.device))
        self.graph = None
        self.graph_replay_counter = torch.zeros(1, dtype=torch.int64, device=input_ids.device)
        self.expanded_mla_keepalive = ()
        self.source_weight_keepalive = ()
        self.attention_capture_inputs = None
        self.replays = 0
        self.signature = None
        self.last_done = None

    def forward(self):
        if self.static_ids.shape[0] == 0:
            # No local attention/dense/shared computation is required, but
            # every expert owner must still process peer tokens in 58 LL pairs.
            for layer in self.model.model.layers[3:]:
                layer.mlp.r1_deepep_adapter(self.empty_hidden)
            return self.empty_output
        return self.model(self.static_ids, attention_mask=None,
            position_ids=self.batch.positions, use_cache=False,
            output_attentions=False, output_hidden_states=False, return_dict=True)

    def warm_and_capture(self):
        import torch
        from .r1_decode_cuda_graph import whole_model_attention_scope
        error = None
        try:
            with torch.no_grad(), torch.cuda.stream(self.stream):
                _prewarm_local_kernels(self.model, self.batch, self.manager,
                                       self.pool, stream=self.stream)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        self.owner.coordinator.agree("local-JIT-warm", error=error)
        error = None
        try:
            with torch.no_grad(), torch.cuda.stream(self.stream), whole_model_attention_scope(self.outputs):
                for _ in range(2):
                    self.forward()
            _stream_complete(self.stream)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        self.owner.coordinator.agree("all-rank-model-warm", error=error)
        error = None
        try:
            self.graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), whole_model_attention_scope(self.outputs):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.output = self.forward()
                    self.graph_replay_counter.add_(1)
            # Graphs do not retain arbitrary input tensors. A later eager call
            # on another stream may replace the module's MLA cache entry.
            # Keep its old allocation (and source aliases) alive until graph
            # invalidation waits for the final replay and releases this record.
            if self.batch is not None:
                entries = tuple(getattr(layer.self_attn.module,
                    "_mla_decode_weight_cache_entry", None) for layer in self.model.model.layers)
                self.expanded_mla_keepalive = tuple(entry for entry in entries if entry is not None)
            self.attention_capture_inputs = self.outputs.capture_inputs
            if self.batch is not None and not self.attention_capture_inputs:
                raise RuntimeError("Whole-decode attention did not retain its captured external tensor inputs")
            aliases = [parameter.detach() for parameter in self.model.parameters()
                       if parameter.device.type == "cuda"]
            for layer in self.model.model.layers:
                attn = getattr(layer.self_attn, "module", layer.self_attn)
                shared = getattr(layer.mlp, "shared_experts", None)
                for module in (attn, getattr(shared, "module", shared)):
                    for item in getattr(module, "_r1_native_gemm_weights", ()):
                        aliases.extend(value.detach() for value in (item.weight, item.scale))
            for layer in self.model.model.layers[3:]:
                adapter = layer.mlp.r1_deepep_adapter
                native_weights = getattr(adapter, "graph_weight_tensors", None)
                if callable(native_weights):
                    aliases.extend(value.detach() for value in native_weights()
                                   if value.device.type == "cuda")
                    continue
                for key in ("gate_list", "up_list", "down_list", "gate_scale_list",
                            "up_scale_list", "down_scale_list"):
                    aliases.extend(value.detach() for value in getattr(layer.mlp, key)
                                   if value.device.type == "cuda")
            self.source_weight_keepalive = tuple(aliases)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        # No rank replays any captured LL operation unless every rank ended
        # capture successfully. A Python capture failure cannot desynchronize
        # the next rank's dispatch/combine sequence.
        self.owner.coordinator.agree("all-rank-model-capture", error=error)

    def replay(self, input_ids, callback):
        import torch
        started = time.perf_counter()
        self.static_ids.copy_(input_ids)
        copied = time.perf_counter()
        with _trace_range(getattr(self.owner, "trace_enabled", False), "graph_replay_submit"):
            self.graph.replay()
        replayed = time.perf_counter()
        self.replays += 1
        with _trace_range(getattr(self.owner, "trace_enabled", False), "KV_clone_Host_callbacks"):
            if self.batch is not None:
                self.outputs.stage_after_replay(callback, replay_ordinal=self.replays)
        staged = time.perf_counter()
        self.last_done = torch.cuda.Event()
        self.last_done.record(torch.cuda.current_stream(input_ids.device))
        self.last_timing_ms = {"input_ids_staging": (copied - started) * 1000,
                              "graph_replay_submit": (replayed - copied) * 1000,
                              "KV_clone_Host_callbacks": (staged - replayed) * 1000}
        return self.output


class R1WholeDecodeState:
    def __init__(self, worker):
        import torch.distributed as dist
        if worker.world_size != 8:
            raise ValueError("The R1 whole-decode candidate requires the single EP8 profile")
        group = dist.new_group(ranks=list(range(worker.world_size)), backend="gloo",
                               timeout=timedelta(seconds=180))
        self.coordinator = WholeDecodeCoordinator(group,
            world_size=worker.world_size, rank=worker.rank)
        self.record = None
        self.invalidations = 0
        self.captures = 0
        self.total_replays = 0
        self.compare_enabled = comparison_requested()
        self.compared_nonzero_rows = False
        self.diagnostic_eager_passes = 0
        self.weight_guard = None
        self.last_step_timing_ms = {}
        self.trace_enabled = os.environ.get("BATCHGEN_R1_DECODE_TRACE", "0") == "1"

    def _weight_guard_for(self, model):
        if not fast_weight_guard_requested():
            return None
        if self.weight_guard is None:
            self.weight_guard = CapturedWeightSignature(model)
        return self.weight_guard

    def compare_first_nonzero_replay(self, worker, record):
        """All EP ranks join one eager pass when any rank needs its first check.

        A rank with zero rows must still execute its ghost LL pairs, even
        though it has no local logits/KV to compare. No Host append or sequence
        advancement occurs in the separate collector used here.
        """
        import torch
        from .r1_decode_cuda_graph import WholeModelKVOutputs, whole_model_attention_scope
        rows = int(record.static_ids.shape[0])
        needed = rows > 0 and not self.compared_nonzero_rows
        if not self.coordinator.agree("diagnostic-selection", changed=needed):
            return
        error, graph_logits, graph_kv = None, None, None
        try:
            graph_logits = record.output.logits.detach().clone()
            graph_kv = {layer: value.detach().clone()
                        for layer, value in record.outputs.kv_by_layer.items()}
            expected = set(record.outputs.expected_layers)
            if set(graph_kv) != expected:
                raise RuntimeError("Graph KV diagnostic is missing required layers")
            diagnostic = WholeModelKVOutputs(record.outputs.expected_layers,
                torch.cuda.current_stream(record.static_ids.device).cuda_stream)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        self.coordinator.agree("diagnostic-snapshot", error=error)
        error, eager_output = None, None
        try:
            with torch.no_grad(), whole_model_attention_scope(diagnostic):
                eager_output = record.forward()
            _stream_complete(torch.cuda.current_stream(record.static_ids.device))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        self.coordinator.agree("diagnostic-eager-forward", error=error)
        error, report = None, None
        try:
            if set(diagnostic.kv_by_layer) != expected:
                raise RuntimeError("Eager KV diagnostic is missing required layers")
            if diagnostic.last_staged_replay != 0:
                raise RuntimeError("Eager diagnostic unexpectedly appended Host KV")
            report = {"rank": worker.rank, "local_rows": rows,
                "kind": "same_LL_graph_vs_eager_NOT_cross_framework",
                "performance_measurement": False,
                "logits": _compare_tensor(graph_logits, eager_output.logits),
                "last_token_KV": {str(layer): _compare_tensor(graph_kv[layer],
                    diagnostic.kv_by_layer[layer]) for layer in record.outputs.expected_layers},
                "checked_layers": len(expected), "extra_Host_appends": 0,
                "source_state_advanced": False, "all_ranks_joined_LL": True,
                "graph_replay_ordinal": record.replays, "status": "PASSED"}
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        self.coordinator.agree("diagnostic-numeric-comparison", error=error)
        self.diagnostic_eager_passes += 1
        if rows:
            self.compared_nonzero_rows = True
        logging.info("[R1_WHOLE_GRAPH_COMPARE] %s", json.dumps(report, sort_keys=True))

    def invalidate(self, *, reason):
        record = self.record
        if record is not None and record.last_done is not None:
            _wait_event(record.last_done)
        self.record = None
        # The guard owns original and prepared tensors. Release it with the
        # graph so non-reused phase models/KV can actually be reclaimed.
        self.weight_guard = None
        self.invalidations += 1
        logging.info("[R1_WHOLE_GRAPH_INVALIDATE] reason=%s generation=%d", reason, self.invalidations)

    def run(self, worker, input_ids):
        import torch
        from batchgen.models.wrappers.attention import AttnWrapperBase
        from .r1_decode_cuda_graph import forward_scope
        error, batch, signature, pool, callback = None, None, None, None, None
        self.trace_enabled = bool(getattr(worker, "_r1_decode_trace_active",
            os.environ.get("BATCHGEN_R1_DECODE_TRACE", "0") == "1"))
        timing = {}
        run_started = time.perf_counter()
        with ExitStack() as scopes:
            try:
                validate_profile(model_identifier=worker.huggingface_ckpt_name,
                    enabled=worker.args.enable_cuda_graph,
                    disabled=worker.args.disable_cuda_graphs, deepep=worker.args.enable_deepep)
                model = worker.model
                manager = AttnWrapperBase.gpu_paged_kv_manager
                if manager is None:
                    raise RuntimeError("R1 whole-decode requires the active paged GPU KV manager")
                gids = [int(gid) for gid in manager._gpu_page_table_manager.slot_to_seq_id]
                lengths = [int(worker.global_batch.get_sequence_by_global_idx(gid).current_context_length)
                           for gid in gids]
                from batchgen.r1_alignment_controls import effective_context_limit
                context_cap = effective_context_limit(int(model.config.max_position_embeddings),
                                                       worker.huggingface_ckpt_name)
                micro_batch_size = int(model.model.layers[0].self_attn.decode_attn_micro_batch_size)
                if len(model.model.layers) != 61 or model.config.first_k_dense_replace != 3:
                    raise ValueError("The whole-decode candidate requires full R1 61-layer topology")
                pool = worker.parallel_manager._r1_deepep_pool
                if any(getattr(layer.mlp, "r1_deepep_adapter", None) is None
                       for layer in model.model.layers[3:]):
                    raise RuntimeError("All 58 routed layers must use the actual DeepEP LL adapter")
                if pool.phase_configurations < 1:
                    raise RuntimeError("R1 LL pool has not completed a decode phase configuration")
                metadata_started = time.perf_counter()
                with _trace_range(self.trace_enabled, "attention_metadata"):
                    batch = scopes.enter_context(forward_scope(model, manager, gids, lengths,
                        context_cap=context_cap, micro_batch_size=micro_batch_size))
                timing["attention_metadata"] = (time.perf_counter() - metadata_started) * 1000
                if batch is not None:
                    timing.update(batch.last_prepare_timing_ms)
                if input_ids.shape != (len(gids), 1) or input_ids.dtype != torch.int64:
                    raise ValueError("Whole-decode input IDs must match the actual local [rows,1] metadata")
                callback = AttnWrapperBase.kv_append_callback
                if batch is not None and callback is None:
                    raise RuntimeError("Whole-decode real replay requires the existing Host KV callback")
                pool.external_valid_rows = True
                pool.set_valid_rows(len(gids))
                signature_started = time.perf_counter()
                with _trace_range(self.trace_enabled, "signature_guards"):
                    signature = _signature(worker, model, manager, batch, pool,
                                           input_ids, self.invalidations, self._weight_guard_for(model))
                timing["signature_guards"] = (time.perf_counter() - signature_started) * 1000
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
            changed = self.record is None or self.record.signature != signature
            agreed_started = time.perf_counter()
            with _trace_range(self.trace_enabled, "Gloo_agreement"):
                rebuild = self.coordinator.agree("metadata-and-signature", changed=changed, error=error)
            timing["Gloo_agreement"] = (time.perf_counter() - agreed_started) * 1000
            rebuild_started = time.perf_counter()
            if rebuild:
                self.invalidate(reason="all-rank-signature-agreement")
                for layer in model.model.layers:
                    layer.self_attn._r1_whole_decode_owner = self
                error, record = None, None
                try:
                    record = _WholeRecord(self, worker, model, manager, batch, pool, input_ids)
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                self.coordinator.agree("static-buffer-allocation", error=error)
                record.warm_and_capture()
                # Local warmup can create a stream-specific MLA BF16 cache.
                # Pin the final addresses, rather than the pre-warm identity.
                error = None
                try:
                    record.signature = _signature(worker, model, manager, batch, pool,
                                                   input_ids, self.invalidations, self._weight_guard_for(model))
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                self.coordinator.agree("final-capture-signature", error=error)
                self.record = record
                self.captures += 1
                logging.info("[R1_WHOLE_GRAPH_CAPTURE] rank=%d captures=%d rows=%d MoE=DeepEP_LL phase=%d",
                    worker.rank, self.captures, len(gids), pool.phase_configurations)
            timing["capture_rebuild"] = (time.perf_counter() - rebuild_started) * 1000 if rebuild else 0.0
            result = self.record.replay(input_ids, callback)
            timing.update(self.record.last_timing_ms)
            timing["whole_decode_host_total"] = (time.perf_counter() - run_started) * 1000
            timing["graph_rebuilt"] = int(rebuild)
            self.last_step_timing_ms = timing
            self.total_replays += 1
            if self.compare_enabled:
                self.compare_first_nonzero_replay(worker, self.record)
            if self.total_replays == 1 or self.total_replays % 64 == 0:
                logging.info("[R1_WHOLE_GRAPH_REPLAY] rank=%d replays=%d captures=%d rows=%d Host_KV=post_replay",
                    worker.rank, self.total_replays, self.captures, len(gids))
            return result


def run_whole_decode(worker, input_ids):
    """Own forward_scope so one-rank metadata errors also abort collectively."""
    state = getattr(worker, "_r1_whole_decode_state", None)
    if state is None:
        state = R1WholeDecodeState(worker)
        worker._r1_whole_decode_state = state
    return state.run(worker, input_ids)
