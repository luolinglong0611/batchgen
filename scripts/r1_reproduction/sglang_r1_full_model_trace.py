"""Opt-in SG0.5.5.post3 native-row trace bridge; imports no GPU code by default.

Installation happens from a separately generated DeepseekV2 source bootstrap.
Original methods and returned logits are never numerically modified.
"""
from contextvars import ContextVar
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import sys
import weakref

STOCK = {
    "models/deepseek_v2.py": "620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49",
    "layers/logits_processor.py": "c95dc2d82cb88c3f35230e46840b03fc20f17192b3dfd0a6654c17ad00a185d4",
    "managers/schedule_batch.py": "cde09ea7f562c227ba629c0bc48ce0ec24a8bc31c5f7fca43e73f65e0cc05bd6",
    "model_executor/forward_batch_info.py": "2939c4308912f046717626dfab400080a2161e46fd214509e3b844289d2801a9",
    "model_executor/model_runner.py": "c16d0d2b133e0e1efd7d338034a7908e662bbadab1dcef418418c08c28910c45",
}
COMBINED_MODEL_SHA256 = "b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd"
BOOTSTRAP = '''\n# Explicit diagnostic bootstrap; disabled unless R1_FULL_MODEL_TRACE=1.\nfrom scripts.r1_reproduction.sglang_r1_full_model_trace import install as _r1_install_full_model_trace\n_r1_install_full_model_trace()\n'''
_CURRENT = ContextVar("r1_sg_actual_forward_trace", default=None)
_INSTALLATION = None


def record_runtime_once(runner, mode):
    """Observe materialized state after a real forward, once per model/phase."""
    if not all(os.environ.get(key) for key in (
            "R1_RUNTIME_AUDIT_DIR", "R1_RUNTIME_RUN_ID", "R1_RUNTIME_DEPLOYMENT_SHA")):
        return
    recorded = getattr(runner, "_r1_runtime_recorded_models", None)
    if recorded is None:
        recorded = weakref.WeakKeyDictionary()
        runner._r1_runtime_recorded_models = recorded
    phases = recorded.setdefault(runner.model, set())
    if mode not in phases:
        collector = importlib.import_module("scripts.r1_reproduction.r1_actual_runtime")
        if collector.record_sg(runner.model, runner, getattr(runner, "server_args", None), mode) is not None:
            phases.add(mode)
_SEEN_SCHEDULE = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flag():
    value = os.environ.get("R1_FULL_MODEL_TRACE", "0")
    if value not in ("0", "1"):
        raise ValueError("R1_FULL_MODEL_TRACE must be explicitly 0 or 1")
    return value == "1"


def cpu_list(value):
    if value is None:
        raise ValueError("Actual CPU/GPU batch ledger tensor missing")
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    result = list(value)
    if any(type(x) is not int for x in result):
        raise ValueError("Native batch ledger requires integer values")
    return result


def phase(mode, global_mode=None):
    if mode.is_idle():
        if global_mode is None or global_mode.is_idle():
            return None
        return phase(global_mode)
    if mode.is_decode():
        return "decode"
    if mode.name == "EXTEND":
        return "prefill"
    raise ValueError("Trace bridge rejects mixed/speculative/split/unknown forward modes")


def freeze_schedule_rows(schedule, model_batch):
    """Copy the real Req ledger in exactly the order passed to ModelWorkerBatch."""
    if getattr(schedule, "enable_overlap", None) is not False:
        raise ValueError("Trace prefix ledger requires overlap scheduling disabled")
    mode = schedule.forward_mode
    actual_phase = phase(mode, getattr(model_batch, "global_forward_mode", None))
    lengths = cpu_list(model_batch.seq_lens_cpu)
    pools = cpu_list(model_batch.req_pool_indices)
    actual_input = cpu_list(model_batch.input_ids)
    reqs = schedule.reqs
    if len(reqs) != len(lengths) or len(pools) != len(reqs):
        raise ValueError("Actual Req/sequence-length/pool-index row counts differ")
    if mode.is_idle() and reqs:
        raise ValueError("Idle batch unexpectedly contains owned requests")
    rows, expected_input = [], []
    for i, req in enumerate(reqs):
        original, generated = cpu_list(req.origin_input_ids), cpu_list(req.output_ids)
        full = original + generated
        processed = lengths[i]
        if (not 0 < processed <= len(full) or type(req.req_pool_idx) is not int or pools[i] != req.req_pool_idx
                or any(not 0 <= token < 129280 for token in full)):
            raise ValueError("Native request prefix/context or actual pool mapping differs")
        prefix = full[:processed]
        if mode.is_decode():
            if processed != len(full):
                raise ValueError("Decode prefix is not the complete actual Req ledger")
            expected_input.append(prefix[-1])
        else:
            starts = model_batch.extend_prefix_lens
            lens = model_batch.extend_seq_lens
            if len(starts) != len(reqs) or len(lens) != len(reqs):
                raise ValueError("Missing actual extend partition")
            start, count = starts[i], lens[i]
            if type(start) is not int or type(count) is not int or not 0 <= start < processed or start + count != processed:
                raise ValueError("Actual prefill prefix/extend lengths do not cover the consumed prefix")
            if cpu_list(req.fill_ids) != prefix:
                raise ValueError("Req.fill_ids is not the actual processed origin/output prefix")
            expected_input.extend(prefix[start:processed])
        # Immutable tuples cross ModelWorkerBatch -> ForwardBatch. Partial
        # prefill chunks are recorded, but never mislabeled sampled predictions.
        rows.append((str(req.rid), tuple(prefix), processed, pools[i], processed == len(full), actual_phase))
    if actual_input != expected_input:
        raise ValueError("Actual model input_ids differ from the frozen native request/extend ledger")
    return tuple(rows)


def validate_runner(runner):
    args = runner.server_args
    required = {"disable_cuda_graph": True, "disable_overlap_schedule": True,
                "enable_piecewise_cuda_graph": False, "enable_two_batch_overlap": False,
                "enable_single_batch_overlap": False, "enable_dp_lm_head": True}
    if runner.tp_size != 16 or runner.pp_size != 1 or runner.is_draft_worker:
        raise RuntimeError("SG trace bridge requires non-draft TP16/PP1")
    if getattr(args, "speculative_algorithm", None) is not None:
        raise RuntimeError("Speculative execution cannot use this exact-row qualification bridge")
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise RuntimeError("SG trace bridge requires explicit setting: " + name)


def install_wrappers(schedule_cls, forward_cls, runner_cls, logits_cls, trace):
    """Install against admitted classes; separated for accelerator-free tests."""
    schedule_original = schedule_cls.get_model_worker_batch
    forward_original = forward_cls.__dict__["init_new"].__func__
    runner_original = runner_cls.forward
    logits_original = logits_cls.forward
    signatures = {
        "schedule": (schedule_original, ["self", "seq_lens_cpu_cache"]),
        "forward": (forward_original, ["cls", "batch", "model_runner"]),
        "runner": (runner_original, ["self", "forward_batch", "skip_attn_backend_init", "pp_proxy_tensors", "reinit_attn_backend", "split_forward_count"]),
        "logits": (logits_original, ["self", "input_ids", "hidden_states", "lm_head", "logits_metadata", "aux_hidden_states"]),
    }
    for name, (function, expected) in signatures.items():
        if list(inspect.signature(function).parameters) != expected or getattr(function, "_r1_actual_rows_trace", False):
            raise RuntimeError("Frozen stock method signature/installation mismatch: " + name)

    @functools.wraps(schedule_original)
    def get_worker(self, seq_lens_cpu_cache=None):
        global _SEEN_SCHEDULE
        result = schedule_original(self, seq_lens_cpu_cache)
        result._r1_trace_rows = freeze_schedule_rows(self, result)
        result._r1_trace_consumed_input_ids = tuple(cpu_list(result.input_ids))
        _SEEN_SCHEDULE = True
        return result

    @functools.wraps(forward_original)
    def init_new(cls, batch, model_runner):
        result = forward_original(cls, batch, model_runner)
        if hasattr(batch, "_r1_trace_rows"):
            result._r1_trace_rows = batch._r1_trace_rows
            result._r1_trace_consumed_input_ids = batch._r1_trace_consumed_input_ids
            if result.batch_size != len(result._r1_trace_rows):
                raise ValueError("ForwardBatch.init_new changed the actual logical request row count")
        return result

    @functools.wraps(logits_original)
    def logits_forward(self, input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states=None):
        result = logits_original(self, input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states)
        current = _CURRENT.get()
        if current is not None:
            if logits_metadata is not current["batch"] or not hasattr(result, "next_token_logits"):
                raise RuntimeError("Actual LogitsProcessor output lost its exact ForwardBatch provenance")
            result._r1_trace_logits_origin = current["token"]
        return result

    @functools.wraps(runner_original)
    def runner_forward(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None,
                       reinit_attn_backend=False, split_forward_count=1):
        if not hasattr(forward_batch, "_r1_trace_rows"):
            if _SEEN_SCHEDULE:
                raise RuntimeError("Scheduled actual forward is missing native Req row provenance")
            # Stock model-load profiling precedes any scheduled request. It is
            # not suite evidence and cannot be assigned fabricated request IDs.
            return runner_original(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors, reinit_attn_backend, split_forward_count)
        validate_runner(self)
        rows = forward_batch._r1_trace_rows
        if len(rows) != forward_batch.batch_size or cpu_list(forward_batch.seq_lens_cpu) != [row[2] for row in rows]:
            raise ValueError("Actual ForwardBatch rows/context changed after native ledger copy")
        if cpu_list(forward_batch.req_pool_indices) != [row[3] for row in rows]:
            raise ValueError("Actual ForwardBatch request-pool row order changed")
        if tuple(cpu_list(forward_batch.input_ids)) != forward_batch._r1_trace_consumed_input_ids:
            raise ValueError("Actual ForwardBatch input token tensor changed after the frozen scheduler ledger")
        mode = phase(forward_batch.forward_mode, forward_batch.global_forward_mode)
        if mode is None:
            trace.flush_layer_stats(self.tp_rank, self.tp_size)
            return runner_original(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors, reinit_attn_backend, split_forward_count)
        trace.register_model_hooks(self.model, self.tp_rank, self.tp_size, "dynamic")
        trace.set_execution_phase(self.tp_rank, self.tp_size, mode)
        token = object()
        context = {"batch": forward_batch, "token": token}
        reset = _CURRENT.set(context)
        try:
            output = runner_original(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors, reinit_attn_backend, split_forward_count)
        finally:
            _CURRENT.reset(reset)
        if not isinstance(output, tuple) or len(output) != 2 or output[1]:
            raise RuntimeError("Graph/PP/unknown actual ModelRunner output cannot qualify this eager bridge")
        result = output[0]
        if getattr(result, "_r1_trace_logits_origin", None) is not token:
            raise RuntimeError("Returned sampler logits did not originate in the actual admitted LogitsProcessor call")
        logits = result.next_token_logits
        # Use the real returned logits AFTER stock post_forward_mlp_sync_batch
        # trimmed DP padding. Never invent a slice or gather across global TP.
        if logits is None or logits.dim() != 2 or tuple(logits.shape) != (len(rows), 129280):
            raise RuntimeError("Actual post-trim full-vocabulary logits/request rows differ")
        eligible = [i for i, row in enumerate(rows) if row[4]]
        if eligible:
            selected = logits if len(eligible) == len(rows) else logits[eligible]
            trace.capture_logits([rows[i][0] for i in eligible], [rows[i][1] for i in eligible],
                                 selected, self.tp_rank, self.tp_size, mode)
        record_runtime_once(self, mode)
        self._r1_trace_forwards = getattr(self, "_r1_trace_forwards", 0) + 1
        if self._r1_trace_forwards % 64 == 0:
            trace.flush_layer_stats(self.tp_rank, self.tp_size)
        return output

    for function in (get_worker, init_new, logits_forward, runner_forward):
        function._r1_actual_rows_trace = True
    schedule_cls.get_model_worker_batch = get_worker
    forward_cls.init_new = classmethod(init_new)
    logits_cls.forward = logits_forward
    runner_cls.forward = runner_forward


def install_completion_wrapper(req_cls, trace):
    """Observe the stock stop decision exactly once; never make that decision."""
    original = req_cls.check_finished
    if (list(inspect.signature(original).parameters) != ["self", "new_accepted_len"]
            or getattr(original, "_r1_native_completion_trace", False)):
        raise RuntimeError("Frozen Req.check_finished signature/installation mismatch")

    @functools.wraps(original)
    def check_finished(self, new_accepted_len=1):
        result = original(self, new_accepted_len)
        if self.finished_reason is not None and not getattr(self, "_r1_completion_observed", False):
            # Consult the already initialized process group. No import, group
            # construction, collective, model call or sampling change here.
            dist = sys.modules.get("torch.distributed")
            if dist is None or not dist.is_initialized():
                raise RuntimeError("Native completion has no actual initialized rank identity")
            trace.trace_sg_completion(self, int(dist.get_rank()), int(dist.get_world_size()))
            self._r1_completion_observed = True
        return result

    check_finished._r1_native_completion_trace = True
    req_cls.check_finished = check_finished


def install():
    global _INSTALLATION
    if not flag():
        return {"status": "disabled", "gpu_or_SG_imports": False}
    if _INSTALLATION is not None:
        return _INSTALLATION
    if importlib.metadata.version("sglang") != "0.5.5.post3":
        raise RuntimeError("SG full-model trace requires exact SGLang0.5.5.post3")
    expected_model = os.environ.get("R1_SG_TRACE_MODEL_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_model) or expected_model != COMBINED_MODEL_SHA256:
        raise RuntimeError("Explicit exact admitted generated SG model source SHA required")
    modules = {path: importlib.import_module("sglang.srt." + path[:-3].replace("/", ".")) for path in STOCK}
    for path, module in modules.items():
        expected = expected_model if path == "models/deepseek_v2.py" else STOCK[path]
        if sha(module.__file__) != expected:
            raise RuntimeError("Actual installed SG trace source SHA mismatch: " + path)
    model_path = Path(modules["models/deepseek_v2.py"].__file__)
    if not model_path.read_text().endswith(BOOTSTRAP):
        raise RuntimeError("Generated SG model is missing the exact diagnostic bootstrap")
    trace = importlib.import_module("scripts.r1_reproduction.r1_full_model_trace")
    install_wrappers(modules["managers/schedule_batch.py"].ScheduleBatch,
                     modules["model_executor/forward_batch_info.py"].ForwardBatch,
                     modules["model_executor/model_runner.py"].ModelRunner,
                     modules["layers/logits_processor.py"].LogitsProcessor, trace)
    install_completion_wrapper(modules["managers/schedule_batch.py"].Req, trace)
    _INSTALLATION = {"status": "SG_actual_native_row_hooks_installed_NOT_qualified", "scope": "full_model",
                     "stock_sources": STOCK, "generated_model_sha256": expected_model,
                     "bridge_sha256": sha(__file__), "actual_model_source": str(model_path),
                     "arithmetic_or_sampling_changed": False, "formal_performance_permitted": False}
    return _INSTALLATION
