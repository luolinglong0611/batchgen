"""Opt-in full-model qualification trace hooks; disabled imports are CPU-only.

This helper does not force tokens or change model outputs. Parent integration
must call the public hooks at actual admissions, forward/logits and completion
sites. Registration alone is never evidence of layer execution.
"""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
import weakref

_RECORDERS = {}
_HANDLES = {}
_MODEL_REFS = {}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def enabled():
    flag = os.environ.get("R1_FULL_MODEL_TRACE", "0")
    if flag not in ("0", "1"):
        raise ValueError("R1_FULL_MODEL_TRACE must be explicitly 0 or 1")
    return flag == "1"


def request_key(request_id):
    return str(request_id).split("@", 1)[0]


class Recorder:
    def __init__(self, rank, world_size):
        self.rank, self.world_size = int(rank), int(world_size)
        self.run_id = os.environ["R1_FULL_MODEL_TRACE_RUN_ID"]
        self.directory = Path(os.environ["R1_FULL_MODEL_TRACE_DIR"])
        if not self.directory.is_absolute() or not self.run_id or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid explicit trace identity")
        suite_path = Path(os.environ["R1_FULL_MODEL_TRACE_SUITE"])
        deployment_path = Path(os.environ["R1_FULL_MODEL_TRACE_DEPLOYMENT"])
        self.suite_sha = os.environ["R1_FULL_MODEL_TRACE_SUITE_SHA256"]
        self.deployment_sha = os.environ["R1_FULL_MODEL_TRACE_DEPLOYMENT_SHA256"]
        if file_sha(suite_path) != self.suite_sha or file_sha(deployment_path) != self.deployment_sha:
            raise ValueError("Trace suite/deployment identity mismatch")
        self.suite = json.loads(suite_path.read_text())
        self.clock_identity = None
        if self.suite.get("collection_protocol", {}).get("time_basis") == "node_monotonic_causal_v2":
            causal = importlib.import_module("scripts.r1_reproduction.r1_causal_evidence")
            self.clock_identity = causal.identity()
            if self.clock_identity["helper_sha256"] != self.suite["collection_protocol"]["causal_tool_sha256"]:
                raise ValueError("Actual causal identity helper differs from frozen suite")
        deployment = json.loads(deployment_path.read_text())
        if deployment["world_size"] != self.world_size or deployment["num_hidden_layers"] != 61:
            raise ValueError("Trace requires the declared full R1 world and 61 layers")
        self.sources = {}
        self.source_paths = {}
        for role, source in deployment["source_files"].items():
            actual = file_sha(source["path"])
            if actual != source["sha256"]:
                raise ValueError("Loaded trace deployment source mismatch: " + role)
            self.sources[role] = actual
            self.source_paths[role] = source["path"]
        self.jobs = {job["id"]: job for job in self.suite["jobs"]}
        if len(self.jobs) != len(self.suite["jobs"]):
            raise ValueError("Duplicate suite jobs")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filename = self.directory / f"rank-{self.rank}-pid-{os.getpid()}.jsonl"
        self.stream = self.filename.open("x", buffering=1)
        self.ordinal = 0
        self.pending = {}
        self.active_phase = None
        self.layer_totals = {}
        self.layer_flushed_calls = {}
        self.emit("trace_start", source_files=self.sources, helper_sha256=file_sha(__file__),
                  actual_pid=os.getpid(), actual_num_hidden_layers_required=61,
                  **({"clock_identity": self.clock_identity} if self.clock_identity is not None else {}))

    def emit(self, event, **fields):
        self.ordinal += 1
        record = {"schema": "r1-full-model-trace-v1", "event": event,
                  "run_id": self.run_id, "suite_sha256": self.suite_sha,
                  "deployment_sha256": self.deployment_sha, "rank": self.rank,
                  "world_size": self.world_size, "ordinal": self.ordinal,
                  "observed_unix_ns": time.time_ns(), **fields}
        if self.clock_identity is not None:
            record.update(observed_monotonic_ns=time.monotonic_ns(),
                          clock_identity_sha256=value_sha(self.clock_identity))
        self.stream.write(canonical(record) + "\n")
        return record


def recorder(rank, world_size):
    if not enabled():
        return None
    key = (os.getpid(), int(rank), int(world_size))
    if key not in _RECORDERS:
        _RECORDERS[key] = Recorder(rank, world_size)
    return _RECORDERS[key]


def ids_list(value):
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).cpu().tolist()
    result = list(value)
    if any(type(v) is not int or not 0 <= v < 129280 for v in result):
        raise ValueError("Invalid/out-of-vocabulary native token ID")
    return result


def _stop_ids(value):
    if value is None:
        return []
    if type(value) not in (list, tuple, set, frozenset):
        raise ValueError("Native stop IDs must be an observed CPU collection")
    return sorted(set(ids_list(value)))


def _stop_strings(value):
    if value is None:
        return []
    if type(value) not in (list, tuple, set) or any(type(item) is not str for item in value):
        raise ValueError("Native stop strings must be an observed collection")
    return sorted(value)


def _stop_common(ignore, maximum, eos):
    if type(ignore) is not bool or type(maximum) is not int or maximum <= 0 or not eos:
        raise ValueError("Missing/invalid actual ignore_eos, output budget or EOS IDs")
    return {"schema": "r1-native-stop-policy-v1", "status": "observed",
            "ignore_eos": ignore, "max_new_tokens": maximum, "eos_token_ids": eos}


def bg_stop_policy(worker, seq):
    """Read actual worker/sequence state; never infer EOS from the fixture."""
    try:
        result = _stop_common(seq.ignore_eos, seq.max_decode_length, _stop_ids(worker.eos_token_ids))
        method = worker._get_finish_reason
        module_globals = getattr(getattr(method, "__func__", method), "__globals__", {})
        breaker = module_globals.get("REP_DETECTION")
        if type(breaker) is not bool or type(seq._rep_detected) is not bool:
            raise ValueError("Actual BG repetition state unavailable")
        result.update(repetition_breaker_enabled=breaker and not seq.ignore_eos,
                      repetition_detected=seq._rep_detected,
                      evidence=["worker.eos_token_ids", "SequenceEntry.ignore_eos/max_decode_length/_rep_detected",
                                "loaded completion worker module REP_DETECTION"])
        return result
    except (AttributeError, TypeError, ValueError) as error:
        return {"schema": "r1-native-stop-policy-v1", "status": "missing", "reason": str(error)}


def sg_stop_policy(req):
    """Mirror the actual Req token/string/grammar sources without changing them."""
    try:
        model_eos = _stop_ids(req.eos_token_ids)
        tokenizer = req.tokenizer
        tokenizer_eos, additional = [], []
        if tokenizer is not None:
            token = tokenizer.eos_token_id
            tokenizer_eos = _stop_ids([] if token is None else [token])
            additional = _stop_ids(tokenizer.additional_stop_token_ids)
        sampling = req.sampling_params
        result = _stop_common(sampling.ignore_eos, sampling.max_new_tokens,
                              sorted(set(model_eos + tokenizer_eos)))
        result.update(model_eos_token_ids=model_eos, tokenizer_eos_token_ids=tokenizer_eos,
                      additional_stop_token_ids=additional, stop_token_ids=_stop_ids(sampling.stop_token_ids),
                      stop_strs=_stop_strings(sampling.stop_strs), stop_regex_strs=_stop_strings(sampling.stop_regex_strs),
                      grammar_active=req.grammar is not None,
                      evidence=["Req.eos_token_ids/tokenizer.eos_token_id/tokenizer.additional_stop_token_ids",
                                "Req.sampling_params ignore_eos/max_new_tokens/stop_token_ids/stop_strs/stop_regex_strs",
                                "Req.grammar"])
        return result
    except (AttributeError, TypeError, ValueError) as error:
        return {"schema": "r1-native-stop-policy-v1", "status": "missing", "reason": str(error)}


def trace_sg_completion(req, rank, world_size):
    """Called only after stock Req.check_finished; all output remains native."""
    if not enabled():
        return
    r = recorder(rank, world_size)
    key = request_key(req.rid)
    if key not in r.jobs:
        return
    try:
        original, generated = ids_list(req.origin_input_ids), ids_list(req.output_ids)
        finished_len = req.finished_len
        if finished_len is not None:
            if type(finished_len) is not int or not 0 < finished_len <= len(generated):
                raise ValueError("Invalid actual Req.finished_len")
            generated = generated[:finished_len]
        reason = req.finished_reason
        raw = reason.to_json()
        if type(raw) is not dict or type(reason.is_error) is not bool or not generated:
            raise ValueError("Missing actual native finish reason or output")
        flush_layer_stats(rank, world_size)
        r.emit("completion", request_id=key, native_request_id=str(req.rid),
               native_generated_ids=generated, generated_count=len(generated), decoded_length=len(generated),
               original_input_ids_sha256=value_sha(original), original_input_length=len(original),
               last_native_token=generated[-1], raw_native_finish_reason=raw,
               native_finish_reason_class=type(reason).__name__, native_finish_is_error=reason.is_error,
               stop_policy=sg_stop_policy(req))
    except Exception as exc:
        r.emit("trace_error", location="SG_completion", request_id=key,
               error_type=type(exc).__name__, message=str(exc))


def record_input_tokens(request_id, input_ids, rank, world_size):
    r = recorder(rank, world_size)
    if r is None:
        return
    key = request_key(request_id)
    if key not in r.jobs:
        return
    ids = ids_list(input_ids)
    r.emit("input_tokens", request_id=key, native_request_id=str(request_id),
           input_count=len(ids), input_ids_sha256=value_sha(ids),
           input_ids_equal_fixture=ids == r.jobs[key]["input_ids"])


def _first_tensor(value):
    if hasattr(value, "detach") and hasattr(value, "numel") and hasattr(value, "is_floating_point"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for name in ("hidden_states", "last_hidden_state", "attn_output", "output"):
            if name in value:
                result = _first_tensor(value[name])
                if result is not None:
                    return result
    return None


def tensor_summary(tensor):
    if tensor is None or not tensor.is_floating_point():
        return {"observed": False, "reason": "no floating activation tensor"}
    flat = tensor.detach().reshape(-1)
    bad = checked = 0
    # Bound temporary GPU allocations and inspect all returned hidden values.
    for start in range(0, flat.numel(), 1024**2):
        part = flat[start: start + 1024**2]
        bad += int((~part.isfinite()).sum().item())
        checked += part.numel()
    return {"observed": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "numel": tensor.numel(), "checked_elements": checked,
            "nonfinite_elements": bad, "validity_scope": "all_returned_hidden_tensor_elements"}


def hidden_input(args, kwargs):
    explicit = _first_tensor(kwargs.get("hidden_states"))
    if explicit is not None and explicit.is_floating_point():
        return explicit
    # SG decoder forwards commonly receive integer positions before floating
    # hidden_states; choosing args[0] would inspect positions instead.
    for value in args:
        candidate = _first_tensor(value)
        if candidate is not None and candidate.is_floating_point():
            return candidate
    return None


def register_model_hooks(model, rank, world_size, phase):
    """Attach actual execution hooks to 61 layers and attention/MLP children.

    Call after PSM configure_prefill/configure_decoding or SG model creation.
    Attention+MLP child events cover packed BG prefill that bypasses layer.forward.
    """
    r = recorder(rank, world_size)
    if r is None:
        return []
    if phase not in ("prefill", "decode", "dynamic"):
        raise ValueError("Actual model phase must be prefill/decode, or dynamic with set_execution_phase")
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None or len(layers) != 61:
        raise ValueError("Full R1 actual model.layers must contain exactly 61 layers")
    key = (id(model), int(rank), phase)
    previous = _MODEL_REFS.get(id(model))
    if previous is not None and previous() is not model:
        # PSM destroys/recreates phase models. Python can reuse an id; stale
        # handles must neither suppress new hooks nor retain old GPU weights.
        for old_key in [item for item in _HANDLES if item[0] == id(model)]:
            del _HANDLES[old_key]
    _MODEL_REFS[id(model)] = weakref.ref(model)
    if key in _HANDLES:
        return _HANDLES[key]
    if any(existing[0] == id(model) and existing[1] == int(rank) for existing in _HANDLES):
        raise ValueError("Do not attach two fixed phase labels to one model; use dynamic phase selection")
    handles = []
    for index, layer in enumerate(layers):
        for component, module in (("layer", layer), ("attention", layer.self_attn), ("mlp", layer.mlp)):
            def hook(mod, args, kwargs, output, layer_idx=index, kind=component):
                try:
                    actual_phase = r.active_phase if phase == "dynamic" else phase
                    if actual_phase not in ("prefill", "decode"):
                        raise ValueError("Actual forward mode was not set before the dynamic model call")
                    summary = tensor_summary(_first_tensor(output))
                    input_summary = tensor_summary(hidden_input(args, kwargs))
                    r.pending.setdefault(actual_phase, set()).add((layer_idx, kind))
                    stat_key = (actual_phase, layer_idx, kind)
                    if not hasattr(r, "layer_totals"):
                        r.layer_totals = {}
                    stat = r.layer_totals.setdefault(stat_key, {"forward_calls": 0, "input_checked_elements": 0,
                        "output_checked_elements": 0, "nonfinite_elements": 0, "unobserved_activations": 0,
                        "last_observed_unix_ns": 0, "last_input": None, "last_output": None,
                        "last_nonempty_observed_unix_ns": 0})
                    stat["forward_calls"] += 1
                    stat["input_checked_elements"] += input_summary.get("checked_elements", 0)
                    stat["output_checked_elements"] += summary.get("checked_elements", 0)
                    stat["nonfinite_elements"] += input_summary.get("nonfinite_elements", 0) + summary.get("nonfinite_elements", 0)
                    stat["unobserved_activations"] += int(input_summary.get("observed") is not True or summary.get("observed") is not True)
                    stat["last_observed_unix_ns"] = time.time_ns()
                    if getattr(r, "clock_identity", None) is not None:
                        stat["last_observed_monotonic_ns"] = time.monotonic_ns()
                    if summary.get("numel", 0) > 0:
                        stat["last_nonempty_observed_unix_ns"] = stat["last_observed_unix_ns"]
                        if getattr(r, "clock_identity", None) is not None:
                            stat["last_nonempty_observed_monotonic_ns"] = stat["last_observed_monotonic_ns"]
                    stat["last_input"], stat["last_output"] = input_summary, summary
                    if stat["nonfinite_elements"] or stat["unobserved_activations"]:
                        r.emit("layer_failure", phase=actual_phase, layer_index=layer_idx, component=kind, **stat)
                except Exception as exc:
                    r.emit("trace_error", location="actual_layer_hook", phase=phase,
                           layer_index=layer_idx, component=kind,
                           error_type=type(exc).__name__, message=str(exc))
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
    _HANDLES[key] = handles
    config = getattr(model, "config", getattr(base, "config", None))
    r.emit("model_hooks_registered", phase=phase, actual_layer_count=len(layers),
           actual_hook_count=len(handles), model_class=type(model).__module__ + "." + type(model).__name__,
           actual_config={name: getattr(config, name, None) for name in
                          ("hidden_size", "num_hidden_layers", "vocab_size", "n_routed_experts")})
    return handles


def set_execution_phase(rank, world_size, phase):
    """For a single SG model, call before the real ForwardBatch forward."""
    r = recorder(rank, world_size)
    if r is not None:
        if phase not in ("prefill", "decode"):
            raise ValueError("Unsupported actual ForwardBatch phase")
        r.active_phase = phase


def flush_layer_stats(rank, world_size):
    """Call on every rank at completed-batch boundaries, before owner filtering.

    Every activation is still inspected on each real forward. Only the journal
    is aggregated, avoiding millions of JSON lines in the 8192-token stage.
    """
    r = recorder(rank, world_size)
    if r is not None:
        if not hasattr(r, "layer_flushed_calls"):
            r.layer_flushed_calls = {}
        for (phase, layer, component), stat in sorted(r.layer_totals.items()):
            key = (phase, layer, component)
            if r.layer_flushed_calls.get(key) != stat["forward_calls"]:
                r.emit("layer_summary", phase=phase, layer_index=layer, component=component, **stat)
                r.layer_flushed_calls[key] = stat["forward_calls"]


def capture_logits(request_ids, prefix_ids, logits, rank, world_size, phase):
    """Capture the exact pre-sampling logits; save full rows for teacher jobs."""
    r = recorder(rank, world_size)
    if r is None:
        return
    if logits.dim() != 2 or logits.shape != (len(request_ids), 129280) or len(prefix_ids) != len(request_ids):
        r.emit("trace_error", location="logits_shape", actual_shape=list(logits.shape), rows=len(request_ids))
        return
    components = sorted([list(item) for item in r.pending.get(phase, set())])
    r.pending[phase] = set()
    for row, raw_id in enumerate(request_ids):
        key = request_key(raw_id)
        if key not in r.jobs:
            continue
        job = r.jobs[key]
        ids = ids_list(prefix_ids[row])
        step = len(ids) - len(job["input_ids"])
        if step == 0:
            record_input_tokens(raw_id, ids, rank, world_size)
        values = logits[row].detach().float().cpu().contiguous()
        finite = values.isfinite()
        nonfinite = int((~finite).sum().item())
        top = values.topk(2) if not nonfinite else None
        full = None
        if job["kind"] == "teacher":
            raw = values.numpy().astype("<f4", copy=False).tobytes()
            filename = f"logits-{key}-rank{rank}-step{step}-{r.ordinal + 1}.f32"
            with (r.directory / filename).open("xb") as output:
                output.write(raw)
            full = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
                    "dtype": "float32_le", "elements": len(values), "bytes": len(raw)}
        greedy = int(top.indices[0]) if top is not None else None
        last = (step + 1 == job["max_new_tokens"] or (greedy == 1 and not job["ignore_eos"]))
        source_end = {role: file_sha(path) for role, path in r.source_paths.items()} if last else None
        if last:
            flush_layer_stats(rank, world_size)
        r.emit("logits", request_id=key, native_request_id=str(raw_id), phase=phase,
               input_ids_sha256=value_sha(ids), context_length=len(ids), generated_before_step=step,
               vocab_size=len(values), nonfinite_elements=nonfinite,
               greedy_token_id=greedy,
               top2_token_ids=[int(v) for v in top.indices] if top is not None else None,
               top2_margin=float(top.values[0] - top.values[1]) if top is not None else None,
               actual_components_since_previous_logits=components, full_logits=full, source_files_at_last_step=source_end)


def trace_bg_logits(worker, logits, batch_sequences):
    """Call at _select_tokens entry, before any sampling or early return."""
    if not enabled():
        return
    r = recorder(worker.rank, worker.world_size)
    try:
        sequences = list(batch_sequences or [])
        if len(sequences) != logits.shape[0]:
            raise ValueError("Missing exact BG logits row -> SequenceEntry mapping")
        if not sequences:
            return
        phase = getattr(getattr(getattr(worker, "model", None), "config", None), "phase", None)
        if phase not in ("prefill", "decode"):
            raise ValueError("Actual BG model.config.phase is required, including re-entry prefill")
        eligible = []
        for index, seq in enumerate(sequences):
            completed = worker._is_sequence_completed(seq)
            if type(completed) is not bool:
                raise ValueError("Actual worker completion predicate did not return bool")
            if completed:
                state = {name: getattr(seq, name) for name in (
                    "decoded_length", "original_prompt_length", "current_context_length", "max_decode_length",
                    "eos_reached", "_rep_detected")}
                state["status"] = getattr(seq.status, "name", str(seq.status))
                r.emit("inactive_forward", request_id=request_key(seq.uuid), native_request_id=str(seq.uuid),
                       phase=phase, eligible=False, completed=True,
                       eligibility_source="actual worker._is_sequence_completed(seq)",
                       generated_before_step=seq.decoded_length, sequence_state=state,
                       model_context_length=worker.model_context_length,
                       actual_logits_row=index, actual_logits_shape=list(logits.shape))
            else:
                eligible.append(index)
        # Observe only rows that the native worker will append to its ledger.
        # The original forward, sampler and completed-row computation still
        # run unchanged; no old duplicate step is deduplicated or truncated.
        if eligible:
            live = [sequences[index] for index in eligible]
            ids = [ids_list(worker._trajectory_model_input(seq)) for seq in live]
            selected = logits if len(eligible) == len(sequences) else logits[eligible]
            capture_logits([seq.uuid for seq in live], ids, selected, worker.rank, worker.world_size, phase)
    except Exception as exc:
        r.emit("trace_error", location="BG_select_tokens", error_type=type(exc).__name__, message=str(exc))


def trace_bg_completion(worker, seq, decoded_tokens):
    """Call while the owner still holds the original prompt and native output."""
    if not enabled():
        return
    r = recorder(worker.rank, worker.world_size)
    key = request_key(seq.uuid)
    if key not in r.jobs:
        return
    try:
        flush_layer_stats(worker.rank, worker.world_size)
        generated = ids_list(decoded_tokens)
        original = ids_list(worker._trajectory_tokens(seq, start=0, length=int(seq.original_prompt_length)))
        r.emit("completion", request_id=key, native_request_id=str(seq.uuid),
               assigned_rank=int(seq.assigned_rank), native_generated_ids=generated,
               generated_count=len(generated), decoded_length=int(seq.decoded_length),
               original_input_ids_sha256=value_sha(original), original_input_length=len(original),
               finish_reason=worker._get_finish_reason(seq),
               last_native_token=generated[-1] if generated else None,
               stop_policy=bg_stop_policy(worker, seq))
    except Exception as exc:
        r.emit("trace_error", location="BG_completion", request_id=key,
               error_type=type(exc).__name__, message=str(exc))
