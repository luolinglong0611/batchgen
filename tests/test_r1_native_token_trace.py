"""CPU completion diagnostics without importing the CUDA worker stack."""

import ast
import hashlib
import importlib.util
import itertools
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen/batchgen_worker.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRACE = load("r1_native_token_trace_cpu", ROOT / "batchgen/worker/r1_native_token_trace.py")
BOOK = load("r1_native_token_query_book_cpu", ROOT / "batchgen/query_book.py")


def sequence(**changes):
    values = dict(uuid="request-0", batch_id="batch-1", global_idx=3, assigned_rank=2,
                  original_prompt_length=2, prompt_length=2, decoded_length=2,
                  max_decode_length=16, current_context_length=4, eos_reached=True,
                  _rep_detected=False, status="PREFILLED")
    values.update(changes)
    return SimpleNamespace(**values)


def record(seq=None, prompt=None, generated=None):
    return TRACE.build_native_token_record(
        seq or sequence(), rank=2, model_context_length=128,
        prompt_tokens=prompt if prompt is not None else torch.tensor([11, 22], dtype=torch.int32),
        generated_tokens=generated if generated is not None else torch.tensor([42, 154827], dtype=torch.int32),
    )


def test_native_ids_are_untrimmed_and_prompt_hash_is_portable():
    first = record()
    second = record(prompt=torch.tensor([11, 999, 22, 999], dtype=torch.int64)[::2])
    assert first == second
    assert first["generated_ids"] == [42, 154827]
    assert first["original_prompt_ids_sha256"] == hashlib.sha256(b"[11,22]").hexdigest()
    assert first["prompt_ids_hash_encoding"] == "sha256:json-compact-integer-array:utf8"
    assert first["decoded_length_matches_ids_count"] is True
    assert first["finish_reason"] == "stop"
    assert json.loads(json.dumps(first)) == first


@pytest.mark.parametrize("count", [0, 8192, 8193, 10000])
def test_native_ids_have_explicit_bounded_truncation_and_actual_counts(count, monkeypatch):
    # Observe every list conversion: the full generated tensor must never be
    # copied to Python before applying the cap.
    sizes = []
    tolist = torch.Tensor.tolist

    def tracked(tensor):
        sizes.append(tensor.numel())
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", tracked)
    result = record(sequence(decoded_length=count + 1), generated=torch.arange(count, dtype=torch.int64))
    assert sizes == [2, min(count, 8192)]
    assert result["generated_ids"] == list(range(min(count, 8192)))
    assert result["generated_ids_count"] == count
    assert result["generated_ids_logged_count"] == min(count, 8192)
    assert result["generated_ids_omitted_count"] == max(0, count - 8192)
    assert result["generated_ids_truncated"] == (count > 8192)
    assert result["decoded_length"] == count + 1
    assert result["decoded_length_matches_ids_count"] is False


@pytest.mark.parametrize("bad", [
    torch.empty(2, device="meta", dtype=torch.int32),
    torch.tensor([1., 2.], dtype=torch.float32),
    torch.tensor([True, False]),
    torch.tensor([[1, 2]], dtype=torch.int32),
    [1, 2],
])
@pytest.mark.parametrize("field", ["prompt", "generated"])
def test_non_cpu_or_non_integer_vector_rejected_before_value_reads(bad, field, monkeypatch):
    def forbidden(*args):
        raise AssertionError("invalid inputs must fail before any value read")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    with pytest.raises(ValueError):
        record(**{field: bad})


def test_prompt_length_mismatch_and_negative_ids_are_visible_errors():
    with pytest.raises(ValueError, match="count differs"):
        record(sequence(original_prompt_length=3))
    with pytest.raises(ValueError, match="non-negative"):
        record(generated=torch.tensor([-1], dtype=torch.int32))


@pytest.mark.parametrize("repetition,decode_cap,context_cap,eos", list(itertools.product([False, True], repeat=4)))
def test_finish_reason_matches_real_handler_without_lifespan_side_effects(repetition, decode_cap, context_cap, eos):
    source = ROOT / "batchgen/worker/completion.py"
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "CompletionHandler")
    events = []
    namespace = {"CompletionContext": object, "SeqEvent": SimpleNamespace(COMPLETED="done"),
                 "lifespan": SimpleNamespace(dump_lifespan=lambda *args: events.append(args),
                                             has_ctx_mismatch=lambda _: False)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    seq = sequence(_rep_detected=repetition, max_decode_length=2 if decode_cap else 16,
                   current_context_length=128 if context_cap else 4, eos_reached=eos,
                   _lifespan_log=[], log_event=lambda *args: events.append(args))
    actual = TRACE.completion_finish_reason(seq, 128)
    assert events == []
    expected = namespace["CompletionHandler"].get_finish_reason(SimpleNamespace(rank=2, model_context_length=128), seq)
    assert actual == expected
    assert events  # The production method is deliberately not called by trace.


def gather_function(dist):
    node = next(node for node in ast.walk(ast.parse(WORKER.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == "_gather_completed_tokens")
    namespace = {"List": list, "dist": dist, "os": os, "logging": logging, "json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace[node.name]


@pytest.fixture
def completion_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "batchgen.worker.r1_native_token_trace", TRACE)
    monkeypatch.delenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", raising=False)
    # Any accidental CUDA discovery is a regression, even on a CPU host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA query in CPU trace"))
    seq = sequence()
    book = BOOK.QueryBook(256, page_tokens=8)
    slot = book.bind(seq.uuid, max_tokens=64)
    book.write_prompt(slot, torch.tensor([11, 22], dtype=torch.int32))
    book.append_token(slot, 42)
    book.append_token(slot, 154827)
    calls = []

    def decoded(entry):
        calls.append("generated")
        return book.generated_tokens(book.slot_for(entry.uuid))

    def prompt(entry, *, length):
        calls.append("prompt")
        output = torch.empty(length, dtype=torch.int32)
        book.copy_span_to(book.slot_for(entry.uuid), output, length=length)
        return output

    def decode(tokens):
        calls.append("detokenize")
        return "answer"  # Existing API removes EOS; trace must retain it.

    def gather(outputs, tokens):
        calls.append("collective")
        outputs[0] = tokens

    worker = SimpleNamespace(model_name="deepseek-ai/DeepSeek-R1", rank=2,
                             model_context_length=128, world_size=1,
                             _uuid_to_local_map={seq.uuid: 0}, query_book={0: object()},
                             global_batch=SimpleNamespace(get_sequence=lambda _: seq),
                             _trajectory_decoded_tokens=decoded, _trajectory_tokens=prompt,
                             _decode_tokens_to_string=decode)
    return worker, seq, book, calls, gather_function(SimpleNamespace(all_gather_object=gather))


def logged_records(caplog):
    return [json.loads(item.getMessage().split(" ", 1)[1]) for item in caplog.records
            if item.getMessage().startswith("[R1_NATIVE_TOKENS] ")]


@pytest.mark.parametrize("model,flag", [
    ("deepseek-ai/DeepSeek-R1", None), ("deepseek-ai/DeepSeek-R1", "0"),
    ("deepseek-ai/DeepSeek-R1", "true"), ("deepseek-ai/DeepSeek-V3", "1"),
    ("deepseek-ai/DeepSeek-R1-0528", "1"), (None, "1"),
])
def test_disabled_or_wrong_model_has_no_extra_tensor_processing(completion_worker, model, flag, monkeypatch, caplog):
    worker, seq, book, calls, gather = completion_worker
    worker.model_name = model
    if flag is not None:
        monkeypatch.setenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", flag)
    # .tolist(), prompt reads, or lazy import on the disabled path would fail.
    monkeypatch.delitem(sys.modules, "batchgen.worker.r1_native_token_trace")
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *args: pytest.fail("extra tensor processing"))
    assert gather(worker, [seq.uuid]) == {seq.uuid: "answer"}
    assert calls == ["generated", "detokenize", "collective"]
    assert not logged_records(caplog)


def test_enabled_trace_reads_active_owner_and_precedes_detokenization(completion_worker, monkeypatch, caplog):
    worker, seq, book, calls, gather = completion_worker
    monkeypatch.setenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", "1")
    with caplog.at_level(logging.INFO):
        assert gather(worker, [seq.uuid]) == {seq.uuid: "answer"}
    assert calls == ["generated", "prompt", "detokenize", "collective"]
    entry, = logged_records(caplog)
    assert entry == record(seq)
    assert seq.status == "PREFILLED"  # Prefill budget completion is valid here.
    assert book.has_sequence(seq.uuid)
    # Completion release also removes the local owner/query-book binding.
    book.release(book.slot_for(seq.uuid))
    worker._uuid_to_local_map.clear()
    worker.query_book.clear()
    calls.clear()
    caplog.clear()
    assert gather(worker, [seq.uuid]) == {}
    assert calls == ["collective"]
    assert not logged_records(caplog)


@pytest.mark.parametrize("case", ["other_owner", "missing_binding", "missing_sequence"])
def test_trace_never_reads_prompt_for_nonowner_or_unbound_sequence(completion_worker, monkeypatch, caplog, case):
    worker, seq, book, calls, gather = completion_worker
    monkeypatch.setenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", "1")
    if case == "other_owner":
        seq.assigned_rank = 1
    elif case == "missing_binding":
        worker.query_book.clear()
    else:
        worker.global_batch.get_sequence = lambda _: None
    gather(worker, [seq.uuid])
    assert "prompt" not in calls
    assert not logged_records(caplog)


def test_reentry_preserves_original_prompt_hash_and_ledger_generated_ids(completion_worker, monkeypatch, caplog):
    worker, seq, book, calls, gather = completion_worker
    monkeypatch.setenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", "1")
    slot = book.slot_for(seq.uuid)
    book.append_prompt(slot, torch.tensor([77, 88], dtype=torch.int32))
    book.append_token(slot, 99)
    book.begin_reentry_turn(slot, prompt_length=7)
    seq.prompt_length = 7
    seq.decoded_length = 3
    with caplog.at_level(logging.INFO):
        gather(worker, [seq.uuid])
    entry, = logged_records(caplog)
    assert entry["original_prompt_length"] == 2
    assert entry["original_prompt_ids_sha256"] == hashlib.sha256(b"[11,22]").hexdigest()
    assert entry["generated_ids"] == [42, 154827, 99]
    assert entry["decoded_length_matches_ids_count"] is True


def test_diagnostic_failure_does_not_change_text_or_collective(completion_worker, monkeypatch, caplog):
    worker, seq, book, calls, gather = completion_worker
    monkeypatch.setenv("BATCHGEN_R1_NATIVE_TOKEN_TRACE", "1")

    def fail(*args, **kwargs):
        calls.append("prompt-error")
        raise RuntimeError("private diagnostic detail should not leak into the log")

    worker._trajectory_tokens = fail
    assert gather(worker, [seq.uuid]) == {seq.uuid: "answer"}
    assert calls == ["generated", "prompt-error", "detokenize", "collective"]
    assert "[R1_NATIVE_TOKENS_ERROR] rank=2 uuid=request-0 error=RuntimeError" in caplog.text
    assert "private diagnostic detail" not in caplog.text
    assert not logged_records(caplog)
