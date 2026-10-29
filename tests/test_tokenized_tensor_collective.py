"""Exercise the real int64 transport and failure agreement with eight Gloo ranks."""
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "batchgen/worker/tokenized_tensor_collective.py"


def _load_helper():
    # Spawned Gloo children must not import batchgen's GPU package initializer.
    spec = importlib.util.spec_from_file_location("_tokenized_collective_under_test", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _items(rank, world, count, active, length=lambda i: (i * 7) % 13):
    return [{"idx": i, "length": length(i),
        "input_ids": torch.arange(length(i), dtype=torch.int64) * 3 - 5 + i * 10**12}
        for i in range(rank, count, world) if rank in active]


def _contains_tensor(value):
    if isinstance(value, torch.Tensor): return True
    if isinstance(value, dict): return any(_contains_tensor(x) for x in value.values())
    if isinstance(value, (tuple, list)): return any(_contains_tensor(x) for x in value)
    return False


def _assert_same(actual, expected):
    assert len(actual) == len(expected)
    for got_rank, want_rank in zip(actual, expected):
        assert len(got_rank) == len(want_rank)
        for got, want in zip(got_rank, want_rank):
            assert set(got) == {"idx", "length", "input_ids"}
            assert got["idx"] == want["idx"] and got["length"] == want["length"]
            assert got["input_ids"].dtype == torch.int64 and got["input_ids"].device.type == "cpu"
            assert got["input_ids"].ndim == 1 and got["input_ids"].is_contiguous()
            assert torch.equal(got["input_ids"], want["input_ids"])


def _gloo_worker(rank, world, init_method, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world,
        init_method=init_method, timeout=timedelta(seconds=60))
    helper = _load_helper()
    checks = []
    try:
        cases = [("all_empty", 0, tuple(range(world))), ("empty_ranks_and_zero_length", 3, tuple(range(world))),
                 ("ragged_large_int64", 23, tuple(range(world)))]
        cases += [("wave_" + str(start), 23, (start, start + 1)) for start in range(0, world, 2)]
        for name, count, active in cases:
            local = _items(rank, world, count, active)
            expected = [None] * world
            dist.all_gather_object(expected, local)
            original_object = dist.all_gather_object
            object_messages = []

            def metadata_only(out, payload, **kwargs):
                assert not _contains_tensor(payload)
                object_messages.append(payload)
                return original_object(out, payload, **kwargs)

            dist.all_gather_object = metadata_only
            try:
                actual = helper.gather_tokenized_tensors(local, device="cpu", active_ranks=active, num_sequences=count)
            finally:
                dist.all_gather_object = original_object
            _assert_same(actual, expected)
            assert len(object_messages) == (2 if any(x["length"] for rows in expected for x in rows) else 1)
            # Views remain valid once the input and the helper's CUDA/local references die.
            del local
            _assert_same(actual, expected)
            nonempty = [x["input_ids"] for rows in actual for x in rows if x["length"]]
            if nonempty:
                assert len({x.untyped_storage().data_ptr() for x in nonempty}) == 1
            checks.append(name)
        active = tuple(range(world))
        for name in ("wrong_dtype", "noncontiguous", "wrong_length", "wrong_owner", "tokenizer_error",
                     "bad_active", "metadata_disagreement", "device_budget", "allocation_error"):
            local = _items(rank, world, world, active, lambda _: 2)
            args = {"device": "cpu", "active_ranks": active, "num_sequences": world}
            if rank == 3:
                if name == "wrong_dtype": local[0]["input_ids"] = local[0]["input_ids"].to(torch.int32)
                if name == "noncontiguous": local[0]["input_ids"] = torch.arange(4, dtype=torch.int64)[::2]
                if name == "wrong_length": local[0]["length"] = 7
                if name == "wrong_owner": local[0]["idx"] = 0
                if name == "tokenizer_error": args["local_error"] = "deliberate tokenizer failure"
                if name == "bad_active": args["active_ranks"] = None
                if name == "metadata_disagreement": args["num_sequences"] = world + 1
            if name == "device_budget": args["max_device_buffer_bytes"] = 8
            original_zeros = torch.zeros
            if name == "allocation_error" and rank == 3:
                def fail_allocation(*_args, **_kwargs):
                    # Object-collective metadata also allocates an 8-element
                    # size vector. Fail only the 2-token data allocation.
                    if _args and _args[0] == 2:
                        raise RuntimeError("deliberate allocation failure")
                    return original_zeros(*_args, **_kwargs)
                torch.zeros = fail_allocation
            try:
                with pytest.raises((ValueError, RuntimeError)) as captured:
                    helper.gather_tokenized_tensors(local, **args)
            finally:
                torch.zeros = original_zeros
            errors = [None] * world
            dist.all_gather_object(errors, str(captured.value))
            assert len(set(errors)) == 1, errors
            assert "before token transfer" in errors[0] or "metadata disagrees" in errors[0] or "byte budget" in errors[0]
            checks.append(name)
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "passed": checks}))
    finally:
        dist.destroy_process_group()


def test_real_eight_rank_gloo_matches_object_transport_and_agrees_on_errors(tmp_path, monkeypatch):
    if not dist.is_available() or not dist.is_gloo_available(): pytest.skip("Gloo unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_gloo_worker, args=(8, "file://" + str(tmp_path / "gloo-init"), str(tmp_path)), nprocs=8, join=True)
    results = [json.loads((tmp_path / f"rank-{r}.json").read_text()) for r in range(8)]
    assert all(len(x["passed"]) == 16 for x in results)
    assert len({tuple(x["passed"]) for x in results}) == 1


def test_import_does_not_load_torch_or_initialize_a_device():
    code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('collective',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(HELPER_PATH)], check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=30)


def test_worker_opt_in_preserves_waves_and_uses_only_compact_helper(monkeypatch):
    spec = importlib.util.spec_from_file_location("_legacy_tokenizer_test_helpers", ROOT / "tests/test_worker_tokenization_memory.py")
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    helper = _load_helper()
    monkeypatch.setitem(sys.modules, "batchgen.worker.tokenized_tensor_collective", helper)
    monkeypatch.setenv("BATCHGEN_TOKENIZER_COMPACT_COLLECTIVE", "1")
    monkeypatch.setenv("BATCHGEN_TOKENIZER_PARALLEL_RANKS", "2")
    monkeypatch.setenv("BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET", "12")
    texts = ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx"]
    worker, uuids = legacy.make_worker(texts, rank=2, world_size=4)
    worker.torch_device = torch.device("cpu")
    seen = []

    def compact(items, *, device, active_ranks, num_sequences, local_error):
        assert device == torch.device("cpu") and num_sequences == 8 and local_error is None
        assert bool(items) == (worker.rank in active_ranks)
        seen.append(active_ranks)
        return [legacy.strided_payload(texts, r, 4) if r in active_ranks else [] for r in range(4)]

    monkeypatch.setattr(helper, "gather_tokenized_tensors", compact)
    monkeypatch.setattr(legacy.FAKE_DIST, "all_gather_object", lambda *_a, **_kw: pytest.fail("legacy tensor object transport reached"), raising=False)
    legacy.run_tokenize(worker, uuids)
    assert seen == [(0, 1), (2, 3)]
    for uuid, text in zip(uuids, texts):
        assert worker._trajectory_book.tokens(worker._trajectory_book.slot_for(uuid)).tolist() == legacy.encode(text)


def test_worker_serialized_path_remains_legacy_even_when_opt_in_set(monkeypatch):
    spec = importlib.util.spec_from_file_location("_legacy_serialized_tokenizer_test_helpers", ROOT / "tests/test_worker_tokenization_memory.py")
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    helper = _load_helper()
    monkeypatch.setitem(sys.modules, "batchgen.worker.tokenized_tensor_collective", helper)
    monkeypatch.setenv("BATCHGEN_TOKENIZER_COMPACT_COLLECTIVE", "1")
    monkeypatch.delenv("BATCHGEN_TOKENIZER_PARALLEL_RANKS", raising=False)
    monkeypatch.delenv("BATCHGEN_TOKENIZER_PARALLEL_CHAR_BUDGET", raising=False)
    monkeypatch.setitem(legacy.TOKENIZE_NAMESPACE, "_TOKENIZER_SERIALIZE_CHARS_PER_RANK", 1)
    worker, uuids = legacy.make_worker(["abc", "def"], rank=0, world_size=1)
    seen = []
    monkeypatch.setattr(helper, "gather_tokenized_tensors", lambda *_a, **_kw: pytest.fail("compact transport reached serialized path"))
    monkeypatch.setattr(legacy.FAKE_DIST, "broadcast_object_list", lambda payload, src: seen.append(src), raising=False)
    legacy.run_tokenize(worker, uuids)
    assert seen == [0]
