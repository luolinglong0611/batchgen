"""Audit R1's real service tokenizer against the pinned HF assets, without engines.

Run with torch, tokenizers and transformers installed. No GPU or network needed.
"""
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "batchgen/models/deepseek/deepseekv3"
spec = importlib.util.spec_from_file_location(
    "r1_workload_tokenizer_test", ROOT / "scripts/r1_reproduction/prepare_workload.py"
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


@pytest.fixture
def tokenizers():
    service, _ = workload.load_service_tokenizer(ASSETS / "tokenizer.json")
    hf = transformers.AutoTokenizer.from_pretrained(str(ASSETS), local_files_only=True)
    return service, hf


def worker_encode(service, texts, **kwargs):
    return service(texts, return_tensors=None, padding=False,
                   truncation=False, return_attention_mask=False, **kwargs)["input_ids"]


def test_plain_single_and_worker_batch_match_hf(tokenizers):
    service, hf = tokenizers
    texts = ["Hello world", "", "你好，世界！", "A longer sentence with several words."]
    expected = [hf.encode(text, add_special_tokens=True) for text in texts]
    assert expected[0] == [0, 19923, 2058]
    assert [service.encode(text) for text in texts] == expected
    assert worker_encode(service, texts) == expected
    assert worker_encode(service, "Hello world") == [expected[0]]
    assert service.encode("Hello world", add_special_tokens=False) == [19923, 2058]
    padded = service(texts, return_tensors="pt", padding=True)
    for index, ids in enumerate(expected):
        assert padded["input_ids"][index][padded["attention_mask"][index].bool()].tolist() == ids


@pytest.mark.parametrize("generation_prompt", [False, True])
def test_rendered_chat_and_worker_do_not_duplicate_bos(tokenizers, generation_prompt):
    service, hf = tokenizers
    messages = [{"role": "user", "content": "Hello world"}]
    expected = hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt)
    text = service.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    assert service.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt) == expected
    assert service.encode(text) == expected
    assert worker_encode(service, [text])[0] == expected
    assert expected.count(service.bos_token_id) == 1
    # HF encode(rendered, True) is NOT the desired chat-template contract.
    assert hf.encode(text)[:2] == [service.bos_token_id] * 2


def test_explicit_bos_boundaries_are_idempotent_without_collapsing_content(tokenizers):
    service, hf = tokenizers
    bos = service.bos_token
    plain = hf.encode("Hello world", add_special_tokens=False)
    for count in (1, 2):
        text = bos * count + "Hello world"
        expected = [service.bos_token_id] * count + plain
        assert service.encode(text) == expected
        assert service.encode(text, add_special_tokens=False) == expected
        assert worker_encode(service, [text])[0] == expected
    interior = "Hello " + bos + " world"
    assert service.encode(interior) == hf.encode(interior)


@pytest.mark.parametrize("add_bos,add_eos", [(False, False), (True, False), (False, True), (True, True)])
def test_configured_bos_eos_flags_and_existing_boundaries(tokenizers, add_bos, add_eos):
    service, hf = tokenizers
    service._config.update(add_bos_token=add_bos, add_eos_token=add_eos)
    service._setup_configured_special_tokens()
    hf.add_bos_token, hf.add_eos_token = add_bos, add_eos
    expected = hf.encode("Hello world")
    assert service.encode("Hello world") == expected
    assert worker_encode(service, ["Hello world"])[0] == expected
    text = (service.bos_token if add_bos else "") + "Hello world" + (service.eos_token if add_eos else "")
    assert service.encode(text) == expected
    assert worker_encode(service, [text])[0] == expected
    if add_eos:
        assert service.encode(text + service.eos_token) == expected + [service.eos_token_id]
    assert service.encode(text, add_special_tokens=False) == hf.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("mode", ["raw", "chat"])
def test_exact_length_workload_matches_both_service_paths(tokenizers, mode):
    service, hf = tokenizers
    row = {"context": "The evidence is stored in a blue notebook. " * 3000,
           "question": "Where is the evidence?"}
    for target, _ in workload.PROFILES:
        result, reason = workload.fit_context(hf, "Read: {context}\nQuestion: {input}\nAnswer:", row, target, mode)
        assert result is not None, reason
        prompt, ids, _, _ = result
        assert len(ids) == target
        assert service.encode(prompt) == ids
        assert worker_encode(service, [prompt])[0] == ids


def test_actual_manifest_records_independent_asset_revisions_and_rejects_tampering(tmp_path):
    manifest = ROOT / "scripts/r1_reproduction/r1_model_manifest.json"
    files = [ASSETS / "tokenizer.json", ASSETS / "tokenizer_config.json"]
    pins = workload.tokenizer_file_pins(files, manifest)
    assert pins["tokenizer.json"]["revision"] == "20f2865710d161dc4a32c4e8c3d7c42c7b9852f8"
    assert pins["tokenizer_config.json"]["revision"] == "88aa7f6c55672ea426d6c45b0dccbd0e82c38f15"
    config = tmp_path / "tokenizer_config.json"
    shutil.copyfile(files[1], config)
    with config.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="does not match pinned"):
        workload.tokenizer_file_pins([files[0], config], manifest)


def test_offline_wrapper_load_restores_module_registry():
    names = ["batchgen", "batchgen.config", "batchgen.config.tokenizer_registry",
             "batchgen.config.base_tokenizer", "batchgen.config.fast_tokenizer", "_r1_audit_deepseek_tokenizer"]
    before = {name: sys.modules.get(name) for name in names}
    workload.load_service_tokenizer(ASSETS / "tokenizer.json")
    assert {name: sys.modules.get(name) for name in names} == before
