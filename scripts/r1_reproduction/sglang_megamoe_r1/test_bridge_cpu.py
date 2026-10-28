"""CPU numerical, storage and patch-scope checks. Not GPU qualification."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

D = Path(__file__).resolve().parent
SOURCE = D / "candidate/python/sglang/srt/r1_fp8_megamoe_bridge.py"


@pytest.fixture
def bridge(monkeypatch):
    spec = importlib.util.spec_from_file_location("r1_sg_bridge_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv(module.ENVIRONMENT, raising=False)
    return module


def hf_config():
    return SimpleNamespace(architectures=["DeepseekV3ForCausalLM"], hidden_size=7168,
        moe_intermediate_size=2048, n_routed_experts=256, num_experts_per_tok=8,
        num_hidden_layers=61, first_k_dense_replace=3)


def test_opt_in_is_explicit_and_original_flag_off_does_nothing(bridge, monkeypatch):
    monkeypatch.setattr(bridge, "_runtime", lambda: pytest.fail("Flag-off imported GPU runtime"))
    assert not bridge.allow_standard_fp8_model(None, "fp8")
    assert not bridge.maybe_prepare_routed(None, None)
    monkeypatch.setenv(bridge.ENVIRONMENT, "true")
    with pytest.raises(ValueError, match="0 or 1"):
        bridge.requested()


def test_model_guard_restricts_original_r1_fp8_without_global_config_rewrite(bridge, monkeypatch):
    monkeypatch.setenv(bridge.ENVIRONMENT, "1")
    config = hf_config()
    original = copy.deepcopy(vars(config))
    assert bridge.allow_standard_fp8_model(config, "fp8")
    assert vars(config) == original
    with pytest.raises(ValueError, match="topology"):
        bridge.allow_standard_fp8_model(config, "mxfp8")
    config.n_routed_experts = 384
    with pytest.raises(ValueError, match="n_routed_experts"):
        bridge.allow_standard_fp8_model(config, "fp8")


def old_c1_reference(weight, scales):
    """Independent per-128-block reference, not the bridge reshape algorithm."""
    n, k = weight.shape
    source = (weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.bfloat16)
    q = torch.empty_like(weight)
    sf = torch.empty_like(scales)
    for i in range(n // 128):
        for j in range(k // 128):
            block = source[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            scale = torch.pow(2.0, torch.ceil(torch.log2(block.abs().float().max().clamp_min(1e-4) / 448)))
            sf[i, j] = scale
            q[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (block * (1 / scale)).to(torch.float8_e4m3fn)
    return q, sf


@pytest.mark.parametrize("zero", [False, True])
def test_per32_representation_exactly_preserves_effective_c1_weight(bridge, zero):
    torch.manual_seed(260028)
    weight = (torch.randn((256, 512)) if not zero else torch.zeros((256, 512))).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.0013, 0.0021, 0.0039, 0.0057], [0.0034, 0.0071, 0.0027, 0.0018]])
    before_weight, before_scale = weight.float().clone(), scales.clone()
    q, sf_u8 = bridge.c1_expert_to_mxfp8(weight, scales)
    old_q, old_sf = old_c1_reference(weight, scales)
    assert q.dtype == torch.float8_e4m3fn and sf_u8.dtype == torch.uint8
    assert sf_u8.shape == (256, 16) and sf_u8.is_contiguous()
    effective_new = q.float() * torch.pow(2.0, sf_u8.int() - 127).repeat_interleave(32, 1)
    effective_old = old_q.float() * old_sf.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(q.float(), old_q.float())
    assert torch.equal(effective_new, effective_old)
    assert torch.equal(weight.float(), before_weight) and torch.equal(scales, before_scale)


def test_bad_original_block_metadata_is_rejected(bridge):
    weight = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
    for scale in (torch.zeros((1, 1)), torch.full((1, 1), float("nan"))):
        with pytest.raises(ValueError, match="positive and finite"):
            bridge.c1_expert_to_mxfp8(weight, scale)
    with pytest.raises(ValueError, match="128x128"):
        bridge.c1_expert_to_mxfp8(weight, torch.ones((1, 2)))


class TorchCPUProxy:
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False, synchronize=lambda *args: None)
    def __getattr__(self, name):
        return getattr(torch, name)


def fake_layer():
    layer = SimpleNamespace(layer_id=3)
    for name, shape in (("w13_weight", (2, 512, 512)), ("w2_weight", (2, 512, 256))):
        weight = torch.nn.Parameter(torch.ones(shape, dtype=torch.float8_e4m3fn), requires_grad=False)
        scale = torch.nn.Parameter(torch.full((shape[0], shape[1] // 128, shape[2] // 128), 0.0013), requires_grad=False)
        weight.weight_loader, scale.weight_loader = "original-weight-loader", "original-scale-loader"
        setattr(layer, name, weight)
        setattr(layer, name + "_scale_inv", scale)
    return layer


def test_only_layer_local_method_and_scale_storage_change_without_duplicate_bank(bridge, monkeypatch):
    monkeypatch.setenv(bridge.ENVIRONMENT, "1")
    config = SimpleNamespace(weight_block_size=[128, 128], is_checkpoint_fp8_serialized=True, use_mxfp8=False)
    method = SimpleNamespace(quant_config=config, use_mxfp8=False, convert_mxfp8_to_block=False)
    other_method = SimpleNamespace(quant_config=config, use_mxfp8=False)
    layer = fake_layer()
    ptrs = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    original_config = copy.deepcopy(vars(config))
    backend = SimpleNamespace(is_flashinfer_megamoe=lambda: True)
    calls = []
    def prepare(target):
        calls.append(target)
        assert method.use_mxfp8
        assert target.w13_weight_scale_inv.dtype == torch.uint8
        assert target.w13_weight_scale_inv.shape == (2, 512, 16)
        assert target.w2_weight_scale_inv.shape == (2, 512, 8)
    monkeypatch.setattr(bridge, "_runtime", lambda: (TorchCPUProxy(), backend, backend, prepare))
    # Exact R1 GPU geometry is checked independently; this test isolates
    # transformation/lifetime on small real CPU tensors without fake CUDA.
    monkeypatch.setattr(bridge, "_validate_routed", lambda *args: None)
    assert bridge.maybe_prepare_routed(method, layer)
    assert calls == [layer] and method.use_mxfp8
    assert vars(config) == original_config and not other_method.use_mxfp8
    assert ptrs == (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    assert layer.w13_weight.weight_loader == "original-weight-loader"
    assert layer.w13_weight_scale_inv.weight_loader == "original-scale-loader"
    assert layer._r1_fp8_megamoe_bridge["weight_bits"] == 8
    with pytest.raises(RuntimeError, match="twice"):
        bridge.maybe_prepare_routed(method, layer)


def test_capture_and_wrong_backend_fail_before_weight_mutation(bridge, monkeypatch):
    monkeypatch.setenv(bridge.ENVIRONMENT, "1")
    backend = SimpleNamespace(is_flashinfer_megamoe=lambda: False)
    monkeypatch.setattr(bridge, "_runtime", lambda: (TorchCPUProxy(), backend, backend, None))
    with pytest.raises(ValueError, match="both FlashInfer"):
        bridge.maybe_prepare_routed(None, None)
    backend.is_flashinfer_megamoe = lambda: True
    proxy = TorchCPUProxy()
    proxy.cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setattr(bridge, "_runtime", lambda: (proxy, backend, backend, None))
    with pytest.raises(RuntimeError, match="outside CUDA Graph"):
        bridge.maybe_prepare_routed(None, None)


def test_patch_changes_only_config_guard_and_routed_postload_not_dense():
    relative = "python/sglang/srt/layers/quantization/fp8.py"
    before = ast.parse((D / "reference/sglang_v0522_fp8.upstream.py").read_text())
    after = ast.parse((D / "candidate" / relative).read_text())
    old_cls = {n.name:n for n in before.body if isinstance(n, ast.ClassDef)}
    new_cls = {n.name:n for n in after.body if isinstance(n, ast.ClassDef)}
    for name in old_cls:
        if name != "Fp8MoEMethod":
            assert ast.dump(old_cls[name]) == ast.dump(new_cls[name]), name
    old_methods = {n.name:n for n in old_cls["Fp8MoEMethod"].body if isinstance(n, ast.FunctionDef)}
    new_methods = {n.name:n for n in new_cls["Fp8MoEMethod"].body if isinstance(n, ast.FunctionDef)}
    for name in old_methods:
        if name not in ("process_weights_after_loading_block_quant", "apply"):
            assert ast.dump(old_methods[name]) == ast.dump(new_methods[name]), name
    old_apply, new_apply = old_methods["apply"], new_methods["apply"]
    assert ast.dump(ast.Module(body=old_apply.body[:-1], type_ignores=[])) == ast.dump(
        ast.Module(body=new_apply.body[:-3], type_ignores=[]))
    assert ast.dump(old_apply.body[-1].value) == ast.dump(new_apply.body[-3].value)
    pins = json.loads((D / "PINS.json").read_text())
    assert pins["upstream_commit"] == "baa9befa381511cd4d56f83980951e5a45fffc6d"
    for item in pins["files"]:
        assert hashlib.sha256((D / "candidate" / item["path"]).read_bytes()).hexdigest() == item["candidate_sha256"]


def test_actual_original_c1_functions_match_bridge(bridge):
    spec = importlib.util.spec_from_file_location("actual_c1_source_oracle", D / "oracle_functions.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    old = helpers.c1_functions(torch)
    torch.manual_seed(281127)
    w = torch.randn((256, 512)).to(torch.float8_e4m3fn)
    s = torch.tensor([[0.0013, 0.0021, 0.0039, 0.0057], [0.0034, 0.0071, 0.0027, 0.0018]])
    bf16 = old["block_quant_dequant"](w, s, [128, 128], torch.bfloat16)
    old_q, old_sf = old["quant_weight_ue8m0"](bf16, [128, 128])
    q, sf = bridge.c1_expert_to_mxfp8(w, s)
    assert torch.equal(old_q.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(torch.pow(2.0, sf.int() - 127), old_sf.repeat_interleave(128, -2).repeat_interleave(4, -1))


def test_apply_log_is_once_per_capture_mode_after_native_return_no_gpu_values(bridge, monkeypatch, caplog):
    capture = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture[0])
    mega = SimpleNamespace(_fleet_params=SimpleNamespace(max_tokens_per_rank=8192))
    layer = SimpleNamespace(layer_id=3, moe_ep_rank=0, _flashinfer_megamoe_layer=mega)
    dispatch = SimpleNamespace(hidden_states=torch.zeros((0, 7168), dtype=torch.bfloat16))
    with caplog.at_level("INFO", logger=bridge.logger.name):
        bridge.report_apply_success(layer, dispatch, None)
        bridge.report_apply_success(layer, dispatch, None)
        capture[0] = True
        bridge.report_apply_success(layer, dispatch, None)
        bridge.report_apply_success(layer, dispatch, None)
    logs = [r.message for r in caplog.records if "R1_FP8_MEGAMOE_APPLY" in r.message]
    assert len(logs) == 2
    payloads = [json.loads(r.split("] ", 1)[1]) for r in logs]
    assert [p["capturing"] for p in payloads] == [False, True]
    assert all(p["native_runner_returned"] and not p["observation_is_replay_counter"] for p in payloads)
