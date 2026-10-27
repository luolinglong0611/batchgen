import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import batchgen.runtime_preflight as preflight

SITE = Path("/opt/conda/lib/python3.11/site-packages")


def test_unknown_model_type_fails_closed(monkeypatch):
    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_detect_model_type_from_identifier", lambda _: "new_model")

    with pytest.raises(preflight.RuntimePreflightError, match="no declared runtime contract"):
        preflight.run_runtime_preflight(SimpleNamespace(model="new-model"))


def test_preflight_checks_contract_before_core_engine(monkeypatch):
    calls = []

    monkeypatch.setattr(preflight, "_check_torch", lambda: calls.append("torch"))
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: "gpt_oss")
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: calls.append("tokenizer"))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: SITE)
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))

    def fake_import(name, *, origin=None):
        calls.append((name, origin))
        if name == "batchgen.core_engine":
            return SimpleNamespace(__file__=str(SITE / "batchgen/core_engine.so"))
        if name == "libucx":
            return SimpleNamespace(load_library=lambda: calls.append("ucx-load"))
        if name == "deep_gemm":
            return SimpleNamespace(fp8_mqa_logits=lambda max_seqlen_k: None)
        return SimpleNamespace(__file__=str(SITE / "module.py"))

    monkeypatch.setattr(preflight, "_import_required", fake_import)
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _: "0.1.5.post3")

    assert preflight.run_runtime_preflight(SimpleNamespace(model="openai/gpt-oss-120b")) == "gpt_oss"
    assert calls[0] == "torch"
    assert calls.index("ucx-load") < calls.index(("batchgen.core_engine", "batchgen"))
    assert ("batchgen_kernels.attention._C_fused_ops", "batchgen") in calls
    assert ("flash_attn_interface", "site") in calls


def test_non_aot_core_engine_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_import_required", lambda name, *, origin=None: SimpleNamespace(
        __file__="/tmp/core_engine.py"
    ))

    with pytest.raises(preflight.RuntimePreflightError, match="not an AOT native module"):
        preflight._check_core_engine()


def test_ucx_load_failure_fails():
    def broken():
        raise OSError("libucp.so.0: cannot open shared object file")

    with pytest.raises(preflight.RuntimePreflightError, match="libucp"):
        preflight._check_ucx(SimpleNamespace(load_library=broken))


# --- provenance: same-root rule ------------------------------------------------


@pytest.fixture
def worktrees(tmp_path, monkeypatch):
    """Two source worktrees plus a pyroot that symlinks into worktree A."""

    a, b = tmp_path / "wt_a", tmp_path / "wt_b"
    for wt in (a, b):
        (wt / "batchgen").mkdir(parents=True)
        (wt / "batchgen_kernels/attention").mkdir(parents=True)
        (wt / "core").mkdir()
    pyroot = tmp_path / "pyroot_a"
    pyroot.mkdir()
    os.symlink(a / "batchgen", pyroot / "batchgen")
    os.symlink(a / "batchgen_kernels", pyroot / "batchgen_kernels")
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: a.resolve())
    return SimpleNamespace(a=a, b=b, pyroot=pyroot)


def _mod(path):
    return SimpleNamespace(__file__=str(path))


def test_pyroot_kernels_from_same_worktree_pass(worktrees):
    ext = worktrees.pyroot / "batchgen_kernels/attention/_C_fused_ops.so"
    (worktrees.a / "batchgen_kernels/attention/_C_fused_ops.so").touch()
    resolved = preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)
    assert resolved == (worktrees.a / "batchgen_kernels/attention/_C_fused_ops.so").resolve()


def test_kernels_from_other_worktree_fail(worktrees):
    ext = worktrees.b / "batchgen_kernels/attention/_C_fused_ops.so"
    with pytest.raises(preflight.RuntimePreflightError, match="same install or worktree"):
        preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)


def test_kernels_from_site_packages_fail_for_source_batchgen(worktrees):
    ext = SITE / "batchgen_kernels/attention/_C_fused_ops.so"
    with pytest.raises(preflight.RuntimePreflightError, match="same install or worktree"):
        preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)


def test_third_party_native_shadowed_by_worktree_fails(worktrees):
    with pytest.raises(preflight.RuntimePreflightError, match="outside site-packages"):
        preflight._require_origin(
            "flash_mla", _mod(worktrees.a / "flash_mla/__init__.py"), preflight._ORIGIN_SITE
        )


# --- core_engine freshness in a source worktree --------------------------------


def _core_engine_at(monkeypatch, so_path):
    monkeypatch.setattr(
        preflight,
        "_import_required",
        lambda name, *, origin=None: _mod(so_path),
    )


def test_stale_core_engine_fails(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.cpython-311-x86_64-linux-gnu.so"
    so.touch()
    src = worktrees.a / "core/batchgen.cpp"
    src.touch()
    os.utime(so, (1000, 1000))
    os.utime(src, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    with pytest.raises(preflight.RuntimePreflightError, match="older than"):
        preflight._check_core_engine()


def test_fresh_core_engine_passes(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.cpython-311-x86_64-linux-gnu.so"
    so.touch()
    src = worktrees.a / "core/batchgen.cpp"
    src.touch()
    os.utime(src, (1000, 1000))
    os.utime(so, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    preflight._check_core_engine()


def test_installed_core_engine_skips_source_freshness(monkeypatch):
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: SITE)
    _core_engine_at(monkeypatch, SITE / "batchgen/core_engine.so")

    preflight._check_core_engine()


def test_missing_core_dir_fails_in_source_mode(worktrees, monkeypatch):
    (worktrees.a / "core").rmdir()
    so = worktrees.a / "batchgen/core_engine.so"
    so.touch()
    _core_engine_at(monkeypatch, so)

    with pytest.raises(preflight.RuntimePreflightError, match="no core/ directory"):
        preflight._check_core_engine()


def test_non_source_files_do_not_mark_core_engine_stale(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.so"
    so.touch()
    note = worktrees.a / "core/NOTES.md"
    note.touch()
    os.utime(so, (1000, 1000))
    os.utime(note, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    preflight._check_core_engine()


def test_newer_external_header_marks_core_engine_stale(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.so"
    so.touch()
    header = worktrees.a / "external/dep/api.h"
    header.parent.mkdir(parents=True)
    header.touch()
    os.utime(so, (1000, 1000))
    os.utime(header, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    with pytest.raises(preflight.RuntimePreflightError, match="older than"):
        preflight._check_core_engine()


def test_kernels_dev_jit_is_rejected(monkeypatch):
    monkeypatch.setenv("BATCHGEN_KERNELS_DEV", "1")

    with pytest.raises(preflight.RuntimePreflightError, match="BATCHGEN_KERNELS_DEV"):
        preflight.run_runtime_preflight(SimpleNamespace(model="openai/gpt-oss-120b"))


def test_batchgen_root_resolves_through_pyroot_symlink(worktrees, monkeypatch):
    # Exercise the real _batchgen_root: batchgen imported via pyroot/batchgen.
    monkeypatch.undo()
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))
    fake = SimpleNamespace(__file__=str(worktrees.pyroot / "batchgen/__init__.py"))
    monkeypatch.setitem(__import__("sys").modules, "batchgen", fake)

    assert preflight._batchgen_root() == worktrees.a.resolve()
    assert preflight._is_source_root(worktrees.a.resolve())


def test_batchgen_root_in_site_packages_is_installed(tmp_path, monkeypatch):
    site_dir = tmp_path / "site-packages"
    (site_dir / "batchgen").mkdir(parents=True)
    monkeypatch.setattr(preflight, "_site_roots", lambda: (site_dir.resolve(),))
    fake = SimpleNamespace(__file__=str(site_dir / "batchgen/__init__.py"))
    monkeypatch.setitem(__import__("sys").modules, "batchgen", fake)

    assert preflight._batchgen_root() == site_dir.resolve()
    assert not preflight._is_source_root(site_dir.resolve())


@pytest.mark.parametrize(
    ("capability", "sm_count", "expect_wgmma"),
    [((9, 0), 78, True), ((9, 0), 132, False), ((10, 0), 78, False)],
)
def test_wgmma_decode_selected_only_on_h20(monkeypatch, capability, sm_count, expect_wgmma):
    imported = []

    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: "gpt_oss")
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: None)
    monkeypatch.setattr(preflight, "_check_core_engine", lambda: None)
    monkeypatch.setattr(preflight, "_check_deepgemm", lambda: None)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(preflight.torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(
        preflight.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=sm_count),
    )

    def fake_import(name, *, origin=None):
        imported.append(name)
        if name == "libucx":
            return SimpleNamespace(load_library=lambda: None)
        if name == "batchgen_kernels.attention.decode":
            return SimpleNamespace(attention_decode_bf16=lambda: None)
        return SimpleNamespace(__file__=str(SITE / "module.py"))

    monkeypatch.setattr(preflight, "_import_required", fake_import)

    preflight.run_runtime_preflight(SimpleNamespace(model="openai/gpt-oss-120b"))
    assert ("batchgen_kernels.attention.decode" in imported) == expect_wgmma


@pytest.fixture
def explicit_r1_profile(monkeypatch):
    monkeypatch.setenv("BATCHGEN_R1_SM103_MLA_COMPAT", "flashinfer-fa2")
    monkeypatch.setenv("BATCHGEN_R1_SM103_FP8_COMPAT", "1")
    monkeypatch.setenv("DISABLE_MMA_V5", "1")
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "get_device_capability", lambda *_: (10, 3))
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _: "0.5.2")
    return SimpleNamespace(model="deepseek-ai/DeepSeek-R1", disable_cuda_graphs=True,
                           enable_cuda_graph=False, kv_dtype="bfloat16")


def test_explicit_r1_profile_declares_both_required_aot_modules(explicit_r1_profile):
    contract = preflight._selected_contract("deepseek_v3", explicit_r1_profile)
    assert contract.modules == ("flashinfer", "libucx")
    assert contract.extensions == (
        "batchgen_kernels.common._C_mgn_ops",
        "batchgen_kernels.moe._C_fused_moe_token_permutation",
    )
    assert not contract.require_deepgemm


@pytest.mark.parametrize("attention,gemm", [("0", "1"), ("flashinfer-fa2", "0")])
def test_partial_r1_profile_never_selects_an_implicit_fallback(monkeypatch, explicit_r1_profile, attention, gemm):
    monkeypatch.setenv("BATCHGEN_R1_SM103_MLA_COMPAT", attention)
    monkeypatch.setenv("BATCHGEN_R1_SM103_FP8_COMPAT", gemm)
    with pytest.raises(preflight.RuntimePreflightError, match="requires both explicit"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("disable,enable", [(False, False), (False, True), (True, True)])
def test_explicit_r1_profile_refuses_every_graph_enabled_or_implicit_case(explicit_r1_profile, disable, enable):
    explicit_r1_profile.disable_cuda_graphs = disable
    explicit_r1_profile.enable_cuda_graph = enable
    with pytest.raises(preflight.RuntimePreflightError, match="disable-cuda-graphs"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("kv_dtype", [None, "auto", "bf16", "float8_e4m3fn", "float16"])
def test_explicit_r1_profile_requires_bfloat16_kv(explicit_r1_profile, kv_dtype):
    explicit_r1_profile.kv_dtype = kv_dtype
    with pytest.raises(preflight.RuntimePreflightError, match="BF16 KV"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("available,capability", [(False, (10, 3)), (True, (10, 0)), (True, (9, 0)), (True, (8, 9))])
def test_explicit_r1_profile_requires_actual_cuda_sm103(monkeypatch, explicit_r1_profile, available, capability):
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(preflight.torch.cuda, "get_device_capability", lambda *_: capability)
    with pytest.raises(preflight.RuntimePreflightError, match="actual CUDA capability 10.3"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("version", ["0.5.1", "0.6.13", "0.5.2.post1"])
def test_explicit_r1_profile_requires_the_exact_common_fi_version(monkeypatch, explicit_r1_profile, version):
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _: version)
    with pytest.raises(preflight.RuntimePreflightError, match="flashinfer-python==0.5.2"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("model_type,model", [("deepseek_v3", "deepseek-ai/DeepSeek-V3"),
                                            ("gpt_oss", "openai/gpt-oss-120b"),
                                            ("deepseek_v2", "deepseek-ai/DeepSeek-R1")])
def test_explicit_r1_profile_cannot_affect_another_model(explicit_r1_profile, model_type, model):
    explicit_r1_profile.model = model
    with pytest.raises(preflight.RuntimePreflightError, match="restricted to DeepSeek-R1"):
        preflight._selected_contract(model_type, explicit_r1_profile)


def test_explicit_r1_profile_requires_the_triton_arch_compatibility_flag(monkeypatch, explicit_r1_profile):
    monkeypatch.delenv("DISABLE_MMA_V5")
    with pytest.raises(preflight.RuntimePreflightError, match="DISABLE_MMA_V5=1"):
        preflight._selected_contract("deepseek_v3", explicit_r1_profile)


@pytest.mark.parametrize("model_type", ["deepseek_v3", "gpt_oss", "mixtral"])
def test_native_default_contract_is_identical_without_compatibility_flags(monkeypatch, model_type):
    monkeypatch.delenv("BATCHGEN_R1_SM103_MLA_COMPAT", raising=False)
    monkeypatch.delenv("BATCHGEN_R1_SM103_FP8_COMPAT", raising=False)

    def unexpected(*args):
        raise AssertionError("Native default must not probe the SM103/FI compatibility implementation")

    monkeypatch.setattr(preflight.torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(preflight.importlib.metadata, "version", unexpected)
    assert preflight._selected_contract(model_type, SimpleNamespace(model="native-default")) is preflight._CONTRACTS[model_type]
