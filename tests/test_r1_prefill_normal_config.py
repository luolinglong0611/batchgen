"""Normal DeepEP config ownership and legal SM-bound contracts; CPU-only."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "batchgen/models/deepseek/deepseekv3"
PACKAGE = "r1_normal_prefill_config_contract"
package = ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".resident_prefill_tuning",
                                             DIRECTORY / "resident_prefill_tuning.py")
tuning = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuning
spec.loader.exec_module(tuning)


class Buffer:
    num_sms = 20
    defaults = {role: object() for role in ("dispatch", "combine")}

    @staticmethod
    def get_dispatch_config(world_size):
        assert world_size == 8
        return Buffer.defaults["dispatch"]

    @staticmethod
    def get_combine_config(world_size):
        assert world_size == 8
        return Buffer.defaults["combine"]

    @staticmethod
    def set_num_sms(*args):
        pytest.fail("Normal tuning must never mutate DeepEP's global policy")


def test_default_returns_original_library_config_objects():
    for role in ("dispatch", "combine"):
        assert tuning.normal_config(Buffer, None, role=role, requested_sms=0) is Buffer.defaults[role]


@pytest.mark.parametrize("sms", [20, 40, 80, 120, 144])
def test_explicit_configs_change_only_SMS_and_do_not_touch_LL_global(sms):
    config_type = lambda *args: args
    dispatch = tuning.normal_config(Buffer, config_type, role="dispatch", requested_sms=sms, device_sm_count=148)
    combine = tuning.normal_config(Buffer, config_type, role="combine", requested_sms=sms, device_sm_count=148)
    assert dispatch == (sms, 6, 256, 6, 128)
    assert combine == (sms, 4, 256, 6, 128)
    assert Buffer.num_sms == 20


def test_common_override_and_role_overrides_are_independent():
    e = {tuning.NORMAL_SMS_ENV: "80", tuning.COMBINE_SMS_ENV: "40"}
    assert tuning.requested_normal_sms(e) == {"dispatch": 80, "combine": 40}
    e[tuning.DISPATCH_SMS_ENV] = "0"
    assert tuning.requested_normal_sms(e) == {"dispatch": 0, "combine": 40}
    assert tuning.requested_normal_sms({}) == {"dispatch": 0, "combine": 0}


@pytest.mark.parametrize("value", ["-2", "1", "3", "2.0", " 40", "４０", "NaN", "", 40])
def test_malformed_SM_requests_are_rejected(value):
    with pytest.raises(ValueError):
        tuning.requested_normal_sms({tuning.NORMAL_SMS_ENV: value})


@pytest.mark.parametrize("sms", [150, 256])
def test_override_cannot_exceed_actual_GPU_sms(sms):
    with pytest.raises(ValueError, match="active GPU"):
        tuning.normal_config(Buffer, tuple, role="dispatch", requested_sms=sms, device_sm_count=148)


def test_unsupported_config_constructor_and_non_EP8_fail_closed():
    with pytest.raises(RuntimeError, match="Config"):
        tuning.normal_config(Buffer, None, role="dispatch", requested_sms=40, device_sm_count=148)
    with pytest.raises(ValueError, match="EP8"):
        tuning.normal_config(Buffer, tuple, role="dispatch", requested_sms=40, world_size=16, device_sm_count=148)


def test_exchange_uses_candidate_buffer_size_hints_and_leaves_global_policy(monkeypatch):
    import torch
    spec = importlib.util.spec_from_file_location(PACKAGE + ".resident_prefill", DIRECTORY / "resident_prefill.py")
    resident = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resident)
    constructed = []

    class Config:
        def __init__(self, *args):
            self.args = args
        def get_nvl_buffer_size_hint(self, hidden_bytes, world):
            return self.args[0] * 1000
        def get_rdma_buffer_size_hint(self, hidden_bytes, world):
            return 0

    class LocalBuffer(Buffer):
        def __init__(self, group, **kw):
            constructed.append(kw)
        def dispatch(self, x, *, expert_alignment, topk_weights, num_tokens_per_expert):
            pass
        def combine(self, x, handle, *, previous_event):
            pass

    monkeypatch.setenv(tuning.NORMAL_SMS_ENV, "80")
    monkeypatch.delenv(tuning.DISPATCH_SMS_ENV, raising=False)
    monkeypatch.setenv(tuning.COMBINE_SMS_ENV, "40")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=148))
    monkeypatch.setattr(resident.importlib, "import_module", lambda name: SimpleNamespace(Buffer=LocalBuffer, Config=Config))
    exchange = resident.R1NormalPrefillExchange(SimpleNamespace(size=lambda: 8))
    assert exchange.dispatch_config.args == (80, 6, 256, 6, 128)
    assert exchange.combine_config.args == (40, 4, 256, 6, 128)
    assert constructed == [{"num_nvl_bytes": 80000, "num_rdma_bytes": 0,
                            "low_latency_mode": False, "explicitly_destroy": True}]
    assert Buffer.num_sms == 20
