"""Opt-in single-model R1 prefill/decode phase reuse.

One decode-capable model owns attention/shared/dense weights and their native
FP8 banks. Only the routed MoE forward branch changes. The worker still drains
Host writes, destroys old graphs and releases/rebuilds GPU KV at boundaries.
No second phase model or copy of any weight bank is retained here.
"""
import os
import types


ENVIRONMENT = "BATCHGEN_R1_PHASE_REUSE"


def enabled_for(manager):
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(ENVIRONMENT + " must be explicitly0 or1")
    if value == "0":
        return False
    config = manager.engine_config
    if (manager.world_size != 8 or config.EP_Config.enable_offloading
            or config.Basic_Config.attn_mode != 3
            or not manager._r1_native_moe() or not manager.prefill_uses_resident_ep()
            or os.environ.get("BATCHGEN_R1_ORDINARY_GEMM") != "deep_gemm"
            or not config.Module_Batching_Config.r1_deepep_ll
            or os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1"):
        raise ValueError("R1 phase reuse requires single EP8, native ordinary/MoE GEMMs, "
                         "resident prefill, DeepEP LL and no weight offload")
    if getattr(manager, "persistent_phase_instances", False):
        raise ValueError("Choose R1 single-model phase reuse without dual persistent phase instances")
    return True


def phase_moe_forward(moe, hidden_states):
    phase = moe._r1_runtime_phase
    if phase == "prefill":
        return moe._r1_phase_prefill_impl(moe, hidden_states)
    if phase == "decode":
        return moe._r1_phase_decode_impl(moe, hidden_states)
    raise RuntimeError("R1 resident model has no active prefill/decode phase")


class R1SharedPhaseState:
    """A bounded owner for exactly one model, with fixed decode capacity."""
    def __init__(self, model, *, first_moe_layer, capacity, generation, prefill_impl):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("R1 shared phase model needs a fixed positive decode capacity")
        self.model, self.capacity, self.generation = model, capacity, generation
        self.layers = tuple(model.model.layers[first_moe_layer:])
        self.phase = None
        for layer in self.layers:
            moe = layer.mlp
            if (getattr(moe, "r1_deepep_adapter", None) is None
                    or getattr(moe, "_r1_resident_prefill_adapter", None) is None):
                raise RuntimeError("R1 single-model reuse requires both native MoE adapters")
        for layer in self.layers:
            moe = layer.mlp
            moe._r1_phase_decode_impl = type(moe).forward
            moe._r1_phase_prefill_impl = prefill_impl
            moe._r1_runtime_phase = None
            moe.forward = types.MethodType(phase_moe_forward, moe)
        model._r1_weight_generation = generation
        model._r1_weight_storage_immutable = True

    def activate(self, phase, *, requested_rows=None):
        if self.model is None:
            raise RuntimeError("Released R1 shared phase model cannot be reactivated")
        if phase not in ("prefill", "decode"):
            raise ValueError("R1 shared phase target must be prefill or decode")
        if (requested_rows is not None
                and (type(requested_rows) is not int or not 0 <= requested_rows <= self.capacity)):
            raise ValueError("R1 reused decode rows exceed the fixed process capacity")
        # No conversion/reload is allowed here. A source mutation must be
        # handled by explicit model release/rebuild, never a stale graph replay.
        for layer in self.model.model.layers:
            owners = (layer.self_attn, getattr(layer.mlp, "shared_experts", None))
            for owner in owners:
                module = getattr(owner, "module", owner)
                for entry in getattr(module, "_r1_native_gemm_weights", ()):
                    if not entry.valid():
                        raise RuntimeError("R1 shared phase activation found stale ordinary weights")
        self.model.config.phase = phase
        nested = getattr(self.model.model, "config", None)
        if nested is not None:
            nested.phase = phase
        for layer in self.layers:
            layer.mlp._r1_runtime_phase = phase
        self.phase = phase
        return self.model

    def release(self):
        if self.model is not None:
            self.model._r1_weight_storage_immutable = False
        self.model, self.layers, self.phase = None, (), None
