"""Opt-in EP8 normal-DeepEP tuning, without changing its global LL policy.

The installed DeepEP v1.2.1 Config does not expose its fields to Python.
Only the reviewed EP8 normal defaults are reproduced when an SM override is
requested. Zero preserves the installed library's original config object.
"""
import os


NORMAL_SMS_ENV = "BATCHGEN_R1_PREFILL_NORMAL_SMS"
DISPATCH_SMS_ENV = "BATCHGEN_R1_PREFILL_DISPATCH_SMS"
COMBINE_SMS_ENV = "BATCHGEN_R1_PREFILL_COMBINE_SMS"
_EP8_NORMAL_CHUNKS = {
    "dispatch": (6, 256, 6, 128),
    "combine": (4, 256, 6, 128),
}


def requested_normal_sms(environ=None):
    """Return role-local overrides; zero means the library recommendation."""
    environ = os.environ if environ is None else environ

    def read(name, default):
        raw = environ.get(name, default)
        if (not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal()
                or (int(raw) != 0 and (int(raw) < 2 or int(raw) % 2))):
            raise ValueError(f"{name} must be0 or a positive even SM count")
        return int(raw)

    common = str(read(NORMAL_SMS_ENV, "0"))
    return {"dispatch": read(DISPATCH_SMS_ENV, common),
            "combine": read(COMBINE_SMS_ENV, common)}


def normal_config(buffer_type, config_type, *, role, requested_sms,
                  world_size=8, device_sm_count=None):
    """Construct one local config without Buffer.set_num_sms/global mutation.

    The library requires an even dispatch/combine block count. The candidate
    additionally limits it to the active device's SM count, even though CUDA
    can queue more blocks; this makes the sweep meaningful and bounded.
    """
    if world_size != 8 or role not in _EP8_NORMAL_CHUNKS:
        raise ValueError("R1 normal SM tuning requires EP8 and dispatch/combine")
    if (type(requested_sms) is not int or requested_sms < 0
            or requested_sms and (requested_sms < 2 or requested_sms % 2)):
        raise ValueError("Normal DeepEP SM count must be0 or positive and even")
    if not requested_sms:
        return getattr(buffer_type, "get_" + role + "_config")(world_size)
    if type(device_sm_count) is not int or device_sm_count < requested_sms:
        raise ValueError("Normal DeepEP SM count exceeds the active GPU SM count")
    if not callable(config_type):
        raise RuntimeError("Installed DeepEP lacks the reviewed Config constructor")
    # Reviewed get_{dispatch,combine}_config(8) from DeepEP v1.2.1; only
    # the first constructor argument changes. No collective/math policy changes.
    return config_type(requested_sms, *_EP8_NORMAL_CHUNKS[role])
