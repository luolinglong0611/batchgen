"""Explicit selection of the R1 SM103 attention compatibility implementation."""
import os


ENVIRONMENT = "BATCHGEN_R1_SM103_MLA_COMPAT"
FLASHINFER_FA2 = "flashinfer-fa2"


def common_attention_enabled():
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", FLASHINFER_FA2):
        raise ValueError(f"{ENVIRONMENT} must be 0 or {FLASHINFER_FA2!r}, got {value!r}")
    return value == FLASHINFER_FA2
