"""Run the frozen R1 CPU contracts without importing the GPU package initializer.

Only the top-level package namespace is predeclared. Actual submodules are read
from this checkout; individual tests own any accelerator API doubles. Pass
pytest arguments or test paths to override the default validated test suite.
"""

from pathlib import Path
import os
import sys
import types


TESTS = (
    "tests/test_r1_phase_reuse.py",
    "tests/test_phase_instance_reuse.py",
    "tests/test_phase_switch_control_flow.py",
    "tests/test_r1_resident_prefill.py",
    "tests/test_r1_native_ordinary_gemm.py",
    "tests/test_r1_decode_cuda_graph.py",
    "tests/test_r1_decode_fast_metadata.py",
    "tests/test_r1_whole_decode_cuda_graph.py",
    "tests/test_r1_whole_graph_eager_compare.py",
    "tests/test_r1_sm103_fp8_compat.py",
    "tests/test_r1_prefill_preparation_reuse.py",
    "tests/test_r1_prefill_retirement.py",
    "tests/test_r1_prefill_rope.py",
    "tests/test_worker_tokenization_memory.py",
    "tests/test_tokenization_policy.py",
    "tests/worker/test_step_timing.py",
    "tests/worker/test_decode_trace.py",
    "tests/test_decode_step_timing_wiring.py",
    "tests/test_r1_whole_kv_contiguous_stage.py",
    "tests/test_r1_shared_input_quant_reuse.py",
    "tests/test_tokenized_tensor_collective.py",
    "tests/test_r1_blackwell_prefill.py",
    "tests/test_r1_prefill_normal_config.py",
    "tests/test_r1_prefill_silu_quant_contract.py",
    "tests/test_r1_static_cast_reuse.py",
    "tests/test_r1_rope_handoff_contract.py",
    "tests/test_r1_deepgemm_moe.py",
    "tests/test_r1_decode_fused_native_rope.py",
)


def main():
    import pytest

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    package = types.ModuleType("batchgen")
    package.__path__ = [str(root / "batchgen")]
    sys.modules["batchgen"] = package
    return pytest.main(["-q", "--tb=short", *(sys.argv[1:] or TESTS)])


if __name__ == "__main__":
    raise SystemExit(main())
