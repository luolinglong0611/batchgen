"""Candidate enablement and input ownership contracts, no CUDA qualification."""
import importlib.util
import ast
from pathlib import Path

import pytest
import torch


path = Path(__file__).resolve().parents[1] / 'batchgen/models/deepseek/deepseekv3/prefill_silu_quant.py'
spec = importlib.util.spec_from_file_location('r1_prefill_silu_contract', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def test_fused_candidate_default_off_and_strict_opt_in(monkeypatch):
    monkeypatch.delenv(candidate.ENV, raising=False)
    assert not candidate.requested() and candidate._KERNEL is None
    monkeypatch.setenv(candidate.ENV, '1')
    assert candidate.requested()
    monkeypatch.setenv(candidate.ENV, 'true')
    with pytest.raises(ValueError):
        candidate.requested()


def test_fused_candidate_refuses_CPU_before_importing_or_initializing_triton():
    with pytest.raises(ValueError, match='CUDA BF16'):
        candidate.fused_silu_quantize(torch.zeros((128, 4096), dtype=torch.bfloat16))
    assert candidate._KERNEL is None


def test_routing_dependent_row_count_and_scale_stride_never_specialize_JIT():
    """Expert counts vary every layer; they must not create fresh kernels."""
    tree = ast.parse(path.read_text())
    fused = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fused')
    constants = [a.arg for a in fused.args.args if isinstance(a.annotation, ast.Attribute)
                 and a.annotation.attr == 'constexpr']
    assert constants == ['K', 'GROUPS', 'BLOCK_K']
    assert 'ROWS' not in [a.arg for a in fused.args.args]
    assert next(a for a in fused.args.args if a.arg == 'ALIGNED_ROWS').annotation is None
    decorator = next(d for d in fused.decorator_list if isinstance(d, ast.Call))
    assert ast.literal_eval(next(k.value for k in decorator.keywords if k.arg == 'do_not_specialize')) == ['ALIGNED_ROWS']
    body_names = {n.id for statement in fused.body for n in ast.walk(statement) if isinstance(n, ast.Name)}
    assert all(a.arg in body_names for a in fused.args.args)
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_silu_quantize')
    assert not any(k.arg == 'ROWS' for n in ast.walk(wrapper) if isinstance(n, ast.Call) for k in n.keywords)
