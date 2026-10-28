"""No CUDA: exact candidate signature on empty vs live page-table contracts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "batchgen/attention/mla/r1_whole_decode_cuda_graph.py"
spec = importlib.util.spec_from_file_location("checked_whole_zero_rank", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def inputs():
    # get_layer_kv_with_page_table raises on the real manager when its active
    # table is None, including a fresh rank which has never owned a sequence.
    manager = NS(_gpu_page_table_manager=NS(gpu_table=None), _k_cache=[])
    def get_layer(i):
        if manager._gpu_page_table_manager.gpu_table is None:
            raise RuntimeError("GPU page table is not initialized")
        return torch.ones(1, 64, 1, 576), None, None
    manager.get_layer_kv_with_page_table = get_layer
    model = NS(model=NS(layers=[object()] * 61))
    pool = NS(capacity=4, phase_configurations=1)
    return model, manager, pool


def test_initial_zero_row_rank_needs_no_kv_table(monkeypatch):
    model, manager, pool = inputs()
    monkeypatch.setattr(module, "_weight_signature", lambda model: ("unchanged-weights",))
    signature = module._signature(NS(), model, manager, None, pool,
        torch.empty(0, 1, dtype=torch.int64), 0)
    assert signature[-2] == ()
    pool.phase_configurations += 1
    assert module._signature(NS(), model, manager, None, pool,
        torch.empty(0, 1, dtype=torch.int64), 0) != signature


def test_live_rank_still_refuses_a_missing_kv_table(monkeypatch):
    model, manager, pool = inputs()
    monkeypatch.setattr(module, "_weight_signature", lambda model: ("unchanged-weights",))
    with pytest.raises(RuntimeError, match="GPU page table is not initialized"):
        module._signature(NS(), model, manager, object(), pool,
            torch.zeros(1, 1, dtype=torch.int64), 0)
