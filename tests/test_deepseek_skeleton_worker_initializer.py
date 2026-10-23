"""Real worker monkeypatch integration, with every patched torch class restored."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SKELETON = load('r1_worker_skeleton', ROOT / 'batchgen/models/deepseek/deepseekv3/skeleton.py')
UTILS = load('r1_worker_utils', ROOT / 'batchgen/utils.py')


@contextmanager
def worker_initializer(repeats=1):
    # Production mutates process-global classes. Snapshot own attributes (not
    # inherited values) so this integration test cannot affect other tests.
    missing = object()
    attrs = ('__init__', 'reset_parameters', '_old_init', '_old_reset_parameters')
    classes = {cls for cls in vars(nn.modules).values()
               if isinstance(cls, type) and issubclass(cls, nn.Module)}
    saved = {cls: {key: vars(cls).get(key, missing) for key in attrs} for cls in classes}
    try:
        for _ in range(repeats):
            UTILS.config_torch_module_initializer()
        yield
    finally:
        for cls, values in saved.items():
            for key, value in values.items():
                if value is missing:
                    if key in vars(cls):
                        delattr(cls, key)
                else:
                    setattr(cls, key, value)


@pytest.mark.parametrize('repeats', [1, 2])
@pytest.mark.parametrize('kind,bias', [('linear', True), ('linear', False), ('embedding', False)])
def test_actual_worker_patch_preserves_logical_shapes_and_zero_storage(kind, bias, repeats):
    factory = getattr(SKELETON, 'make_' + kind)
    kwargs = {'bias': bias} if kind == 'linear' else {'padding_idx': 0}
    normal = factory(SimpleNamespace(), 7, 5, **kwargs)
    expected = {name: tuple(p.shape) for name, p in normal.named_parameters()}
    with worker_initializer(repeats):
        with torch.device('meta'):
            raw = (nn.Linear if kind == 'linear' else nn.Embedding)(7, 5, **kwargs)
        assert all(tuple(p.shape) == (1,) for p in raw.parameters())  # Reproduce defect.
        model = factory(SimpleNamespace(batchgen_skeleton=True), 7, 5, **kwargs)
        SKELETON.capture_parameter_shapes(model)
        assert model._batchgen_expected_parameter_shapes == expected
        assert all(p.shape == (0,) and p.device.type == 'cpu'
                   and p.untyped_storage().nbytes() == 0 for p in model.parameters())
        model.to(dtype=torch.float64)
        assert all(p.untyped_storage().nbytes() == 0 for p in model.parameters())
        with pytest.raises(RuntimeError, match='Missing resident weight'):
            SKELETON.validate_parameter_residency(model, set())
        for name, parameter in normal.named_parameters():
            model.get_parameter(name).data = parameter.detach().clone()
        SKELETON.validate_parameter_residency(model, set())
        x = torch.randn(2, 7) if kind == 'linear' else torch.tensor([0, 1, 6])
        torch.testing.assert_close(model(x), normal(x))
        model.weight.data = torch.zeros(1, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match='Weight shape mismatch'):
            SKELETON.validate_parameter_residency(model, set())
    # The test must undo the process-global patch including nested wrappers.
    assert nn.Linear(7, 5).weight.shape == (5, 7)


def test_r1_embedding_exact_checkpoint_dimensions_remain_storage_free():
    with worker_initializer():
        model = SKELETON.make_embedding(SimpleNamespace(batchgen_skeleton=True), 129280, 7168)
        SKELETON.capture_parameter_shapes(model)
        assert SKELETON.expected_parameter_shape(model, 'weight') == (129280, 7168)
        assert model.weight.numel() == 0 and model.weight.untyped_storage().nbytes() == 0


def test_non_skeleton_path_retains_existing_worker_initializer_behavior():
    with worker_initializer():
        module = SKELETON.make_linear(SimpleNamespace(), 7, 5)
        assert module.weight.shape == (1,)
        assert not hasattr(module, '_batchgen_parameter_shapes')
