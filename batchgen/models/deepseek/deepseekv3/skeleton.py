"""Storage-free large parameters for BatchGen's explicit weight-loading path.

This does not change ordinary model construction. Small parameters and buffers
still initialize normally; only Linear/Embedding parameters use empty CPU
placeholders, which support the serving code's later ``parameter.data = weight``.
"""

import torch
from torch import nn


def _make_module(config, factory, *args, **kwargs):
    if not getattr(config, "batchgen_skeleton", False):
        return factory(*args, **kwargs)
    module = factory(*args, **kwargs, device="meta")
    module._batchgen_parameter_shapes = {}
    for name, parameter in list(module.named_parameters(recurse=False)):
        module._batchgen_parameter_shapes[name] = tuple(parameter.shape)
        module.register_parameter(name, nn.Parameter(
            torch.empty(0, dtype=parameter.dtype, device="cpu"),
            requires_grad=parameter.requires_grad,
        ))
    return module


def make_linear(config, *args, **kwargs):
    return _make_module(config, nn.Linear, *args, **kwargs)


def make_embedding(config, *args, **kwargs):
    return _make_module(config, nn.Embedding, *args, **kwargs)


def expected_parameter_shape(model, name):
    expected = getattr(model, "_batchgen_expected_parameter_shapes", {})
    if name in expected:
        return expected[name]
    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    shapes = getattr(module, "_batchgen_parameter_shapes", {})
    if parameter_name in shapes:
        return shapes[parameter_name]
    return tuple(module.get_parameter(parameter_name).shape)


def capture_parameter_shapes(model):
    """Remember small norm/router shapes as well, before any .data replacement."""
    model._batchgen_expected_parameter_shapes = {
        name: expected_parameter_shape(model, name)
        for name, _ in model.named_parameters()
    }


def validate_parameter_residency(model, allowed_empty_names):
    """Refuse missing resident weights before wrappers or device transfer.

The caller must enumerate the exact streamed or non-local parameter names.
Empty tensors have zero elements, so a later ``model.to(cuda)`` cannot allocate
the full logical matrices that were recorded only as Python shape metadata.
"""
    for name, parameter in model.named_parameters():
        if parameter.is_meta:
            raise RuntimeError(f"Unmaterialized meta parameter: {name}")
        if parameter.numel() == 0:
            if name not in allowed_empty_names:
                raise RuntimeError(f"Missing resident weight: {name}")
        elif tuple(parameter.shape) != expected_parameter_shape(model, name):
            raise RuntimeError(
                f"Weight shape mismatch for {name}: {tuple(parameter.shape)} "
                f"!= {expected_parameter_shape(model, name)}"
            )
