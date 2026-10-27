"""Reuse one real position maximum within a serial eager MLA decode forward.

The owner must bind the same normal (versioned) position tensor for every layer
and keep it unchanged until the forward returns. This scope caches only the
scalar maximum: each layer must still apply its own rotary-cache bounds check.
Outside an explicit scope, the original reduction and scalar readback remain.
Unsafe writes through .data or an external writer bypass version counters and
are unsupported while the scope is active.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_DECODE_POSITION_STATE = ContextVar("mla_decode_position_state", default=None)


def _torch():
    import torch
    return torch


def _position_signature(tensor):
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise RuntimeError(
            "Scoped MLA positions require a normal versioned tensor; "
            "use the unscoped path for inference tensors"
        ) from exc
    return (version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.layout, tensor.dtype, tensor.device)


def _stream_key(tensor):
    torch = _torch()
    if tensor.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise ValueError("Scoped MLA position validation requires eager CUDA execution")
    return (tensor.device, torch.cuda.current_stream(tensor.device).cuda_stream)


class _DecodePositionState:
    def __init__(self):
        self.tensor = None
        self.signature = None
        self.stream_key = None
        self.maximum = None

    def clear(self):
        # The yielded state may survive in user code or an exception traceback.
        # It must not retain a tensor or a reusable scalar after this forward.
        self.tensor = None
        self.signature = None
        self.stream_key = None
        self.maximum = None

    def bind(self, tensor):
        if not isinstance(tensor, _torch().Tensor):
            raise ValueError("Scoped MLA positions must be a Torch tensor")
        self.tensor = tensor
        self.signature = _position_signature(tensor)
        self.stream_key = _stream_key(tensor)
        if tensor.numel():
            self.maximum = tensor.max().item()
        # Detect changes even if a caller mutates while the initial read waits.
        self.validate(tensor)

    def validate(self, tensor):
        if tensor is not self.tensor:
            raise ValueError("Position tensor identity changed inside one MLA decode forward")
        if _position_signature(tensor) != self.signature:
            raise ValueError("Position tensor metadata or version changed inside one MLA decode forward")
        if _stream_key(tensor) != self.stream_key:
            raise ValueError("CUDA device or stream changed inside one MLA decode forward")


@contextmanager
def mla_decode_position_scope(q_position_ids):
    """Read the position maximum once, before any layer projection is launched.

    Normal tensor version counters detect in-place changes, including changes
    through aliases. Replacing the tensor with a view, changing its geometry or
    switching CUDA streams is rejected. Inference tensors without a version
    counter must use the unscoped path. Empty batches need no scalar readback;
    their attention path must return before requesting a maximum. No state is
    reused across model forwards, and nested scopes are deliberately unsupported.
    """
    if _DECODE_POSITION_STATE.get() is not None:
        raise RuntimeError("Nested MLA decode-position scopes are unsupported")
    state = _DecodePositionState()
    token = _DECODE_POSITION_STATE.set(state)
    try:
        state.bind(q_position_ids)
        yield state
    finally:
        _DECODE_POSITION_STATE.reset(token)
        state.clear()


def decode_position_max(q_position_ids):
    """Return the real maximum, retaining unscoped reduction semantics."""
    state = _DECODE_POSITION_STATE.get()
    if state is None:
        return q_position_ids.max().item()
    state.validate(q_position_ids)
    if state.maximum is None:
        raise ValueError("Empty MLA positions have no maximum; return from attention before validation")
    return state.maximum
