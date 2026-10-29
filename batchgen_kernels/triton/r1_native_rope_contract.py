"""CPU metadata contract for the opt-in R1 FP32 native decode RoPE kernel.

No accelerator import, index-value readback, allocation or synchronization.
The caller owns valid positions in [0, cos.shape[0]); bounds follow the same
per-step CPU planner contract as the previous R1 compatibility kernel.
"""
import math


def _byte_interval(tensor):
    start = tensor.data_ptr()
    elements = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return start, start + elements * tensor.element_size()


def validate_native_fp32_inputs(kv, q, cos, sin, positions, weight, latent, rope, eps, *, torch):
    tensors = (kv, q, cos, sin, positions, weight)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError('R1 fused native RoPE requires Tensor inputs')
    if (type(latent) is not int or type(rope) is not int or (latent, rope) != (512, 64)
            or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps < 0):
        raise ValueError('R1 fused native RoPE requires latent512/rope64 and finite nonnegative eps')
    if (kv.ndim != 3 or kv.shape[0] <= 0 or tuple(kv.shape[1:]) != (1, 576)
            or kv.dtype != torch.bfloat16 or kv.device.type != 'cuda'):
        raise ValueError('R1 fused native RoPE requires BF16 CUDA KV [B,1,576], B>0')
    rows, device = kv.shape[0], kv.device
    if (kv.stride(-1) != 1 or kv.stride(0) < 576
            or tuple(q.shape) != (rows, 128, 1, 64) or q.dtype != torch.bfloat16
            or q.device != device or not q.is_contiguous()):
        raise ValueError('R1 fused native RoPE requires nonoverlapping KV rows and contiguous BF16 Q [B,128,1,64]')
    for name, value in (('cos', cos), ('sin', sin)):
        if (value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 64
                or value.device != device or value.dtype not in (torch.bfloat16, torch.float32)
                or not value.is_contiguous()):
            raise ValueError(f'R1 fused native {name} requires contiguous CUDA BF16/FP32 [positions,64]')
    if cos.shape != sin.shape:
        raise ValueError('R1 fused native cos/sin shapes differ')
    if (tuple(positions.shape) != (rows, 1) or positions.device != device
            or positions.dtype not in (torch.int32, torch.int64) or not positions.is_contiguous()):
        raise ValueError('R1 fused native positions require contiguous CUDA int32/int64 [B,1]')
    if (tuple(weight.shape) != (512,) or weight.device != device
            or weight.dtype not in (torch.bfloat16, torch.float32) or not weight.is_contiguous()):
        raise ValueError('R1 fused native RMS weight requires contiguous CUDA BF16/FP32 [512]')
    # Q is modified inplace. It must not overwrite any read-only input while
    # another CTA is still reading it. Pair/head tiles themselves are disjoint.
    q_start, q_end = _byte_interval(q)
    for name, source in (('KV', kv), ('cos', cos), ('sin', sin), ('positions', positions), ('RMS weight', weight)):
        begin, end = _byte_interval(source)
        if q_start < end and begin < q_end:
            raise ValueError(f'R1 fused native inplace Q aliases read-only {name}')
    return rows, 128
