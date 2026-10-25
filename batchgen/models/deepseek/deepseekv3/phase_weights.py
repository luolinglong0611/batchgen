"""Exact-value R1 phase preparation; weights and scale arithmetic stay unchanged."""
import logging
import os
import re
import time


def enabled():
    value = os.getenv('BATCHGEN_R1_PHASE_PREPARE', '0')
    if value not in ('0', '1'):
        raise ValueError('BATCHGEN_R1_PHASE_PREPARE must be 0 or 1')
    return value == '1'


def prepare_worker_skeleton(source, model_name):
    """Dequantize the nine dense weights once per node before worker mmap.

    Copy the dictionary, never the checkpoint's mapped storages. Removing only
    these scales makes the ordinary skeleton loader bind the exact same BF16
    values directly. Other models and the disabled path retain their payload.
    """
    if not enabled():
        return source
    if model_name.lower() != 'deepseek-ai/deepseek-r1':
        raise ValueError('Phase preparation requires original DeepSeek-R1')
    import torch
    from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
    start = time.perf_counter()
    names = [(f'model.layers.{layer}.mlp.{proj}.weight', shape)
             for layer in range(3)
             for proj, shape in [('gate_proj', (18432, 7168)),
                                 ('up_proj', (18432, 7168)),
                                 ('down_proj', (7168, 18432))]]
    # Validate the entire input before doing substantial work.
    for key, shape in names:
        weight, scale = source[key], source[key + '_scale_inv']
        if (weight.device.type != 'cpu' or weight.dtype != torch.float8_e4m3fn
                or tuple(weight.shape) != shape or not weight.is_contiguous()
                or scale.device.type != 'cpu' or scale.dtype != torch.float32
                or tuple(scale.shape) != (shape[0] // 128, shape[1] // 128)):
            raise ValueError(f'Unexpected original R1 dense weight/scale: {key}')
    result = dict(source)
    for key, _ in names:
        result[key] = deepseek_v3_dequantization(source[key], source[key + '_scale_inv'])
        del result[key + '_scale_inv']
    logging.info('[R1_PHASE_PREPARE] node dense dequantization once: %.3fs, 9 weights, %d BF16 bytes',
                 time.perf_counter() - start, sum(result[k].numel() * 2 for k, _ in names))
    return result


def pack_phase_scales(source, phase, rank, world_size, device):
    """One bulk scale transfer per phase instead of ~45,000 tiny copies.

    Views own their flat GPU storage. Rebuild at each phase boundary, so no
    stale scales or prefill-only GPU allocation persists into decode. This
    changes neither scale values nor decode's resident scale set.
    """
    if not enabled():
        return None
    import torch
    if phase not in ('prefill', 'decode') or world_size != 16 or not 0 <= rank < 16:
        raise ValueError('Phase scale packing requires the qualified R1 EP16 layout')
    start = time.perf_counter()
    selected = []
    for key, value in source.items():
        match = re.fullmatch(r'model\.layers\.([0-9]+)\.(self_attn|mlp\.shared_experts|mlp\.experts\.([0-9]+))\.(.+)\.weight_scale_inv', key)
        if not match or int(match[1]) >= 61:
            continue
        if phase == 'decode' and match[3] is not None and not rank * 16 <= int(match[3]) < (rank + 1) * 16:
            continue
        if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError(f'Unexpected R1 scale: {key}')
        selected.append((key, value))
    if not selected:
        raise ValueError('No R1 scales selected')
    packed = torch.cat([value.reshape(-1) for _, value in selected]).to(device)
    result, offset = {}, 0
    for key, value in selected:
        result[key] = packed[offset:offset + value.numel()].view(value.shape)
        offset += value.numel()
    logging.info('[R1_PHASE_PREPARE] %s scales: %d tensors, %d bytes, one transfer, %.3fs',
                 phase, len(result), packed.numel() * packed.element_size(), time.perf_counter() - start)
    return result
