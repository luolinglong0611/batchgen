"""Explicit R1 SG dense-only alignment with native BG main BF16 math.

Default import/disabled paths do not import Torch, inspect a device, or mutate SG.
The original checkpoint stays FP8. Only layers 0..2 are derived into BF16 after
loading; attention/shared/routed experts are excluded. No qualification implied.
"""
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import weakref

ENV = 'SGLANG_R1_DENSE_BF16'
PROVENANCE_ENV = 'SGLANG_R1_DENSE_PROVENANCE'
PROVENANCE_SHA_ENV = 'SGLANG_R1_DENSE_PROVENANCE_SHA256'
SG_MODEL_SHA = 'b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd'
SG_FP8_SHA = '31faac58174b3ee8298504c140e00647f502db2205b7c0e6156a8c0a407595d2'
BG_DEQUANT_SHA = '2872df05675ad7032e1ee1dbf028e8e54e218193e800822c15fadcb047dabfbf'
MODEL_MANIFEST_SHA = 'd9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253'
WEIGHT_IDENTITY = 'd156c2047e21d830fe9cdd5f20ad346324455c4a4bbf2c94ddbf82b9f5860982'
FORMULA = 'BF16(FP32(W_fp8[i,j])*S_fp32[floor(i/128),floor(j/128)])'
ORDER = ['gate_linear_BF16', 'silu_materialized_BF16', 'up_linear_BF16',
         'multiply_materialized_BF16', 'down_linear_BF16']
_INSTALLATION = None
_ORIGINALS = None
_MODULES = weakref.WeakKeyDictionary()
SHA = re.compile(r'^[0-9a-f]{64}$')


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(ENV + ' must be explicitly 0 or 1')
    return value == '1'


def expected_tensors():
    result = {}
    for layer in range(3):
        for part, shape in (('gate_proj', [18432, 7168]), ('up_proj', [18432, 7168]),
                            ('down_proj', [7168, 18432])):
            name = f'model.layers.{layer}.mlp.{part}.weight'
            result[name] = ('float8_e4m3fn', shape)
            result[name + '_scale_inv'] = ('float32', [x // 128 for x in shape])
    return result


def validate_provenance(value):
    if (value.get('schema') != 'r1-dense-fp8-provenance-v1'
            or value.get('model_manifest_sha256') != MODEL_MANIFEST_SHA
            or value.get('model_weight_identity_sha256') != WEIGHT_IDENTITY):
        raise ValueError('Original sealed R1 model identity is missing or differs')
    records = value.get('tensor_records')
    if not isinstance(records, dict) or set(records) != set(expected_tensors()):
        raise ValueError('Exactly all 18 dense source weight/scale tensor records required')
    for name, (dtype, shape) in expected_tensors().items():
        record = records[name]
        if (record.get('dtype') != dtype or record.get('shape') != shape
                or not SHA.fullmatch(str(record.get('raw_sha256', '')))):
            raise ValueError('Dense source dtype/shape/raw SHA mismatch: ' + name)
    return value


def load_provenance():
    path = os.environ.get(PROVENANCE_ENV)
    expected = os.environ.get(PROVENANCE_SHA_ENV, '')
    if not path or not SHA.fullmatch(expected):
        raise RuntimeError('Explicit frozen dense provenance path and SHA256 required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Dense provenance file changed')
    return validate_provenance(json.loads(raw)), {'path': str(Path(path).resolve()), 'sha256': expected}


def _source(value, expected):
    path = inspect.getsourcefile(value)
    if path is None:
        raise RuntimeError('Dense alignment source identity unavailable')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Frozen dense alignment source mismatch: ' + path)
    return {'path': str(Path(path).resolve()), 'sha256': expected}


def dense_layer_id(arguments):
    """Pure CPU role guard. Shared/other widths are excluded; ambiguous dense fails."""
    if arguments.get('intermediate_size') != 18432:
        return None
    prefix = arguments.get('prefix', '')
    match = re.fullmatch(r'model\.layers\.([012])\.mlp', prefix)
    if (not match or arguments.get('hidden_size') != 7168
            or arguments.get('hidden_act') != 'silu'
            or type(arguments.get('tp_size')) is not int or arguments['tp_size'] != 1
            or type(arguments.get('tp_rank')) is not int or arguments['tp_rank'] != 0):
        raise ValueError('Dense BF16 requires explicit original layer0..2 prefix/geometry/TP1')
    return int(match.group(1))


def _metadata(tensor):
    return {'dtype': str(tensor.dtype).removeprefix('torch.'),
            'shape': list(tensor.shape), 'strides': list(tensor.stride()),
            'device': str(tensor.device)}


def tensor_raw_sha256(tensor, torch, chunk_bytes=16 * 1024 * 1024):
    """Read original values only, chunked D2H; no casts before byte hashing."""
    if sys.byteorder != 'little' or not tensor.is_contiguous():
        raise ValueError('Dense raw hashing requires little-endian contiguous original tensors')
    row_bytes = tensor.shape[-1] * tensor.element_size()
    rows = max(1, chunk_bytes // row_bytes)
    digest = hashlib.sha256()
    for first in range(0, tensor.shape[0], rows):
        chunk = tensor.detach()[first:first + rows].contiguous().cpu()
        if not bool(torch.isfinite(chunk.float()).all().item()):
            raise ValueError('Nonfinite original dense weight/scale; never sanitize')
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify(tensor, name, provenance, torch):
    record = provenance['tensor_records'][name]
    if (str(tensor.dtype).removeprefix('torch.') != record['dtype']
            or list(tensor.shape) != record['shape'] or not tensor.is_contiguous()
            or tensor.device.type != 'cuda'):
        raise ValueError('Actual original dense tensor contract differs: ' + name)
    if tensor_raw_sha256(tensor, torch) != record['raw_sha256']:
        raise ValueError('Actual loaded dense tensor values differ from original: ' + name)
    return {'name': name, **_metadata(tensor), 'raw_sha256': record['raw_sha256'],
            'raw_values_verified_against_original': True}


class _RejectFp8AfterDenseDerivation:
    def apply(self, *args, **kwargs):
        raise RuntimeError('Derived BF16 dense requires main-ordered direct F.linear; FP8 bypass rejected')

    def process_weights_after_loading(self, *args, **kwargs):
        raise RuntimeError('Dense derivation must follow complete native FP8 weight loading')


def align_dense_mlp(module, layer_id, provenance, provenance_identity):
    """Same function used by the installed SG forward and isolated GPU qualifier."""
    existing = _MODULES.get(module)
    if existing is not None:
        return existing
    if type(layer_id) is not int or layer_id not in range(3) or getattr(module, 'tp_size', None) != 1:
        raise ValueError('Only explicit dense layers0..2 at TP1 can be converted')
    validate_provenance(provenance)
    torch = importlib.import_module('torch')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Dense derivation cannot occur during CUDA graph capture')
    gu, down = module.gate_up_proj, module.down_proj
    if (getattr(gu, 'tp_size', None) != 1 or getattr(down, 'tp_size', None) != 1
            or getattr(gu, 'bias', None) is not None or getattr(down, 'bias', None) is not None):
        raise ValueError('Dense derived linear modules require actual TP1 and no bias')
    if tuple(gu.weight.shape) != (36864, 7168) or tuple(down.weight.shape) != (7168, 18432):
        raise ValueError('Dense merged gate/up or down weight shape differs')
    device = gu.weight.device
    if (device.type != 'cuda' or tuple(torch.cuda.get_device_capability(device)) != (10, 3)
            or any(t.device != device for t in (down.weight, gu.weight_scale_inv, down.weight_scale_inv))):
        raise RuntimeError('Dense BF16 alignment requires all source tensors on one actual SM103 device')
    if (tuple(gu.weight_scale_inv.shape) != (288, 56)
            or tuple(down.weight_scale_inv.shape) != (56, 144)):
        raise ValueError('Original FP32 block128 scale shapes differ')
    prefix = f'model.layers.{layer_id}.mlp.'
    tensors = {'gate_proj.weight': gu.weight[:18432], 'up_proj.weight': gu.weight[18432:],
               'down_proj.weight': down.weight,
               'gate_proj.weight_scale_inv': gu.weight_scale_inv[:144],
               'up_proj.weight_scale_inv': gu.weight_scale_inv[144:],
               'down_proj.weight_scale_inv': down.weight_scale_inv}
    proofs = [_verify(t, prefix + name, provenance, torch) for name, t in tensors.items()]
    dequant_module = importlib.import_module('batchgen_kernels.triton.fp8_quantize')
    source = _source(dequant_module.deepseek_v3_dequantization, BG_DEQUANT_SHA)
    # Original BG main formula; two independent halves preserve original block boundaries.
    with torch.inference_mode():
        gate = dequant_module.deepseek_v3_dequantization(tensors['gate_proj.weight'], tensors['gate_proj.weight_scale_inv'])
        up = dequant_module.deepseek_v3_dequantization(tensors['up_proj.weight'], tensors['up_proj.weight_scale_inv'])
        down_bf16 = dequant_module.deepseek_v3_dequantization(tensors['down_proj.weight'], tensors['down_proj.weight_scale_inv'])
        merged = torch.cat((gate, up), dim=0)
        if any(t.dtype != torch.bfloat16 or t.device != device for t in (gate, up, down_bf16, merged)):
            raise RuntimeError('BG dequantization did not return actual BF16 on the original device')
        new_gu = torch.nn.Parameter(merged, requires_grad=False)
        new_down = torch.nn.Parameter(down_bf16, requires_grad=False)
    old = (gu.weight, down.weight, gu.quant_method, down.quant_method)
    try:
        gu.weight, down.weight = new_gu, new_down
        gu.quant_method = _RejectFp8AfterDenseDerivation()
        down.quant_method = _RejectFp8AfterDenseDerivation()
    except BaseException:
        gu.weight, down.weight, gu.quant_method, down.quant_method = old
        raise
    record = {'status': 'derived_actual_weights_NOT_numerical_or_model_qualified',
              'layer_index': layer_id, 'original_tensor_proofs': proofs,
              'provenance': provenance_identity, 'dequant_source': source, 'dequant_formula': FORMULA,
              'math_order': ORDER, 'actual_weight_metadata': {'gate_up': _metadata(new_gu), 'down': _metadata(new_down)},
              'completed_forward_calls': 0, 'last_input': None,
              'counter_scope': 'host_forward_API_returns_NOT_CUDA_synchronization',
              'helper_source': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'persistent_weight_bytes_increase': 396361728,
              'lazy_first_forward_conversion': True, 'must_complete_before_formal_warmup_and_BCT': True,
              'formal_test_permitted': False}
    _MODULES[module] = record
    module._r1_dense_bf16_receipt = record
    return record


def ordered_dense_forward(module, x):
    """BG main's *dense* evaluation and BF16 materialization order, without FP8 A."""
    torch = importlib.import_module('torch')
    functional = importlib.import_module('torch.nn.functional')
    if (x.dtype != torch.bfloat16 or x.shape[-1] != 7168
            or x.device != module.gate_up_proj.weight.device
            or module.gate_up_proj.weight.dtype != torch.bfloat16
            or module.down_proj.weight.dtype != torch.bfloat16):
        raise ValueError('Derived dense forward requires actual BF16 input on its weights device')
    gate = functional.linear(x, module.gate_up_proj.weight[:18432])
    activated = functional.silu(gate)
    up = functional.linear(x, module.gate_up_proj.weight[18432:])
    intermediate = activated * up
    output = functional.linear(intermediate, module.down_proj.weight)
    if any(t.dtype != torch.bfloat16 for t in (gate, activated, up, intermediate, output)):
        raise RuntimeError('Dense main-order BF16 intermediate materialization differs')
    return output


def dense_backend_receipt():
    """Read-only: installation alone and uncalled modules stay unverified."""
    return json.loads(json.dumps({'installation': _INSTALLATION, 'modules': list(_MODULES.values()),
            'actual_converted_layer_ids': sorted({r['layer_index'] for r in _MODULES.values()}),
            'completed_forward_calls': sum(r['completed_forward_calls'] for r in _MODULES.values()),
            'status': 'unverified' if not _MODULES else 'partial_actual_NOT_qualified',
            'formal_test_permitted': False}))


def install():
    global _INSTALLATION, _ORIGINALS
    if not enabled():
        return dense_backend_receipt()
    if _INSTALLATION is not None:
        return dense_backend_receipt()
    provenance, provenance_identity = load_provenance()
    sg = importlib.import_module('sglang.srt.models.deepseek_v2')
    fp8 = importlib.import_module('sglang.srt.layers.quantization.fp8')
    model_source = _source(sg.DeepseekV2MLP, SG_MODEL_SHA)
    fp8_source = _source(fp8.Fp8LinearMethod, SG_FP8_SHA)
    args = sg.get_global_server_args()
    if getattr(args, 'disable_cuda_graph', None) is not True or getattr(args, 'enable_piecewise_cuda_graph', None) is not False:
        raise RuntimeError('Explicit dense BF16 candidate requires all declared SG graphs OFF')
    cls = sg.DeepseekV2MLP
    original_init, original_forward = cls.__init__, cls.forward
    signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        arguments = dict(signature.bind(self, *args, **kwargs).arguments)
        layer_id = dense_layer_id(arguments)
        original_init(self, *args, **kwargs)
        if layer_id is not None:
            self._r1_dense_bf16_layer_id = layer_id

    @functools.wraps(original_forward)
    def forward(self, x, forward_batch=None, should_allreduce_fusion=False,
                use_reduce_scatter=False, gemm_output_zero_allocator=None):
        layer_id = getattr(self, '_r1_dense_bf16_layer_id', None)
        if layer_id is None:
            return original_forward(self, x, forward_batch, should_allreduce_fusion,
                                    use_reduce_scatter, gemm_output_zero_allocator)
        if should_allreduce_fusion or use_reduce_scatter or gemm_output_zero_allocator is not None:
            raise RuntimeError('Dense BF16 main-order candidate rejects unqualified fusion/scatter/allocator')
        record = align_dense_mlp(self, layer_id, provenance, provenance_identity)
        output = ordered_dense_forward(self, x)
        record['completed_forward_calls'] += 1
        record['last_input'] = _metadata(x)
        return output
    try:
        cls.__init__, cls.forward = init, forward
    except BaseException:
        cls.__init__, cls.forward = original_init, original_forward
        raise
    _ORIGINALS = (cls, original_init, original_forward)
    _INSTALLATION = {'status': 'installed_NOT_executed_NOT_qualified', 'SG_model_source': model_source,
                     'helper_source': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                     'SG_fp8_source': fp8_source, 'provenance': provenance_identity,
                     'scope': 'only model.layers.0..2.mlp; original main BG dense policy retained',
                     'persistent_bytes_increase_per_rank': 1189085184,
                     'temporary_conversion_workspace_NOT_included': True,
                     'formal_test_permitted': False}
    return dense_backend_receipt()


def rollback_unexecuted_installation():
    global _INSTALLATION, _ORIGINALS
    if _MODULES:
        raise RuntimeError('Cannot roll back dense installation after model weights were converted')
    if _ORIGINALS is not None:
        cls, original_init, original_forward = _ORIGINALS
        cls.__init__, cls.forward = original_init, original_forward
    _INSTALLATION = _ORIGINALS = None
