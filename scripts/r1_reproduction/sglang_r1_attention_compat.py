"""Explicit SG native FlashInfer fa2 selection and observable call contracts.

Import is CPU-only. install() must precede SG attention-backend construction;
it changes no cache layout, token addresses, Q/K/V contents or cache bytes.
"""
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path


ENV = 'SGLANG_R1_COMMON_FI_FA2'
SG_BACKEND_SHA = 'f24df800a605d1d349d1d11c1a78921ace252778a2a10e980492d5fb83abad2d'
_INSTALLATION = None
_WRAPPERS = []
_ROLLBACK = None


def _identity(value):
    path = inspect.getsourcefile(value)
    if path is None:
        raise RuntimeError('Cannot identify the native FlashInfer/SG source')
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _bf16(value):
    return str(value) in ('torch.bfloat16', 'bfloat16')


def _positive_scale(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Common attention requires an explicit finite positive sm_scale')
    return float(value)


def _arguments(method, args, kwargs):
    bound = inspect.signature(method).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _checked_plan(kind, values):
    if kind == 'ragged':
        expected = {'head_dim_qk': 192, 'head_dim_vo': 128, 'num_qo_heads': 128,
                    'num_kv_heads': 128, 'causal': True, 'pos_encoding_mode': 'NONE',
                    'use_fp16_qk_reduction': False, 'window_left': -1}
        for key, value in expected.items():
            if values.get(key) != value:
                raise ValueError('SG common ragged plan mismatch: ' + key)
        if values.get('qo_indptr') is not values.get('kv_indptr'):
            raise ValueError('SG common fresh prefill requires identical Q/KV indptr objects')
        for key in ('custom_mask', 'packed_custom_mask', 'prefix_len_ptr', 'token_pos_in_items_ptr'):
            if values.get(key) is not None:
                raise ValueError('SG common fresh prefill rejects masks/prefix metadata')
        scale = values.get('sm_scale')
        if scale is not None:
            scale = _positive_scale(scale)
        dims = {'head_dim_qk': 192, 'head_dim_vo': 128, 'layout': 'NHD', 'causal': True,
                'sm_scale_at_plan': scale}
    else:
        for key, value in {'head_dim_ckv': 512, 'head_dim_kpe': 64, 'page_size': 1, 'num_heads': 128, 'causal': False}.items():
            if values.get(key) != value:
                raise ValueError('SG common MLA plan mismatch: ' + key)
        dims = {'head_dim_ckv': 512, 'head_dim_kpe': 64, 'page_size': 1,
                'causal': values.get('causal'), 'sm_scale': _positive_scale(values.get('sm_scale'))}
    q_type, kv_type = values.get('q_data_type'), values.get('kv_data_type')
    if not _bf16(q_type) or (kv_type is not None and not _bf16(kv_type)):
        raise TypeError('SG common attention requires BF16 Q/KV')
    if values.get('logits_soft_cap') not in (None, 0, 0.0):
        raise ValueError('SG common attention rejects logits soft cap')
    return dict(dims, q_dtype=str(q_type), kv_dtype=str(kv_type or q_type))


def _planned_module_identity(wrapper):
    module = getattr(wrapper, '_cached_module', None)
    if module is None:
        return None
    operations = {}
    for name in ('plan', 'run', 'ragged_run', 'paged_run'):
        value = getattr(module, name, None)
        if value is not None:
            operations[name] = {'module': getattr(value, '__module__', None),
                                'qualname': getattr(value, '__qualname__', None),
                                'registered_name': getattr(value, '_name', None),
                                'repr': repr(value)[:300]}
    return {'type': type(module).__module__ + '.' + type(module).__qualname__,
            'module_uri': getattr(module, 'uri', None), 'operations': operations}


def _loaded_kernel_objects():
    maps = Path('/proc/self/maps')
    if not maps.exists():
        return []
    paths = set()
    try:
        lines = maps.read_text().splitlines()
    except OSError as error:
        return [{'status': 'kernel_mapping_identity_unavailable', 'error': repr(error)}]
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5] and 'flashinfer' in fields[5].lower():
            paths.add(fields[5])
    result = []
    for path in sorted(paths):
        try:
            result.append({'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        except OSError as error:
            result.append({'path': path, 'status': 'kernel_object_identity_unavailable', 'error': repr(error)})
    return result


def _wrap_native_wrapper(wrapper, kind):
    if getattr(wrapper, '_backend', None) != 'fa2':
        raise RuntimeError('SG FlashInfer wrapper did not retain explicit fa2')
    record = {'kind': kind, 'constructor_backend': 'fa2', 'completed_plans': 0,
              'completed_run_api_calls': 0, 'last_plan': None, 'last_run': None}
    _WRAPPERS.append((wrapper, record))
    original_plan = wrapper.plan

    @functools.wraps(original_plan)
    def plan(*args, **kwargs):
        contract = _checked_plan(kind, _arguments(original_plan, args, kwargs))
        result = original_plan(*args, **kwargs)
        if getattr(wrapper, '_backend', None) != 'fa2':
            raise RuntimeError('SG attention plan changed the explicit fa2 backend')
        record.update(completed_plans=record['completed_plans'] + 1, last_plan=contract,
                      planned_module=_planned_module_identity(wrapper))
        return result

    wrapper.plan = plan
    if hasattr(wrapper, 'begin_forward'):
        wrapper.begin_forward = plan

    def instrument_run(name):
        original = getattr(wrapper, name)

        @functools.wraps(original)
        def run(*args, **kwargs):
            if record['last_plan'] is None or getattr(wrapper, '_backend', None) != 'fa2':
                raise RuntimeError('SG common attention run requires an observed fa2 plan')
            values = _arguments(original, args, kwargs)
            names = ('q', 'k', 'v') if kind == 'ragged' else ('q_nope', 'q_pe', 'ckv_cache', 'kpe_cache')
            tensors = [values[key] for key in names]
            dimensions = (192, 192, 128) if kind == 'ragged' else (512, 64, 512, 64)
            if any(t.shape[-1] != dim or not _bf16(t.dtype) for t, dim in zip(tensors, dimensions)):
                raise ValueError('SG common attention actual tensor dtype/head dimension mismatch')
            if kind == 'ragged':
                if any(len(t.shape) != 3 or t.shape[1] != 128 for t in tensors) or len({t.shape[0] for t in tensors}) != 1:
                    raise ValueError('SG common fresh ragged actual Q/K/V token/head shapes differ')
                scale = _positive_scale(getattr(wrapper, '_sm_scale', None))
                if getattr(wrapper, '_causal', None) is not True:
                    raise ValueError('SG common fresh ragged run requires causal=True')
            else:
                if (any(len(t.shape) != 3 or t.shape[1] != 128 for t in tensors[:2])
                        or any(len(t.shape) != 3 or t.shape[1] != 1 for t in tensors[2:])):
                    raise ValueError('SG common MLA actual query heads or token-page view differs')
                scale = record['last_plan']['sm_scale']
            result = original(*args, **kwargs)
            if getattr(wrapper, '_backend', None) != 'fa2':
                raise RuntimeError('SG attention run changed the explicit fa2 backend')
            record.update(completed_run_api_calls=record['completed_run_api_calls'] + 1,
                          last_run={'api': name, 'shapes': [list(t.shape) for t in tensors],
                                    'dtypes': [str(t.dtype) for t in tensors], 'sm_scale': scale,
                                    'backend': getattr(wrapper, '_backend', None)})
            return result

        setattr(wrapper, name, run)

    for name in ('run', 'run_return_lse'):
        if hasattr(wrapper, name):
            instrument_run(name)
    return wrapper


def install():
    global _INSTALLATION, _ROLLBACK
    if os.environ.get(ENV) != '1':
        raise RuntimeError('SG common FI attention requires explicit ' + ENV + '=1')
    if importlib.metadata.version('sglang') != '0.5.5.post3' or importlib.metadata.version('flashinfer-python') != '0.5.2':
        raise RuntimeError('SG common attention requires SG0.5.5post3 and FlashInfer0.5.2')
    if _INSTALLATION is not None:
        return receipt()
    backend = importlib.import_module('sglang.srt.layers.attention.flashinfer_mla_backend')
    source = _identity(backend.FlashInferMLAAttnBackend)
    if source['sha256'] != SG_BACKEND_SHA:
        raise RuntimeError('SG FlashInfer MLA backend source SHA mismatch')
    server = importlib.import_module('sglang.srt.server_args').get_global_server_args()
    expected = {'attention_backend': 'flashinfer', 'page_size': 64, 'disable_cuda_graph': True,
                'disable_radix_cache': True, 'disable_chunked_prefix_cache': True,
                'flashinfer_mla_disable_ragged': False, 'enable_piecewise_cuda_graph': False,
                'chunked_prefill_size': -1, 'disaggregation_mode': 'null'}
    for key, value in expected.items():
        if getattr(server, key, None) != value:
            raise RuntimeError('SG common fresh FI attention requires setting ' + key + '=' + str(value))
    for key in ('prefill_attention_backend', 'decode_attention_backend'):
        if getattr(server, key, None) not in (None, 'flashinfer'):
            raise RuntimeError('SG common attention rejects a conflicting ' + key)
    mla_class, ragged_class = backend.BatchMLAPagedAttentionWrapper, backend.BatchPrefillWithRaggedKVCacheWrapper
    mla_source, ragged_source = _identity(mla_class), _identity(ragged_class)
    cls = backend.FlashInferMLAAttnBackend
    original_init, original_metadata = cls.__init__, cls.init_forward_metadata

    def mla_factory(*args, **kwargs):
        if kwargs.get('use_cuda_graph', False):
            raise RuntimeError('SG common FI candidate requires Graph OFF')
        kwargs['backend'] = 'fa2'
        return _wrap_native_wrapper(mla_class(*args, **kwargs), 'mla')

    def ragged_factory(*args, **kwargs):
        if kwargs.get('use_cuda_graph', False) or (args[1] if len(args) > 1 else kwargs.get('kv_layout', 'NHD')) != 'NHD':
            raise RuntimeError('SG common ragged candidate requires NHD and Graph OFF')
        kwargs['backend'] = 'fa2'
        return _wrap_native_wrapper(ragged_class(*args, **kwargs), 'ragged')

    @functools.wraps(original_init)
    def initialize(self, model_runner, *args, **kwargs):
        if model_runner.page_size != 64 or not _bf16(model_runner.dtype) or not _bf16(model_runner.kv_cache_dtype):
            raise RuntimeError('SG common native MLA allocator must be BF16 page64 with a page1 token view')
        original_init(self, model_runner, *args, **kwargs)
        self.fmha_backend = 'fa2'
        if self.enable_chunk_kv:
            raise RuntimeError('SG common fresh prefill cannot enable chunk/prefix KV paths')

    @functools.wraps(original_metadata)
    def metadata(self, forward_batch):
        decode = forward_batch.forward_mode.is_decode_or_idle()
        if not decode and (not forward_batch.forward_mode.is_extend()
                           or any(forward_batch.extend_prefix_lens_cpu)):
            raise RuntimeError('SG common fresh ragged prefill rejects cached prefix/speculative modes')
        result = original_metadata(self, forward_batch)
        if not decode and not self.forward_metadata.use_ragged:
            raise RuntimeError('SG did not select the required ragged prefill path')
        return result

    originals = (backend, cls, mla_class, ragged_class, original_init, original_metadata,
                 mla_factory, ragged_factory, initialize, metadata)
    try:
        backend.BatchMLAPagedAttentionWrapper = mla_factory
        backend.BatchPrefillWithRaggedKVCacheWrapper = ragged_factory
        cls.__init__, cls.init_forward_metadata = initialize, metadata
    except BaseException:
        backend.BatchMLAPagedAttentionWrapper, backend.BatchPrefillWithRaggedKVCacheWrapper = mla_class, ragged_class
        cls.__init__, cls.init_forward_metadata = original_init, original_metadata
        raise
    _ROLLBACK = originals
    _INSTALLATION = {'status': 'installed_NOT_model_or_GPU_qualified',
                     'sg_backend_source': source, 'native_mla_source': mla_source,
                     'native_ragged_source': ragged_source,
                     'backend': 'fa2', 'sg_allocator_page_size': 64, 'decode_kernel_page_size': 1,
                     'native_page64_allocator_qualified': False,
                     'cache_mutation_by_hook': False, 'formal_test_permitted': False}
    return receipt()


def receipt():
    return {'installation': _INSTALLATION,
            'wrappers': [dict(record, current_backend=getattr(wrapper, '_backend', None)) for wrapper, record in _WRAPPERS],
            'loaded_flashinfer_kernel_objects': _loaded_kernel_objects() if _INSTALLATION is not None else [],
            'formal_test_permitted': False}


def rollback_unconstructed_installation():
    """Restore only this hook's unconsumed registrations after setup failure."""
    global _INSTALLATION, _ROLLBACK
    if _ROLLBACK is None:
        return
    if _WRAPPERS:
        raise RuntimeError('Cannot roll back SG attention after native wrappers were constructed')
    backend, cls, mla, ragged, old_init, old_metadata, factory_mla, factory_ragged, init, metadata = _ROLLBACK
    if (backend.BatchMLAPagedAttentionWrapper is not factory_mla
            or backend.BatchPrefillWithRaggedKVCacheWrapper is not factory_ragged
            or cls.__init__ is not init or cls.init_forward_metadata is not metadata):
        raise RuntimeError('Another owner changed the SG attention registrations')
    backend.BatchMLAPagedAttentionWrapper, backend.BatchPrefillWithRaggedKVCacheWrapper = mla, ragged
    cls.__init__, cls.init_forward_metadata = old_init, old_metadata
    _INSTALLATION, _ROLLBACK = None, None
