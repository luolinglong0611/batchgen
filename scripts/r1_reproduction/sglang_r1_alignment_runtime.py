"""Opt-in SG R1 layout/FP8 compatibility helpers; import has no GPU effects.

Installed only by the separately prepared DeepseekV2 model source candidate.
This is a correctness/alignment candidate, not stock SG or GPU qualification.
"""
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path


SHARED_ENV = 'SGLANG_R1_LOCAL_SHARED_TP1'
_LINEAR_INSTALLATION = None
_LINEAR_CALLS = 0
_DENSE_HELPER = None


def _file_identity(path):
    if path is None:
        raise RuntimeError('Cannot identify the common linear/SG FP8 source')
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def local_shared_enabled():
    value = os.environ.get(SHARED_ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(SHARED_ENV + ' must be explicitly 0 or 1')
    return value == '1'


def require_r1_config(config, quant_config):
    expected = {'model_type': 'deepseek_v3', 'hidden_size': 7168,
                'n_routed_experts': 256, 'n_shared_experts': 1,
                'num_experts_per_tok': 8, 'moe_intermediate_size': 2048,
                'first_k_dense_replace': 3, 'hidden_act': 'silu'}
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError('R1 alignment config mismatch: ' + name)
    if getattr(config, 'architectures', None) != ['DeepseekV3ForCausalLM']:
        raise ValueError('R1 alignment requires the original R1 architecture')
    if (quant_config is None or quant_config.get_name() != 'fp8'
            or list(quant_config.weight_block_size or []) != [128, 128]
            or not getattr(quant_config, 'is_checkpoint_fp8_serialized', False)):
        raise ValueError('R1 alignment requires serialized FP8 weights and FP32 block128 scales')


def require_shared_contract(config, quant_config, args, *, tp_size, ep_size,
                            attention_tp_size, dp_attention, a2a_none,
                            runner_triton, deepgemm_enabled):
    require_r1_config(config, quant_config)
    if not (tp_size == ep_size == 16 and attention_tp_size == 1 and dp_attention
            and a2a_none and runner_triton and not deepgemm_enabled):
        raise RuntimeError('R1 local shared requires TP16/DP-attention16/EP16, no A2A, Triton routed MoE, DeepGEMM off')
    required = {'disable_shared_experts_fusion': True, 'disable_cuda_graph': True,
                'disable_overlap_schedule': True,
                'enable_dp_lm_head': True, 'moe_dense_tp_size': 1, 'pp_size': 1,
                'enable_two_batch_overlap': False, 'enable_single_batch_overlap': False,
                'enable_piecewise_cuda_graph': False,
                'enable_flashinfer_allreduce_fusion': False}
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise RuntimeError('R1 local shared unsupported server setting: ' + name)
    if getattr(args, 'speculative_algorithm', None) is not None:
        raise RuntimeError('R1 local shared does not support speculative execution')


def capture_local_shared_input(moe, global_input, residual, forward_batch):
    """Copy the normalized owned rows with SG's actual DP padding mapping."""
    if not getattr(moe, '_r1_local_shared_tp1', False):
        return None
    if residual is None or global_input.shape[-1] != residual.shape[-1]:
        raise RuntimeError('R1 local shared requires a local residual-shaped token buffer')
    if getattr(moe.shared_experts, 'tp_size', None) != 1:
        raise RuntimeError('R1 local shared weights were not constructed at TP1')
    torch = importlib.import_module('torch')
    local_input = torch.empty_like(residual, memory_format=torch.contiguous_format)
    importlib.import_module('sglang.srt.layers.dp_attention').dp_scatter(
        local_input, global_input, forward_batch)
    return local_input


def finish_local_shared(moe, local_input, local_routed_output):
    """Add once, after routed SUM and scatter; do not modify the residual."""
    if local_input is None:
        return local_routed_output
    shared = moe.shared_experts(local_input)
    if (shared.shape != local_routed_output.shape or shared.dtype != local_routed_output.dtype
            or shared.device != local_routed_output.device):
        raise RuntimeError('R1 local shared/routed output contracts differ')
    return local_routed_output + shared


def common_block_fp8_linear(input, weight, block_size, weight_scale,
                            input_scale=None, bias=None):
    """Use the same BF16->FP8 quantizer and ordinary GEMM entry as BG."""
    global _LINEAR_CALLS
    if list(block_size) != [128, 128] or input_scale is not None or bias is not None:
        raise ValueError('R1 common linear requires dynamic block128 FP8 without bias')
    common = importlib.import_module('batchgen.gemm.r1_sm103_compat')
    output = common.w8a16_gemm(weight, weight_scale, input.reshape(-1, input.shape[-1]))
    _LINEAR_CALLS += 1
    return output.reshape(*input.shape[:-1], weight.shape[0])


def install_fp8_linear_backend(config, quant_config):
    """Install before R1 model construction; never rewrite loaded weights."""
    global _LINEAR_INSTALLATION, _DENSE_HELPER
    require_r1_config(config, quant_config)
    required_environment = {'BATCHGEN_R1_SM103_FP8_COMPAT': '1', 'DISABLE_MMA_V5': '1',
                            'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
                            'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '0'}
    for name, value in required_environment.items():
        if os.environ.get(name) != value:
            raise RuntimeError('R1 common linear requires explicit ' + name + '=' + value)
    if importlib.metadata.version('sglang') != '0.5.5.post3':
        raise RuntimeError('R1 common linear requires SGLang 0.5.5.post3')
    dg = importlib.import_module('sglang.srt.layers.deep_gemm_wrapper')
    if dg.ENABLE_JIT_DEEPGEMM or dg.DEEPGEMM_SCALE_UE8M0:
        raise RuntimeError('DeepGEMM/UE8M0 was enabled before the common linear hook')
    common = importlib.import_module('batchgen.gemm.r1_sm103_compat')
    if not callable(getattr(common, 'w8a16_gemm', None)):
        raise RuntimeError('Common BG w8a16_gemm entry is unavailable')
    cls = importlib.import_module('sglang.srt.layers.quantization.fp8').Fp8LinearMethod
    dense = None
    if os.environ.get('SGLANG_R1_DENSE_BF16', '0') != '0':
        dense = importlib.import_module('scripts.r1_reproduction.sglang_r1_dense_bf16')
        dense.enabled()  # Reject misspelled/non-boolean opt-in; disabled imports remain untouched.
    if getattr(cls.__init__, '_r1_common_linear_hook', False):
        if dense is not None and dense.dense_backend_receipt()['installation'] is None:
            raise RuntimeError('Dense BF16 opt-in must be frozen before the first common backend installation')
        return linear_backend_receipt()
    common_source = _file_identity(getattr(common, '__file__', None))
    sg_fp8_source = _file_identity(inspect.getsourcefile(cls))
    original = cls.__init__

    @functools.wraps(original)
    def aligned_init(self, quant_config):
        original(self, quant_config)
        if self.block_quant:
            require_r1_config(config, quant_config)
            if self.use_marlin:
                raise RuntimeError('Marlin would bypass the R1 common FP8 linear backend')
            self.w8a8_block_fp8_linear = common_block_fp8_linear

    aligned_init._r1_common_linear_hook = True
    installation = {
        'status': 'hook_installed_NOT_model_or_GPU_qualified',
        'sglang_version': '0.5.5.post3', 'common_module_path': getattr(common, '__file__', None),
        'common_source': common_source, 'sg_fp8_source': sg_fp8_source,
        'common_entry': 'batchgen.gemm.r1_sm103_compat.w8a16_gemm',
        'scope': 'Fp8LinearMethod block128 attention/dense/shared only; no grouped MoE, absorbed attention, or BF16 weight quantization',
        'loader_ue8m0_requantization': 'disabled_by_required_DeepGEMM_off',
        'formal_test_permitted': False,
    }
    attention, attention_preexisting = None, False
    dense_preexisting = dense is not None and dense.dense_backend_receipt()['installation'] is not None
    try:
        if os.environ.get('SGLANG_R1_COMMON_FI_FA2') == '1':
            attention = importlib.import_module('scripts.r1_reproduction.sglang_r1_attention_compat')
            attention_preexisting = attention.receipt()['installation'] is not None
            attention.install()
        if dense is not None:
            dense.install()
        cls.__init__ = aligned_init
    except BaseException:
        cls.__init__ = original
        if dense is not None and not dense_preexisting:
            dense.rollback_unexecuted_installation()
        if attention is not None and not attention_preexisting:
            attention.rollback_unconstructed_installation()
        raise
    _LINEAR_INSTALLATION = installation
    _DENSE_HELPER = dense
    return linear_backend_receipt()


def linear_backend_receipt():
    return {'installation': _LINEAR_INSTALLATION, 'completed_common_linear_calls': _LINEAR_CALLS,
            'dense_BF16_candidate': _DENSE_HELPER.dense_backend_receipt() if _DENSE_HELPER is not None else None,
            'formal_test_permitted': False}
