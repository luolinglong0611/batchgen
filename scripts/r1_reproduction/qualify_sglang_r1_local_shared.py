#!/usr/bin/env python3
"""Real SG DP/shared branch qualification, with no model server.

Default only prints a plan. Distributed execution is owned by the parent
guard; it imports the installed generated SG adapter and uses its real decoder
and MoE forward methods, native SG collectives, and one original shared MLP.
Routed contributions are synthetic, not routed-expert kernel qualification.
"""
import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import traceback
from types import SimpleNamespace, MethodType


SOURCE_SHA = {
    'sglang.srt.distributed.parallel_state': '274ba74d9a9ee5f4f182de0be11c30fbcc4862c146fd9f917c526b4619478fed',
    'sglang.srt.layers.dp_attention': 'ea7fa7d5035aa8f9ada8049ff055a67dd87115b376bba6c0b13046b9eefd14d0',
    'sglang.srt.layers.communicator': 'e3f911a97cb960c875a06014ab149d75bff8736ddebdf4a6ad0a3654003c6407',
}
EXTRA_SOURCE_SHA = {
    'sglang.srt.server_args': '34dfebfe793e7e7409ee69a71c77d3550039e20ed71bff0e954e23c0042bdfdf',
    'sglang.srt.layers.linear': 'e985537aa15162f49747c99753ea84aad74cebff614013933fad61246a4809d3',
    'sglang.srt.layers.quantization.fp8': '31faac58174b3ee8298504c140e00647f502db2205b7c0e6156a8c0a407595d2',
    'sglang.srt.layers.moe.utils': '8fbe57a0b6021720145a3abbbf8d69c11031b9d5c3a32845d006c46b8df0b47a',
}
HIDDEN, INTERMEDIATE = 7168, 2048
SHARED_MODEL_SHA = 'a00712388404cf3a22a70f770f8bbaf1adc2ed7127182dda7fd47fb94c4897cf'
TRACED_MODEL_SHA = 'b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd'
TRACE_BOOTSTRAP = '\n# Explicit diagnostic bootstrap; disabled unless R1_FULL_MODEL_TRACE=1.\nfrom scripts.r1_reproduction.sglang_r1_full_model_trace import install as _r1_install_full_model_trace\n_r1_install_full_model_trace()\n'


def generated_source_contract(data, expected):
    actual = hashlib.sha256(data).hexdigest()
    if expected not in (SHARED_MODEL_SHA, TRACED_MODEL_SHA) or actual != expected:
        raise ValueError('Only the explicitly admitted alignment or alignment-plus-trace model is supported')
    if actual == TRACED_MODEL_SHA:
        suffix = TRACE_BOOTSTRAP.encode()
        if not data.endswith(suffix) or hashlib.sha256(data[:-len(suffix)]).hexdigest() != SHARED_MODEL_SHA:
            raise ValueError('Combined model must retain the exact alignment body and disabled diagnostic bootstrap')
    return adapter_source_contract(data.decode())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def original_tensor_specs(layer):
    base = f'model.layers.{layer}.mlp.shared_experts.'
    specs = {}
    for part in ('gate_proj', 'up_proj', 'down_proj'):
        shape = [HIDDEN, INTERMEDIATE] if part == 'down_proj' else [INTERMEDIATE, HIDDEN]
        specs[base + part + '.weight'] = {'shape': shape, 'dtype': 'torch.float8_e4m3fn'}
        specs[base + part + '.weight_scale_inv'] = {
            'shape': [shape[0] // 128, shape[1] // 128], 'dtype': 'torch.float32'}
    return specs


def case_counts(world):
    if world not in (2, 4, 16):
        raise ValueError('Only explicit 2/4-rank prechecks or the real16-rank layout are supported')
    return [('equal', [2] * world), ('unequal_empty', [r % 4 for r in range(world)]),
            ('last_owner_only', [0] * (world - 1) + [3])]


def effective_counts(counts, mode):
    if mode not in ('SUM_LEN', 'MAX_LEN') or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Invalid explicit DP padding fixture')
    return list(counts) if mode == 'SUM_LEN' else [max(counts)] * len(counts)


def fixture_sign_code(owner, token):
    """Unique six-bit row identity, with one exactly summable magnitude."""
    if type(owner) is not int or type(token) is not int or not (0 <= owner < 16 and 0 <= token < 4):
        raise ValueError('Fixture code is restricted to the frozen 16x4 row space')
    code = owner * 4 + token
    return tuple(-1 if (code >> bit) & 1 else 1 for bit in range(6))


def validate_numeric_contract(value):
    if value.get('reference') != 'original_fp8_weights_dequantized_fp64_swiglu':
        raise ValueError('Explicit FP64 shared reference contract required')
    for key in ('atol', 'rtol', 'max_failure_fraction'):
        x = value.get(key)
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0:
            raise ValueError('Invalid predeclared numerical bound: ' + key)
    if value['max_failure_fraction'] >= 1:
        raise ValueError('Numerical contract cannot permit all outputs to fail')
    return value


def aggregate_rank_records(records, world):
    expected_cases = {(name, mode) for name, _ in case_counts(world) for mode in ('SUM_LEN', 'MAX_LEN')}
    if len(records) != world or {r.get('rank') for r in records} != set(range(world)):
        raise ValueError('Every real rank must supply exactly one closed result')
    source_sets, weight_sets, numeric_sets = set(), set(), set()
    for record in records:
        if (record.get('status') != 'passed_real_shared_branch_NOT_full_model_qualified'
                or record.get('world_size') != world or record.get('source_sha256') != sha(__file__)
                or record.get('tp_members') != list(range(world))
                or record.get('attention_tp_size') != 1 or record.get('expert_parallel_size') != world
                or record.get('attention_dp_size') != world or record.get('actual_shared_tp_size') != 1):
            raise ValueError('Rank execution/source/topology did not pass the current contract')
        if {(c['case'], c['padding']) for c in record['cases']} != expected_cases or len(record['cases']) != len(expected_cases):
            raise ValueError('Rank lacks a unique complete SUM_LEN/MAX_LEN case set')
        numeric = record['numerical_contract']
        if numeric is not None:
            validate_numeric_contract(numeric)
        for case in record['cases']:
            if (case['routed_SUM_calls'] != 1 or case['local_shared_calls'] != 1
                    or not case['shared_added_once_bitwise'] or not case['separate_shared_oracle_bitwise']
                    or not case['native_gather_matches_independent_rows']
                    or (numeric is not None and case['reference_failures'] > numeric['max_failure_fraction'] * case['checked_FP64_reference_elements'])):
                raise ValueError('Rank branch-composition/numerical gate failed')
        source_sets.add(json.dumps({name: value['sha256'] for name, value in record['sources'].items()}, sort_keys=True))
        weight_sets.add(json.dumps(record['weights'], sort_keys=True))
        numeric_sets.add((record['numerical_contract_sha256'], json.dumps(numeric, sort_keys=True)))
    if len(source_sets) != 1 or len(weight_sets) != 1 or len(numeric_sets) != 1:
        raise ValueError('Ranks used different sources, original tensors or numerical criteria')
    return {'status': 'passed_complete_shared_branch_rank_set_NOT_full_model_qualified',
            'world_size': world, 'observed_ranks': list(range(world)),
            'partial_world_precheck_passed': world in (2, 4), 'final16_layout_qualified': world == 16,
            'independent_shared_MLP_numerical_qualified': numeric is not None,
            'routed_expert_kernel_qualified': False, 'full_model_qualified': False,
            'formal_test_permitted': False}


def adapter_source_contract(text):
    # Bind the actual generated methods, not merely an environment flag.
    import ast
    tree = ast.parse(text)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    cls = classes.get('DeepseekV2DecoderLayer')
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None) if cls else None
    if method is None:
        raise ValueError('Missing generated SG DecoderLayer.forward')
    calls = {name: [n.lineno for n in ast.walk(method) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == name]
             for name in ('_r1_capture_local_shared_input', '_r1_finish_local_shared')}
    scatter = [n.lineno for n in ast.walk(method) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'postprocess_layer']
    if any(len(v) != 1 for v in calls.values()) or len(scatter) != 1 or not (
            calls['_r1_capture_local_shared_input'][0] < scatter[0] < calls['_r1_finish_local_shared'][0]):
        raise ValueError('Generated SG shared/scatter ordering contract mismatch')
    return {'decoder_forward_AST_sha256': hashlib.sha256(ast.dump(method).encode()).hexdigest()}


def plan(args):
    return {'schema': 'r1-native-local-shared-qualification-v1', 'status': 'plan_only_NOT_executed',
            'world_size': args.world_size, 'final_layout_world_size': 16,
            'stage': 'final16_layout' if args.world_size == 16 else 'partial_world_precheck_only',
            'scope': 'original_shared_MLP_and_real_SG_DP_adapter_with_synthetic_routed_contributions',
            'routed_expert_kernel_qualified': False, 'full_model_qualified': False,
            'formal_test_permitted': False, 'cases': case_counts(args.world_size),
            'padding_modes': ['SUM_LEN', 'MAX_LEN'], 'shared_layer': args.layer,
            'original_tensor_specs': original_tensor_specs(args.layer),
            'oracle': 'independent_dyadic_routed_rows_plus_separate_same_native_shared_call_bitwise',
            'independent_shared_MLP_numerical_qualified': False,
            'numerical_contract_required_before_execution': False,
            'optional_FP64_gate_requires_explicit_pinned_contract': True,
            'limits': {'external_wall_seconds': 900, 'per_rank_torch_GPU_bytes': 3 * 1024**3,
                       'per_rank_external_GPU_bytes': 4 * 1024**3, 'per_rank_RSS_bytes': 8 * 1024**3},
            'source_sha256': sha(__file__), 'gpu_operations': False,
            'requires_parent_guard': 'R1_SG_SHARED_GUARDED=1', 'output_dir': str(args.output_dir)}


def tensor_sha(torch, value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def source_record(module, expected):
    path = Path(inspect.getsourcefile(module))
    actual = sha(path)
    if actual != expected:
        raise ValueError('Installed SG source identity mismatch: ' + str(path))
    return {'path': str(path), 'sha256': actual}


def load_weights(torch, args, device):
    if sha(args.weights_bundle) != args.weights_sha256 or sha(args.tensor_manifest) != args.tensor_manifest_sha256:
        raise ValueError('Original shared weight bundle/manifest SHA mismatch')
    manifest = json.loads(args.tensor_manifest.read_text())
    if manifest.get('bundle_sha256') != args.weights_sha256 or manifest.get('layer_index') != args.layer:
        raise ValueError('Shared bundle provenance/layer identity mismatch')
    safetensors = importlib.import_module('safetensors.torch')
    tensors = safetensors.load_file(str(args.weights_bundle), device='cpu')
    specs = original_tensor_specs(args.layer)
    if set(tensors) != set(specs) or set(manifest.get('tensors', {})) != set(specs):
        raise ValueError('Original shared bundle must contain exactly six original checkpoint tensors')
    records = {}
    for name, spec in specs.items():
        value = tensors[name]
        actual = tensor_sha(torch, value)
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype'] or actual != manifest['tensors'][name]['sha256']:
            raise ValueError('Original shared tensor identity/shape/dtype mismatch: ' + name)
        if not torch.isfinite(value.float()).all() or (name.endswith('scale_inv') and not (value > 0).all()):
            raise ValueError('Original shared tensors/scales must be finite and scales positive')
        records[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': actual}
    return {name: value.to(device) for name, value in tensors.items()}, records, manifest


def server_args_defaults(module, world):
    cls = module.ServerArgs
    value = cls.__new__(cls)  # Do not call model-path/HF configuration discovery.
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            setattr(value, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:
            setattr(value, field.name, field.default_factory())
    for key, selected in {'tp_size': world, 'dp_size': world, 'ep_size': world, 'pp_size': 1,
                          'enable_dp_attention': True, 'moe_dense_tp_size': 1, 'enable_dp_lm_head': True,
                          'moe_a2a_backend': 'none', 'moe_runner_backend': 'triton',
                          'disable_cuda_graph': True, 'disable_overlap_schedule': True,
                          'disable_shared_experts_fusion': True, 'enable_single_batch_overlap': False,
                          'enable_two_batch_overlap': False, 'enable_piecewise_cuda_graph': False,
                          'enable_flashinfer_allreduce_fusion': False, 'device': 'cuda'}.items():
        setattr(value, key, selected)
    return value


def execute(args):
    if os.environ.get('R1_SG_SHARED_GUARDED') != '1' or socket.gethostname() not in args.allowed_hostname:
        raise RuntimeError('Real shared qualification requires an allowed host and parent-owned distributed guard')
    env = {'BATCHGEN_R1_SM103_FP8_COMPAT': '1', 'DISABLE_MMA_V5': '1',
           'SGLANG_ENABLE_JIT_DEEPGEMM': '0', 'SGLANG_ENABLE_FLASHINFER_FP8_GEMM': '0',
           'SGLANG_R1_LOCAL_SHARED_TP1': '1', 'SGLANG_R1_COMMON_FI_FA2': '0', 'R1_FULL_MODEL_TRACE': '0'}
    if any(os.environ.get(key) != value for key, value in env.items()):
        raise RuntimeError('Exact isolated shared-probe environment is required')
    rank, world, local_rank = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != args.world_size or world not in (2, 4, 16):
        raise ValueError('Launcher/world-size contract mismatch')
    numeric = None
    if args.numerical_contract is not None:
        if sha(args.numerical_contract) != args.numerical_contract_sha256:
            raise ValueError('Numerical acceptance contract SHA mismatch')
        numeric = validate_numeric_contract(json.loads(args.numerical_contract.read_text()))
    if importlib.metadata.version('sglang') != '0.5.5.post3':
        raise RuntimeError('Real shared qualification requires SGLang0.5.5post3')
    torch = importlib.import_module('torch')
    if torch.__version__ != '2.9.0+cu130':
        raise RuntimeError('Common Torch identity mismatch')
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)
    device = torch.device('cuda', local_rank)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError('Real shared compatibility qualification is restricted to SM103')
    torch.cuda.set_per_process_memory_fraction((3 * 1024**3) / torch.cuda.get_device_properties(device).total_memory, device)
    # ModelRunner establishes these globals before communicator/model imports.
    server_module = importlib.import_module('sglang.srt.server_args')
    server = server_args_defaults(server_module, world)
    server_module.set_global_server_args_for_scheduler(server)
    moe_config = importlib.import_module('sglang.srt.layers.moe.utils')
    moe_config.initialize_moe_config(server)
    modules = {name: importlib.import_module(name) for name in SOURCE_SHA}
    sources = {name: source_record(module, SOURCE_SHA[name]) for name, module in modules.items()}
    parallel = modules['sglang.srt.distributed.parallel_state']
    parallel.set_custom_all_reduce(False)
    parallel.init_distributed_environment(world, rank, 'env://', local_rank, 'nccl', timeout=180)
    parallel.initialize_model_parallel(world, world, 1, backend='nccl', torch_compile=False)
    dp = modules['sglang.srt.layers.dp_attention']
    dp.initialize_dp_attention(server, SimpleNamespace(hidden_size=HIDDEN, dtype=torch.bfloat16))
    model = importlib.import_module('sglang.srt.models.deepseek_v2')
    sources.update({name: source_record(importlib.import_module(name), expected)
                    for name, expected in EXTRA_SOURCE_SHA.items()})
    sources['generated_model'] = source_record(model, args.adapter_source_sha256)
    ast_contract = generated_source_contract(Path(sources['generated_model']['path']).read_bytes(), args.adapter_source_sha256)
    runtime = importlib.import_module('scripts.r1_reproduction.sglang_r1_alignment_runtime')
    config = SimpleNamespace(model_type='deepseek_v3', architectures=['DeepseekV3ForCausalLM'],
                             hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE,
                             n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8,
                             first_k_dense_replace=3, hidden_act='silu')
    quant = importlib.import_module('sglang.srt.layers.quantization.fp8').Fp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme='dynamic', weight_block_size=[128, 128])
    runtime.install_fp8_linear_backend(config, quant)
    if world == 16:
        runtime.require_shared_contract(config, quant, server, tp_size=parallel.get_tensor_model_parallel_world_size(),
            ep_size=parallel.get_moe_expert_parallel_world_size(), attention_tp_size=dp.get_attention_tp_size(),
            dp_attention=dp.is_dp_attention_enabled(), a2a_none=moe_config.get_moe_a2a_backend().is_none(),
            runner_triton=moe_config.get_moe_runner_backend().is_triton(),
            deepgemm_enabled=model.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM)
    tensors, weight_records, manifest = load_weights(torch, args, device)
    base = f'model.layers.{args.layer}.mlp.shared_experts.'
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            shared = model.DeepseekV2MLP(HIDDEN, INTERMEDIATE, 'silu', quant, reduce_results=False, tp_rank=0, tp_size=1)
    finally:
        torch.set_default_dtype(previous_dtype)
    for part, start in (('gate_proj', 0), ('up_proj', INTERMEDIATE)):
        shared.gate_up_proj.weight.data[start:start + INTERMEDIATE].copy_(tensors[base + part + '.weight'])
        shared.gate_up_proj.weight_scale_inv.data[start // 128:(start + INTERMEDIATE) // 128].copy_(tensors[base + part + '.weight_scale_inv'])
    shared.down_proj.weight.data.copy_(tensors[base + 'down_proj.weight'])
    shared.down_proj.weight_scale_inv.data.copy_(tensors[base + 'down_proj.weight_scale_inv'])
    for name in ('gate_up_proj', 'down_proj'):
        layer = getattr(shared, name)
        if layer.quant_method.w8a8_block_fp8_linear is not runtime.common_block_fp8_linear:
            raise RuntimeError('Actual shared linear did not bind the common original-scale backend')
        layer.quant_method.process_weights_after_loading(layer)
        if (layer.tp_size != 1 or layer.tp_rank != 0 or layer.params_dtype != torch.bfloat16
                or layer.input_scale is not None or layer.bias is not None):
            raise RuntimeError('Actual shared linear TP1/BF16 dynamic/no-bias contract mismatch')
    if shared.down_proj.reduce_results:
        raise RuntimeError('TP1 shared down projection must not reduce')
    for part, start in (('gate_proj', 0), ('up_proj', INTERMEDIATE)):
        if (tensor_sha(torch, shared.gate_up_proj.weight[start:start + INTERMEDIATE]) != weight_records[base + part + '.weight']['sha256']
                or tensor_sha(torch, shared.gate_up_proj.weight_scale_inv[start // 128:(start + INTERMEDIATE) // 128]) != weight_records[base + part + '.weight_scale_inv']['sha256']):
            raise RuntimeError('Actual merged shared projection changed an original FP8 tensor/scale')
    if (tensor_sha(torch, shared.down_proj.weight) != weight_records[base + 'down_proj.weight']['sha256']
            or tensor_sha(torch, shared.down_proj.weight_scale_inv) != weight_records[base + 'down_proj.weight_scale_inv']['sha256']):
        raise RuntimeError('Actual shared down projection changed an original FP8 tensor/scale')
    before_weights = {name: tensor_sha(torch, parameter) for name, parameter in shared.named_parameters()}
    # Optional independent mathematical gate, never implied by composition pass.
    dequant = {}
    if numeric is not None:
        for part in ('gate_proj', 'up_proj', 'down_proj'):
            data, scale = tensors[base + part + '.weight'], tensors[base + part + '.weight_scale_inv']
            dequant[part] = data.double() * scale.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    comm = modules['sglang.srt.layers.communicator']
    modes = comm.LayerScatterModes(comm.ScatterMode.TP_ATTN_FULL, comm.ScatterMode.TP_ATTN_FULL,
        comm.ScatterMode.FULL, comm.ScatterMode.TP_ATTN_FULL, comm.ScatterMode.TP_ATTN_FULL)
    native_comm = comm.LayerCommunicator(modes, torch.nn.Identity(), torch.nn.Identity(), allow_reduce_scatter=False)
    results = []
    original_sum = model.tensor_model_parallel_all_reduce
    for case, counts in case_counts(world):
        for padding in ('SUM_LEN', 'MAX_LEN'):
            actual_counts = effective_counts(counts, padding)
            local_count = actual_counts[rank]
            dp.set_dp_buffer_len(sum(actual_counts), local_count, padding == 'MAX_LEN', actual_counts)
            dp.set_is_extend_in_batch(padding == 'SUM_LEN')
            batch = SimpleNamespace(dp_padding_mode=getattr(dp.DpPaddingMode, padding),
                dp_local_start_pos=None, dp_local_num_tokens=None,
                global_num_tokens_gpu=torch.tensor(actual_counts, device=device, dtype=torch.int64),
                input_ids=torch.zeros(local_count, dtype=torch.int64, device=device))
            def fixture_input(owner):
                value = torch.zeros((actual_counts[owner], HIDDEN), dtype=torch.bfloat16, device=device)
                for token in range(counts[owner]):
                    signs = torch.tensor(fixture_sign_code(owner, token), dtype=torch.bfloat16, device=device)
                    value[token] = signs.repeat((HIDDEN + 5) // 6)[:HIDDEN] * 0.0625
                return value

            x = fixture_input(rank)
            independently_gathered = torch.cat([fixture_input(owner) for owner in range(world)])
            # This extra invocation is outside the adapter-call instrumentation.
            # It is an independent composition oracle, not independent MLP math.
            separate_shared = shared(x).clone()
            observed = {'sums': [], 'shared': [], 'residual': None}

            class Routed(torch.nn.Module):
                quant_method = SimpleNamespace()

                def forward(self, value, *_args, **_kwargs):
                    return value / (8 * world)

            class MoE(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shared_experts, self.experts = shared, Routed()
                    self._r1_local_shared_tp1, self._fuse_shared_experts_inside_sbo = True, False
                    self.tp_size, self.num_fused_shared_experts = world, 0
                    self.routed_scaling_factor = 1.0
                    self.gate = lambda *_: None
                    self.topk = lambda *_: None
                    self.topk.empty_topk_output = lambda *_: None
                    self.forward_normal = MethodType(model.DeepseekV2MoE.forward_normal, self)
                    self._forward_shared_experts = MethodType(model.DeepseekV2MoE._forward_shared_experts, self)

                def forward(self, *args, **kwargs):
                    return model.DeepseekV2MoE.forward(self, *args, **kwargs)

            def checked_sum(value):
                if not torch.equal(value, independently_gathered / (8 * world)):
                    raise RuntimeError('Routed SUM input differs from independent dyadic contributions')
                observed['sums'].append(tensor_sha(torch, value))
                result = original_sum(value)
                # Identical dyadic contributions must sum exactly, so route
                # arithmetic cannot hide an extra shared contribution.
                if not torch.equal(result, global_input / 8):
                    raise RuntimeError('Native routed SUM contains more than the synthetic routed contributions')
                return result

            def prepare_mlp(*args):
                nonlocal global_input
                global_input, residual = native_comm.prepare_mlp(*args)
                if not torch.equal(global_input, independently_gathered):
                    raise RuntimeError('Actual native DP gather differs from independent row ownership fixture')
                observed['residual'] = residual.clone()
                return global_input, residual

            def shared_hook(_module, inputs, output):
                observed['shared'].append((inputs[0].clone(), output.clone()))

            global_input = None
            handle = shared.register_forward_hook(shared_hook)
            proxy_comm = SimpleNamespace(prepare_attn=lambda hidden, residual, *_: (hidden, residual),
                prepare_mlp=prepare_mlp, postprocess_layer=native_comm.postprocess_layer,
                should_fuse_mlp_allreduce_with_next_layer=native_comm.should_fuse_mlp_allreduce_with_next_layer,
                should_use_reduce_scatter=native_comm.should_use_reduce_scatter)
            proxy = SimpleNamespace(mlp=MoE(), layer_communicator=proxy_comm,
                self_attn=lambda positions, hidden_states, **_: hidden_states)
            model.tensor_model_parallel_all_reduce = checked_sum
            try:
                y, residual = model.DeepseekV2DecoderLayer.forward(proxy,
                    torch.arange(local_count, device=device), x, batch, torch.zeros_like(x), None)
            finally:
                handle.remove()
                model.tensor_model_parallel_all_reduce = original_sum
            if len(observed['sums']) != 1 or len(observed['shared']) != 1:
                raise RuntimeError('Actual adapter must execute one routed SUM and one local shared call')
            shared_input, shared_output = observed['shared'][0]
            if not torch.equal(shared_input, x) or not torch.equal(residual, observed['residual']):
                raise RuntimeError('Shared branch used nonlocal input or changed the residual')
            if not torch.equal(shared_output, separate_shared):
                raise RuntimeError('Actual shared output differs from separate same-native-shared invocation')
            if not torch.equal(y, x / 8 + separate_shared):
                raise RuntimeError('Actual adapter did not add the local shared output exactly once after SUM/scatter')
            checked, failed, max_error = 0, 0, None
            if numeric is not None:
                valid = x[:counts[rank]].double()
                g, u = valid @ dequant['gate_proj'].T, valid @ dequant['up_proj'].T
                reference = (torch.nn.functional.silu(g) * u) @ dequant['down_proj'].T
                target = valid / 8 + reference
                error = (y[:counts[rank]].double() - target).abs()
                tolerance = numeric['atol'] + numeric['rtol'] * target.abs()
                failed, checked = int((error > tolerance).sum()), error.numel()
                max_error = float(error.max()) if checked else 0.0
                if failed > numeric['max_failure_fraction'] * checked:
                    raise RuntimeError('Shared layer exceeded the predeclared independent FP64 numerical contract')
            if not torch.isfinite(y).all() or not torch.equal(y[counts[rank]:], torch.zeros_like(y[counts[rank]:])):
                raise RuntimeError('Shared layer produced nonfinite or nonzero unused rows')
            results.append({'case': case, 'padding': padding, 'valid_local_tokens': counts[rank],
                'actual_local_rows': local_count, 'actual_global_rows': global_input.shape[0],
                'routed_SUM_calls': 1, 'local_shared_calls': 1, 'shared_added_once_bitwise': True,
                'separate_shared_oracle_bitwise': True, 'native_gather_matches_independent_rows': True,
                'separate_oracle_calls': 1,
                'checked_FP64_reference_elements': checked, 'reference_failures': failed,
                'max_abs_to_FP64': max_error})
    if before_weights != {name: tensor_sha(torch, value) for name, value in shared.named_parameters()}:
        raise RuntimeError('Shared FP8 weights/scales changed during qualification')
    torch.cuda.synchronize()
    summary = {'rank': rank, 'world_size': world, 'hostname': socket.gethostname(),
               'stage': 'final16_layout' if world == 16 else 'partial_world_precheck_only',
               'cases': results, 'sources': sources, 'adapter_contract': ast_contract,
               'weights': weight_records, 'weights_provenance': manifest,
               'numerical_contract_sha256': args.numerical_contract_sha256,
               'numerical_contract': numeric,
               'independent_shared_MLP_numerical_qualified': numeric is not None,
               'linear_backend_receipt': runtime.linear_backend_receipt(),
               'tp_members': list(parallel.get_tp_group().ranks), 'attention_tp_size': dp.get_attention_tp_size(),
               'expert_parallel_size': parallel.get_moe_expert_parallel_world_size(),
               'attention_dp_size': dp.get_attention_dp_size(),
               'actual_shared_tp_size': shared.tp_size,
               'actual_shared_parameters': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                            for name, value in shared.named_parameters()},
               'peak_GPU_bytes': torch.cuda.max_memory_allocated(), 'gpu_operations': True,
               'full_model_qualified': False, 'routed_expert_kernel_qualified': False,
               'formal_test_permitted': False}
    torch.distributed.barrier()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--aggregate', action='store_true', help='Read all rank results only; never import GPU libraries')
    ap.add_argument('--world-size', type=int, choices=(2, 4, 16), default=16)
    ap.add_argument('--allowed-hostname', action='append', default=[])
    ap.add_argument('--layer', type=int, default=3)
    ap.add_argument('--adapter-source-sha256')
    ap.add_argument('--weights-bundle', type=Path)
    ap.add_argument('--weights-sha256')
    ap.add_argument('--tensor-manifest', type=Path)
    ap.add_argument('--tensor-manifest-sha256')
    ap.add_argument('--numerical-contract', type=Path)
    ap.add_argument('--numerical-contract-sha256')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.aggregate:
        records = [json.loads((args.output_dir / f'rank-{rank}.json').read_text()) for rank in range(args.world_size)]
        print(json.dumps(aggregate_rank_records(records, args.world_size), indent=2)); return 0
    if not args.execute:
        print(json.dumps(plan(args), indent=2)); return 0
    if not args.output_dir.is_absolute() or any(getattr(args, name) is None for name in (
            'adapter_source_sha256', 'weights_bundle', 'weights_sha256', 'tensor_manifest',
            'tensor_manifest_sha256')):
        ap.error('Execute requires an absolute output directory and all explicit source/weight pins')
    if (args.numerical_contract is None) != (args.numerical_contract_sha256 is None):
        ap.error('Optional FP64 gate requires both an explicit contract and its SHA')
    rank = int(os.environ.get('RANK', '-1'))
    output = args.output_dir / f'rank-{rank}.json'
    if not args.output_dir.is_dir() or output.exists():
        raise ValueError('Parent must create the exclusive output directory; rank output must be new')
    result = dict(plan(args), status='failed_distributed_shared_probe', execution_requested=True, gpu_operations=None)
    code = 1
    try:
        result.update(execute(args), status='passed_real_shared_branch_NOT_full_model_qualified',
                      rank_contract_passed=True, final16_layout_qualified=False,
                      full_rank_set_aggregation_required=True)
        code = 0
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc(), final16_layout_qualified=False)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rank': rank, 'status': result['status'], 'output': str(output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
