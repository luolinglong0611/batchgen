#!/usr/bin/env python3
"""Extract six original shared FP8 tensors on CPU; default is read-only plan.

No checkpoint conversion, re-quantization, CUDA initialization or model request.
The coordinator pins the original checkpoint config/index and may additionally
provide a JSON mapping of selected shard filenames to their full-file SHA256.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def probe_module():
    path = Path(__file__).with_name('qualify_sglang_r1_local_shared.py')
    spec = importlib.util.spec_from_file_location('r1_shared_original_tensor_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_shards(root, index, specs):
    weight_map = index.get('weight_map', {})
    result = {}
    root = root.resolve(strict=True)
    for name in specs:
        relative = weight_map.get(name)
        if not isinstance(relative, str) or not relative.endswith('.safetensors'):
            raise ValueError('Original index lacks an explicit safetensors shard: ' + name)
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError('Unsafe original shard path')
        path = (root / relative_path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Original shard must be a file within the checkpoint root')
        result.setdefault(relative, []).append(name)
    return result


def check_config(config):
    expected = {'hidden_size': 7168, 'moe_intermediate_size': 2048,
                'n_shared_experts': 1, 'first_k_dense_replace': 3}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError('Original R1 shared checkpoint dimensions mismatch')
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'fp8' or quant.get('weight_block_size') != [128, 128]:
        raise ValueError('Original serialized FP8 block128 config required')


def prepare(args):
    probe = probe_module()
    root = args.checkpoint_dir.resolve(strict=True)
    config_path, index_path = root / 'config.json', root / 'model.safetensors.index.json'
    config_sha, index_sha = sha(config_path), sha(index_path)
    if config_sha != args.config_sha256 or index_sha != args.index_sha256:
        raise ValueError('Exact original config/index SHA mismatch')
    check_config(json.loads(config_path.read_text()))
    specs = probe.original_tensor_specs(args.layer)
    shards = selected_shards(root, json.loads(index_path.read_text()), specs)
    shard_pins = json.loads(args.shard_sha_manifest.read_text()) if args.shard_sha_manifest else None
    if shard_pins is not None and any(not isinstance(shard_pins.get(name), str) or len(shard_pins[name]) != 64 for name in shards):
        raise ValueError('Explicit full-file hashes required for every selected shard')
    receipt = {'schema': 'r1-original-shared-weight-bundle-v1', 'status': 'plan_only_NOT_executed',
               'source_checkpoint_dir': str(root), 'layer_index': args.layer,
               'config_sha256': config_sha, 'index_sha256': index_sha,
               'selected_shards': shards, 'original_tensor_specs': specs,
               'source_full_shard_hashes_required': shard_pins is not None,
               'source_full_shard_hashes_verified': False,
               'gpu_operations': False, 'checkpoint_conversion': False,
               'source_sha256': sha(__file__), 'probe_source_sha256': sha(probe.__file__),
               'output_dir': str(args.output_dir) if args.output_dir else None}
    if not args.execute:
        return receipt
    if args.output_dir is None or not args.output_dir.is_absolute():
        raise ValueError('CPU extraction requires an absolute fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        # Hide GPUs before importing the CPU tensor reader. No CUDA query occurs.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU extraction refuses an initialized CUDA context')
        tensors, records, shard_records = {}, {}, {}
        for relative, names in shards.items():
            path = root / relative
            full_sha = sha(path) if shard_pins is not None else None
            if shard_pins is not None and full_sha != shard_pins[relative]:
                raise ValueError('Original full checkpoint shard SHA mismatch: ' + relative)
            shard_records[relative] = {'bytes': path.stat().st_size, 'sha256': full_sha,
                                      'full_file_hash_verified': shard_pins is not None}
            with safe_open(str(path), framework='pt', device='cpu') as source:
                for name in names:
                    value = source.get_tensor(name).contiguous()
                    spec = specs[name]
                    if value.device.type != 'cpu' or list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
                        raise ValueError('Original shared tensor CPU shape/dtype mismatch: ' + name)
                    if not torch.isfinite(value.float()).all() or (name.endswith('scale_inv') and not (value > 0).all()):
                        raise ValueError('Original tensor nonfinite or scale nonpositive: ' + name)
                    tensors[name] = value
                    records[name] = dict(spec, sha256=probe.tensor_sha(torch, value), original_shard=relative)
        bundle = args.output_dir / 'shared-original.safetensors'
        temporary = args.output_dir / 'shared-original.safetensors.partial'
        save_file(tensors, str(temporary), metadata={'scope': 'original tensors only; no conversion'})
        bundle_sha = sha(temporary)
        temporary.rename(bundle)
        receipt.update(status='prepared_original_shared_CPU_bundle_NOT_GPU_qualified',
                       bundle_path=str(bundle), bundle_sha256=bundle_sha, tensors=records,
                       source_shards=shard_records, source_full_shard_hashes_verified=shard_pins is not None,
                       provenance_limit=None if shard_pins is not None else
                       'Config/index are pinned and original logical tensor hashes observed; full source-shard hashes were not checked by this utility')
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU extraction unexpectedly initialized CUDA')
        manifest = args.output_dir / 'shared-original.manifest.json'
        manifest.write_text(json.dumps(receipt, indent=2) + '\n')
        return dict(receipt, manifest_path=str(manifest), manifest_sha256=sha(manifest))
    except BaseException as error:
        (args.output_dir / 'failed.json').write_text(json.dumps(dict(receipt,
            status='failed_CPU_extraction_NOT_qualified', error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--index-sha256', required=True)
    parser.add_argument('--layer', type=int, choices=range(3, 61), default=3)
    parser.add_argument('--shard-sha-manifest', type=Path,
                        help='Optional JSON object {original shard filename: full-file SHA256}')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
