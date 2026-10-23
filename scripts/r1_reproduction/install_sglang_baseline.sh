#!/usr/bin/env bash
# Install only inside an allocated container. No host packages/driver changes.
# Usage: bash install_sglang_baseline.sh --root /ABS/NEW/sglang-baseline \
#          --cuda-home /ABS/private-cuda-13.0 --mirror aliyun --jobs 8
#
# This is a CUDA13 adaptation of the paper's SGLang release, following its
# official Docker recipe, not a claim that the original paper ran on B300.
# Python 3.11 is intentional (upstream Docker uses 3.12). DP-attention/TP baseline
# does not use the Docker's optional DeepEP/NVSHMEM build or diffusion extras.
# No third-party source/metadata checks are patched. Four upstream metadata
# exceptions are recorded explicitly; any other incompatibility is fatal.
# Existing environments are never overwritten. Keep the complete root directory
# (lock, Triton wheel, source commit, exports, audit and logs) for reproducibility.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
baseline_root=
cuda_home=
mirror=aliyun
jobs=8
while (($#)); do
  case "$1" in
    --root) baseline_root=${2:?missing root}; shift 2 ;;
    --cuda-home) cuda_home=${2:?missing CUDA home}; shift 2 ;;
    --mirror) mirror=${2:?missing mirror}; shift 2 ;;
    --jobs) jobs=${2:?missing jobs}; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -f /.dockerenv ]] || { echo 'ERROR: run inside the allocated container'; exit 2; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'ERROR: this recipe targets Linux x86_64'; exit 2; }
[[ $baseline_root == /* && $cuda_home == /* ]] || { echo 'ERROR: --root and --cuda-home must be absolute'; exit 2; }
[[ $jobs =~ ^[1-8]$ ]] || { echo 'ERROR: --jobs must be 1..8'; exit 2; }
case "$mirror" in
  aliyun) package_index=https://mirrors.aliyun.com/pypi/simple/ ;;
  pypi) package_index=https://pypi.org/simple/ ;;
  *) echo 'ERROR: --mirror must be aliyun or pypi'; exit 2 ;;
esac
for tool in uv git curl sha256sum g++ make; do
  command -v "$tool" >/dev/null || { printf 'ERROR: missing required tool %s\n' "$tool"; exit 2; }
done
[[ -x $cuda_home/bin/nvcc ]] || { echo 'ERROR: private nvcc is missing'; exit 2; }
baseline_env=$baseline_root/baseline-env
[[ ! -e $baseline_env ]] || { echo 'ERROR: baseline-env already exists; refusing to overwrite or silently upgrade it'; exit 2; }
[[ ! -e $baseline_root/installation ]] || { echo 'ERROR: installation records already exist; use a fresh --root'; exit 2; }
mkdir -p "$baseline_root/installation" "$baseline_root/wheels" "$baseline_root/sources"
audit_dir=$baseline_root/installation
exec > >(tee -a "$audit_dir/install.log") 2>&1
stage=preflight
py=
snapshot() {
  local status=$1
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s}\n' "$stage" "$status" > "$audit_dir/exit-status.json"
  if [[ -n $py && -x $py ]]; then
    uv --no-config pip freeze --python "$py" > "$audit_dir/environment.freeze.txt" 2>&1 || true
    "$py" -m pip inspect > "$audit_dir/pip-inspect.json" 2> "$audit_dir/pip-inspect.err" || true
    "$py" -m pip check > "$audit_dir/pip-check.txt" 2>&1
    printf '%s\n' "$?" > "$audit_dir/pip-check.exit-code"
    uv --no-config pip check --python "$py" > "$audit_dir/uv-check.txt" 2>&1
    printf '%s\n' "$?" > "$audit_dir/uv-check.exit-code"
  fi
  if ((status)); then
    printf 'FAILED at stage %s; see %s. No version fallback was attempted.\n' "$stage" "$audit_dir"
  fi
  exit "$status"
}
# The EXIT snapshot must collect failures without hiding the original exit code.
trap 'saved_exit=$?; set +e; snapshot "$saved_exit"' EXIT

# Ignore inherited package routing and BatchGen Python/compiler overrides only
# in this script's process. Do not expose or modify machine credential config.
unset PYTHONPATH PYTHONHOME UV_INDEX UV_EXTRA_INDEX_URL UV_INDEX_URL UV_DEFAULT_INDEX
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL TRITON_WHEEL_NAME TRITON_WHEEL_VERSION_SUFFIX
unset DISABLE_MMA_V5 TRITON_PTXAS_PATH TRITON_PTXAS_BLACKWELL_PATH
export PIP_CONFIG_FILE=/dev/null
export CUDA_HOME=$cuda_home
export XDG_CACHE_HOME=$baseline_root/cache
export TRITON_CACHE_DIR=$baseline_root/cache/triton-runtime
export MAX_JOBS=$jobs
export FLASHINFER_CUBIN_DOWNLOAD_THREADS=$jobs
export FLASHINFER_LOGGING_LEVEL=warning
uv --version > "$audit_dir/uv-version.txt"
"$cuda_home/bin/nvcc" --version | tee "$audit_dir/nvcc-version.txt"
if ! grep -q 'release 13\.0,' "$audit_dir/nvcc-version.txt"; then
  echo 'ERROR: use the validated private CUDA 13.0 toolkit, not system CUDA 13.2'; exit 1
fi

sglang_commit=7b2fb3d47c9dc43367c6fd3bf9396f7ba45e6883
triton_commit=4caa0328bf8df64896dd5f6fb9df41b0eb2e750a
docker_url=https://raw.githubusercontent.com/sgl-project/sglang/$sglang_commit/docker/Dockerfile
kernel_url=https://github.com/sgl-project/whl/releases/download/v0.3.17.post1/sgl_kernel-0.3.17.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl
curl -fL --retry 3 --connect-timeout 15 --max-time 120 "$docker_url" -o "$audit_dir/upstream-Dockerfile"
printf '%s  %s\n' 352f52b5ff8880b355a76bb0c5c2fc0a558df55763f6be86eed08019fd515794 "$audit_dir/upstream-Dockerfile" | sha256sum -c -
curl -fIL --retry 2 --connect-timeout 15 --max-time 120 "$kernel_url" > "$audit_dir/sgl-kernel-wheel-headers.txt"
cp "$script_dir/sglang_baseline_requirements.in" "$audit_dir/requirements.in"
cp "$script_dir/sglang_baseline_overrides.txt" "$audit_dir/overrides.txt"

stage=create-isolated-python
uv --no-config venv --python 3.11 --managed-python "$baseline_env"
py=$baseline_env/bin/python
export VIRTUAL_ENV=$baseline_env
export PATH=$baseline_env/bin:$cuda_home/bin:$PATH
"$py" -VV > "$audit_dir/python-version.txt"

stage=resolve-exact-wheel-lock
# A historical cutoff prevents an unconstrained auxiliary dependency from
# selecting a 2026 API. Every resolved dependency is then pinned and hashed.
# The explicit CuTe prerelease pin is the only intended prerelease request.
# --torch-backend routes PyTorch only; it does not expose a global extra index.
uv --no-config pip compile "$audit_dir/requirements.in" \
  --override "$audit_dir/overrides.txt" --python "$py" \
  --default-index "$package_index" --torch-backend cu130 \
  --exclude-newer 2025-11-17T23:59:59Z --no-build --generate-hashes \
  --output-file "$audit_dir/resolved-wheels.lock"

stage=install-locked-binaries
# This finite --no-deps install consumes the COMPLETE transitive lock. Upstream
# metadata remains unchanged and is checked explicitly after the source overlay.
uv --no-config pip install --python "$py" --no-deps --no-build --require-hashes \
  --default-index "$package_index" --torch-backend cu130 \
  -r "$audit_dir/resolved-wheels.lock"

stage=build-official-triton-commit
triton_source=$baseline_root/sources/triton-$triton_commit
git init "$triton_source"
git -C "$triton_source" remote add origin https://github.com/triton-lang/triton.git
git -C "$triton_source" fetch --depth 1 origin "$triton_commit"
git -C "$triton_source" checkout --detach FETCH_HEAD
[[ $(git -C "$triton_source" rev-parse HEAD) == "$triton_commit" ]] || { echo 'ERROR: Triton commit mismatch'; exit 1; }
git -C "$triton_source" status --porcelain > "$audit_dir/triton-before-build.status"
[[ ! -s $audit_dir/triton-before-build.status ]] || { echo 'ERROR: Triton source is not clean'; exit 1; }
# Build dependencies above match python/requirements.txt in this commit. Its own
# build downloads the commit-pinned LLVM/CUDA tools; no safety checks are edited.
"$py" -m pip wheel --no-deps --no-build-isolation --no-index \
  --wheel-dir "$baseline_root/wheels" "$triton_source"
mapfile -t triton_wheels < <(find "$baseline_root/wheels" -maxdepth 1 -name 'triton-3.5.0*.whl' -type f)
[[ ${#triton_wheels[@]} == 1 ]] || { echo 'ERROR: expected exactly one fixed-source Triton wheel'; exit 1; }
sha256sum "${triton_wheels[0]}" > "$audit_dir/triton-wheel.sha256"
uv --no-config pip install --python "$py" --no-deps --reinstall-package triton "${triton_wheels[0]}"
git -C "$triton_source" diff --exit-code

stage=audit-metadata-exceptions
"$py" - "$audit_dir" "$baseline_env" "$triton_commit" "$sglang_commit" "$package_index" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

out, prefix, triton_commit, sglang_commit, index = sys.argv[1:]
if Path(sys.prefix).resolve() != Path(prefix).resolve() or sys.version_info[:2] != (3, 11):
    raise RuntimeError('Incorrect Python environment')
expected = {
    'sglang': '0.5.5.post3', 'torch': '2.9.0+cu130', 'torchaudio': '2.9.0+cu130',
    'torchvision': '0.24.0+cu130', 'sgl-kernel': '0.3.17.post1+cu130',
    'flashinfer-python': '0.5.2', 'flashinfer-cubin': '0.5.2',
    'nvidia-cutlass-dsl': '4.3.0.dev0', 'nvidia-nccl-cu13': '2.28.3',
    'transformers': '4.57.1',
}
for package, version in expected.items():
    if md.version(package) != version:
        raise RuntimeError(f'{package}: expected {version}, found {md.version(package)}')
triton_version = md.version('triton')
if Version(triton_version).base_version != '3.5.0':
    raise RuntimeError('Fixed-source Triton version mismatch')
allowed = {
    ('sglang', 'torch', '==2.8.0', '2.9.0+cu130'),
    ('sglang', 'torchaudio', '==2.8.0', '2.9.0+cu130'),
    ('sglang', 'nvidia-cutlass-dsl', '==4.2.1', '4.3.0.dev0'),
    ('torch', 'nvidia-nccl-cu13', '==2.27.7', '2.28.3'),
}
environment = {**default_environment(), 'extra': ''}
exceptions, unexpected = [], []
for distribution in md.distributions():
    owner = canonicalize_name(distribution.metadata['Name'])
    for raw in distribution.requires or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate(environment):
            continue
        dependency = canonicalize_name(req.name)
        try:
            installed = md.version(req.name)
        except md.PackageNotFoundError:
            installed = None
        if installed is None or (req.specifier and not req.specifier.contains(installed, prereleases=True)):
            record = {'package': owner, 'requirement': raw, 'installed': installed}
            key = (owner, dependency, str(req.specifier), installed)
            (exceptions if key in allowed else unexpected).append(record)
report = {
    'status': 'failed' if unexpected else 'installed_with_documented_upstream_metadata_exceptions',
    'expected_versions': expected, 'triton_version': triton_version,
    'triton_source_commit': triton_commit, 'sglang_source_commit': sglang_commit,
    'metadata_exceptions': exceptions, 'unexpected_incompatibilities': unexpected,
    'index': index, 'torch_backend': 'cu130',
    'historical_dependency_cutoff': '2025-11-17T23:59:59Z',
    'scope': 'Environment installation audit; not full-model correctness or performance qualification',
}
Path(out, 'metadata-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
if unexpected:
    raise RuntimeError('Unexpected dependency incompatibilities; see metadata-audit.json')
PY

stage=runtime-import-smoke
"$py" - <<'PY' > "$audit_dir/runtime-smoke.json"
import importlib.metadata as md
import json
import torch
import triton
import sgl_kernel
import flashinfer
if torch.__version__ != '2.9.0+cu130' or torch.version.cuda != '13.0':
    raise RuntimeError('Incorrect Torch/CUDA build')
devices = [dict(index=i, name=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i))
           for i in range(torch.cuda.device_count())]
if not devices or any(tuple(d['capability']) != (10, 3) for d in devices):
    raise RuntimeError('Expected allocated SM103 GPUs')
x = torch.arange(16, device='cuda', dtype=torch.bfloat16).reshape(4, 4)
if not torch.isfinite(x @ x).all():
    raise RuntimeError('CUDA BF16 smoke failed')
torch.cuda.synchronize()
print(json.dumps({'devices': devices, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'triton': md.version('triton'), 'flashinfer': md.version('flashinfer-python')}))
PY
"$py" -m sglang.launch_server --help > "$audit_dir/launch-server-help.txt"

stage=prefetch-official-flashinfer-cubins
"$py" -m flashinfer --download-cubin

stage=complete-with-documented-metadata-exceptions
printf 'Baseline environment installed at %s\n' "$baseline_env"
printf 'Review %s/metadata-audit.json plus pip-check.txt and uv-check.txt. Expected upstream exceptions remain visible.\n' "$audit_dir"
