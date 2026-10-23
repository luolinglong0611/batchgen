#!/usr/bin/env bash
# Install only inside an allocated container. No host packages/driver changes.
# Usage: bash install_sglang_baseline.sh --root /ABS/NEW/sglang-baseline \
#          --cuda-home /ABS/private-cuda-13.0 --mirror pypi --download-mirror aliyun --jobs 8
# Optional: --sgl-kernel-wheel /ABS/sgl_kernel-0.3.17.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl
# Optional: --bootstrap-triton-wheel /ABS/triton-3.5.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# Optional: --llvm-system-suffix almalinux-x64 (official LLVM build for glibc >= 2.28)
# Optional: --cxx-runtime-dir /ABS/verified-private-cxx-runtime (never replaces system libraries)
# Optional: --skip-gpu-smoke (hides GPUs; defers GPU checks and cubin prefetch)
# Optional: --upstream-dockerfile /ABS/previous-audit/upstream-Dockerfile
# The local wheel must match the official release's fixed size and SHA256. Only
# its transport location changes; package metadata and dependency pins do not.
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
mirror=pypi
download_mirror=
jobs=8
sgl_kernel_wheel=
bootstrap_triton_wheel=
upstream_dockerfile=
llvm_system_suffix=almalinux-x64
skip_gpu_smoke=0
cxx_runtime_dir=
while (($#)); do
  case "$1" in
    --root) baseline_root=${2:?missing root}; shift 2 ;;
    --cuda-home) cuda_home=${2:?missing CUDA home}; shift 2 ;;
    --mirror) mirror=${2:?missing mirror}; shift 2 ;;
    --download-mirror) download_mirror=${2:?missing download mirror}; shift 2 ;;
    --jobs) jobs=${2:?missing jobs}; shift 2 ;;
    --sgl-kernel-wheel) sgl_kernel_wheel=${2:?missing wheel}; shift 2 ;;
    --bootstrap-triton-wheel) bootstrap_triton_wheel=${2:?missing Triton wheel}; shift 2 ;;
    --llvm-system-suffix) llvm_system_suffix=${2:?missing LLVM suffix}; shift 2 ;;
    --cxx-runtime-dir) cxx_runtime_dir=${2:?missing private C++ runtime}; shift 2 ;;
    --skip-gpu-smoke) skip_gpu_smoke=1; shift ;;
    --upstream-dockerfile) upstream_dockerfile=${2:?missing Dockerfile}; shift 2 ;;
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
# Resolution needs authoritative upload dates for the historical cutoff. A
# separately selected binary mirror only consumes the complete, hash-pinned lock.
download_mirror=${download_mirror:-$mirror}
case "$download_mirror" in
  aliyun) download_index=https://mirrors.aliyun.com/pypi/simple/ ;;
  pypi) download_index=https://pypi.org/simple/ ;;
  *) echo 'ERROR: --download-mirror must be aliyun or pypi'; exit 2 ;;
esac
[[ $llvm_system_suffix == almalinux-x64 || $llvm_system_suffix == ubuntu-x64 ]] || { echo 'ERROR: unsupported official LLVM system suffix'; exit 2; }
for tool in uv git curl sha256sum g++ make python3; do
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
unset DISABLE_MMA_V5 TRITON_PTXAS_PATH TRITON_PTXAS_BLACKWELL_PATH TRITON_APPEND_CMAKE_ARGS
export PIP_CONFIG_FILE=/dev/null
export CUDA_HOME=$cuda_home
export XDG_CACHE_HOME=$baseline_root/cache
export TRITON_CACHE_DIR=$baseline_root/cache/triton-runtime
export MAX_JOBS=$jobs
# Use upstream's supported selector, never patch its downloaded tools or checks.
export TRITON_LLVM_SYSTEM_SUFFIX=$llvm_system_suffix
# The official LLVM static libraries may need a newer C++ runtime than the
# allocated container provides. This explicit private path is process-local.
if [[ -n $cxx_runtime_dir ]]; then
  [[ $cxx_runtime_dir == /* && -d $cxx_runtime_dir/lib ]] || { echo 'ERROR: invalid private C++ runtime directory'; exit 2; }
  python3 - "$cxx_runtime_dir" "$audit_dir" <<'PY_CXX'
import hashlib,json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
receipt=json.loads((root/'provenance.json').read_text())
expected={
 'libstdc++.so':'d3c4bd998c94a632c3887c8605ca95e6b102a3bee5fd9d9f6075d130dd3d809d',
 'libstdc++.so.6':'d3c4bd998c94a632c3887c8605ca95e6b102a3bee5fd9d9f6075d130dd3d809d',
 'libgcc_s.so.1':'4f74724a5d9026964eafefc57e46f3363a24c5ad8c83d52b6b7fb270fcb64655',
}
for name,digest in expected.items():
    path=root/'lib'/name
    if not path.resolve().is_relative_to((root/'lib').resolve()):
        raise RuntimeError('Private runtime symlink escapes library directory')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Private C++ runtime SHA mismatch: '+name)
packages={p['package']:p for p in receipt['packages']}
for name,digest in {
 'libstdcxx-ng':'123d97faccdd39aa65a7beaf0dd031dad61e3ed45149c8ed91b07ae3aea6d86b',
 'libgcc-ng':'cad9fc19ea75b6be4a4a47ac742136aa63410c86584b580ee549639ad8817d79',
}.items():
    item=packages[name]
    if item['version']!='12.3.0' or item['archive_sha256']!=digest:
        raise RuntimeError('Unexpected private runtime provenance')
    if not item['source_url'].startswith('https://conda.anaconda.org/conda-forge/linux-64/'):
        raise RuntimeError('Unexpected C++ runtime source')
(out/'private-cxx-runtime.json').write_text(json.dumps({
 'runtime_directory':str(root),'library_sha256':expected,'provenance':receipt,
 'system_libraries_modified':False},indent=2)+'\n')
PY_CXX
  export LD_LIBRARY_PATH=$cxx_runtime_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export LIBRARY_PATH=$cxx_runtime_dir/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
  # Upstream setup.py supports this selector; preserve source and checks.
  export TRITON_APPEND_CMAKE_ARGS
  TRITON_APPEND_CMAKE_ARGS=$(python3 - "$cxx_runtime_dir/lib" <<'PY_CXX_FLAGS'
import shlex,sys
flags='-L'+sys.argv[1]+' -Wl,-rpath,'+sys.argv[1]
print(shlex.join(['-DCMAKE_'+kind+'_LINKER_FLAGS='+flags for kind in ('EXE','SHARED','MODULE')]))
PY_CXX_FLAGS
)
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$cxx_runtime_dir/lib" > "$audit_dir/private-cxx-runtime.env"
fi
export BASELINE_SKIP_GPU_SMOKE=$skip_gpu_smoke
if ((skip_gpu_smoke)); then export CUDA_VISIBLE_DEVICES=; fi
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
if [[ -n $upstream_dockerfile ]]; then
  [[ $upstream_dockerfile == /* && -f $upstream_dockerfile ]] || { echo 'ERROR: --upstream-dockerfile must be an existing absolute file'; exit 2; }
  cp "$upstream_dockerfile" "$audit_dir/upstream-Dockerfile"
else
  # Older container curl lacks --retry-all-errors; retry transient TLS failures
  # with a finite bound without changing the pinned source.
  for attempt in 1 2 3 4; do
    if curl -fL --connect-timeout 15 --max-time 120 "$docker_url" -o "$audit_dir/upstream-Dockerfile"; then
      break
    fi
    [[ $attempt -lt 4 ]] || { echo 'ERROR: official Dockerfile download failed after 4 attempts'; exit 1; }
    sleep 2
  done
fi
printf '%s  %s\n' 352f52b5ff8880b355a76bb0c5c2fc0a558df55763f6be86eed08019fd515794 "$audit_dir/upstream-Dockerfile" | sha256sum -c -
python3 - "$audit_dir" "$docker_url" "$upstream_dockerfile" <<'PY_DOCKERFILE'
import json,sys
from pathlib import Path
out,origin,local=sys.argv[1:]
Path(out,'upstream-Dockerfile-source.json').write_text(json.dumps({
    'official_source_url':origin,
    'verified_sha256':'352f52b5ff8880b355a76bb0c5c2fc0a558df55763f6be86eed08019fd515794',
    'transport':'verified-local-copy' if local else 'official-source-url',
    'local_source_path':str(Path(local).resolve()) if local else None,
    'source_modified':False},indent=2)+'\n')
PY_DOCKERFILE
if [[ -z $sgl_kernel_wheel ]]; then
  curl -fIL --retry 2 --connect-timeout 15 --max-time 120 "$kernel_url" > "$audit_dir/sgl-kernel-wheel-headers.txt"
fi
cp "$script_dir/sglang_baseline_requirements.in" "$audit_dir/requirements.in"
cp "$script_dir/sglang_baseline_overrides.txt" "$audit_dir/overrides.txt"
cp "$script_dir/sglang_baseline_overrides.txt" "$audit_dir/source-overrides.txt"
stage=verify-official-sgl-kernel-artifact
python3 - "$audit_dir" "$sgl_kernel_wheel" "$kernel_url" <<'PY'
import hashlib,json,sys,zipfile
from email.parser import BytesParser
from pathlib import Path

out,local,origin=sys.argv[1:]
out=Path(out)
expected_sha='51a97eb73076f9e2b6bfc2dea0c152bd20ae1ce9914017de096efb5ed9a24371'
expected_size=380986924
record={'official_release_url':origin,
        'official_asset_api':'https://api.github.com/repos/sgl-project/whl/releases/assets/317144081',
        'official_asset_id':317144081,'expected_sha256':expected_sha,'expected_size':expected_size,
        'package':'sgl-kernel','filename_version':'0.3.17.post1+cu130',
        'metadata_version':'0.3.17.post1',
        'transport':'verified-local-wheel' if local else 'official-release-url',
        'metadata_modified':False,'dependencies_modified':False}
source=origin+'#sha256='+expected_sha
if local:
    wheel=Path(local)
    if not wheel.is_absolute() or not wheel.is_file():
        raise RuntimeError('--sgl-kernel-wheel must be an existing absolute file')
    expected_name='sgl_kernel-0.3.17.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl'
    if wheel.name != expected_name or wheel.stat().st_size != expected_size:
        raise RuntimeError('Local sgl-kernel filename/size does not match the official release')
    digest=hashlib.sha256()
    with wheel.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected_sha:
        raise RuntimeError('Local sgl-kernel SHA256 mismatch')
    with zipfile.ZipFile(wheel) as archive:
        metadata=[name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata)!=1:raise RuntimeError('Expected exactly one package metadata entry')
        meta=BytesParser().parsebytes(archive.read(metadata[0]))
        if meta['Name'].replace('_','-')!='sgl-kernel' or meta['Version']!='0.3.17.post1':
            raise RuntimeError('Local wheel package identity mismatch')
        record['wheel_requires_dist']=meta.get_all('Requires-Dist',[])
    record.update(local_path=str(wheel.resolve()),verified_sha256=digest.hexdigest(),verified_size=wheel.stat().st_size)
    source=wheel.resolve().as_uri()+'#sha256='+expected_sha
path=out/'overrides.txt'
lines=path.read_text().splitlines()
indices=[i for i,line in enumerate(lines) if line.startswith('sgl-kernel @ ')]
if len(indices)!=1:raise RuntimeError('Expected exactly one existing sgl-kernel override')
lines[indices[0]]='sgl-kernel @ '+source
path.write_text('\n'.join(lines)+'\n')
record['effective_requirement']=lines[indices[0]]
(out/'sgl-kernel-source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY

# The PyTorch index re-uploaded Triton 3.5.0 in January 2026; pin its original
# October 2025 PyPI artifact for bootstrap, then replace it with the fixed source
# build below. This preserves the historical dependency cutoff.
stage=verify-bootstrap-triton-artifact
python3 - "$audit_dir" "$bootstrap_triton_wheel" <<'PY_TRITON'
import hashlib,json,sys,zipfile
from email.parser import BytesParser
from pathlib import Path
out,local=sys.argv[1:];out=Path(out)
name='triton-3.5.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
origin='https://files.pythonhosted.org/packages/3d/78/949a04391c21956c816523678f0e5fa308eb5b1e7622d88c4e4ef5fceca0/'+name
sha='f34bfa21c5b3a203c0f0eab28dcc1e49bd1f67d22724e77fb6665a659200a4ec'
size=170433488
record={'official_metadata_url':'https://pypi.org/pypi/triton/3.5.0/json',
        'official_wheel_url':origin,'expected_sha256':sha,'expected_size':size,
        'upload_time_iso_8601':'2025-10-13T16:37:57.132491Z',
        'package':'triton','version':'3.5.0','scope':'bootstrap-only; replaced by fixed-source build',
        'transport':'verified-local-wheel' if local else 'official-PyPI-url',
        'metadata_modified':False,'dependencies_modified':False}
source=origin+'#sha256='+sha
if local:
    wheel=Path(local)
    if not wheel.is_absolute() or not wheel.is_file() or wheel.name!=name or wheel.stat().st_size!=size:
        raise RuntimeError('Local bootstrap Triton filename/size mismatch')
    digest=hashlib.sha256()
    with wheel.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=sha:raise RuntimeError('Local bootstrap Triton SHA256 mismatch')
    with zipfile.ZipFile(wheel) as archive:
        names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names)!=1:raise RuntimeError('Expected one bootstrap Triton METADATA')
        meta=BytesParser().parsebytes(archive.read(names[0]))
        if meta['Name']!='triton' or meta['Version']!='3.5.0':raise RuntimeError('Bootstrap Triton identity mismatch')
    source=wheel.resolve().as_uri()+'#sha256='+sha
    record.update(local_path=str(wheel.resolve()),verified_sha256=digest.hexdigest(),verified_size=size)
path=out/'overrides.txt'
lines=path.read_text().splitlines()
if any(line.startswith(('triton==','triton @ ')) for line in lines):raise RuntimeError('Unexpected existing Triton override')
lines.append('triton @ '+source);path.write_text('\n'.join(lines)+'\n')
record['effective_requirement']=lines[-1]
(out/'bootstrap-triton-source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY_TRITON

# SGLang pins torchao 0.9.0, absent from the cu130 index selected by
# --torch-backend. Keep that dependency unchanged and pin its official PyPI wheel.
python3 - "$audit_dir" "$download_index" <<'PY_TORCHAO'
import json,sys
from pathlib import Path
out,index=sys.argv[1:];out=Path(out)
name='torchao-0.9.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl'
origin='https://files.pythonhosted.org/packages/7d/fe/a24225d30775192a4c5d9cea3ecb95e6adc69d0a8b5ed98eb8e58d362344/'+name
sha='bc708910301a9f98344d43f3fe2aa6d5e1fab706d772b6df47ff05087d664145'
transport=origin
if index=='https://mirrors.aliyun.com/pypi/simple/':
    transport=origin.replace('https://files.pythonhosted.org/packages/','https://mirrors.aliyun.com/pypi/packages/')
record={'official_metadata_url':'https://pypi.org/pypi/torchao/0.9.0/json',
        'official_wheel_url':origin,'transport_url':transport,
        'expected_sha256':sha,'expected_size':5652091,
        'upload_time_iso_8601':'2025-02-28T13:54:15.239390Z',
        'package':'torchao','version':'0.9.0',
        'metadata_modified':False,'dependencies_modified':False}
path=out/'overrides.txt';lines=path.read_text().splitlines()
if any(line.startswith(('torchao==','torchao @ ')) for line in lines):raise RuntimeError('Unexpected existing torchao override')
lines.append('torchao @ '+transport+'#sha256='+sha);path.write_text('\n'.join(lines)+'\n')
record['effective_requirement']=lines[-1]
(out/'torchao-source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY_TORCHAO

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
  --default-index "$download_index" --torch-backend cu130 \
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
"$py" - "$audit_dir" "$baseline_env" "$triton_commit" "$sglang_commit" "$package_index" "$download_index" <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

out, prefix, triton_commit, sglang_commit, index, download_index = sys.argv[1:]
if Path(sys.prefix).resolve() != Path(prefix).resolve() or sys.version_info[:2] != (3, 11):
    raise RuntimeError('Incorrect Python environment')
expected = {
    'sglang': '0.5.5.post3', 'torch': '2.9.0+cu130', 'torchaudio': '2.9.0+cu130',
    # Official cu130 filename contains +cu130; its unchanged METADATA does not.
    'torchvision': '0.24.0+cu130', 'sgl-kernel': '0.3.17.post1',
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
    'resolution_index': index, 'binary_download_index': download_index, 'torch_backend': 'cu130',
    'historical_dependency_cutoff': '2025-11-17T23:59:59Z',
    'triton_llvm_system_suffix': os.environ['TRITON_LLVM_SYSTEM_SUFFIX'],
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
import os
import torch
import triton
import sgl_kernel
import flashinfer
if torch.__version__ != '2.9.0+cu130' or torch.version.cuda != '13.0':
    raise RuntimeError('Incorrect Torch/CUDA build')
skip_gpu = os.environ.get('BASELINE_SKIP_GPU_SMOKE') == '1'
devices = []
if not skip_gpu:
    devices = [dict(index=i, name=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i))
               for i in range(torch.cuda.device_count())]
    if not devices or any(tuple(d['capability']) != (10, 3) for d in devices):
        raise RuntimeError('Expected allocated SM103 GPUs')
    x = torch.arange(16, device='cuda', dtype=torch.bfloat16).reshape(4, 4)
    if not torch.isfinite(x @ x).all():
        raise RuntimeError('CUDA BF16 smoke failed')
    torch.cuda.synchronize()
print(json.dumps({'gpu_qualification': not skip_gpu, 'devices': devices, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'triton': md.version('triton'), 'flashinfer': md.version('flashinfer-python')}))
PY
"$py" -m sglang.launch_server --help > "$audit_dir/launch-server-help.txt"

if ((skip_gpu_smoke)); then
  printf '%s\n' '{"gpu_qualification":false,"cubin_prefetch":"deferred","reason":"--skip-gpu-smoke"}' > "$audit_dir/deferred-gpu-qualification.json"
  stage=complete-with-deferred-gpu-qualification
else
  stage=prefetch-official-flashinfer-cubins
  "$py" -m flashinfer --download-cubin
  stage=complete-with-documented-metadata-exceptions
fi
printf 'Baseline environment installed at %s\n' "$baseline_env"
printf 'Review %s/metadata-audit.json plus pip-check.txt and uv-check.txt. Expected upstream exceptions remain visible.\n' "$audit_dir"
