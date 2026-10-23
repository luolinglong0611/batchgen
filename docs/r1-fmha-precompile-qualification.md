# R1 fmha_gen CPU qualification for the October 2026 two-node run

`scripts/r1_reproduction/fmha_precompile_qualification.py` preserves the exact
CPU qualification script used for this run. Its SHA-256 is
`b4e051784b101aed580c7f345082d92fd22008b43b70de426b51d9fb70a74b31`.
The canonical copy is byte-for-byte identical to the tested private v2 script;
it is not a general installer or a configurable multi-host tool.

The script admits only the two recorded Linux container hostnames, the BG
interpreter under `/home/admin/batchgen-r1-repro-20261023/env`, pinned helper
and input files, and the exclusive output directory
`/home/admin/batchgen-r1-repro-20261023/cuda-fmha-precompile-v2`. It refuses to
reuse that directory. That directory already exists on both tested nodes, so
running this archived recipe again will fail admission. A different machine,
toolchain, or experimental round requires a separately reviewed candidate with
new pinned paths, input hashes, and output directory. Nothing here installs
packages or changes the existing toolkit or Torch sources.

## Why the overlay and preload are required

The original full-model JIT used private nvcc 13.0.88 with a BG-environment
CRT 13.4.92 header selected through `CPATH`. The generated CUDA stub called
`__cudaLaunch` with two arguments, but the selected header defined the
one-argument macro. The qualification keeps nvcc, CRT, NVVM, and runtime at
private 13.0.88, with the existing private CCCL 13.0.85. It does not patch the
macro or relax any check.

The entire BG CUDA include directory is excluded. The new `include-aux`
directory contains four exact regular-file copies of BG auxiliary headers:
`cublasLt.h`, `cublas_api.h`, `cublas_v2.h`, and `nvrtc.h`. Their transitive
CUDA headers resolve through the private toolkit. The include probe records
and resolves the actual CRT path; any other CRT path fails qualification.
Node0's five historical auxiliary headers in its private include directory
are inventoried and pinned; node1's absence of those headers is also checked.
The script does not silently fill or alter those existing directories.

Torch's `libtorch_global_deps.so`, `libtorch_cuda.so`, and `libc10_cuda.so`
have `DT_RPATH` entries for the BG CUDA libraries. An ordinary
`LD_LIBRARY_PATH` preference did not select the private runtime. The v2
environment therefore explicitly preloads private `libcudart.so.13` using
`LD_PRELOAD`. This is a recorded runtime environment change. Two independent
CPU-only Torch imports on each node confirmed a single private runtime
mapping before and after import, with `torch.cuda.is_initialized()` false.

## Preserved results and limits

The first CPU candidate remains failed on both nodes:

- Node0 compiled and loaded `fmha_gen.so`, but its runtime mapping was BG
  13.0.48, so the private-runtime gate failed.
- Node1 failed compilation of `fmhaReduction.cu` because the original
  three-header overlay lacked `cublas_v2.h`.

The v2 candidate used a fresh FlashInfer workspace and successfully compiled
and loaded the actual FlashInfer 0.6.13 `fmha_gen` module on both nodes. It
resolved the MLA decode entrypoint without invoking it. Both reports recorded
a cold cache, no changed inputs, only private runtime mappings, and Torch
CUDA uninitialized before and after the operation. Both owned worker cohorts
were reaped without remaining live session processes. Each build took approximately
37 seconds and sampled session RSS remained below 1.8 GiB.

The successful statuses are deliberately named
`compiled_and_loaded_CPU_only_NOT_runtime_qualified` and
`passed_CPU_build_load_NOT_runtime_qualified`. Compilation and loading do not
qualify GPU execution, numerical accuracy, full-model quality, or speed.
Existing strict attention and NaN-tail failures remain failures. The two
generated libraries have different SHA-256 values and must be pinned per
node; the matching build recipe is not a claim of identical binary output.

The receipts include source and input hashes, copied-header provenance,
compiler versions, preserved stub/include evidence, actual `build.ninja`
and library hashes, `/proc/self/maps`, and the guardian's final cleanup audit.
`environment.json` is the complete explicit environment dictionary. A later
runtime experiment must preserve its toolchain, overlay, workspace, and
preload and independently verify the libraries actually loaded.

The script reuses pinned `r1_launch_safety.py` and `r1_launch_process.py`.
It uses one compile job, explicit CUDA-hidden variables, and a 900-second
build deadline. The guardian may then use a further 180+10+30 seconds for
controlled cleanup. The 8 GiB memory limit is a sampled session RSS sum at
0.5-second intervals, not a hard cgroup bound. Python socket auditing rejects
Python network operations; it is not a native network namespace. An explicit
architecture avoids CUDA architecture discovery. The script neither queries
a GPU nor launches a model or HTTP service.

The local frozen evidence for this run is under
`.repro-private/cuda-fmha-jit-audit-v1/`, including
`cpu-v1-failure-receipt.json`, `cpu-v2-pass-receipt.json`, the per-node original
JSON receipts, and the four preload-import probe results. These private raw
artifacts are separate from the committed source. The CPU v2 summary SHA-256
is `edbe3ddc4249e78f4c8b9b5bf50480fc058e3fc65e8a3e7b9cc619203c739cff`.

## Local tests

The six standard-library tests exercise file bytes and mutation detection,
regular-file header copies and the runtime symlink, environment isolation,
real Python socket auditing, normalized include traces, and explicit CUDA
visibility admission. They import the canonical script without running its
Linux build entrypoint or importing Torch.

```sh
python3 -m unittest discover -s tests -p test_fmha_precompile_qualification.py -v
```

These preparation tests supplement the recorded Linux build evidence; they
do not substitute for it.
