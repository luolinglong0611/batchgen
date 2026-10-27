# DeepSeek-R1 compatibility and comparison tools

These changes are based on BatchGen `33ce112`. The implementation separates
platform compatibility, explicit comparison controls, and optional decode
performance changes. CPU contracts and isolated operator checks are distinct
from a full-model correctness or throughput result.

## Platform and attention selection

- `BATCHGEN_HOST_REGISTER_CHUNK_GIB`: unset/0 retains one CUDA host
  registration. A positive integer enables page-aligned, tensor-aware ranges;
  an individual tensor must fit in a range. Registration failure rolls back
  completed ranges, and release uses the original CUDA device. This setting
  does not resize Host KV or change admission/scheduling.
  The Host KV worker view also honors the opt-in, grouping only complete
  physical layers so existing page and token transfers never cross separately
  registered ranges. A 2304--2560 GiB, 61-layer BF16 MLA pool requires a
  64 GiB limit; 32 GiB cannot contain one physical layer.
- `BATCHGEN_R1_SM103_FP8_COMPAT=1`: explicitly selects the existing Triton
  W8A8 implementation with original FP8 operands and FP32 scales. It requires
  actual compute capability 10.3 and `DISABLE_MMA_V5=1`. Unset/0 retains
  DeepGEMM; grouped/masked DeepGEMM options are not implemented by the adapter.
- `BATCHGEN_R1_SM103_MLA_COMPAT=flashinfer-fa2`: explicitly selects BF16
  FlashInfer 0.5.2 FA2 attention, with Graph OFF and native page64 storage
  exposed as valid token-page views. The server profile requires both explicit
  compatibility selectors, DeepSeek-R1, SM103 and BF16 KV. RoPE uses the
  matching native-interleaved FP32 arithmetic. The SDPA helper is an independent
  correctness oracle, not a throughput backend selected by this profile.

## Comparison and evidence

`BATCHGEN_R1_CONTEXT_CAP` and `BATCHGEN_R1_GPU_KV_TOKENS` are optional explicit
comparison controls; they are not paper defaults. The fixed GPU KV control
checks actual allocator capacity and all-rank agreement. The tokenizer respects
configured BOS/EOS boundaries without duplicating an already rendered chat BOS.

The scripts in this directory prepare/seal checkpoints, create identical
workloads, inspect effective runtime parameters, and collect/finalize model and
cross-node traces. SGLang adapters are independently opt-in and pinned to the
historical comparison environment. Their altered backends, allocator geometry,
shared-expert handling, and graph restrictions must be reported as experimental
conditions. Plan output and a registered hook alone are not evidence of model
execution or numerical qualification.

Most qualification tools default to a read-only plan. Read the relevant tool's
`--help` and protocol document before explicit guarded execution. The remote
command helper requires a caller-supplied verified target, container ID and
hostname; no credentials or live connection bindings are included here.

## Decode preparation and attention microbatches

`BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP=1` explicitly selects the existing
paged decode path (`attn_mode=3`) with DP8 attention and all 32 local routed
experts resident per rank on an eight-GPU deployment. It disables the unused
decode weight/KV rings while retaining the actual world size and native paged
KV allocator. EP offloading must remain disabled, including its zero-ratio
flag. This is a new configuration adaptation, not a new scheduling mechanism
or a claim that the model and peak workspaces fit a particular GPU. Unset/0
keeps the previous planner behavior. Set `BATCHGEN_MAX_DECODE_RANK_BSZ` and
`BATCHGEN_MAX_RANK_BSZ` to the same positive integer when tuning admission and
communication buffers (default 128); a legacy-only override cannot raise the
worker's admission cap. Existing SM103 compatibility and Graph OFF requirements
still apply to that platform profile.

The FA2 forward scope reuses metadata, workspace and planning only within one
stable eager forward. MLA position validation shares the scalar maximum within
that forward while preserving each layer's RoPE-bound check. Resident MLA
weights may reuse one complete BF16 dequantized matrix per module; this uses
additional device memory. Load/unload, device moves and tracked source changes
invalidate the cache; unversioned/streamed weights retain dequantization.

`BATCHGEN_DECODE_ATTN_MICRO_BATCH_SIZE` defaults to 0. A positive integer
splits paged BF16 decode attention into row slices, restores original row order,
and returns the assembled batch before the layer's MoE call. This requires
Graph OFF. It does not add a scheduler, increase GPU KV capacity, or implement
per-layer Host-history swapping beyond the existing resident KV working set.

## Local validation

The compatibility, alignment and preparation test modules run with CPU Torch,
Transformers, Tokenizers and pytest; CUDA/Linux execution checks have explicit
platform/opt-in gates. For host-registration bookkeeping alone:

```sh
c++ -std=c++17 -pthread -I. tests/cpp/test_host_registration.cpp -o /tmp/test_host_registration
/tmp/test_host_registration
```

Generated experiment results, logs, checkpoint data, native binaries and private
connection state are stored separately from this code.
