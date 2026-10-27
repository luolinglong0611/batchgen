# R1 shared-expert adaptation and phase-layout contract

2026-10-26; BG upstream base `33ce112983f6fae019803b8ae1eb0ef3aae6e8fa`.
This is a prepared source candidate plus CPU semantic tests. No SG model was
instantiated, no GPU/distributed collective was run, and no installed SG source
was changed by this work. The current parameter table must retain **planned /
unverified** until fresh runtime receipts exist.

## Controls and BatchGen mechanisms

| Item | Classification | Required handling |
|---|---|---|
| Original FP8 checkpoint bytes, FP32 block scales, tokenizer, prompts, output length and sampling | Common control | Freeze identical logical tensors, input/output contract and before/after loader identities |
| Dense, attention and shared-expert TP1 compute on owned DP tokens | Common module-layout control for the new comparison | SG can be adapted as below; do not leave shared TP16 and call it TP1 |
| Decode routed expert TP1 / EP16, 16 experts per rank | Common decode layout control | Freeze group members and actual local IDs/shapes on all ranks |
| BF16-to-FP8 block quantization and original weight-scale semantics | Common numerical control | Use the common ordinary-linear entry; disable SG UE8M0 requantization and verify actual data |
| Graph state, overlap/fusion state, memory resource limits, external admission cap | Common conditions | First qualification uses graph/overlap/fusion off. Equal memory fractions alone do not prove equal usable resources |
| BG prefill pure DP, full expert range available per rank through streamed/offloaded weights | **BatchGen phase mechanism** | Retain and register it; do not silently replace it with resident EP16 |
| BG decode resident partitioned EP16 with phase switching | **BatchGen phase mechanism**, with decode partition degree controlled | Record transition costs, residency and route; do not remove the phase switch to make process labels equal |
| Internal microbatch packing, prefill/decode scheduling, weight/KV transfer timing and placement | **BatchGen treatment variables** | Same external workload/resource/capacity contract; export internal trajectories as outcomes/conditions |
| SG prefill resident EP16 and chunk scheduling | Baseline system condition | Register the difference. Giving SG the BG streaming scheduler would implement the mechanism inside the baseline |
| Different routed grouped Triton kernels, merged projection shape, SiLU/attention implementation | Declared kernel conditions, with numerical qualification | Do not label them common kernels. They need mathematical/scale correctness; distinct implementation names alone do not imply unequal model parameters |

Thus "parameter-aligned" can mean the same external model/workload, original
weight/scale contract, resource controls and explicitly common module layouts,
while retaining declared system mechanisms. It must not mean that both
systems have identical prefill residency or every internal batch trajectory.
An experiment that intentionally equalizes the phase strategy is a separate
ablation and must be named as such.

## Exact preserved source evidence

Paths in this paragraph are relative to `/Users/lilele/Desktop/batchgen`.
They describe preserved sources, not the replacement containers' installed
objects:

| Source | SHA256 | Relevant lines |
|---|---|---|
| `.repro-private/baseline-recovery/diagnostic-source/sglang/srt/models/deepseek_v2.py` | `620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49` | 455-498 MLP TP; 675-691 shared constructor; 747-873 normal MoE/SUM; 2829-2863 gather/MLP/scatter; 3454-3532 UE8M0 loader |
| `.repro-private/sglang-moe-largeM-audit-v1/sources/sglang/srt/layers/communicator.py` | `e3f911a97cb960c875a06014ab149d75bff8736ddebdf4a6ad0a3654003c6407` | 252-268 sparse FULL input; 487-529 prepare/scatter; 532-567 fusion selection; 720-765 norm before DP gather; 884-905 reduce-scatter vs scatter |
| `.repro-private/sglang-formal-A-control-v2/dp_attention-node0.py` | `ea7fa7d5035aa8f9ada8049ff055a67dd87115b376bba6c0b13046b9eefd14d0` | 380-395 actual per-rank offset/count; 510-545 gather/scatter mapping and zeroed unused output |
| `.repro-private/baseline-recovery/formal-source-audit/sglang/srt/layers/moe/fused_moe_triton/layer.py` | `096364a11cfad23f98775445631aa3d6c504ba648d3f3baaf55f939441237a0d` | 168-180 local experts and expert TP partition |

In the new main checkout, `Parallel_Strategy_Manager.py:132-237` constructs
prefill with `phase="prefill"` and weight-copy tasks for attention, shared
experts and every routed expert. `:293-341` constructs decode, with `256/16`
resident experts per rank for world size >8. `modeling_deepseek_v3.py:4187-4190`
selects separate prefill/decode MoE classes. Its decode `:1912-1931` finishes
routed EP inference then adds `self.shared_experts(identity)` on local input.
The new main file identities are recorded in `MAIN_NATIVE_KERNEL_PLAN.md`.

## Minimal SG local-shared algorithm

For local normalized tokens `X_r`, shared MLP `S`, and routed contributions
`E_j`, the required local MoE result is:

```text
Y_r = slice_r(SUM_j E_j(gather(X))) + S(X_r)
```

Stock SG's no-A2A path instead has TP-sharded shared weights and sums their
partial shared results together with routed results. Merely forcing full
shared weights at TP1 computes `SUM_j [E_j(X) + S(X)]`, giving 16 shared
contributions. Dividing by 16 changes compute/dataflow and can change rounding;
it is not the selected solution.

The candidate changes only the SG model path and a separate helper:

1. Construct shared gate/up/down projections with `tp_rank=0, tp_size=1`.
2. After native `prepare_mlp` has performed norm and DP gather, use native
   `dp_scatter` into a new residual-shaped local buffer. This preserves SG's
   own MAX_LEN/SUM_LEN mapping and avoids hand-computed offsets or buffer aliasing.
3. Run native routed MoE with its shared branch suppressed. Bypass dual-stream
   and shared-in-SBO dispatch. Preserve original router, routing scale, expert
   computation and routed SUM. Require all-reduce followed by local scatter;
   the candidate disables the communicator's reduce-scatter option for these
   sparse layers and rejects unexpected fusion/reduce-scatter dispatch.
4. After `postprocess_layer` has returned local routed output, call TP1 shared
   on the saved local normalized input and add it once. Leave the residual
   unchanged. Shared results never enter the global SUM.

This preserves the mathematical branch sum and BG's routed-then-shared
addition ordering. Native FP8 numerical equivalence, real padding behavior,
actual collective groups, checkpoint loading and distributed liveness still
require qualification; CPU mocks cannot establish them.

No `communicator.py` or BG core file is changed. The bounded candidate rejects
DeepEP/Mooncake, TP/EP other than16, attention TP other than1, fused shared
experts, graph/piecewise graph, pipeline/speculative execution, SBO/TBO and
FlashInfer all-reduce fusion. This keeps the initial correctness surface small;
enabling those paths requires separate tests and an explicit condition change.

## Common ordinary-linear FP8 hook

`sglang_r1_alignment_runtime.py` installs its hook before `DeepseekV2Model`
construction. It wraps `Fp8LinearMethod.__init__` and binds serialized FP8
block128 linear instances to
`batchgen.gemm.r1_sm103_compat.w8a16_gemm(W_fp8, W_scale_fp32, X_bf16)`.
Weights/scales pass by identity. The common module performs the same native
main BF16 activation quantization and default W8A8 GEMM used by the BG adapter.
SG's own activation quantizer is bypassed for these linears, so a shared
"block128" label is not being substituted for actual formula equality.

Required controls are `SGLANG_ENABLE_JIT_DEEPGEMM=0` and
`SGLANG_ENABLE_FLASHINFER_FP8_GEMM=0`. The hook verifies effective DeepGEMM and
UE8M0 flags are false; this also makes SG's loader requant helper condition
false. It rejects Marlin and non-serialized FP8 block methods, and leaves
unquantized/BF16 methods alone. It does not secretly quantize BF16 checkpoint
weights. Before/after loader hashes remain necessary evidence.

Covered: attention projections implemented through `Fp8LinearMethod`, the
first dense MLPs, and the TP1 shared MLP. Excluded: absorbed attention BMM,
the attention operator, routed grouped MoE GEMMs, non-block128 or biased
linears, and an alternative quantization format. SG may merge gate/up or
Q/KV projections while BG calls separate matrices, so actual GEMM shapes,
FP8 quantized inputs, scale rows and activation outputs must be compared.
`linear_backend_receipt()` reports installation and completed calls; mere
installation is not execution evidence.

## Native SG FlashInfer: allocator64, kernel token-pages1

The official SG0.5.5post3 FI-MLA source has SHA256
`f24df800a605d1d349d1d11c1a78921ace252778a2a10e980492d5fb83abad2d`.
Its SM100 ragged default is `cutlass` and its MLA wrapper default is `auto`;
there is no suitable CLI guarantee of the common explicit `fa2` implementation.
`sglang_r1_attention_compat.py` therefore forces the **native SG wrapper
factories** to `backend="fa2"`, verifies the retained backend after planning,
and observes actual plan/run dimensions, BF16 dtypes and softmax scales. It
does not route SG through BG's page64 reshaping shim or alter any cache bytes.

This SG backend source does not assert allocator page1. It retains
`model_runner.page_size`, while its ordinary non-speculative FI decode plan
always uses kernel page1 and token addresses from `req_to_token`. Preserved
`model_runner.py:1777-1784,1880-1907` (SHA256
`c16d0d2b133e0e1efd7d338034a7908e662bbadab1dcef418418c08c28910c45`)
constructs MLAPool with the allocator's page size and selects the paged
allocator when that size exceeds1. Preserved `memory_pool.py:1423-1431,1482`
(SHA256 `e5f75d3b14b1c375da839894abf3e6fd6b61a65ccec0ad3dab2dc4844a4a32fb`)
stores BF16 MLA as `[size+page_size,1,576]` token-major rows. Thus allocator64
with FI kernel token-pages1 is a source-supported candidate; allocator64 must
not be mislabeled allocator1 simply because the kernel plan uses1.

The helper now requires allocator64 and kernel page1. No page-size assertion
is removed. Its native allocation qualification remains unverified until the
coordinator runs `qualify_sglang_r1_page64_allocator.py` in a bounded exclusive
single-SM103 job. That script uses the **real** SG paged allocator, MLAPool,
req-to-token index kernel and FI0.5.2 fa2 operator. It covers lengths63/64/65,
2047/2048/2049,8191/8192/8193, decode append across a boundary, ownership,
free/reuse, valid-KV immutability and NaN-tail versus zero-tail output parity.
It freezes a primitive-only uniform-attention oracle bound of one BF16 ULP
per output against FP64, and hashes the full cache before/after each native
run. Thus a finite constant wrong output or an unused-tail write cannot pass
merely because two poisoned/clean runs agree.
Its default only prints a plan; GPU execution has not been performed here.

Common fresh prefill additionally requires radix/prefix caching disabled,
chunked prefill disabled (`-1`), ragged selection enabled, and Graph OFF.
The wrapper verifies ragged NHD Q/K192, V128 with128 heads; decode verifies
latent512, RoPE64, kernel page1, BF16 and single-query noncausal planning.
Native SG metadata/cache/index algorithms remain in use. The receipt includes
native source identities, planned `_cached_module` registered-op details and
loaded FlashInfer shared-object hashes when observable. Missing binary/module
identity or missing actual run observations keeps kernel equivalence
unverified; a common wrapper name alone is insufficient.

`--chunked-prefill-size -1` does not itself enforce the BG prefill budget8192.
That separate cap must limit the **sum of valid tokens in a scheduled prefill
microbatch** to8192, rather than allowing many8192-token prompts in one batch.
The coordinator must audit SG's actual prefill admission budget separately.

## Planned settings and actual layout receipts

The following is a fragment, not a complete launch command or a runtime result.
The attention backend, page/context/admission budgets and numerical criteria
must come from the coordinator's current common contract.

```text
SGLANG_R1_LOCAL_SHARED_TP1=1
BATCHGEN_R1_SM103_FP8_COMPAT=1
DISABLE_MMA_V5=1
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_ENABLE_FLASHINFER_FP8_GEMM=0
SGLANG_R1_COMMON_FI_FA2=1

--tp-size 16 --dp-size 16 --enable-dp-attention --ep-size 16
--moe-a2a-backend none --moe-runner-backend triton
--moe-dense-tp-size 1 --enable-dp-lm-head
--disable-shared-experts-fusion --disable-cuda-graph --disable-radix-cache
--disable-overlap-schedule
--dtype bfloat16 --quantization fp8 --kv-cache-dtype auto
--attention-backend flashinfer --page-size 64 --chunked-prefill-size -1
--disable-chunked-prefix-cache
```

Keep `enable_two_batch_overlap`, `enable_single_batch_overlap`,
`enable_piecewise_cuda_graph`, `enable_flashinfer_allreduce_fusion` false,
`speculative_algorithm=None`, `pp_size=1`. The internal
`allow_reduce_scatter=False` on adapted sparse layers is part of the source
candidate, not an existing CLI flag.

| Module / phase | Planned layout and shape | Required actual evidence |
|---|---|---|
| Attention, both phases | Attention TP1; 16 DP owners | Actual group members, projection shards, owned/valid/padded token counts |
| Dense layers 0-2 | Full local dense MLP, TP1 | Gate/up/down logical shapes based on `config.intermediate_size`; no dense TP SUM |
| Shared MLP, sparse layers | TP1 full weights on local input; shared gate+up `[4096,7168]`, down `[7168,2048]`; corresponding FP32 scales `[32,56]` and `[56,16]` | Projection `tp_rank/tp_size`, actual weight/scale logical-byte identities, common backend call, shared input row count; shared absent from routed SUM |
| Routed decode | EP16, expert TP1; IDs `[16*r,16*r+16)`, full intermediate2048 | All16 rank IDs/offsets, w13 `[16,4096,7168]` and w2 `[16,7168,2048]` logical shapes, original FP32 scales |
| Embedding / LM head | Full local weights via DP attention / DP LM head | Actual weight rows/padding, group selection and logits ownership |
| BG routed prefill | Pure DP with full256 logical expert range and streamed weights | Actual selected prefill class, weight-copy tasks, residency/transfer bytes/timing |
| SG routed prefill | Resident EP16, routed expert TP1 | Actual expert partition and collectives; explicitly different from BG phase mechanism |
| Sparse communication | SG normalize/gather -> routed SUM -> local scatter -> local shared add | Trace operation order, shapes/dtypes/group members; valid/padded offsets and absence of duplicated shared SUM |
| KV allocation / attention | Both allocator64; both common FI decode kernel page1; SG native `[tokens,1,576]`, BG page64 token view | Actual native SG allocator boundary/free/reuse pass, request-to-token index mapping, native FI source/compiled identity, head dimensions/dtypes/scale and tail parity |

Do not replace actual receipts with the expected shapes in this table. CPU
recovery, a source hash, an HTTP200, or a single operator result cannot mark
all these rows aligned.

## Concrete preparation and remaining qualification

Default invocation below prints the exact source diff and identities without
writes. `--execute --output-dir <new-absolute-directory>` writes the original,
candidate, helper, diff and receipt into an exclusive directory; it does not
replace installed SG or start a worker:

```text
python3 scripts/r1_reproduction/prepare_sglang_r1_local_shared.py \
  --source <stock-sg-site-packages>/sglang/srt/models/deepseek_v2.py
```

The generator accepts only the exact stock hash above, and refuses already
patched/unknown sources. Any additional necessary compatibility patch needs
an explicit composed diff and new source receipt; do not bypass the identity
check. The deployed workspace must expose both the runtime helper's
`scripts.r1_reproduction` namespace and the common `batchgen.gemm` module to
every worker, then record their imported paths and hashes.

CPU tests exercise the generated decoder's ordering with actual helper code,
an independent nonlinear FP64 SwiGLU oracle, 16 owners, equal/unequal/empty
ranks, two abstract padding mappings and poisoned unused rows; they also
check duplicate-SUM failure, unsafe fusion rejection, original source
immutability, hook argument identity and refusal of hidden requantization.
They use mock communication and do not execute SG's CUDA `dp_scatter`.

Next, after environment/kernel preparation, run an isolated distributed layer
qualification with real SG scatter/gather and original weights: separate
routed/shared outputs, all16 owners, variable and empty ranks, routing skew,
valid and poisoned unused tails, SUM_LEN and MAX_LEN. Verify collective
liveness and unchanged input/weight/scale bytes. Compare stock shared-TP16
mathematics against the adapted TP1 result and an independent reference;
retain expected reduction/quantization error rather than claiming bitwise
identity across different reduction orders. Qualify each projection and
SiLU branch under the common FP8 contract before teacher-forced logits.
Formal performance remains gated on those results and the other current
attention, capacity, runtime-parameter and measurement conditions.
