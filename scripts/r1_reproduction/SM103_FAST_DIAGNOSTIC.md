# Explicit SM103 fast full-model diagnostic

The R1 launcher defaults to `triton_precision`. To collect full-model diagnostic
outputs with FA4 prefill and FlashInfer MLA decode, select
`--attention-implementation fa4_flashinfer`. The plan then explicitly sets all
three opt-ins:

```
BATCHGEN_R1_SM103_EXPERIMENTAL=1
BATCHGEN_R1_SM103_ATTENTION=fa4_flashinfer
BATCHGEN_R1_SM103_FAST_DIAGNOSTIC=1
```

This selection does not qualify model accuracy or performance. The same-input
attention-only diagnostics retain the original strict gate (`atol=1e-5`,
`rtol=0.016`, failure fraction below `1e-4`). Precision passed the tested rows.
FA4 prefill at lengths 63/64/65 failed with fraction about 0.01897; FlashInfer
decode at 2047/2048/2049 failed at about 0.03200, and at 8191/8192/8193 at about
0.03746. The fast candidates were finite and byte-identical across repeats for
zero and finite padding. With deliberately poisoned NaN padding, FlashInfer
produced nonfinite outputs in both nonaligned decode rows at each length family;
precision remained finite. These failures remain unresolved and preserved.

Preflight still rejects CUDA graphs and prefix-cache reuse. It verifies the
actual FA4 and FlashInfer entry points, pins FA4 4.0.0b36, FlashInfer Python/cubin
0.6.13 and Cutlass DSL 4.8.1, and retains original FP8 weights, FP32 block scales,
CUDA 13 and native-library checks. There is no implicit backend fallback. This
option changes only attention selection; topology, host KV capacity and the
normal server command remain the same.

The diagnostic launcher supervises its directly owned HTTP child with WNOWAIT.
A stop request first signals that leader so normal worker and native-memory
release can run for 300 seconds. Any necessary escalation targets only the
primary process group while its original leader is still unreaped. Other
groups are audited and explicitly require operator cleanup if still live.
The structured audit is saved before reaping; no numeric signal follows reap.

Use an independently reviewed clean BatchGen environment when launching: the
launcher retains inherited variables outside its explicit environment overrides.
Keep skeleton mmap off for the first full-model probe, record native completion
IDs with `BATCHGEN_R1_NATIVE_TOKEN_TRACE=1`, and inspect the actual run's rank
coverage and padding assumptions. Neither a successful startup nor finite text
is evidence that the numerical or model-quality gates passed. Diagnostic timings
are not the formal 6000-request BatchGen/SGLang comparison.

## Full-model observations, 2026-10-24

`r1-fast-diagnostic-v3` reached generation on both allocated eight-GPU nodes
with the original DeepSeek-R1 671B FP8 checkpoint. It retains TP1/EP16,
DP-attention16, BF16 KV, 64 GiB host KV per node, skeleton mmap disabled,
and graphs/prefix reuse disabled. The container reports SM103 and the NVML
name `NVIDIA L20D`; the requested platform is two B300 servers. These are
diagnostics on this hardware, not measurements on the paper's H20/H200 setup.

The launch uses `NCCL_RAS_ENABLE=0` and a coherent private CUDA 13.0.88
compiler/CRT/NVVM/runtime with explicit libcudart preload. This resolved the
previous full-model attempt's FlashInfer `fmha_gen` compiler/header conflict.
The per-node compiled binaries have different hashes and are individually
pinned. Before/after snapshots cover the same 16 worker identities and show
the expected runtime path and, after decode, each node's expected fmha path.
Container overlay device-number differences and denied `map_files` access
prevent a stronger mapped-file identity proof; the original strict audit
failure remains recorded.

| Frozen probe | Observed result | Scope |
| --- | --- | --- |
| First 16 identical arithmetic requests, cap 128 | Complete native traces on ranks 0–15, one request each; all 16 nonempty and length-capped; 2,048 generated tokens | First token 33001 agrees; all 16 full native sequences differ; pairwise common prefixes span 14–78 tokens |
| Six basic chat questions, two copies each, cap 2048 | All 12 stop naturally, have a nonempty answer after `</think>`, and match API/native token counts; 11,532 generated tokens | Manual review finds all 12 final answers correct for arithmetic, algebra, Chinese ratio, recall, Python arithmetic, and elementary logic |

The second probe's actual rank coverage is 0–11. Its two copies share a mixed
batch and have different rank/history conditions; this is not an isolated
determinism test. All six pairs have different native sequences. Neither
these 12 basic answers nor collection success establishes general model
quality. The 128-token cap, rather than a demonstrated wrong final answer,
prevented the first probe from answering its question. The client elapsed
times include phase transitions and prescribed warmup was absent; they are
not the formal BCT/speedup result.

Preserved client hashes:

- First16 `run.json`: `0acf84f7bc28dcc204f39dc55b11a84468d4b224072854fe15c689ea0e554b7d`.
- Longchat `run.json`: `aed93ab8a0b26f46b39a5df9e58ac5e21a7c78f21c8a16f63038483c31739103`.
- Longchat native/answer observation: `9c3190315673481d9e903b9c90f43916d5d24aad4df5ee36ebb9d993ab4c59c7`.

The strict attention failures above remain unresolved. A CPU-only replay of
the captured decode tensors reproduced the FlashInfer 6,291/196,608 failing
elements with both FP32 and FP64 auxiliary references. Of those failures,
6,236 have reference magnitude below 0.1; this locates much of the error near
zero but does not change the original threshold or turn it into a pass.
The corresponding Triton output agrees bitwise with rounded FP64 in
196,607/196,608 elements. Numerical gates and capacity-tuned SGLang A/B
performance measurements remain separate work.

The subsequent64-item paired clipped-context QA collection is now frozen; see
[QA_SANITY.md](QA_SANITY.md). BatchGen returned all64 items, with63 validated
and one repetition termination classified as a failure by the unchanged client.
There were11 empty outputs and7 length-capped outputs. All64 native traces
match the corresponding API counts and finish reasons. The original failed
guardian and absent remote score remain preserved alongside an explicitly
local-only score recomputation. These observations do not qualify quality or
performance. Both owned v3 services were then stopped normally and all16 GPUs
were independently observed idle before the next experiment.

A subsequent isolated SGLang replay used the **same captured BF16 query, cache,
page table, sequence lengths2047/2048/2049 and original R1 YaRN scale**. The only
cache-layout adaptation was checked losslessly. SGLang0.5.5.post3's native
FlashInfer0.5.2 TRTLLM MLA decode produced outputs bitwise equal to the archived
BatchGen FlashInfer0.6.13 outputs in all196,608 elements, for both repeats.
The SGLang runtime and fmha library paths were observed, and its original cubin
loader returned two byte buffers whose hashes matched the requested hashes.
No model weights, new generation requests or BatchGen runtime preload were used.

Both outputs still fail the unchanged strict FP32-reference gate in
6,291/196,608 elements; normalized RMS error to that unrounded reference is
0.0023038945. These are different metrics: the3.199768% value is the fraction
of elements exceeding the envelope, not a3.2% relative error. The same-input
result establishes that this tested decode discrepancy is shared with the
baseline, rather than introduced uniquely by BatchGen's FlashInfer version.
It does not qualify prefill, other sequence lengths, NaN padding, whole-model
quality or performance. The original gate failure and diagnostic exit2 remain.
Replay report SHA256:
`97deaa37a130afb6d7157cfd1b8a8e2e91712ea784ac8ef2d382b80cf9e2a9af`.

## v4 source inventory and core-engine provenance

The frozen `r1-fast-diagnostic-v4` ready1 records contain437 Python-source
entries on node0 and432 on node1. All432 shared path/hash entries agree; their
sorted compact JSON SHA256 is
`3c6d447a03c8588046e03114738a01943eb5d3c983368ec1c69b5a12cf2c9f78`.
The whole source trees are **not identical**. Node0 alone contributes four
`batchgen/op_builder/*.py` inventory entries and Kimi Linear's `mla_triton.py`.
The launcher uses `batchgen.rglob('*.py')`: node0's op_builder is a directory,
whereas node1's is a symlink to the root op_builder. Resolved init, all_ops and
core_engine Python sources match. The Kimi addition is outside the selected
DeepSeek-R1 registry/parameter-server path.

The two actual builder sources differ. Node0 SHA256 is
`5feb4ed24eb4b0f76f1d850b549a4c478efaa890e4e7493229e63994c9a8277d`;
node1 SHA256 is
`a4ce53f9dec21035508f5b2f6a469c660878a3df7ed0bf4da06b90e2bce9a5be`.
Their saved diff changes only `compute_capability_args`: numeric architecture
sorting, parsing multi-digit major versions such as10.x, and using that major
for BF16 eligibility. These are real compilation-flag differences; this audit
does not establish equivalent rebuilds. The shared, pinned engine_loader
imports the AOT `batchgen.core_engine` and rejects implicit JIT. No running
model source or binary was changed to resolve this inventory check.

Both core-engine files read at the frozen observation are271,668,832 bytes with SHA256
`f0f0e5af58fb7c6f404e1e390b626556c5355497179b18af133bc729ed8a6a29`.
Historical owned-process snapshots observe this pathname mapped in9 of10
processes on each node. The recorded identities are:

| Node | Path stat `(major,minor,inode)` | `/proc/maps` `(major,minor,inode)` | Backing-byte access |
| --- | --- | --- | --- |
|0|`(0,336,56263687)`|`(259,17,56263687)`|`map_files` readlink matches the pathname; stat/hash returns EPERM|
|1|`(0,261,48413633)`|`(259,9,48413633)`|`map_files` readlink matches the pathname; stat/hash returns EPERM|

The inode agrees within each node but the device identities differ. Therefore
the on-disk SHA does **not** prove equality of the inaccessible mapped backing
bytes, and no loaded-byte SHA is claimed. The snapshots have Unix timestamps
1792792847.4684129/node0 and1792792847.278983/node1; they are historical
observations, not a statement about any future cohort or current service state.

The new metadata validator permits exactly the frozen432-entry common corpus
and those five node0-only path/hash pairs, while retaining both original full
maps with `whole_trees_identical=false`. Unknown extras, removals, node reversal
or changes to the common corpus remain rejected. This explicit diagnostic
provenance exception does not qualify model accuracy, performance, future
rebuilds or unseen binaries; all earlier numerical and quality failures remain
unchanged. Source review SHA256:
`62111f61adb5708ddc2670091d823c67569c62280a43750ac0fcdadaf96af916`.
Mapped evidence node0/node1 SHA256:
`79f49d01de092f8a4c126d23b756cd1c1b1fedcce2613a52597b724020da43ec` /
`f6aca55f85879bbca73ba185f299763324ef9c951c45cef331bcae9710871ee7`.


The first v4 fixed-length pilot was admitted after new health/empty-queue
observations on both nodes at2026-10-23 22:01:12 UTC. Its32 requests use the
unchanged first32 input8192/output2048 payload, one separate excluded warmup,
traceOFF and the original no-retry batch client. The CPU guard launched at
22:03:21 UTC with PID1543390/start33490563; its child is
PID1543467/start33491338. Admission SHA256 is
`9600fe231365337a23e2d6cb8936d63af4663c5c67360c2245c7515ac78d045d`;
actual launch SHA256 is
`d235ca9274065f05f44ed96db0744f4e05c23fe8c73b947804e382a688c1f361`.
At the saved22:07:57 UTC observation, the excluded warmup remained active
with no guard violation. There is no completed BG BCT/throughput result from
this pilot at that observation. Single-request decode log rates do not
establish32/512/6000-request throughput or paper speedup.
