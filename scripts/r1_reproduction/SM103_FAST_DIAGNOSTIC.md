# Explicit SM103 fast full-model diagnostic

## Latest decode synchronization and graph investigation — 2026-10-25

Both nodes now serve `r1-fast-diagnostic-v19` at runtime `b9cf8a5cb988c48a809455b1d163ff120d015033`. Both fixed 512 profiles completed and were independently imported and verified; fresh final observations show both services healthy with zero active batches or trajectories. The runtime common source map contains 435 checked files, with the existing five node0-only helper paths kept explicit.

**Historical strict attention numerical and NaN-tail failures remain unresolved. This performance diagnostic does not qualify full-model numerical or semantic quality.**

Hardware, TP1/EP16/DP-attention16 layout, host KV 192 GiB/node, attention backend, FP8 checkpoint, existing GEMM tiles and framework graph setting remain unchanged. The three new implementation opt-ins enable GPU row counts, local routed-MoE graphs and local-only decode expert skeletons. Both versions use the same first 512 payloads, temperature 0, ignored EOS, no inference retries, one excluded warmup per profile and short-then-long order. Each configuration/profile has one observation. SGLang was not rerun in this follow-up.

| Fixed input / output tokens | Previous BG v17 (s) | BG v19 (s) | Less wall time | Output tokens/s v17 → v19 |
|---|---:|---:|---:|---:|
| 8192 / 2048 | 500.590 | 445.436 | 11.0% | 2094.679 → 2354.045 |
| 2048 / 8192 | 1222.817 | 978.312 | 20.0% | 3430.034 → 4287.288 |

All 1024 measured returned texts, usage objects and finish reasons match v17. Each profile has 512 successes, one prefill wave, exact forced output lengths and zero recomputed admissions. Native upload-to-complete-retrieval is the BCT boundary in both BG runs; no TTFT or TPOT is inferred.

### Decode and phase boundaries

| Native measurement | Short v17 → v19 | Long v17 → v19 |
|---|---:|---:|
| Weighted decode step (ms) | 138.351 → 113.721 | 136.493 → 107.319 |
| Rank0 decode window (s) | 286.402 → 234.367 | 1120.829 → 881.983 |
| Rank0 other wall window (s) | 214.188 → 211.069 | 101.988 → 96.329 |
| Max 16-rank prefill KV-ready (s) | 157.953 → 166.389 | 42.412 → 42.056 |
| Rank0 two switch busy timers (s) | 62.981 → 59.466 | 60.756 → 53.788 |

Prefill was not optimized: the short-profile 16-rank KV-ready maximum increased from 157.953 to 166.389 seconds; the long-profile value was nearly unchanged. The wall-window partition is separate from native prefill and switch timers. Switch busy timers and prefill KV-ready boundaries overlap other classifications and cannot be added to BCT. The short profile still pays a large streamed-prefill/phase cost. Phase instances are rebuilt at each switch. This change removes unused expert construction and scale wrappers.

### Actual graph coverage and local operator qualification

In each measured profile, the native logs show 58 first captures on every one of the 16 ranks at 512 global padded tokens: 928 captures per profile, without repeated capture inside the measurement window. The final imported node-log prefixes independently retain the exact bytes used for this check. All matched complete decode samples use the local-graph replay path.

GPU row counts keep the complete worst-case dispatch capacity and pass a GPU valid-prefix scalar to both existing quantizers. A masked inverse map excludes invalid activation/index tails. This removes the two routed-MoE host scalar readbacks per layer (116 per rank/step on the active baseline path). Gating, dispatch, expert math and weighted scatter are captured together; all-gather and all-reduce remain outside, on the caller stream. Other worker/control synchronization is not claimed removed.

The real checkpoint layer 3 test passed native routing at global token counts 16/256/512/1024, worst-case local skew and empty local experts, with bitwise outputs. Dynamic inputs/routes, retained outputs, tracked source mutations/replacement, stream isolation, poisoned tails and conservative unversioned-weight fallback were checked. The profiler observed zero local scalar readbacks in three actual graph replays per case. Registered weight layouts stay fixed for the lifetime of a decode instance.

| Real-weight local operator, native routing | Dynamic slice (ms/call) | GPU counts eager (ms/call) | Local graph (ms/call) | Dynamic / graph |
|---|---:|---:|---:|---:|
| 16 global tokens | 0.457 | 0.376 | 0.180 | 2.53x |
| 256 global tokens | 0.618 | 0.466 | 0.368 | 1.68x |
| 512 global tokens | 0.614 | 0.461 | 0.426 | 1.44x |
| 1024 global tokens | 0.738 | 0.596 | 0.563 | 1.31x |

These are paired local-operator timings with real weights and synthetic activations, not full-model speedups. The passing worker recorded actual captures and replays. Earlier noncontiguous-fixture and eager-fallback attempts are excluded from graph qualification and timing conclusions.

The decode skeleton creates 16 local MLPs instead of 256 per MoE layer, preserving global expert indices and all local parameter contracts. Nonlocal wrappers and their GPU scale copies are skipped. Boundary-rank constructor checks passed; the full phase effect is reported in the native switch timers above.

### Matched sampled components

Rank0 medians below use the same matching call numbers. Every row aggregates that stage across all model layers in one sampled step. Stream intervals include host launch gaps and peer waits; CPU/stream time, nested rows and ranks must not be added.

| Observed stage (ms/step) | Short v17 → v19 | Long v17 → v19 |
|---|---:|---:|
| Attention, stream | 55.49 → 33.74 | 52.20 → 32.78 |
| Routed MoE parent, stream | 120.75 → 69.89 | 99.13 → 81.89 |
| Shared expert, host call | 7.07 → 5.44 | 6.97 → 6.23 |
| All-gather + all-reduce, stream including waits | 59.98 → 35.79 | 38.54 → 47.16 |
| Local MoE graph span, v19 only | 26.67 | 26.14 |

The attention implementation is unchanged; its lower inclusive stream median is not an isolated attention-kernel speedup. The long-profile collective median increased from 38.54 to 47.16 ms despite the lower BCT, so no communication speedup is inferred from these samples. Individual gate/GEMM/readback Python spans are not observable inside replay; missing inner timings are not zeros. The analyzer rejects partial coverage, duplicate layers and mixed eager-fallback samples. The same sampling call policy is retained, but graph replay changes probe counts. No trace-off experiment isolates instrumentation overhead. Complete-run weighted steps and BCT remain the main comparison.

Host RDMA steady-window coverage and per-host rates remain in the JSON. These host counters and inclusive collective spans do not provide a pure wire-time or congestion attribution. No NCCL/NVLS or parallel-layout setting was changed.

CPU checks passed 19 tests and 4 subtests. All formal CPU clients were independently closed with no inference retry. Returned-field consistency does not establish native token-ID identity or full-model quality. The experiment measures the fixed first 512 prefix, not 6000 requests.

Actual all-rank data, GPU qualification, immutable source/capture evidence and final service state: [SM103_DECODE_GRAPH_RESULTS_20261025.json](SM103_DECODE_GRAPH_RESULTS_20261025.json).
Public result SHA256: `3b262bf6944428fa857900bb96d92224c289897ed523861a1dc1c5ffce60a73e`.

## Historical v16/v17 component investigation — 2026-10-25

The optimized runtime `de7f6f0a117951a0f53ed81bfc1a6dbe65466c28` was restored on both nodes as `r1-fast-diagnostic-v18` and later retired for v19. Its final observations showed both services healthy and idle after a completed native 8-output-token smoke. The measured optimized cohort is v17; v18 uses the same runtime bytes, flags and KV192 setting. The SGLang A v6 reference completed both 512 profiles and was retired before BG restoration.

**This remains a fast diagnostic: historical strict attention numerical and NaN-tail gates are failed and unresolved. Operator parity and returned-text consistency do not qualify full-model quality.**

All paired BG observations retain the same component sampling policy, first 512 requests, temperature 0, ignored EOS, no retries, one excluded warmup per profile and short-then-long order. All 16 physical device UUIDs match across the experiments. Each configuration/profile has one observation. BG BCT includes native upload through full result retrieval; SG BCT includes concurrent HTTP submission through all complete bodies. No TTFT/TPOT comparison is inferred.

| Fixed512 profile | BG sampled baseline v16 (s) | BG optimized v17 (s) | Less BG wall | BG output tokens/s | Fresh SG A v6 (s) | BG / SG wall |
|---|---:|---:|---:|---:|---:|---:|
| 8192 input /2048 output | 538.696 | 500.590 | 7.1% | 1946.506 → 2094.679 | 238.004 | 2.103x |
| 2048 input /8192 output | 1371.160 | 1222.817 | 10.8% | 3058.945 → 3430.034 | 472.929 | 2.586x |

Both optimized BG profiles have 512 successes, exact forced lengths, one prefill wave and zero recomputed admissions. All 1024 measured returned texts, usage objects and finish reasons match the sampled baseline. V15 without probes was 535.621/1344.096s; v16 was 538.696/1371.160s. Those single-run differences do not isolate probe overhead. The v16→v17 pair keeps probe policy equal.

### Prefill and model switching

| Native measurement (seconds) | Short v16 → v17 | Long v16 → v17 |
|---|---:|---:|
| Max 16-rank prefill KV-ready | 157.982 → 157.953 | 42.605 → 42.412 |
| Rank 0 two switch busy timers | 63.861 → 62.981 | 58.282 → 60.756 |
| Rank 0 other wall window | 214.866 → 214.188 | 100.213 → 101.988 |

The optimized complete-run decode windows are 286.402s (short) and 1120.829s (long); the corresponding other wall windows are 214.188s and 101.988s, or 42.8% and 8.3% of BCT. Almost all observed end-to-end reduction comes from the decode window, while the prefill/phase path remains a sizable short-profile cost. All 16 ranks and all 512 sequences are covered by native prefill metrics. Switch busy timers and prefill boundaries overlap wall-window classifications and must not be added to BCT. Pure-DP prefill still streams weights and phase changes rebuild models; these paths were not changed. No prefill MoE/attention subkernel split or host-transfer-byte estimate is claimed: the one/two prefill forwards did not reach the sampled-module cadence.

### Matched decode calls

The table shows rank 0 medians of complete all-layer samples at matching call numbers: 6 short and 11 long. Each row aggregates that stage across the model in one sampled step. Event intervals include host launch gaps and peer waiting; they are not occupied-SM/kernel time. Nested rows, ranks and CPU/GPU intervals must not be added. All 16 ranks are retained in the JSON.

| Sampled stage (ms/step) | Short v16 → v17 | Long v16 → v17 |
|---|---:|---:|
| Attention, stream | 56.75 → 55.49 | 53.17 → 52.20 |
| Routed MoE including collectives, stream | 112.33 → 120.75 | 113.10 → 99.13 |
| Gate/up GEMM, stream | 18.60 → 13.88 | 18.52 → 13.79 |
| Down GEMM, stream | 11.95 → 6.26 | 11.96 → 6.16 |
| Shared expert, host call | 22.56 → 7.07 | 22.09 → 6.97 |
| Token-count readback, host call | 25.33 → 36.29 | 22.65 → 17.53 |
| All-gather + all-reduce, stream including waits | 42.56 → 59.98 | 42.43 → 38.54 |

The optimized short sampled MoE parent median increased, while its GEMMs and shared-expert host calls improved. Across all 16 ranks, the long-profile median of rank medians fell 12.25→6.16ms for down GEMM, 18.42→13.95ms for gate/up and 21.91→6.74ms for shared-expert host calls; short-profile MoE parent rank medians rose 114.05→117.20ms. This prevents interpreting only rank 0 or improved subkernels as a universal parent-span improvement. Inclusive collective spans varied 29.34–98.18ms in the short samples and 24.68–75.33ms in the long samples. Sparse samples are not an unbiased substitute for all-step timing. Weighted complete-run decode steps fell 157.401→138.351ms and 154.821→136.493ms. A scalar readback waits on preceding queued work; its host time is not additional independent compute.

The implementation changes are explicit SM103 opt-ins: independent expert/N tiles for down projection, bitwise-qualified gate/up 32x64x128 and down 16x64x128 tiles, and a per-caller-stream shared-expert CUDA graph with source/version validation and cloned outputs. Default paths remain unchanged. Real checkpoint layer 3 operator cases, dynamic graph inputs, retained outputs, weight/scale mutation, stream isolation and inactive poisoned tails were checked. The new down expert-axis kernel retains its original K loop and row masks; the tile variants were checked for bitwise outputs in the stated fixtures. These are narrower checks than full-model quality.

### Isolated communication and configuration separation

The valid isolated worker uses BG's actual merged native launch environment, exact `_setup_nccl_env` source, CPU StatelessProcessGroup/TCPStore bootstrap and PyNccl API. It uses the same 8 RoCE rails, GID 3,16MiB NCCL buffer, RAS 0 and automatic algorithms. Both NVLS settings produced 16 exact-validation reports and closed all GPU workers. Two earlier fixture attempts with mismatched bootstrap/environment produced zero usable timings and are excluded.

At local batch 32 / world 16, all-gather input is 458752 bytes and output/all-reduce tensor is 7340032 bytes (BF16, hidden 7168). The table uses the median across rank medians.

| NVLS setting | Queued all-gather (ms/op) | Queued all-reduce (ms/op) | Single-call all-gather (ms/op) | Single-call all-reduce (ms/op) |
|---|---:|---:|---:|---:|
| 0 | 0.100 | 0.149 | 0.316 | 0.352 |
| 1 | 0.130 | 0.146 | 0.287 | 0.306 |

Queued measurements average 200 operations with reused buffers; single calls include CPU barrier/launch alignment effects. They are isolated API references, not an exact wire-time oracle, and must not be subtracted from model spans to label the remainder as CPU waiting. NVLS 1 does not provide a consistent improvement at the formal batch 32 message sizes, and it was not enabled in the restored model. Fresh SG server info reports enable_nccl_nvls=false and enable_symm_mem=false; installed source sets NVLS 0 on this path. NVLS-off is therefore shared rather than an established explanation for the gap.

Host RDMA data counters remain a separate observation. Their four-byte units follow the [Linux InfiniBand sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband). Only complete sampled intervals inside steady decode windows are used. Counters cover host traffic, and interval averages cannot exclude bursts, latency, congestion or rank-arrival skew.

| Optimized steady-window host counters | Covered seconds node 0 / node 1 | TX GB/s node 0 / node 1 | RX GB/s node 0 / node 1 | Max single-port interval direction / line rate |
|---|---:|---:|---:|---:|
| input8192-output2048 | 98.1 / 98.1 | 6.84 / 6.83 | 6.85 / 6.83 | 3.55% |
| input2048-output8192 | 881.6 / 881.7 | 6.71 / 6.71 | 6.71 / 6.71 | 3.62% |

These interval averages do not establish sustained link-bandwidth saturation; they also do not rule out short congestion or rank-arrival delays.

| Framework setting/strategy | BG measured/restored | Fresh SG A reference |
|---|---|---|
| Parallel layout | TP1 / EP16 / DP-attention16 | TP16 / EP1 / DP16 with DP attention |
| Prefill / phase strategy | Pure DP streamed weights, native microbatch up to 131072 tokens/rank, phase model reconstruction | GPU-resident TP/DP; requested global chunk 32768 becomes 2048/rank under DP attention |
| Graph configuration | Framework graphs off; local projection and shared-expert graphs on | Whole-model graphs configured on, max batch 128 |
| Native kernels | FA4 prefill + FlashInfer MLA decode, portable Triton FP8 projections/grouped MoE | trtllm_mla + flashinfer_trtllm MoE |
| Capacity setting | Host KV 192 GiB/node, no observed512 eviction | GPU memory fraction .85, max running 2048 |
| Prefix reuse / speculation | Off / off | Off / off |
| Requested context / forced lengths |32768; same fixed payloads |32768; same fixed payloads |
| NVLS configuration path |0 |0 |

The same frozen SG command/environment is independently checked on both nodes against the earlier v5 reference. The native async-burst client has no request semaphore, and the closed raw timings independently show all 512 attempts begin before the first completed body; the candidate name c128 is not a 128-request client cap. Actual SG server info also reports per-rank chunk 2048 in both runs: native DP attention divides the requested 32768 by 16. The global max-running argument 2048 becomes 128 per DP rank. BG still publishes native model context 163840 on node 0; node 1 does not publish that field, while both frameworks receive the same requested 32768 cap and 10240-token fixed workload. Internal capacity/context metadata is not declared identical. Equal workload and physical GPUs do not isolate EP layout, graph coverage, kernel implementation or allocator/buffer settings into separate percentages. The two independent TP8 historical topology was not rerun or mixed into this single TP16 reference.

The remaining measured BG costs include unchanged prefill/phase reconstruction, eager routed-MoE scalar synchronization and partial graph coverage, plus attention and collective/wait intervals. These are implementation/strategy costs beyond changing KV capacity; all 512 now remain active and no recompute is observed. The measured optimizations reduce expert GEMM and shared-expert launch costs, but do not resolve those remaining paths.

CPU attribution checks passed 11 tests. Full numerical/semantic quality and native token-ID equivalence remain unqualified. This fixed first 512 prefix is the reduced experiment, not a measured 6000-request result.

Actual data, all-rank matched medians, operator qualification reports, source hashes, fresh SG settings and restored-service evidence are in [SM103_COMPONENT_RESULTS_20261025.json](SM103_COMPONENT_RESULTS_20261025.json).
Public result SHA256: `acbe33ea3bcb79ae7cb88babecddc6a97ed2202a0cf887595f93d84c4130b76a`.

## Historical V15 completed pair — 2026-10-25

The previous cohort `r1-fast-diagnostic-v15` was measured on both nodes at runtime source
commit `8890115737d5356047a00f7bf28b5c7dd5efadc1`
with host KV 192 GiB per node, the checked attention opt-ins, traceOFF and
mmapON. Its final observations confirmed both HTTP services healthy with
no active requests. This is a **fast diagnostic performance result**:
**historical strict numerical and NaN-padding failures remain unresolved**;
fixed output lengths do not establish semantic quality or model equivalence.

All BG entries below use the original first 512 requests per profile, fixed
lengths, temperature 0, ignored EOS, no inference retries, and one excluded
warmup. BG time is upload to all native result files retrieved; SG time is
concurrent HTTP submission to all bodies retrieved. SG values are historical
matched 512 controls with different kernels/graph coverage/expert layout.
Each configuration has one observation, not a repeated-trial estimate.

| Configuration | 8192 input / 2048 output (s) | 2048 input / 8192 output (s) |
|---|---:|---:|
| BG baseline, KV64 | 2091.080 | 5250.487 |
| BG attention optimization, KV64 | 1630.055 | — |
| BG attention + KV160 | 949.579 | 2620.830 |
| BG scheduler fixes + KV160 | 929.502 | 1778.014 |
| BG final, same code + KV192 | 535.621 | 1344.096 |
| SGLang historical TP16, two nodes | 235.562 | 472.887 |
| SGLang historical two TP8 replicas | 164.280 | 375.742 |

Final output throughput is 1957.682/3120.540 tokens/s.
End-to-end speedups over the original BG are 3.904x/3.906x.
Both final profiles selected 512 in one prefill wave, prepared 512 for decode,
assigned 32 to each of 16 replicas, and recorded zero recomputed admissions.
Actual global host KV capacity is 91,678 pages; the native peak allocation and
available-memory observations are retained in the final comparison JSON.

The controls separate configuration and implementation effects:

- The attention-only KV64 short control improved 1.283x. It removes host scalar/nonzero
  synchronization from valid attention metadata, caches checked absorbed weights and
  captures only local projections; actual paged MLA stays eager with its real maximum
  length.
- KV64→160 improved the same attention implementation another 1.717x on the short profile by
  reducing waves and model rebuilds. KV192 was then tested with exactly the same canonical
  runtime source as KV160.
- Completed-history balancing and average-as-upper-bound padding bugs were repaired. At
  unchanged KV160 the long run improved 2620.830→1778.014s; both changes were applied
  together. The actual first 16 replica counts are 32 each and all 512 entered decode.
- Remaining differences include pure-DP streamed prefill weights, model reconstruction,
  eager routed MoE/CPU scalar reads and partial graph coverage. SG uses native
  `flashinfer_trtllm` MoE and full-step CUDA graphs. There is no measured percentage split
  between remaining MoE, network and CPU launch costs.

Both backends use the original FP8 checkpoint. Native BG logs confirm
IB/RoCE+GDRDMA; eth0 is out-of-band bootstrap. BG explicitly disables NVLS,
and SG NVLS/loaded-NCCL binary equivalence is not assumed. This diagnosis
does not claim that high-speed transport eliminates collective costs.

The metadata changes passed 24 decode-helper tests and five actual assignment
method CPU tests. The attention optimization passed real checkpoint-layer
GPU parity at batch 1/16/32 and lengths 2048/8192 plus isolated invalid-metadata
rejection tests; those layer results do not upgrade the historical full-model
quality gate. All final 512 native results passed raw output/HTTP/usage checks
and the original CPU guard/client were independently closed before export.

Actual comparison SHA256: `87f5be5d693beed82affc9126a6e75e8666a4cb700abdda16989a610f79a6a22`.
Actual evidence is saved under `.repro-private/r1-perf-20261025/`; the JSON
binds both final ready observations, source maps, closed analyses and archives.
Earlier sections below are chronological qualification records of superseded
cohorts; their historical deployment states are not the current v15 state.


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

The same pilot subsequently completed, and its original CPU client and guardian
were independently observed absent at2026-10-23 22:20:48 UTC. All32 measured
responses passed the unchanged structural and exact-length checks:262,144
input tokens,65,536 output tokens,32 length finishes, no missing IDs and no
empty decoded text. Upload through complete result-file retrieval took
**527.178497202 seconds**, giving **124.314630335 output tokens/s**. The
separate one-request warmup took470.056206057 seconds and is excluded. There
was one upload/create per phase and no retry; polling and result download are
included in the primary time. These checks do not establish semantic accuracy.

The closed export preserves28 original files (4,578,943 bytes). The original
client validation was reproduced offline, and both original CPU identities
were absent before and after export. Final guard and summary SHA256 are
`a05206e516322fa203ac0ace40650c3eed717913572c4b464b7df9e300dd918e` /
`4002b167f2374123ba4d6e4d47ac523511c65934a0d0e76eb92d3f4042bd525c`.
Independent analysis SHA256 is
`ed94e3f35dd4502e447c03cb7d8f30ce18141ef402562d5f3e49e6f2b550b40f`;
the local freeze index is
`9efb70c22b083150517a0608c6cf9d50f5aa780c0f7dd9a17da9dae28d58fcae`.
Subsequent22:21:28 UTC observations found both services healthy with no active
batch, intake or scheduled requests. This is a completed small diagnostic
pilot, not a formal6000-request result or a successful speedup reproduction.

An independent comparison with the closed SGLang A graph-v5 first32 pilot
verified the same effective-payload hash
`991176dce4fb76a3bb8bab7db4cc50ffd1e0106890269dd4ec9966c4cc7488cd`.

| First32, input8192/output2048 | End-to-end seconds | Output tokens/s |
| --- | ---: | ---: |
| BatchGen v4, framework CUDA Graph disabled |527.178497202|124.314630335|
| SGLang A v5, graph128/c128 |90.197194855|726.585789119|

This small pilot shows no BatchGen speedup: its elapsed time is5.844733 times
the SGLang A result. BatchGen measures upload through complete batch-result
retrieval; SGLang measures first HTTP attempt through the last complete
response body. Both exclude preparation, offline validation and their separate
one-request warmup. The measured BatchGen upload/create-preparation interval
is0.019738892 seconds, so changing that boundary cannot explain the large
difference. Neither result is a repeated, saturated or isolated-kernel
measurement; the graph settings differ and quality remains unqualified.
Independent comparison receipt SHA256:
`5f3f168503b1bff0f781684db595d439842213dad46bba728d38881fcf2a4d75`.


The next v4 pilot used the unchanged first512 input8192/output2048 payload,
traceOFF and one separate warmup. The warmup completed with one valid response
in427.968786670 seconds. The measured batch then failed during packed prefill:
**0 of512 responses were returned**. Its28.526232560-second observation interval
ends in a failed batch, so it is neither a completed BCT nor a throughput result.
The API preserved an AcceleratorError from global rank10, observed at
`wrappers.py`'s `cu_seqlens[seq_idx].item()` after the SM103 prefill backend
returned. CUDA asynchronous error reporting means this observation does not
identify the faulting kernel or establish offload as the cause.

The original CPU client and guard were independently observed absent at
2026-10-23 22:32:25 UTC. Their25-file export preserves21,304,855 original bytes;
the import receipt explicitly retains `server_requests_may_remain=true`.
Final guard and summary SHA256 are
`a24a757559fcf661cf3f846de27bdfe44b961edbfdb2ef4432cb85135faa00c1` /
`16a3bd580127c81cfe0515005ba5ef858a8dad2f82d3d38d546f1a201b4c5632`.
Failed-export import receipt SHA256:
`c25aa6d13be20e6eedf5ab1fed4a6b94f54267a2bfa7c779278a480701b8933e`.
No successful-prefix analyzer was applied to this failed run. The512-request
input2048/output8192 pilot remains unexecuted and was deferred for fault
localization; it must not be recorded as completed or closed.

## Large packed-prefill GEMM repair, 2026-10-24

Commit `ca2d68dae2d810f9ec943ff617ee6e8fbd44e5cf` widens five W8A8 GEMM
index vectors to int64 before stride multiplication. Dot products, FP8/F32
scale policy, reduction and launch configuration are unchanged. Preserved
original compiler artifacts show signed32 output-row multiplication; the
first unsafe zero-based rows are87382 for Q(N24576) and65536 for KV(N32768).
The original M98304 Q and KV isolated calls both raised illegal memory access.

The corrected candidate completed the M2048 Q regression and M98304 KV test,
with all50,331,648 and3,221,225,472 output elements respectively finite and
byte-identical to low-offset original-kernel chunks. The archived corrected
M98304 Q result likewise compared all2,415,919,104 elements bytewise. Selected
boundary rows also have independent CPU FP64 comparisons; these synthetic
primitive tests do not qualify whole-model accuracy. Local checks confirm
exactly the five casts and otherwise unchanged kernel AST, and four service
tool CPU checks pass. The packed harness's three CPU checks pass against its
frozen original-source fixture; that fixture intentionally pins the old GEMM.

After the user restored SSO, the corrected canonical source was deployed on
both allocated nodes, SHA256
`b4ade5a437be8fcf4195b8e85c07d556bda4a53caea596a1276f3486b9632849`.
The packed12x8192 layer test exited naturally with code0: all output/KV finite,
first/last independent sequences byte-identical, input hashes unchanged. Its
original report SHA256 is
`57dba08885f5d355a89f1cc79e51b5c96570cb93e0b8a2414e47e2b46fa7d6bf`.

The corrected traceOFF v6 two-node service completed the unchanged first512
input8192/output2048 workload. All512 unique IDs passed raw-response
revalidation, with4,194,304 input and1,048,576 output tokens, no failed/missing
requests and all finish reasons `length`. Primary upload-to-complete-results
BCT was2076.056094113 seconds and observed output rate505.080764905 tokens/s;
secondary create-to-results was2075.797665197 seconds. Separate warmup1 took
438.103484386 seconds and is excluded. The original guard/client exited
naturally without cleanup signals and were independently observed absent.
The28-file raw import preserves32,301,216 bytes; offline analysis and freeze
index SHA256 are respectively
`994421f2fa08e533f23b2589f2cedd99d5f7a7b6ea726094af4035c05cf4275f` /
`96bcddedddaf772c1eb7dd66fcd562c9074ce0a99f639f52b46736905360c88e`.

The service log records KV-watermark evictions, recomputed prefill waves and
a final two-request tail. These costs remain included in BCT. Framework
CUDA Graphs remain disabled, with64GiB host KV per node. This is one fixed
length transport pilot; it does not establish model quality, representative
full6000 warmup, repeated performance or paper reproduction. The old failed
v4 run and strict numerical failures remain failed.

The corrected long-output512 input2048/output8192 replay subsequently failed.
Its separate one-request warmup passed exact token-count validation in
1456.368102532 seconds, excluded from measurement. During a later recomputed
prefill wave, the CPU-client guard observed host MemAvailable26.152393341GiB,
below its unchanged32GiB floor, and terminated only its owned CPU client.
The service separately reported worker rank3 exitcode-9, workers-down health503,
NCCL aborts and a failed measured batch. A later all_gather_object attempted a
512GiB GPU allocation during this failure; this is not evidence that the GEMM
index repair regressed, and no kernel OOM cause is established by exitcode-9.

The failed client's22-file import preserves10,937,749 original bytes, including
the missing measurement result/summary state. Guard/import receipt SHA256 are
`7678dea4a77411a98979c536af4cbb0c1d670de2af9bef77751bfdb41b5ab4ad` /
`cbdd9fea18cab5ab26cc475b000bffb496e531efcacd1ccff515d322821009e2`.
No positive analyzer, completed512 BCT or throughput was assigned. Both exact
original service supervisors were retired after the client closed; subsequent
audits confirmed their original sessions absent and all16 GPUs idle.

The fresh traceON v7 cohort subsequently completed all96 full-context quality
responses:48 frozen questions balanced across six tasks, two draws per question,
six sequential batches of16. The original sampling/EOS/cap and scoring rules
were retained. All96 API records joined complete native token vectors, and all
96 responses were extractable without failure categories. The CPU client and
guard exited naturally with code0, no cleanup signals, and independently
observed absence. The59-file original import receipt SHA256 is
`58958e33ddcfc289be16afd7ec1c5b55e87b8a014a3972fd927485cb7aa7f0bd`;
native join analysis SHA256 is
`ee114383507096488a2b52692c0b161aa7ee9616b72dd8bde9ee0ec5567e2221`.

| Engine | Task macro EM | Task macro token F1 |
| --- | ---: | ---: |
| Corrected BG v7 | 0.416666667 | 0.620481676 |
| Frozen SG-B96 | 0.447916667 | 0.637503584 |

Each task below contains eight frozen questions with two draws per question.
These per-task scores are descriptive small-subset observations.

| Task | BG EM correct/16 | SG-B EM correct/16 | BG token F1 | SG-B token F1 |
| --- | ---: | ---: | ---: | ---: |
| 2wikimqa | 11 | 12 | 0.837500 | 0.875000 |
| hotpotqa | 10 | 10 | 0.793448 | 0.802997 |
| multifieldqa_en | 2 | 3 | 0.631100 | 0.671827 |
| musique | 8 | 8 | 0.648504 | 0.632639 |
| narrativeqa | 5 | 6 | 0.390278 | 0.452778 |
| qasper | 4 | 4 | 0.422060 | 0.389780 |

The BG-minus-SG paired differences are-3.125 percentage points EM and
-1.702191 percentage points F1. The frozen10000-draw, task-stratified question
bootstrap averages each question's two draws before resampling. Its descriptive
95% intervals are[-7.291667,0] percentage points EM and[-4.842618,0.849312]
percentage points F1. No noninferiority margin or model-quality pass is inferred;
the earlier strict numerical failures remain unchanged. Paired-score SHA256:
`97e76c0de37eeeafcb7f58b42dcc1cbeb7aab1e5eb1e51f725fc93a9fa8317f7`.
The original SG-B run SHA256 remains
`89138dcf0074e4b00a68e09b7c2a7fb7b341fada815e37bd23cf0fec0ae09f6f`.

After quality collection and native snapshots, idle-only smaps_rollup samples
covered all10 processes in each original service SID. PSS totals were
919.602612/922.506779GiB, with private clean+dirty totals179.973759/182.864304GiB.
These are point samples of the owned service SID, not peaks or whole-machine
physical totals. Node0's54.16-second read outlived the35-second terminal
transport deadline; its original completed result was recovered from the
exclusive saved stage and SHA-verified after the reader independently exited.
Node1 completed in25.05seconds. Neither reader was replayed.

The fresh traceOFF v10 cohort used the existing MAP_PRIVATE skeleton loader,
retaining hostKV64, the corrected GEMM, original requests, per-profile deadlines
and numeric/scoring/resource gates. Its short input8192/output2048 replay
completed all512 unique requests, zero failures/missing IDs, with4,194,304 input
and1,048,576 output tokens. Upload-to-complete-results BCT was2091.080453199
seconds, observed output rate501.451772645 tokens/s; separate warmup1 took
428.953476058 seconds and is excluded. Original guard/client exited naturally
with code0, no signals, and independently observed absence. The leader's
one-second samples recorded minimum MemAvailable219,641,831,424 bytes and
peak client-session RSS396,177,408 bytes; these do not establish a continuous
peer minimum. The28-file raw import preserves32,308,259 bytes. Import,
analysis and freeze-index SHA256 are respectively:
`83c6d5d1d010a248c83ccd4f922f5141381a6297b2127764d06fd76f45f3dbb6`,
`e7c964e5c48d1dfdb58b16d4d7cf9e94d113f8b776b78276666b872299c2c705`,
`57904fd4da8ff96dcf9b8688479643f7d7c1cd8c9084d82737ede5443b926e59`.

All512 measured responses plus the separate warmup were compared against the
frozen v6 mmapOFF run: exact API text hashes, usage and finish reasons matched
for all513 rows, differences0. Canonical per-node source maps remained equal
to the original repaired deployment, including its fixed inventory exceptions.
This is storage-output parity for these actual requests, without a native-ID,
model-quality, steady-performance or paper-reproduction claim. The comparison
report SHA256 is
`6356b4673ea29bdfb8ca7838385b3a50a9add46112c3e402402b84f4df8c9b30`.
The same v10 cohort also completed all512 input2048/output8192 requests,
zero failures/missing IDs, with1,048,576 input and4,194,304 output tokens.
Primary BCT was5250.486727973 seconds, observed output rate798.840986999
tokens/s. Separate warmup1 took1473.195581547 seconds and is excluded. Both
original CPU processes exited naturally without signals and were independently
observed absent. The leader's sampled minimum MemAvailable was198,080,581,632
bytes, with client-session RSS peak121,520,128 bytes. Import, analysis and
freeze-index SHA256 are respectively:
`e316fac1fb2ac1e00c86186504678fb9879ddb82ca53b8b3c603cc2a79da8e13`,
`6d49662398949b31450bef210c0c15705c3a624a1e9d30252254e2cfc36b1adc`,
`d54011783774effd899126b8e055d0060872fddf56c09a936c6474e5483a8a66`.
The earlier v6 long512 failure remains a separate failed observation.

After both complete replays, idle rollups covered all10 original service-SID
processes per node. PSS totals were810.288231/810.137455GiB, with private
clean+dirty totals65.716175/65.286560GiB; MemAvailable after collection was
195.485859/196.678326GiB. These are idle points, with different workload/trace
history from the v7 quality baseline, not peaks or a causal whole-machine
memory reduction. Both35-second terminal transports timed out; both original
completed results were recovered from their exclusive saved stages after
independently observing the readers absent. Neither reader was replayed.
Memory-import and bounded-experiment qualification SHA256 are respectively
`b1300f1e525ac819ecdf25a775f4e986029372142800a926f9c82327635f06aa` /
`252c6d143016014a2be657c4f18c411158c34644b09d9b145215d46dfa23a6e0`.
Full short-profile API-output parity, both complete512 profiles and these
memory observations permit one bounded exploratory1024 round. They do not
establish model-quality, representative-shape, steady-performance or paper
reproduction qualification.

The user reduced the6000 tier to1024 measured requests per profile/engine,
one exploratory round, with unchanged input/output lengths and the exact
original-source first1024 selection shared by BG and SG. Actual CPU inspection
SHA256 is `d48095551a85147792f7d42c8a6d92eef82f15256e6f184760b9a1a8fbf721b1`.
The original6000 source and unexecuted repeated protocol remain preserved.
The first actual BG1024 short-profile client completed all1024 unique requests,
zero failures/missing IDs, with8,388,608 input and2,097,152 output tokens.
Primary BCT was3870.283921406 seconds, observed output rate541.859988204
tokens/s, including the low-concurrency tail. Separate warmup1 took
425.917463692 seconds and is excluded. Both original CPU processes closed
naturally without cleanup signals and their absence was independently observed.
Original archive bytes were transferred, SHA-verified, imported and analyzed.
Import and analysis SHA256 are respectively
`4e903e589e90fc9ed151c1ab6fcfcb623e427236f50e69f9487f457c1dbd0f81` /
`bfc15d941e9f4066140b9a02fb44ac30d67915462798916faadbf2e5bdb64d48`.
The BG1024 long profile also completed all1024 unique requests, zero failures,
with2,097,152 input and8,388,608 output tokens. Primary BCT was10271.692826756
seconds, observed output rate816.672396798 tokens/s. Separate warmup1 took
1492.233446348 seconds and is excluded. Both original CPU processes closed
naturally without cleanup signals and were independently observed absent.
The original archive was SHA-verified, imported and analyzed. Import and
analysis SHA256 are respectively
`aba6870a54fe88eaf535b2506469ae78e60d4adbc7452d0cb68c65c2af16e171` /
`cfe9c4dec0ae92d6ebc3165ea51ad12275c34073a3e65a87163178e0882db7d9`.
These are single exploratory observations. The selected SG-Bv2 comparison
and final repaired BG dual-service restoration are now complete.

## Completed reduced comparison and restored services

Each row contains1024 original-prefix requests, zero failed/missing/duplicate
records, exact prescribed token totals, and independently closed CPU clients.
The two engines use the same first1024 source IDs and payload hashes. Separate
warmup and offline validation are excluded from the primary BCT.

| Profile | BG BCT (s) | SG-B BCT (s) | BG output tokens/s | SG-B output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| input8192-output2048 | 3870.283921 | 262.566693 | 541.859988 | 7987.121212 |
| input2048-output8192 | 10271.692827 | 497.086560 | 816.672397 | 16875.547808 |

Excluded warmup timings were BG1/SG-B1024 requests per profile:
short425.917464/266.008069 seconds and long1492.233446/497.242248 seconds.
SG-B assigns512 warmup and512 measured requests to each endpoint. It performs
one global async measured burst and times through the last complete response.
BG measures native upload/batch/poll/full-response retrieval. These API clocks
and warmup coverage differ.

BG uses TP1/EP16/DP-attention16 across both nodes, framework graphsOFF,
hostKV64GiB per node, mmapON and traceOFF. SG-B uses two independent
TP8/DP8/EP1 replicas, graphsON with maximum128 per DP rank, memory fraction.85
and chunk prefill16384. BG worker native context163840 and SG context32768
also differ; the BG submitted interface limit32768 is a separate setting.
Observed SG-B/BG output-rate ratios are14.740194 and20.663791. They describe
these actual configurations and one exploratory round, without a steady-state,
quality-equivalence, formal performance or paper-reproduction claim.
The original6000-request protocol remains unexecuted and preserved.

All original SG1024 archive bytes were SHA-verified before offline analysis.
Import / analysis SHA256:

- SG-B `input8192-output2048`: `6991653bfbed9c4d4afd6eba639f9f4c98b968c8f714e21931fb7884a5dec5b4` / `fa49510f2bb137ed11cba9fa134e7fb02adc97055ac3cf8fa3c3ea29ce4507de`.
- SG-B `input2048-output8192`: `0f30cae516b8064b5019ac78ad0c39675809e6ec9a237b0118c77e016d7f86e7` / `54a32858e7b757385dc1c772d12dbbf00b8ee52fbb2523428a06701b0e07cf1f`.

The four-run comparison report SHA256 is
`fca03eec787ab78d3f0c3c63dc7528419a384428266f8c3263435d9b29951f7b`.
Its pinned96-response quality summary retains the descriptive EM/F1 and
intervals reported above; no new quality-pass conclusion is inferred. Earlier
strict numerical and NaN-padding failures remain unresolved.

After their respective clients closed, both original BGv10 and SG-Bv2 model
sessions were independently observed absent and all16 GPUs idle before the
next cohort started. Their preserved closure-import receipt SHA256 are
`9ea02bb0f155dfd183ed99fb800d2d0b9152b5e1bc02cc660eabde3cf644dd07` /
`128ae27b222af3e23ae28aa02dfc1c60b7d75d3060941923eedab4ab39cefc13`.

The final `r1-fast-diagnostic-v11` BG services are restored on both nodes.
Actual saved launch bytes, original supervisor/service identities, environment,
healthHTTP200 and empty local intake/scheduling/batch pools were verified.
Both per-node source maps and source-deployment bindings exactly match the
repaired v10 cohort, retaining its fixed inventory differences. Skeleton mmap
is ON, native trace is OFF, hostKV64 and diagnostic attention selection remain
unchanged. No additional inference was submitted during this restoration.

- Node0: supervisorPID1636685, servicePID1636690; actual ready SHA256 `7207dde0df37c79e9ec83fc765ea5b82e25f48cc2e7121ab865b1f87d1354b73`.
- Node1: supervisorPID330105, servicePID330108; actual ready SHA256 `600978974079d86f69ecb075919822711d6fac2376c77131eb2a1ab5093ce7fb`.

Final dual-deployment receipt SHA256:
`1eae614e14ddf356b9863583f8415827ba04f9c39c95f8348d257724d1ce0747`.
Detailed private continuation state is in
`.repro-private/index64-resume-20261024/STATUS.md`.

## 2026-10-25 bottleneck investigation and decode optimization

The restored v11 cohort above was retired without inference before this
investigation. Its independently imported closure receipt is
`e4bf702ee4e8e58ae7426fa45103f6cf65495e11d92be5f14a314889d7f06106`.
The following attribution uses actual rank0 timestamps clipped to each original
512-request measurement, rather than adding time across ranks.

| Original BG512 profile | Batch wall (s) | Decode window (s) | Logged decode steps | Mean step (ms) | Recomputed admissions |
|---|---:|---:|---:|---:|---:|
| 8192 input / 2048 output | 2091.080453 | 1671.107 | 8060 | 206.696 | 42 |
| 2048 input / 8192 output | 5250.486728 | 4929.054 | 22653 | 217.320 | 378 |

Upload/create preparation was 0.272/0.058 seconds. Decode windows account for
79.9%/93.9%; they include control overhead. CUDA-event durations include CPU
launch gaps and do not measure occupied SM time. The original host KV64 GiB
per node supplies 30558 global pages (about1.95M token slots), resulting in
multiple prefill/decode waves and eviction/recomputation. This configuration
cost is separate from the per-step implementation cost.

Historical matched512 SGLang measurements further bound the topology argument:
TP16/DP16 across two nodes took235.561517/472.886556 seconds; two independent
TP8/DP8 replicas took164.280091/375.741605 seconds. Both enable DP attention
and CUDA graphs, use native kernels and disable radix caching. Splitting the
replicas improved these observations by1.434x/1.259x, insufficient to explain
BG's entire gap. These are historical controls, with one warmup request for
TP16 and one per endpoint for TP8, excluded from measurement. BG also uses
cross-node MoE allgather/allreduce and a different expert layout; this is not a
controlled estimate of communication alone.

Source commit `d16b481b7bbaecf9af2499abf2c7bc2b9bbb92b5` adds three explicit
SM103 diagnostic opt-ins: asynchronous GPU metadata assertions, versioned
absorbed-weight caching, and two local projection CUDA graphs per layer/shape.
The paged attention call keeps each actual max sequence length; fixed-bound
whole-attention capture was rejected after one observed bitwise mismatch.
Defaults retain the original eager validation. Invalid metadata remains an
error; a device assertion requires discarding that CUDA context.

Actual checkpoint layer0 GPU qualification covered batch1/16/32 at length2048
and8192: all output/new-KV values were bitwise equal, including changes to
hidden states, positions, sequence lengths and page tables. Retained outputs
were not overwritten by replay; source mutation/replacement invalidated
caches. Isolated invalid-page/coverage/position/max-length cases were rejected.
Layer wall time improved2.170x–2.972x (about1.49–1.79ms to0.51–0.71ms).
This is a layer result, not a full-model speedup or a new quality pass.
Positive GPU report SHA256:
`b9b516f5d3d0a3d8fabc04f3a8169e98875d1108ade644f773048f713d821280`.

Both nodes now have backend SHA256
`7e6cc5e9119421fb634423edd343050d2173fb4dfb63264f3aa69cae38728851`;
all other runtime source hashes, including the earlier W8A8 index64 repair,
were preserved. Actual CPU deployment receipt hashes (node0/node1):
`0f85f08277cee71782fce51d36242247463af99bb937c4aecadc7c1c36230fea` /
`ad600bae9258eec11803bdf6989defde20171d13c401b624fdb7f840d96c3433`.
End-to-end implementation and capacity results are reported below; the layer
measurements alone did not establish those results. Historical strict numerical and
NaN-padding failures remain unresolved.

The actual implementation-only v12 control has completed at the unchanged
hostKV64, with the same original512 payloads, lengths, EOS policy and excluded
warmup1. All512 native results passed independent raw-output/HTTP/usage checks:
1630.054834 seconds,643.276519 output tokens/s, versus2091.080453 seconds and
501.451773 tokens/s before the change (1.283x observed throughput,22.0% less
wall time). The measured weighted decode step fell206.696→147.211ms;
decode windows fell1671.107→1189.679s. The same four prefill admissions
190/190/172/2 and42 recomputed admissions remain, isolating the remaining
capacity/tail problem. Other wall time was440.376s versus419.973s. The final
two requests still needed their own phase, despite more than99% of host pages
being free during that tail. Warmup1 completed in344.776004s and is excluded.
This is one paired before/after observation, not a repeated-trial estimate.

Closed v12 archive/manifest/analysis SHA256:
`050ac48b0a0aa9369712af4436ebb3a091c516280cf31833af03f66ec941cfe0` /
`d936c38b38fdf3af4abd0c25812241a661aef64c8c045876de92fbeed04f91ea` /
`c75c21455f3bba1e6ed1fa4ed187084984a299266595b6e82f64cad424318cab`.
Original guard/client exited0 without signals and were independently observed
absent before export. Both model sessions were subsequently stopped through
verified pidfds and observed closed with all16 GPUs idle; retirement receipt
SHA256 `d4e003090762be986d3e2ac4115f979eb426d6908b4ca80bbdb1a99c376483e7`.
The v13 capacity experiment kept identical runtime source and opt-ins,
changing only host KV to160GiB per node through an explicitly pinned private
wrapper around the unchanged launcher. Its fresh two-node process/environment
observations and completed512 results are recorded below. The controller also
checks peer host memory with a40GiB emergency stop floor; the original local
client memory floor remains32GiB.

The v13 hostKV160 short-profile control has now completed with all512 exact
results:949.578636 seconds and1104.253992 output tokens/s. This is2.202x the
original throughput and1.717x the v12 implementation-only control. The two
prefill admissions were476/84, with4094 logged decode steps instead of8060;
decode windows were643.644s and other time305.935s. Weighted decode timing was
156.601ms; stable428-active intervals were about145–147ms, close to the prior
170-active intervals. The observed gain comes mainly from doing fewer decode
waves with a larger active batch, rather than making each step faster.

The unchanged10% watermark still evicted48 requests in the first wave,
including an observed growth plan with zero remaining growth rows/pages.
Consequently recomputed admissions rose42→48 even though waves fell4→2.
The capacity change must not be described as eliminating recomputation.
Both BG controls retain framework graphs OFF and the same local projection
opt-ins; generic/full-step graph coverage and MoE kernels were not changed.

Short v13 archive/manifest/analysis SHA256:
`f5ec139fd2977142254b6448a750c651dd87fb55093f7fdab688006f736850eb` /
`9b1fd16b0b4c745e8790ee2ddbe1b9683e05ddc724e931cd3cf111eaee8e814c` /
`99d1f03cd0fff3554a79a3c534eb2ab18540f80062aca4149bce4198aa30b522`.
Excluded warmup1 was313.686154s. The sampled local host-memory minimum across
warmup and measurement was108.102GiB; no peer40GiB stop was triggered.
The long512 profile then ran on the same v13 services and source; its
completed results follow below.

The running R1 decode log explicitly reports all16 local experts per rank as
persistent in the multi-node mode. The original/default MoE path still uses
allgather/allreduce over16 ranks, grouped Triton operators and two GPU scalar
readbacks per routed layer (`grouped_dequant_moe_fp8`); this patch changes the
attention path only. SGLang's native fused MoE and full-step graph coverage
remain different. No measured percentage split between remaining MoE,
communication and launch costs is claimed. Upload/create overhead was measured
at about0.27s in both new short runs, and cannot explain hundreds of seconds.

The long v13 control exposed two additional metadata implementation bugs.
`estimate_max_decode_replica_batch` used a ceiling average as an upper bound;
the first512 prefill configured32 rows per replica, but only498 entered decode
while14 remained PREFILLED despite free GPU KV pages. Rank0 actually received18
requests, which implies some other replica received at least33. The complete
16-rank assignment vector was not logged; the CPU regression vector
`(18, 33*14, 32)` is representative, not a claimed observed vector. Once other
rows were evicted the14 eventually entered decode, leaving a long14-row tail
with99% of host pages free.

The default L² balancer also included COMPLETED requests retained by the global
batch iterator. Earlier8192-token requests therefore biased the later2048-token
batch. Commits `0a42dc1631c0a1074746d6f9c2834f884f57b1a2` and
`8890115737d5356047a00f7bf28b5c7dd5efadc1` use actual per-replica candidate
counts for padded-buffer sizing and exclude completed history from L² and
legacy least-count balancing. Active/resumable work still counts; minimum16
and hard cap128 remain. The prefix-sharing assignment branch is unchanged.
The exact candidate decode helper passed24 CPU tests; the actual worker
assignment method passed five CPU metadata tests, including completed8k history,
active/evicted work and legacy balancing. Their report hashes are
`b2ea89d6e5142889e8eb159921c978adf762b0fe4d0ac7fde64363f3962f3002` and
`dd7df5876df298263adf53f9dea7f4e3e89451736657dff8ca8ef83c5e17217f`.
Those fixes remained local while the v13 control retained its original source
until closure; their actual dual deployment and full-model results follow.

Native rank0 phase-switch timers whose completion falls within the original
measurement report246.596s/8 switches for the old short run,267.827s/8 for the
implementation-only control, and132.480s/4 for hostKV160. For the latter,
one observed decode→prefill transition was31.856s and the following
prefill→decode transition34.788s (`persistent=False`). These timers cover the
worker's busy transition path, including model release/rebuild/transfer work;
they overlap the wall-window attribution and must not be added to it. This is
a further implementation cost that grows with the number of waves. The old
long control reported251.755s/6 switches. This does not isolate differences
against SGLang's prefill or expert computation.

The v13 long control has now completed512/512 exact2048/8192 results,
zero failures:2620.830431s and1600.372138 output tokens/s,2.003x the old
long run. Warmup1 took995.181932s and is excluded. Decode windows were
2422.876s, other time197.954s, logged steps16382 and weighted step147.628ms.
Prefill selections were512/84 with84 recomputed admissions; native phase
switch timers reported170.539s/4 switches. The fourteen-row tail therefore
required nearly a second full8192-step wave despite abundant free host KV.
This remains the unmodified scheduler control, before the metadata repairs.

Long v13 archive/manifest/analysis SHA256:
`4d545e21fed654ef96055bd97e8e5c54f7daed42ed169ad1d2e2bbb9dc0801b7` /
`d13ead2b2f705298afd4345370d55f86620797fb617516adfde2bc0968f67665` /
`fcf4f943a73e923e1f8772a395b1947e0237916c030d838141fd30a6b77bf4a4`.
The original CPU guard/client exited0 without signals and were independently
observed absent before archive transfer. These are fixed-length transport
results, not a new semantic-quality or strict numerical pass.

The v13 dual services were subsequently retired normally: both original
supervisors/services absent, native supervision exited0 with child reaped and
no parent cleanup required, all16 GPUs idle. The final metadata source was
then deployed to both nodes at commit
`8890115737d5356047a00f7bf28b5c7dd5efadc1`; actual node0/node1 deployment
receipt hashes are
`af06b4b2bd6c79e87ea89924b1adf371f6e2609592bec9d71e344cf75105847e` /
`56c89207d21cc4b2bcd153bb19947cd26d72d84360714ce2674b6f3f188c6939`.
The432-file common runtime map is now
`00f0cb6c2d200ac60c183dec65aee43c6e9391279c33e66ddae45cce6ba5e10f`;
the five previously documented node0 metadata exceptions remain unchanged.
V14 keeps hostKV160, the same attention opt-ins and all model/math/protocol
settings. Its two512 profiles repeat the original short→long order, each with
one excluded warmup, so completed-history conditions are controlled.

Node0's first two v14 startup attempts stopped at the fresh GPU-idle guard,
before execution-directory or model creation. Subsequent read-only checks
observed all eight GPUs at4MiB,0% utilization and no compute processes; the
successful node0 launch retained the identical guard and reused only exact
SHA-verified support files. Node1 started once. This was prelaunch handling;
no inference request or benchmark was retried. Full-model results are reported below; CPU metadata tests alone did not
establish them.

Actual native NCCL logs confirm `Using network IB`, all eight `mlx5_bond_*`
RoCE devices and cross-node `NET/IB/*/GDRDMA` channels. No `NET/Socket` path
appears in that archived node0 startup. `eth0` is the out-of-band bootstrap;
it is not evidence that tensor communication fell back to TCP. BG explicitly
logs `NCCL_NVLS_ENABLE=0`; no SG NVLS equivalence or communication saturation
claim is made. High-speed transport configuration does not remove the cost
of performing collectives over16 ranks at every routed layer.

The frozen SGLang A/B launch plans both point to
`DeepSeek-R1-original-fp8`, explicitly select FP8 quantization and BF16 dtype,
and use `trtllm_mla` attention plus `flashinfer_trtllm` MoE. They do not switch
to a FP4 checkpoint. Their DP-attention/page64/request-context32768 settings
are explicit. A uses TP16/DP16/EP1 across2 nodes, chunked prefill32768 and
max-running2048; B uses independent TP8/DP8/EP1 servers, chunked prefill16384
and max-running1024 per endpoint. Both enable CUDA graphs with max batch128
and disable radix caching. Those native fused MoE and graph paths differ
from BG's current eager routed-MoE path and local attention-projection graphs.
The SGLang plan does not explicitly set NVLS; no equality with BG's explicit
NVLS0 is assumed. BG's batch-interface context32768 remains distinct from
its native worker context163840; all tested total token lengths fit both.

The optimized dual-node controls use these environment settings on
both nodes (the optimization defaults remain disabled outside this recipe):

```text
BATCHGEN_R1_SM103_ASYNC_VALIDATION=1
BATCHGEN_R1_SM103_CACHE_ABSORB=1
BATCHGEN_R1_SM103_PROJECTION_GRAPHS=1
BATCHGEN_R1_SM103_ALLOW_CAPTURE=0
BATCHGEN_R1_NATIVE_TOKEN_TRACE=0
BATCHGEN_R1_SKELETON_MMAP=1
NCCL_RAS_ENABLE=0
```

The native server arguments retain `--disable-cuda-graphs` and select
`fa4_flashinfer` attention. The pinned wrapper set `--host-kv-cache-size 160`
per node for v13/v14, then192 for v15, with all other runtime flags retained. The framework flag does not disable
the explicitly enabled local projection graphs. Paged MLA still receives
its actual max sequence length outside capture. Deployment also retains the
previously qualified CUDA13 toolchain, registered weights and safe process
supervision; these settings alone are not a replacement for those gates.

V14 short has now completed with512 exact8192/2048 results and zero
failures:929.502397s and1128.104676 output tokens/s,2.2497x the original
run. Warmup1 was308.693921s and is excluded. Compared with v13
949.578636s, the observed additional wall reduction is2.1%; a single run
does not establish statistical significance for that small difference.
Admissions476/84,48 recomputes and4094 logged decode steps are unchanged.
Decode windows were631.452s and other time298.050s; weighted step153.692ms.
Guard/client exited0 without signals and were independently absent before
archive transfer; the dual model services remain healthy for the next profile.

V14 short archive/manifest/analysis SHA256:
`469decca5ff2ffccb83d0ea10f444fa5a347851024e4c5719c06fd981647c9f0` /
`9350aec8fec2d31b7f430a2fa353e395d73da1ba3361c5cc269dc22aa8c57495` /
`9379a120de257b9ec573faf9d24469f6420402dabc3cf786e598a6f17d84d742`.

After observing the full KV160 runs, a further configuration-only KV192
experiment was prepared. Actual leader sampled minima were92.584GiB in the
long control, and peer45-second samples reached99.691GiB. Subtracting the
additional32GiB gives a conservative linear projection of60.584/67.691GiB,
above the unchanged32/40GiB stop floors; this projection is not a runtime
qualification and may miss peer transients. KV160 supplies76398 global pages.
Scaling to192GiB predicts about91678, versus81920 pages needed for512
requests at10240 total tokens, leaving about10.64% free against the10%
watermark. Actual pages, resource minima, waves and output lengths must be
measured. The KV160 scheduler control remained unchanged until its full long512
run closed. KV192 reuses the identical canonical runtime source;
only its explicit launcher capacity setting changes.

V14 long has now completed512 exact2048/8192 results, zero failures,
1778.013895s and2358.982690 output tokens/s. This is1.474x the same-KV160
v13 control (32.2% less wall time), and2.953x the original v10. The actual
first assignment is `[32]*16`, with512 sequences prepared for decode and
none stranded PREFILLED. Its saved live log prefix was subsequently verified
byte-for-byte against the closed archive. Both metadata corrections are
applied together; their individual timing contributions are not isolated.

The10% KV watermark nevertheless evicted84; prefill selections remain512/84
with84 recomputed admissions. Logged decode steps fell16382→10366, since
all84 had already generated tokens before eviction. Decode windows were
1572.757s and other time205.257s; weighted step151.391ms. Leader available
host memory reached93.863GiB; neither resource floor triggered. Warmup1
was980.112342s and is excluded. Original guard/client exited0 without signals
and were independently absent before export.

V14 long archive/manifest/analysis SHA256:
`4856d46845717e28dc47288736ae0ab5ff8476f96423989b5ec01a8fa0297e69` /
`766df9631763a1952a43efaa18fa974be35854bcd9234375b902ac7e73755b72` /
`f8d8cdee8b3441fd335f8a02aa5b26b42b3e5e21061e3ff86c36038adacd4765`.
Actual complete v14 comparison SHA256:
`1d9dccc0c4b8feea68afff41ecf139e59ce3a918bb02f71cc5a31ecaff0982c3`.
The same-code KV192 experiment was launched after actual v14 model retirement.

Source inspection also separates the two phase strategies. R1's
`DeepseekV3ParallelStrategyManager.configure_prefill` explicitly configures
pure DP and builds streamed weight-copy tasks for attention, shared experts
and all routed experts. Its multi-node decode configuration instead keeps
all16 local experts per rank persistent. The decode residency observation
therefore does not imply that prefill has no host weight traffic. The phase
switches remain `persistent=False`; one v15 prefill→decode transition reported
31.918s. Total prefill weight-transfer bytes or a separate transfer/compute
percentage were not measured. Resident-prefill parallelism and avoiding model
reconstruction remain implementation work beyond the capacity setting.

The actual KV192 global capacity is91678 pages. V15 short completed all512
exact8192/2048 results, zero failures:535.621190s and1957.682072 output
tokens/s. This is3.9040x the original and1.7354x the identical-code KV160
control. Warmup1 took312.725862s and is excluded. A single prefill selected
all512, with zero recomputed admissions; logged decode steps fell4094→2047.
During decode all512 remained active, using81920/91678 host pages, leaving
10.64% free. Decode windows were321.811s and other time213.810s; weighted
step156.385ms. Larger KV improves total work/waves rather than per-step
latency. The remaining other-time window includes prefill, model transitions
and unclassified costs; it is not all CPU transfer or occupied GPU compute.
Available leader host memory reached83.676GiB; no resource floor triggered.
Guard/client exited0 without signals and were independently absent before
export; both models remain healthy for the next profile.

V15 short archive/manifest/analysis SHA256:
`4ff4f0df0697a75dc46077b4cb8759513135cc539dd73648aca2838168bf0719` /
`d88df7662bb4ac64d9613999903539331926c78708ff624f42afcccd5d5e2691` /
`b30addd251b6b4dfe9661aef282bd00eb2da1f9cf9d140a96219c788cd37cf67`.
The completed long512 control and final combined result are reported next.
Historical numerical/NaN failures remain.

V15 long completed512/512 exact2048/8192 results with zero failures: 1344.095628s and 3120.539872 output tokens/s.
It used one prefill wave,32 requests per replica and zero recomputed
admissions. Logged decode steps were8191; weighted step was151.455ms.
Decode/other wall windows were1243.460/100.636s. Warmup1 was991.162247s, excluded.
Leader host memory minimum was75.916GiB; peer45-second observations
and unchanged resource-floor status are bound in the final comparison.

V15 long archive/manifest/analysis SHA256:
`c4680103d847343979312e639191eb4dd654b14af8681ec9bef0c55ce7b66a84` /
`accd55c8182cd7eea12d145d159b6398f8b581e197ceaf64b03c472202c7401d` /
`0af02dd18fbf6f0d232b37a500cc460ba0c64fe1018fb0cec649f41a0c02392b`.
Final fresh dual-ready capture SHA256 (node0/node1):
`0cc06d091dfcae86ad6efb1f0e808f5aee86b6f2ac3da060ddf9ce9191f58124` /
`c61038f368e24db2d8ac75d8f5742ab8e6be7fc7ad817513cb1ef05fe218b7b3`.

Native rank0 phase-transition busy timers completed within each final
measurement reported2 switches/62.517s for the short profile and
2 switches/58.696s for the long profile, versus8/246.596s and6/251.755s
in the original controls. These reported busy timers overlap the wall-window
attribution and must not be added to it.


## 2026-10-25: measured component baseline and qualified MoE scheduling

V16 adds opt-in, nonblocking sampled component probes to the V15 runtime;
source commit `bca68d6bb9d0450c8594cbd3018d2e4ed45f214d`, common433 map
`db9175ed5c92bf8ddeab049658c05c82bb01dd8d9b62dec63be126fd2f2f4e93`.
Both512 cases completed, with exact forced lengths and zero failures.
The external settings remain KV192, the same first512 requests, temperature0,
ignore_eos=true, retries0, one excluded warmup per profile, short then long.

| V16 baseline | 8192 input / 2048 output | 2048 input / 8192 output |
|---|---:|---:|
| Client upload through all results, seconds | 538.696482 | 1371.160372 |
| Output tokens/s | 1946.506123 | 3058.944879 |
| Weighted decode step, ms | 157.400782 | 154.821243 |
| Native max-rank prefill KV-ready, seconds | 157.982103 | 42.604712 |
| Rank0 switch busy timers, two transitions, seconds | 63.861 | 58.282 |
| Rank0 prefill model configure, seconds | 20.89 | 19.06 |

Native prefill covers all16 participating ranks. Rank0 model configuration
includes skeleton construction11.0s in both cases, expert setup2.7/2.5s,
and to-device1.6/1.2s. Configuration is part of phase switching. Switch busy
timers and prefill boundaries overlap wall classifications; do not add these
rows to BCT. Prefill subkernel probes did not reach their sampling cadence
in the one/two native forwards, so no prefill MoE-versus-attention split is
claimed. The pure-DP streamed prefill implementation and rebuilding phase
models remain distinct from persistent EP decode.

For complete sampled rank0 decode steps, median attention stream intervals
are56.749/53.171ms and routed-MoE112.331/113.099ms. Within MoE, gate/up GEMM
is18.605/18.518ms, down GEMM11.951/11.957ms, and shared-expert stream time
9.402/8.521ms versus22.556/22.088ms host-call time. All-gather and all-reduce
stream medians are27.793+15.741ms and27.728+15.024ms. These are nested,
sampled stream elapsed intervals, including CPU launch gaps and peer arrival
waiting, rather than occupied GPU/kernel time. They must not be summed with
parent MoE, host durations or other ranks. Instrumented sampled steps can be
slower than the weighted all-step average. All16 ranks have complete samples.
Scalar token-count readback host time also waits for preceding work; its
22–25ms cannot be interpreted as additional compute independent of GEMMs.

Both host port inventories were bound to the allocated containers by all
eight device/GUID pairs. During1027.8s of complete sampled intervals inside
long-output steady decode, each host transmitted/received about5.87/5.88GB/s
in aggregate across eight200Gb/s RoCE ports. The highest single-port interval
average was3.17% of its reported direction-specific line rate. These are
host counters, not exclusive model traffic; averages cannot rule out bursts,
latency, congestion or arrival skew. Linux port_xmit_data/port_rcv_data count
four-byte units, as specified by the [Linux InfiniBand sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband).
The short baseline predates the usable host sampler window; no short-window
wire-rate estimate is fabricated. Native NCCL logs select RDMA, not TCP fallback.

The main implementation changes were qualified in isolated SM103 GPU jobs
using original checkpoint layer3 FP8 weights, native activation quantization
on deterministic BF16 inputs, exact old-kernel controls and paired timings:

- Down projection gives each CTA an expert/N tile instead of serially looping
  all experts. All15 routing/tail cases were bitwise identical and finite;
  poisoned inactive tails stayed zero. At16 active experts with8–32 tokens
  each, same-tile stream speedup was1.50–1.61x. Single/empty expert cases
  can regress slightly; the complete model test is still required.
- Across nine routing cases, every tested tile retained bitwise output parity.
  A predefined mixture of uniform8/16/32/64 and skew selects gate/up32x64x128
  and down16x64x128. Weighted stream gains over the64x32x128 tile are1.661x
  and1.573x respectively; the down tile gain is in addition to expert-axis
  parallelism, not an isolated full-model gain.
- The unchanged resident shared-expert projections and SiLU use a bounded
  per-caller-stream CUDA graph. Batch32 host wall speedup is2.51x. B0/1/16/32/
  17/129, dynamic inputs, retained outputs, source mutation/replacement,
  alternate streams and unversioned-source eager fallback were checked.
  Graph source keys include parameters, cached aliases, scales and versions;
  returned outputs are cloned. Prefill, routing and collectives are excluded.

Runtime commit `de7f6f0a117951a0f53ed81bfc1a6dbe65466c28` preserves default
paths and enables these changes only through explicit SM103 experiment flags.
CPU attribution tests:11 passed. Deployed kernel/helper bytes exactly match
the completed GPU qualification fixtures.
Historical strict attention numerical and NaN-tail gates remain failed;
these narrower operator checks do not qualify full-model quality.

Actual evidence SHA256:
- `components-down-report.json`: `8e4995165ebecbfaea214f20553cb35b4d8d900ef2c699bba4006c12b5cd1770`.
- `components-tiles-report.json`: `3f0e99c17b20b94555fcf9e11541955d4d123b64362824aa1ca1ffd4f4b3fdee`.
- `components-shared-report.json`: `3bde0efad6754c6c68fc1ce8ebcba50d367d102c1c16b30b2958e47c0443ff90`.
- `components-v16-input8192-output2048-analysis.json`: `1c2208cb1c75ea351ea2fbec451b0f31834dbd91101f1673046569971736daa1`.
- `components-v16-input2048-output8192-analysis.json`: `ee32a29178b37f7617405bf46448019d02d7bfaf2c760241839d5019c5c89dec`.
- `rdma-v16-input2048-output8192-analysis.json`: `9844d46998f0dcaf8da6ac14d184c47a41ea9084b263accad431f170ef68642a`.
- `final-v16-actual512-comparison.json`: `b8b743c53acf4d71854e9840b7fa194e167710d2eb94318b1b52e53e4e0d23bc`.
