# SGLang formal candidate plans for 16 allocated GPUs

`plan_sglang_formal_candidate.py` creates local JSON plans only. It never starts
services, contacts the containers, queries GPUs or sends requests. It leaves
the diagnostic launcher/executor and frozen QA/6000-request workload untouched.
Candidate A now completes fixed-length pilots through2048 requests using a
smaller prefill chunk and CUDA Graphs. The earlier larger-chunk MoE failure
and numerical/quality limitations remain recorded below. Candidate B, with
two independent TP8 replicas, is prepared but has not started. No candidate
has completed the formal6000-request performance comparison yet.

The identity remains `sglang0.5.5.post3+R1-routing-compat`, using the original
R1 FP8 weights and F32 block scales. The R1 routing compatibility patch and the
four audited dense-kernel switches remain explicit. This is not stock SGLang.

| Candidate | Physical allocation | Independent services | Per service parallelism | Client endpoints |
| --- | --- | --- | --- | --- |
| A | Two nodes, eight GPUs each | One spanning both nodes | TP16, DP attention16, EP1 | Node0 HTTP only |
| B | Two nodes, eight GPUs each | Two, one on each node | TP8, DP attention8, EP1 | Both nodes' HTTP endpoints |

Both use sixteen GPUs total. B sets `nnodes=1,node-rank=0` on each physical
node and uses that node's own container IP for initialization. Its physical
node1 still uses node1's routing-patch receipt. Both services may reuse HTTP
30000 and rendezvous33155 in separate containers. Admission must also check
known ZMQ ports33156–33160, DP handshake port33168 and runtime-selected sockets; this planner
does not claim to reserve ports or inspect their availability.

## Generate a reviewable plan

From the repository, with the previously collected fixed source audit present:

```bash
python3 scripts/r1_reproduction/plan_sglang_formal_candidate.py \
  --topology A --per-dp-running 16 --per-dp-chunk 4096 \
  --memory-fraction 0.85 --output /tmp/r1-sglang-A-plan.json

python3 scripts/r1_reproduction/plan_sglang_formal_candidate.py \
  --topology B --per-dp-running 32 --per-dp-chunk 4096 \
  --memory-fraction 0.90 --output /tmp/r1-sglang-B-plan.json
```

Output creation is exclusive. Without `--output`, JSON goes to stdout. The
default audit location is `.repro-private/baseline-recovery/formal-source-audit`;
`--source-audit` can point to another byte-identical local export. The combined
receipt, five semantic source/help files, diagnostic plan provider and routing
pins must match their frozen SHA256 values. This checks a historical local
export, not the current remote installation. Command strings are for review;
do not bypass parent admission or process supervision by running them directly.

The 48 combinations are A/B × running16/32/64/128 per DP rank ×
chunk2048/4096/8192 per DP rank × static memory fraction0.85/0.90. The tool emits
one selected candidate at a time. CLI max-running and chunk are multiplied by
the DP count of that service: A multiplies by16; each B replica multiplies by8.

## Fixed-version semantics

These conclusions come from the pinned installed 0.5.5.post3 source export,
whose combined receipt SHA is
`7732753365d15c53297f88a6752bbb36bd1a9a9ac57c4d39a0a7a167f802ae1b`:

- `server_args.py:1383` multiplies schedule conservativeness by0.3, and `:1385`
  divides the CLI chunk by DP. The default effective conservativeness is0.3.
- `tp_worker.py:281` divides CLI max-running by DP, then caps it by the request
  pool. A target of128 is not a guarantee of128 simultaneously resident requests.
- `model_runner.py:483` passes the undivided CLI max-running to request-pool
  allocation. Equal per-DP concurrency therefore does not imply equal A/B
  request-metadata memory. Record actual pool slots and memory per rank.
- `server_args.py:1267` rejects explicit `bf16`/`bfloat16` for `trtllm_mla`, even
  though general CLI help lists them. The plan keeps `--kv-cache-dtype auto`
  with `--dtype bfloat16`; `model_runner.py:1547` selects model dtype unless a
  KV FP8 quantization mode overrides it. Record direct dtype observations and
  source-derived expectations separately. This release logs the KV dtype only
  on TP rank0; absent logs on the other15 ranks are missing direct evidence,
  not evidence of a different dtype or a service failure. The executor does
  not stop a candidate because those log lines are absent.
- `model_runner.py:1355` budgets remaining KV memory after model loading as
  available memory minus total capacity times `(1 - mem_fraction_static)`.
  Fraction0.90 does not dedicate90% to KV alone. No `--max-total-tokens` is
  emitted: the old diagnostic32768-token cap must not carry into formal tuning.
  Capacity is profiled and page-rounded, not known at planning time. BF16 R1
  MLA costs70272 bytes per token per rank before other allocations.
- Default max-prefill tokens16384 is not divided by DP and exceeds every
  proposed effective chunk. Graphs are explicitly disabled initially; piecewise
  graphs, torch compile and speculative decoding remain disabled by defaults.
  Radix/prefix caching and shared-expert fusion are disabled explicitly.

The generated environment is an exact allowlist, rendered using `env -i`, so
inherited credentials or `SGLANG_CUTLASS_MOE` cannot alter the plan. It includes
`NCCL_RAS_ENABLE=0` as the hardware stability adaptation documented in
[NCCL_RAS_RECOVERY.md](NCCL_RAS_RECOVERY.md). This does not upgrade NCCL. The
planner records the expected SGLang package version; loaded-library provenance
must be collected separately for each executed cohort.

The diagnostic provider now also emits explicit `NCCL_RAS_ENABLE=0` and clears
inherited BatchGen runtime preloads and CUDA compiler overrides. This tool's
`DIAGNOSTIC_SHA` follows that reviewed provider source; the audited installed
SGLang source receipt and routing pins remain unchanged. Candidate A/B commands,
capacity choices, and exact allowlisted environments retain their existing
settings. A new diagnostic preparation must generate new source-bound receipts;
historical diagnostic receipts are not rewritten or replayed. The separate
formal adapter below consumes reconstructed formal candidate JSON; the original
diagnostic executor continues to rebuild its own small-capacity plan.

## Explicit formal preparation and execution

`execute_sglang_formal_candidate.py` is a fixed-experiment adapter, defaulting
to local plan-only. Supply `--candidate-plan`, its explicit
`--expected-plan-sha256`, `--source-audit`, physical `--node-rank`, a fresh
`--cohort-id` and an exclusive absolute `--run-dir` under the fixed experiment
root. It reconstructs the entire candidate from SHA-pinned provider/planner and
frozen semantic-source/help files, rather than accepting arbitrary argv/env.
It can be staged as a byte-identical private bundle outside the repository:
keep its seven pinned companion files beside it, and explicitly point
`--source-audit` to the receipt plus five exported source/help files. This is
not a general installer or an authorization to use another machine/model.

Only after parent-owned resource release and separate admission, use
`--prepare` on both containers, exchange the resulting fresh `receipt.json`,
then explicitly use `--execute --peer-receipt ...
--expected-peer-receipt-sha256 ...`. Preparation performs CPU artifact checks,
one eight-device idle snapshot, >=850GiB available CPU memory, local port
checks, and own core-filter0/soft-core-limit0 plus local-disk preflight. Ports
are checked, not reserved; disk probing does not prove future project quota.
Execution checks both node plan digests, source/cohort binding, <=600s receipt
ages, disjoint GPU inventories, and immediate local source/core/memory checks.
Old diagnostic receipts cannot authorize a formal candidate.

The service environment is exactly the candidate allowlist, including RAS0
and SG's own CUDA/runtime paths; no ambient BG preload, compiler setting or
credential is inherited. CPU inspection/probe children additionally hide CUDA.
Start the outer executor itself in a clean environment; an ELF preload cannot
be removed after process startup. It explicitly rejects ambient `LD_PRELOAD`.
The pinned diagnostic monitor/cleanup is reused: 7200s startup, >=128GiB host
memory while running, WNOWAIT leader observation and primary-PG cleanup before
reaping. Other groups in the original SID are audited for parent cleanup;
descendants escaping to a new SID are outside that audit. Parent coordinates
the peer on any node failure. No identity scan/signal occurs after child reap,
including final receipt-write errors.

Each new record carries `formal_receipt_schema=r1-sglang-formal-execution-v1`,
the same `formal_binding`, physical node and role. A uses the real node0 server
and node1's local dummy plus real node0 health/model checks. B requires both
nodes' full-server marker and each local R1 health/model identity; node1 dummy
readiness cannot satisfy B. READY proves initialization/HTTP only. It neither
submits requests nor qualifies quality, actual per-rank KV capacity, tuning or
performance. Separate collectors must retain missing direct dtype evidence
and actual per-DP capacities without inventing unobserved rank values.

## Admission and fair measurement still required

The existing diagnostic executor assumes one two-node service and small
capacity. Its receipts and readiness logic do not authorize either new plan;
B requires a new coordinator aware of two independent services. Before any
execution, the parent must establish resource ownership on both containers,
fresh weight/source/library/patch checks, exact candidate communication and
kernel qualification, lifecycle supervision, runtime BF16 KV evidence and
actual memory/concurrency capacity. Prior short/empty-EOS quality observations
must remain visible; successful HTTP or source-level validation is insufficient.

After functional and capacity admission, tune on separate32→128→512 request
subsets and extend if saturation is not reached. Use eager as the control;
qualify graphs separately before selecting an optimized baseline. Freeze the
winning settings before the unchanged6000-request profiles and at least three
warm repetitions. This script does not choose a winner.

For A, one client sends6000 requests to node0. For B, the existing multi-endpoint
client sends the same6000 requests round-robin,3000 to each service. Measure one
cluster BCT from the first network attempt to the last complete response body;
do not add or average per-replica BCTs. Formal runs require fixed output length
with `ignore_eos=true` and exact input/output count validation. Record BatchGen's
batch API versus SGLang's async-burst transport. Loading, compilation, warmup and
offline scoring are outside BCT. No formal speedup is established by these plans.

CPU tests include actual AST-extracted methods from the pinned source for DP
division, request-pool capping and TRTLLM BF16 rejection, plus the48 topology
combinations, provenance rejection, deterministic exclusive output and no
process/network calls. Fixed-source tests skip only if that private export is
absent; report skips explicitly when validating elsewhere.

## Observed candidate A initialization, 2026-10-24 CST

Cohort `sglang-formal-A-v2` started candidate A with 16 running requests and a
4096-token prefill chunk per DP rank, static fraction0.85, and graphs disabled.
Its candidate SHA256 is
`f9685440ebe09f1b6914b3d42d2f91c8800229953b350371fd4b305be5c8d7c2`.
Both nodes reached READY with the original R1 weights; the model's logged
`DeepseekV3ForCausalLM` class name is its architecture, not a different checkpoint.

The frozen `/get_server_info` response contains16 entries, each reporting
52.98GiB weights,173.06GiB KV cache,2644288 token capacity and0GiB graphs.
These memory figures are rounded runtime observations. The response has no DP
rank identifiers, so its list order is not used as a rank map. Separate startup
logs explicitly report2644288-token KV allocations for all16 ranks. Only rank0
directly logs the effective `torch.bfloat16` cache dtype; other dtype statements
remain source-derived expectations. The rank0 scheduler directly reports
max-running16 and chunk4096; this is not a sustained-load capacity qualification.

Installed `dp_attention.py` independently confirms attention TP size1 and
128 query heads per rank for this TP16/DP16 configuration. That source inference
is not a live tensor capture. The frozen capacity receipt SHA256 is
`ac0614bc225bed5752e60bbd7ffc8206d7cfac179176a5ad25a714bb5a8e6c13`.

Initialization reports NCCL2.28.3. All16 worker map snapshots name SGLang's own
NCCL and CUDA13.0.48 runtime libraries, without BatchGen's runtime preload.
The on-disk NCCL library SHA256 is
`109a1a2fa0e622aa2393ad68dd1fec11e6b6e70934ca4db16de4a21c544e7cfa`;
libcudart is `6876f4841bdbe30b187dcf8d7a7d2d6579bf12a36eeb430c7b28484ccdeed89a`.
Overlay device IDs differ between maps and pathname stats, and `map_files`
access is denied, so these are pathname-byte hashes, not hashes of mapped
backing bytes. BatchGen used NCCL2.27.7; the versions are not matched.

All16 ranks warn that eight dense FP8 shapes lack tuned `NVIDIA_L20D` config
files and use defaults. Preserve this limitation when interpreting tuning
results; initialization does not establish an optimized baseline. The12-request
functional result is recorded in [LONGER_CHAT_PROBE.md](LONGER_CHAT_PROBE.md).

### First fixed-length subset failed at the MoE launch

The8192-input/2048-output protocol completed one warmup request: HTTP200,
exact8192/2048 usage, finish reason `length`. Its247.777878 seconds are warmup,
not benchmark BCT. At03:09:19 CST, the following32-request burst reached
4096-token per-DP prefill and raised `cudaLaunchKernelEx: invalid argument`
in FlashInfer's `trtllm_fused_moe_dev_kernel.cu:178`. The client was stopped;
`requests.jsonl` and `summary.json` were absent. The client writes measurement
rows after all requests return, so absence does not mean no requests were sent.
No measured batch throughput, latency, or speedup is available for this attempt.

The exact installed source SHA256 is
`63c9801e6930db1a394ef9d99729e6084fcad31b8164ec5b0d8baf4e63ec2a15`.
Its FP8 activation launcher sets `grid.z = data.numTokens` at line176, while
[CUDA limits the grid y/z dimensions to65535](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
Source analysis predicts that16 DP prefill chunks of4096 can gather65536
tokens for the MoE call. The failing service did not directly trace that
call's token count; this remains a source-derived shape, although the same
launch boundary was subsequently reproduced in isolation below. The activation
kernel already contains a grid-stride token loop, but no patch is applied yet.

Both original services and the client were independently checked absent;
all16 GPUs returned to4MiB/0% with no visible compute processes. Node1 exited3
and its supervisor completed cleanup; node0 required the existing supervisor's
SIGKILL after its120-second SIGTERM grace. Original failed guard metadata,
including `server_requests_may_remain=true` at client shutdown, is preserved
separately from the later model-service cleanup evidence. Closed-export import
receipt SHA256 values are
`441fd2fd925fd1dd2dba91c87d48885844d9c3f6053d3ea7ffceb8fec645a28a`
and `5521c7be5805475b291fe546b025abb88820f41db297128fcda728d601a6dcae`.

### Isolated native MoE launch boundary

Three independent single-GPU processes called the unchanged FlashInfer0.5.2
FP8 MoE function with synthetic R1 TP16-local shapes: hidden7168,
intermediate128,256 experts and topK8. Each process explicitly recorded its
input shape and queried SGLang's loaded CUDA13.0.48 runtime, which reported
maximum grid Z65535. The original `tune_max_num_tokens=8192` default was kept;
it is not treated as a runtime token limit.

| Explicit input M | Activation grid derived from installed source | Observed result |
| --- | --- | --- |
|4096|[2,8,4096]|All output elements finite; worker exit0|
|32768|[2,8,32768]|All output elements finite; worker exit0|
|65536|[2,8,65536]|Same cu:178 `cudaLaunchKernelEx: invalid argument`; worker exit2|

All three guards completed cleanup without signals or live session members;
the original guard identities were absent and GPU1 returned to4MiB/0% with no
compute processes. These synthetic TP16-local calls do not run a model service
or TP allreduce. Finite output scans are not full numerical, model-quality or
performance qualification. The failed full service's actual M was not captured.
The three report SHA256 values, in ascending M order, are
`24eb3a55c991dfc17528eb82b7ee60327e592a83634d2f5de58b56aff20dd26c`,
`e7b4594f541d33dff440a7f6c658ca7f2353b7fa423eddfd2fdf296d5d576f6d`
and `0ae0c67345f459fc2990d158f604fba26438d103d2a117937afc41535b0d72e4`.
The local derived analysis SHA256 is
`22864f3c9744cc6059e9770827e068333adacbd54572a3bbb70e69c4c896bb9e`.

The next candidate is `A-c16-m085-k2048-eager`, SHA256
`d364ba2d74495172c9cac77c2476f7e44a0a2139283a1532e263f42ad1c4da5c`.
It requests2048 prefill tokens per DP rank (CLI chunk32768), retaining TP16,
DP16,EP1,16 running requests per DP, static fraction0.85 and eager execution.
The all-extend gathered MoE shape is then expected to be M32768; mixed scheduler
batches need separate observation. No kernel, quantization or tuning-default
patch is involved. Fresh full-model functional and fixed-length subset checks
are still required before any performance conclusion.
Both nodes started this candidate as cohort `sglang-formal-A-v3` at
2026-10-23 19:29:58 UTC; READY had not yet been observed at that checkpoint.

### Smaller-chunk eager pilot completed

Both nodes subsequently reached READY. The frozen v3 capacity record reports
52.98 GiB weights, 172.97 GiB KV cache, 2642944 token capacity and no CUDA
Graph allocation in each of the 16 server-info entries. Separate rank logs
confirm the token allocations; server-info list order is not a rank map.
The capacity receipt SHA256 is
`981a8627a777410a53d2f2ff759dc9c33c329925617626733fdf2cca2780f335`.

The first 32 unmodified requests from the 6000-request 8192-input/2048-output
manifest completed successfully with `ignore_eos=true`. All 32 response bodies
were received and validated, with 262144 input and 65536 output tokens.
Measured BCT was 281.105371542 seconds, giving 233.1367758663 output tokens/s.
One successful 243.14531451-second warmup was excluded. The client and its
guard exited, with no cleanup signals or remaining server requests; both model
services remained running for the next subset. The raw summary SHA256 is
`7930a29c6e5e1cfa60b48dd253bd7e21943ddf98540e2c79541aff4f4b03b520`,
and the terminal guard SHA256 is
`af7b3b4bdb6b3fcf14e076b626c1bfd14ace006382eecf7aa38dcfdcd8db53bf`.

This is a small eager-mode tuning observation, not a formal 6000-request result,
an optimized SGLang baseline, model-quality qualification, or a BatchGen speedup.
It establishes that this full-model workload completes after reducing the
prefill chunk; it does not prove every mixed scheduler shape is safe. A separate
128-request burst from the same manifest was started at 2026-10-23 19:47:49 UTC
after fresh readiness snapshots and closure of the preceding client.

The 128-request run completed at 19:57:11 UTC: all 128 responses passed the
fixed-length checks, with 1048576 input and 262144 output tokens. Its BCT was
301.870352212 seconds and output throughput was 868.3992915472 tokens/s;
the 248.08451568-second warmup was excluded. The original guard and client
were absent after natural exit0, with no cleanup signals or remaining requests.
The raw summary SHA256 is
`a5a1c629a8738c32a739ada75fe78de1320ec00f5672b99529a34cbf41950ba6`,
and terminal guard SHA256 is
`318746af653ddc5d707ad1e0bb15ebf5164ed1db16ae42b6e9a3e6cef762a58c`.
Decode logs show eight running requests per DP rank and approximately 3% KV
usage. These pilots do not establish saturation: even 512 requests average
only 32 per DP rank, below a prospective per-rank concurrency target of 128.
The single-request warmup also does not prove all larger-shape compilation is
excluded; formal runs need representative warmup before measured repetitions.

The next separate candidate is `A-c128-m085-k2048-graph-bs128`: 2048 global
running requests and CUDA Graph capture up to 128 per DP rank. It retains the
2048 per-rank prefill chunk, static memory fraction0.85 and original precision.
This is a candidate optimization, not a graph-only ablation against c16 eager.
Its two new source checks cover the existing graph runner and MLA backend;
actual capture memory and replay must still be verified. Only the new tooling
was transferred while v3 ran. Both original v3 supervisors were subsequently
sent identity-bound SIGTERM after the 128 client had closed, to release the
nodes for this separate candidate.

### Graph candidate: first 32-request pilot

The same graph candidate, SHA256
`143d658784dd2306432818df9835811d747291f568c036095f6552b234e5f448`,
started as `sglang-formal-A-v5` on both nodes at 20:06:31–32 UTC.
The preceding v4 preparation was abandoned after node0's port check failed;
neither v4 node launched a model. Fresh paired v5 preparation receipts were
used. All 16 ranks subsequently logged successful graph capture, approximately
23 seconds and 2.58 GiB per rank, with 36.46 GiB free afterward. The actual
server-info response reports graph maximum128, 2048 global running requests,
TP16/DP16 and 2048 effective prefill tokens per DP rank. Its 16 memory entries
report 172.97 GiB KV and 2642944 tokens each; list order is not a rank map.
Separate decode logs from both nodes show `cuda graph: True` on every rank
during the following pilot. This observes capture and replay, not model accuracy.

The original first32 requests, unchanged from the eager pilot, completed at
20:14:42 UTC with all32 HTTP200 responses, one attempt each, exact8192 input
and2048 output tokens, and length termination. BCT was90.197194855 seconds,
or726.5857891186 output tokens/s. The67.89535177-second warmup is excluded.
The client and guard naturally closed with exit0, no cleanup signals and no
remaining requests. All9 raw evidence files were exported and hash-verified;
both original process identities were absent before and after export.
Raw summary SHA256:
`f2f7bb290b5bac1ed1b84fb2a3e0724b2189b13ba71f3dae86cf56fbd588e2da`.
Raw requests SHA256:
`8bc4d68c2f592de6357a8a0792d9e381e5cd63e59d36d54ac8fcc13580cb83e4`.
Closed export SHA256:
`727e7edc56310ca2714ac55f1ef06344a15bb33ca6c5228087fb7463c9262ed2`.

This small c128/graph pilot is separate from the c16/eager configuration, so
the difference from281.105371542 seconds is not a graph-only ablation. Default,
untuned dense FP8 configurations remain in use. There is still no formal6000
result, optimized-baseline claim, quality pass or BatchGen speedup measurement.
A first128 burst began at20:16:02 UTC after fresh two-node readiness records
and closure of the32-request client; its result was pending at this checkpoint.

### Graph candidate: 128- and 512-request pilots

The following runs used the same v5 services and original first-N requests
from the frozen 6000-request manifest, with input8192/output2048 and EOS
ignored. Each had one separately timed warmup request, excluded from BCT.

| Requests | Completion UTC | BCT seconds | Output tokens/s | Warmup seconds |
| --- | --- | --- | --- | --- |
| 128 | 2026-10-23 20:19:24 | 122.210913123 | 2145.013021351 | 68.255964834 |
| 512 | 2026-10-23 20:25:59 | 235.561516607 | 4451.389238376 | 68.520140784 |

Every response was HTTP200 on its first attempt, ended by length, and reported
the exact requested input/output token counts. Both clients and guards closed
with exit0, no cleanup signals and no remaining server requests. Raw evidence
was collected only after checking the original processes absent. The summary
SHA256 values for 128 and 512 respectively are
`626eefc4f2b0916e24709136351f7103c91645cabd9b0b7e77c470f776515647`
and `b260a115532e0df09ad202347c2837229fee19acea8123d0c7129d8ab5ad92d6`.
The request-file SHA256 values respectively are
`11fa05172a7b020b0cdba80b4dc9f133f11f0b5d12970a4b941a43c82a46e5d6`
and `332cf201079d8e4b97e825a9cfcb3d7b406bc414f52e978f2d0258a44dc56e2a`.

Both-node logs again show graph replay on all16 ranks. The512 pilot reaches
32 running requests per DP rank, below the configured128; neither observation
establishes saturation or a fully tuned baseline. The same untuned dense FP8
configuration and single-request warmup limitations apply. These are SGLang
tuning results, with no BatchGen speedup or model-quality claim. A separate
512-request input2048/output8192 run started at20:29:32 UTC; its outcome is
not included in this table.

The input2048/output8192 pilot subsequently completed at20:41:59 UTC. All512
requests passed the same fixed-length checks, with1048576 input tokens and
4194304 output tokens. BCT was472.886555561 seconds, or8869.5775988475 output
tokens/s; its270.281145006-second single-request warmup is excluded. Both
original CPU processes were absent after natural exit0, without cleanup
signals or remaining requests. The raw summary SHA256 is
`9f6515e79ac87fd74e3114f1ce7d35d19dfbcf486142d3cdc09e02eb02131f54`;
raw requests SHA256 is
`f136fac7ce36575e0efe0fb791cacbeeefc2517825ae603f2c53ef394ff7d338`.
Actual decode logs show32 running requests and graph replay on all16 ranks.
The closed export SHA256 is
`ab985cf4c7664ae9ea5d36dbdda6523e219d16702d79366d4c6f1807354fb7df`.
This is still a single tuning observation. An input8192/output2048 first2048
burst started at20:43:05 UTC to test the configured per-rank capacity more
closely; it is not a completed result or a saturation claim.

### Graph candidate: completed 2048-request pilot

The first2048 input8192/output2048 requests completed at20:55:43 UTC on
2026-10-23. BCT was677.421717236 seconds, or6191.570026886 output tokens/s.
The68.485722229-second warmup is excluded. Every measured request returned
HTTP200 on its first attempt with exactly8192 input and2048 output tokens,
ending by length. All2048 submission attempts began within0.211200031 seconds
and before the first complete response. Decode logs showed all16 DP ranks at
128 running requests with graph replay; this observes the configured running
limit, not a globally optimal or saturated baseline.

Both original CPU client identities were absent before and after export,
following natural exit0 without cleanup signals or remaining server requests.
Independent offline checks verified all2049 rows, including warmup, and
recomputed the full-body BCT. The nine raw files total35147319 bytes. Summary
SHA256 is `0161e078bbb7ed36c79562bbe0f5ad219721d165360ba4839f7a389abdc73ed2`;
requests SHA256 is
`30d9a2f52ec7dfc90602f2fa6bbf12c406d499032550dc9d344d606d2b80803f`.
The closed export SHA256 is
`177799c8c40258f9af9b77bf4ab4e259a853d604a79f2535b8e20ea297278c28`;
the local integrity index is
`d99674374168e08df3badaee3c8cf91f38cbbd98d3c55cbbb95762bcbdfde10b`.

This is one unchanged workload prefix, with default dense FP8 tuning and a
single-request warmup. It does not establish model accuracy, formal6000
steady-state performance or a BatchGen speedup. A separate graph-enabled
known-answer12 functional regression started at20:56:54 UTC; its result is
pending at this checkpoint.
