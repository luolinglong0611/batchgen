# BatchGen R1 greedy-output audit — 2026-10-23

The six repeated serial prompts used different worker ranks. Their first
generated token IDs agree, but their final texts differ. This is evidence of
an unresolved output difference across execution placements, not an isolated
same-rank determinism test. The location of the first *text* difference does
not establish the numerical origin: prefill hidden states or KV can already
differ while the first argmax remains unchanged. No quality/performance gate
or previous failed result is changed by this audit.

## Frozen evidence and line-number convention

`ROOT=/home/admin/batchgen-r1-repro-20261023` on each allocated container.
The following live-filesystem copies were SHA-checked against the embedded
phase8 archive manifests after the BatchGen service had exited:

| Node | Container hostname | Log | SHA256 |
|---|---|---|---|
| node0 | `serverless-010066009163.wm226` | `ROOT/r1-diagnostic-v3.log` | `f91bca6dc06474fe8070293d97b5d5225c961496f6f6226d233ef9fa2f946f72` |
| node1 | `serverless-010066009166.wm226` | `ROOT/r1-diagnostic-v3.log` | `685dc6b42234272e191f5ad692da25e93f46ea9a69cc11c0f0abffab1031089d` |

The archives are under
`oss://shark-console/luolinglong.lll/batchgen-r1-repro-20261023/resumed-r1/`:

- `evidence-node0-20261023-phase8-20261023T111001Z.tar.gz`, SHA256
  `cb12e7a7c044b7d131682807ed54678f40c5843e557cf03ab88b1515005ae9ad`.
- `evidence-node1-20261023-phase8-20261023T111010Z.tar.gz`, SHA256
  `a30c3dfcd7c0c8572f2652134ba0b068da5132bef79ee1aad9bc3f8cd1814427`.

All log line references below are one-based indices of
`Path(log).read_text(errors="replace").splitlines()`. This counts carriage-return
progress-bar lines; a raw LF-only line counter may differ. Timestamps are the
logged UTC+08:00 times. Client JSON timestamps are UTC.

Client evidence:

- `ROOT/correctness-batchgen-v4/run.json`, SHA256
  `653cff0920236f8e956f1006900c44aefc191f9d8b5a987ddcaedb54b77a9830`;
  its `r{0,1}-<case>.submitted.jsonl`, `.batch.json`, `.requests.jsonl` siblings.
- `ROOT/concurrency-batchgen-v1/run.json`, SHA256
  `13c1dd19c24e07b4a922186a91e3277c5ba2753bd89331120296fd53c5b94d16`;
  `concurrent16.submitted.jsonl` and `concurrent16.requests.jsonl`.
- `ROOT/bg-v4-collected-confirmation.json` contains the text-prefix comparisons.

## Sampling parameters and serial correlation

All 28 submitted bodies specify `temperature=0.0, top_k=1, top_p=1.0`.
For each serial case, the two request bodies are identical; only the outer
`custom_id` changes. `CompletionRequest` preserves zero; scheduler fallback
uses `is not None`; intake/admission transfers `sampling_params` to
`SequenceEntry`. `BatchGenWorker._decode_batch_all_greedy` checks `>0`, and
`_select_tokens` calls `logits.argmax(dim=-1, keepdim=True)` for these requests.
No stochastic token-sampling branch was identified.

Source anchors at the audited source state: `server/io_struct.py:344–360`,
`server/batch_scheduler.py:656–668, 841, 1013`,
`batchgen_worker.py:1826, 3017, 3082, 14905, 19151` (all relative to `batchgen/`).
The four relevant remote source files matched the local hashes during the
read-only audit.

Each serial client waits for its one submitted item before submitting the next.
Thus its `.batch.json` time window contains exactly one real prefill metric.
The table uses that metric's explicit `rank` and `first_sampled_token_ids`, not
an assumed round-robin assignment. Both passes have one local sequence, one
microbatch, no saved prefix tokens, and the prompt length shown below.

| Case | Prompt tokens | Repeat 0 rank / log line | Repeat 1 rank / log line | First ID, both | Common text prefix, characters |
|---|---:|---|---|---:|---:|
| arithmetic | 17 | 0 / node0:8846 | 6 / node0:10152 | 33001 | 58 |
| algebra | 18 | 1 / node0:9064 | 7 / node0:10366 | 33001 | 109 |
| chinese | 26 | 2 / node0:9287 | 8 / node1:5224 | 16426 | 59 |
| recall | 23 | 3 / node0:9501 | 9 / node1:5402 | 33001 | 198 |
| code | 23 | 4 / node0:9715 | 10 / node1:5580 | 33001 | 239 |
| logic | 28 | 5 / node0:9938 | 11 / node1:5758 | 33001 | 458 |

For example arithmetic differs at `"let's see"` versus `"let me think"`,
and algebra at `"solve for a variable"` versus `"solve an equation like this"`.
These character offsets are not native token indices. Full generated native
token IDs and per-step logits were not recorded in the result API/artifacts.

`[L2_BALANCE]` logs accumulate the completed prompts' squared lengths:
node0:8744 initially has `[289,0,...]`; node0:10051 includes the first six
loads plus the second arithmetic request on rank6; node0:11115 has the first
12 loads. `_assign_admitted_sequences_to_ranks` (`batchgen_worker.py:2078–2139`)
includes previously assigned entries without filtering their completion
status. Retaining historical load may be scheduling policy; this audit does
not establish it as a bug. It does establish a placement confounder in this
repeat experiment. Four pairs also change node.

## Concurrent batch: 16 requests, 14 real prefill ranks

Batch ID: `batch_3382d2a57dd843e5b52c85a5`.
Node0:11352 logs 16 selected sequences; node0:11333 records the resulting L²
distribution. Per-rank prefill metrics account for all 16 requests:

| Rank | Real sequences | Total prompt tokens | First IDs | Metric log line |
|---:|---:|---:|---|---|
| 0 | 1 | 26 | 16426 | node0:11446 |
| 1 | 1 | 23 | 33001 | node0:11448 |
| 2 | 1 | 18 | 33001 | node0:11441 |
| 3 | 1 | 23 | 33001 | node0:11447 |
| 4 | 1 | 23 | 33001 | node0:11440 |
| 5 | 0 | — | — | no real prefill metric; accounted total is already 16 |
| 6 | 1 | 23 | 33001 | node0:11439 |
| 7 | 1 | 23 | 33001 | node0:11438 |
| 8 | 1 | 17 | 33001 | node1:5948 |
| 9 | 1 | 18 | 33001 | node1:5947 |
| 10 | 1 | 18 | 33001 | node1:5946 |
| 11 | 0 | — | — | no real prefill metric; accounted total is already 16 |
| 12 | 1 | 28 | 33001 | node1:5945 |
| 13 | 1 | 28 | 33001 | node1:5944 |
| 14 | 2 | 43 | 16426, 33001 | node1:5942 |
| 15 | 2 | 43 | 16426, 33001 | node1:5943 |

Ranks14/15 each contain a 26-token Chinese prompt and a 17-token arithmetic
prompt. The 23-token rows could be code or recall. Metrics have no UUID or
custom ID, so their individual repeated-request-to-rank mapping cannot be
recovered unambiguously. Do not infer it from output order or from identical
lengths. All ranks participate in the decode MoE collectives, but this run
does not demonstrate real attention input on all 16 DP ranks.

The R1 PSM has no attention TP subdivision; the worker's default G=1 makes
local ownership follow `assigned_rank` (`batchgen_worker.py:2144–2179`).
Decode placement therefore follows prefill ownership under the inspected
no-migration path. Actual per-step decode row order/counts are not logged:
`_sync_decode_moe_rank_counts` only emits them under
`BATCHGEN_MULTI_BATCH_DIAG` (`:16557–16604`). The table above is observed
prefill shape, not a claim of a captured per-step decode trace.

## Phase switches and KV evidence

There are 13 prefill/decode waves: 12 serial and one concurrent. Phase logs
show `persistent=False`; models are freed/reconfigured between phases.
For example node0:8938 switches to decode for serial request1; :9059/:9158
switch to prefill/decode for request2; :11425/:11541 do so for concurrency.
This run is not a repeated-forward test against one unchanged model object.

All 13 planner records on node0 explicitly say
`MIGRATION: No migrations needed after planning`, at lines
`8763, 8980, 9203, 9417, 9631, 9854, 10068, 10282, 10496, 10708, 10920,
11132, 11345`. Neighboring records say host KV is already balanced. Host-KV
occupancy is zero before the subsequent serial admissions. The prefill
metrics above report final offload tasks retired (one per real serial row,
two on concurrent ranks14/15).

This supports absence of planned cross-rank KV migrations in these waves.
It does not establish byte equality across ordinary GPU→host prefill offload,
host→GPU decode load, and incremental decode append. Those operations still
exist in this architecture. There are no per-request page IDs, valid-byte
hashes, or copy-event records sufficient to audit them. End-of-wave status
reports (`node0:8941 … 11314, 11548`) are not full intermediate on-hold or
page-boundary traces.

## Smallest useful next diagnostic, specified before rerunning

Use an opt-in diagnostic trace; do not change greedy selection or old gates.
First compare a fixed prompt/prefix repeatedly on the **same rank and same
batch shape**, then intentionally vary rank and batch position. Log enough
to verify those controls rather than assuming placement.

| Boundary / source location | Minimum fields |
|---|---|
| Admission and assignment, worker `:1826, :2116` | UUID/custom ID, batch ID, global index, prompt-token hash/count, assigned rank/node, effective sampling values, cohort generation/epoch |
| Packed prefill selection/writeback, `:14905, :15050, :15104` | UUID-to-row mapping, packed lengths/positions, phase-switch index, actual first token; attach rank to the same record |
| Decode metadata/selection, `:18440, :18500, :19150` | UUID/row, generation step, input token, context length, position ID, page-table digest, local batch shape and all-rank row counts; selected ID, finite flag, top-2 IDs/logits and margin from the **same logits** used for argmax |
| Existing CPU token append, `:2773, :19275` | Complete native generated token sequence with UUID/step; reuse completed CPU readback rather than adding a GPU synchronization solely for IDs |
| KV load/append boundaries, `:4822, :16321, :3842, :19505` | UUID/owner, valid token range, page IDs and mapping generation, completion event/task; for the small diagnostic only, hash valid KV bytes before offload and after reload (exclude padding) |

Once the first native token divergence is known, replay the identical prefix
and capture the first divergent layer's attention output, MoE router
IDs/weights, and pre/post-collective output. Same-token selection does not
guarantee equal hidden states. Integer atomic dispatch ordering, FP8
requantization, BF16 distributed reduction, weight reload and KV transfer are
separate candidates; none is established as the cause by these logs.

The independent CUB input-extent repair is commit `17b9fcb`; its existence
does not attribute the output difference to that defect. Native/sanitizer
qualification is tracked separately.
