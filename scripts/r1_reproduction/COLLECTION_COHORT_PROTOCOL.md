# Explicit correctness collection cohorts

This is a versioned **client collection protocol**, not a production scheduling
optimization or a throughput measurement. It reduces repeated singleton batch
submission and corresponding BatchGen phase transitions. It changes the request
arrival/forward-shape conditions, so both engines must be freshly collected
under the same frozen schedule. Old serial results remain historical results.

The parent archived the old BGv2 run after22 HTTP records (receipt:
`bg-v2-archive-r2-node0.json`). Its arithmetic r0 stopped naturally at262
tokens, while r1 ended at640 tokens with `finish_reason=repetition`, an unclosed
think block and repeated FINAL text. That is an observed natural-EOS/semantic
hard failure, not merely an unfinished collection. Cohort collection does not
repair or erase it. A separately identified BG dense-layer BF16 versus SG FP8
parameter mismatch also requires its own diagnosis and correction before a
fresh paired qualification; client grouping is not that correction.

`--cohort-size` accepts only `1` or `16`, defaulting to `1`. Set it on both
fixture preparation/import and collection. Choosing16 on an older unannotated
suite fails instead of silently changing that suite's protocol.

```text
qualify_r1_full_model.py prepare --suite-version v2 --cohort-size 16 ... --execute
qualify_r1_full_model.py collect --cohort-size 16 --long-correctness ... --execute
r1_independent_semantic.py import --cohort-size 16 ... --execute
r1_independent_semantic.py collect --cohort-size 16 --run-48x2 ... --execute
```

The new suite changes only collection metadata relative to the corresponding
fixed84-v2 or independent96 inputs: `collection_protocol` and the common
collector-source SHA. Every original job's ID, prompt, native input IDs, answer,
teacher prefix, output budget, EOS rule and sampling fields remain unchanged.
The fixed v2's12 semantic8192 budgets and all72 other probes remain intact.
The independent rule and its48-question paired averaging remain unchanged.
Regenerate and freeze source/suite/deployment hashes before starting new runs;
do not edit running service sources or substitute hashes on an old run.

For fixed84 with long correctness enabled, the deterministic schedule is:

1. The original16 coverage probes together.
2. All66 short ordinary jobs in original suite order, grouped16/16/16/16/2.
3. Each of the original two long profiles alone, in original suite order.

This gives8 cohorts; omitting the long stage gives6. Independent96 uses the
original suite order in six groups of16. Legacy fixed collection remains the
coverage wave followed by each ordinary job separately, including the old long
positions. Legacy independent collection remains its first16 together followed
by80 singletons. Nothing is filtered, deduplicated or reclassified as passed.

BatchGen submits one actual batch per cohort with all original per-request
parameters, waits for terminal completion and fetches every output row before
the next cohort. Reordered output rows are joined by ID, never positional zip.
SGLang submits that cohort through a thread pool with at most16 workers, waits
for all calls to finish, then advances. Cohort membership/order and the barrier
are shared; internal admission order, DP-rank placement, padding and native
forward batching are engine mechanisms and are not claimed identical.

Each new run binds its complete deterministic `collection_schedule`, canonical
SHA and `collection_receipts`. Receipts contain group index/kind/planned IDs,
original start/end times and actual received IDs. BG includes the actual unique
batch ID and checks each result's batch ID. SG includes each actual request-call
interval. The existing overall transport deadline remains shared across the
entire run. Failure leaves incomplete evidence, submits no later group and does
not retry a request that may remain active on the service.

The evidence gate verifies exact schedule membership/order, all receipts,
complete/unique results, monotonic nonoverlapping barriers and original run
time bounds. Every actual input/logit/completion event must fall within its BG
cohort or SG request interval. Native IDs must equal `custom_id@actual_batch_id`
for BG and the original request ID for SG. Missing groups, altered sources,
different groups, duplicated results or evidence across a barrier fail. The
fixed comparison and independent scorer require the same schedule SHA for
both engines. Cross-node finalization inherits these checks through the pinned
original `validate_evidence`; it cannot rewrite receipts or event timestamps.

Older unannotated suites/runs remain readable as
`legacy_serial_receipts_unavailable`. A cohort16 suite cannot use that fallback.
Legacy and cohort16 runs cannot be paired. New cohort16 suites bind both their
collector source and the common helper source; use those frozen sources when
performing later offline validation.

All existing substantive gates remain active: exact native prefix/greedy token
binding, finite full129280-FP32 teacher vectors, teacher prefill phase, original
numeric limits, natural EOS and final-answer extraction, native/API agreement,
full16-rank ×61-layer ×attention/MLP ×prefill/decode coverage, and exact long
output lengths. The coverage16 wave remains first; the long singleton probes
retain sparse/empty-rank execution. Cohorts do not manufacture missing rank
coverage or replace native teacher evidence with API top-k logprobs.

Repeated questions retain separate IDs and separate actual outputs. Repeats
that share a cohort do not establish determinism across temporally separated
model reloads. Their existing answer checks or paired averaging remain exactly
as specified; no new equality claim or relaxed tolerance is introduced.

CPU checks use synthetic/mocked HTTP and thread barriers, never GPU services:

```text
python -m unittest discover -s tests -p test_r1_collection_cohorts.py -v
```
