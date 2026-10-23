`qa_sanity.py` prepares and scores a separate 64-item check. Its exact scope is
`paired clipped-context QA sanity, not original LongBench leaderboard or statistical pass`.
It has no HTTP execution command, service operation, model import or pass threshold.

The source is the pinned `workload-formal-6000-v1/manifest.json` (SHA256
`8dc0c57dbe130c496b85cbe5563ca71df7d58638d7ad9ae5b7044d9e3945baa7`),
using its 8192-input / 2048-output profile. Preparation verifies both source file
hashes before reading and again before writing a new directory. It keeps the
original raw prompt, token IDs, references and clipping provenance. It selects
only English `qasper`, `hotpotqa`, `2wikimqa`, `musique`, `narrativeqa`, `triviaqa`
and `multifieldqa_en` samples actually present in the manifest. It deduplicates
`(dataset, original_id)`, rejects conflicting duplicates, sorts identities by
SHA256 of canonical JSON `[20261023, dataset, original_id]`, then takes 64.
Selection does not depend on model responses or scores.

The new files freeze sample IDs, request prompts, exact input IDs, all reference
aliases, source hashes and generation settings. The source 6000 files are never
modified. The derived generation is greedy (`temperature=0`, `top_p=1`, `top_k=1`),
with EOS allowed and a cap of 2048 new tokens. The source performance workload
used `ignore_eos=true`; this deliberate QA derivation records that change.

CPU-only preparation, after reviewing the source and candidate script:

```bash
python3 scripts/r1_reproduction/qa_sanity.py prepare \
  --source-manifest /home/admin/batchgen-r1-repro-20261023/workload-formal-6000-v1/manifest.json \
  --output /home/admin/batchgen-r1-repro-20261023/qa-sanity-64-v1
```

`plan` only prints an existing-client command. It verifies frozen artifacts and
client source bytes. SGLang uses the existing completions client with closed-loop
concurrency 8, timeout 1200 seconds and zero warmup; BatchGen uses its existing
files/batches client with one batch, zero warmup and a 10240 context limit. Both
receive identical frozen request bodies and order. These admission modes and zero
warmup are for descriptive QA only; their timings are not a BCT/speedup comparison.
Verify deployed model/tokenizer, precision, cache settings, available context and
service readiness separately before executing any printed client command.

```bash
python3 scripts/r1_reproduction/qa_sanity.py plan \
  --manifest /home/admin/batchgen-r1-repro-20261023/qa-sanity-64-v1/manifest.json \
  --backend sglang --endpoint http://127.0.0.1:30000 \
  --output /home/admin/batchgen-r1-repro-20261023/qa-sglang-64-v1
```

SGLang's existing client keeps response bytes in memory until the entire phase
drains. It then saves each parsed complete response object, including usage and
finish reason, under `response` in `requests.jsonl`. A parse failure saves a
`raw_response` string instead. This is full response content, not byte-for-byte
HTTP archival. An interrupted phase can leave no request JSONL; monitor service
logs and the bounded guardian rather than treating absent JSONL as no activity.
BatchGen preserves downloaded output files and its validated
`measurement.requests.jsonl`. Do not discard these source response artifacts.

The scorer binds the client config to the exact QA manifest, effective payload,
request/provenance hashes and frozen client source. Unknown or duplicate IDs and
mismatched source identities are rejected. Missing/failed outputs remain in the
full 64-item denominator with zero descriptive score and explicit counts.

```bash
python3 scripts/r1_reproduction/qa_sanity.py score \
  --manifest /home/admin/batchgen-r1-repro-20261023/qa-sanity-64-v1/manifest.json \
  --backend sglang \
  --client-config /home/admin/batchgen-r1-repro-20261023/qa-sglang-64-v1/config.json \
  --results /home/admin/batchgen-r1-repro-20261023/qa-sglang-64-v1/requests.jsonl \
  --output /home/admin/batchgen-r1-repro-20261023/qa-sglang-64-v1-score.json
```

Two metrics are always reported as fractions in `[0,1]`, globally and per task:

- **A raw completion EM/token-F1:** score the entire returned text. A direct answer
  without thinking markup can score normally. This text is never called a proven
  final answer; reasoning or added explanation can reduce EM/F1.
- **B observable post-think suffix EM/token-F1 and coverage:** take the complete
  tail after the last `</think>`. Missing markup, missing closing tags, or an
  unclosed `<think>` in that tail are explicitly unextractable. Their zero values
  are a conservative lower bound, not evidence that the model's answer was wrong.
  Empty suffixes and multiple closing tags have separate counts. Coverage counts
  observable suffixes, including empty ones; nonempty suffix coverage is also saved.

Normalization is fixed: lowercase, delete ASCII `string.punctuation`, remove
standalone `a`, `an`, `the`, then collapse whitespace. EM compares the complete
normalized strings. Token-F1 uses whitespace-token multiset overlap. Each score
takes the best reference alias independently; no stemming, answer searching in
reasoning, first-line/last-sentence heuristics or gold-dependent extraction occurs.

`finish_reason=length` is counted as `length_capped` even when an observed suffix
happens to contain a complete answer. `stop` is only `reported_stop`; it does not
prove semantic completion or native EOS. A completion token count equal to the cap
is recorded separately. The check may still truncate R1 reasoning at 2048 tokens.
The middle-clipped context may have removed answer evidence, and 64 examples do
not establish statistical equivalence or a model quality pass.
Successful validated outputs supply `length_capped` and `reported_stop`; separate
`reported_*_all_responses` counts also include reported finishes from invalid rows.

`run_qa_sanity_guard.py` bounds the SGLang client to 5400 seconds and 2 GiB
session RSS, with a 128 GiB host available-memory floor. Its default is plan-only;
execution requires explicit QA source/manifest hashes and a separately reviewed
admission for the existing v4 service. That admission records the actual service
PID, PGID, SID and Linux start ticks, original child receipt SHA, tokenizer/KV
and routing compatibility evidence. Map the receipt's `sid`/`starttime_ticks` to
the ownership helper's `session`/`start_ticks` explicitly. The admission is an
operator record, not independent HTTP verification of all deployment properties.
The guard launches only the CPU client and can signal only its new process group;
it neither queries GPUs nor starts/stops the model service. An interrupted client
can leave HTTP requests running, which is reported for the resource owner to check.
Successful collection is scored with the frozen definitions and remains descriptive.

2026-10-23 SGLang v4 collection completed in `ROOT/qa-sglang-64-v1` under the
recorded `sglang0.5.5.post3+R1-routing-compat` deployment. All 64 HTTP responses
passed transport/usage validation; 59 reported `stop`, 5 reported `length`, with
23,086 output tokens total. **16 responses contained one output token and empty
decoded text**; transport success must not be described as 64 answered questions.
The cause of these empty completions is not established by this response data.

| Descriptive metric | Result |
|---|---:|
| A complete raw text EM / token-F1 | 0 / 0.1030309484 |
| B observable suffix EM / token-F1 lower bound | 0.078125 / 0.2331822953 |
| B suffix coverage | 36/64 |

Prepared manifest SHA256: `eb12105962cc77dd1f4ffe8423da91d0ba3e97b98c8af7e41236ceeca2351bb4`.
Response JSONL SHA256: `9d130296868b7bae94d727d616be22238b82c48565e88a27c62e2b0fc47baed4`.
Score SHA256: `498f956b8f3170633dabb39c9a75d620257dc3b4a1d99c8439044f0e13380106`.
The client guardian exited zero with no remaining session members or signals.
The exact artifacts were also fetched locally and verified. Python 3.13's default
floating-point aggregation differs from the original Python 3.11 aggregate by
about 1e-17; explicit left-fold addition reproduces the original score bytes.
No source score, metric or original report was changed. These raw, clipped prompts
and incomplete answer observability cannot establish model quality or statistical
equivalence. The formal performance campaign remains pending.

2026-10-24 BatchGen `r1-fast-diagnostic-v3` completed the same 64 frozen inputs.
All response IDs were present; 63 passed the existing client validation and
`qa-0036` failed because its API finish was `repetition` after 768 tokens.
This was a generated repetition termination, not an HTTP transport failure or
worker crash. The client exited1, so its original guardian remains failed and
the remote scoring step did not run. No generation was retried. Offline scoring
uses the unchanged scorer and all64 items, including the failed item's zero score.

| Observation / descriptive metric | BatchGen v3 | SGLang v4 |
| --- | ---: | ---: |
| Validated responses / denominator | 63/64 | 64/64 |
| API reported stop / length / repetition | 56 / 7 / 1 | 59 / 5 / 0 |
| Empty output text, all responses | 11 | 16 |
| A raw text EM / token-F1 | 0 / 0.1069283559 | 0 / 0.1030309484 |
| B observable suffix EM / token-F1 lower bound | 0.0625 / 0.2054005286 | 0.078125 / 0.2331822953 |
| B suffix coverage | 44/64 | 36/64 |

All64 BatchGen native completion traces match their frozen prompt token-ID
hashes, API token counts and finish reasons, with32 traces from each node.
The sole repetition log matches `qa-0036`, rank10, global index64 and768 tokens.
The11 empty outputs each have one native token; this analysis does not separately
bind the EOS configuration and does not infer an EOS cause from empty text.
The two APIs' identical payloads were checked, but BatchGen's repetition breaker
is an additional termination policy. These raw, clipped-context prompts are
descriptive QA and do not establish statistical equivalence or general quality.
Their different submission modes and zero warmup also preclude a speedup claim.

Preserved hashes:

- BatchGen original guardian: `c6060834a83ef912eaa8fcfef8b9e9131d898faeaa3af01fb49d86a81d39e6db`.
- BatchGen original client summary: `96d6eda69fc9c061df88807e79f29c2f0d91b77be6115a42a76d8143668d1e78`.
- BatchGen local-only recomputed score: `154444fb8f63d56453d393e1c9ed73d733f2f71185fa93b28a70d960d6db095c`.
- Paired offline comparison: `d7dc81a680a513efdd8714cbcd33b86681ab1f1f7e3792bb0067474f4ffce104`.
- Native/API pairing: `6b4dab05da0763a42c24d8acb1e9b8e709894f2b93204430a2bb79e9b233b000`.

The original guardian's `server_requests_may_remain=true` is retained. A later,
separate stop audit confirmed both owned model services exited0 after leader
SIGTERM, all original live session members disappeared, and all16 GPUs had
no compute processes, zero utilization and4MiB each. That later observation
closes resource cleanup; it does not change the failed QA outcome.
