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
