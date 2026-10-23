# Independent R1 chat probes

`longer_chat_probe.py` freezes new protocols from the original six-case fixture
whose exact SHA256 is
`f7ed0434a2894d604115affeed28ba279e59acda0794932d6a21359fa649d53d`.
The original fixture, its chat strings/input IDs/teacher references, existing
6000 workloads, 64-QA data, and numerical gates are never edited. This tool
uses only Python's standard library and existing standard-library client helpers.

| Protocol | Requests | Output cap | Submission |
| --- | --- | --- | --- |
| `r1-longer-chat-2048-v1` | Six original chat cases, two copies each | 2048 | One BatchGen batch, or 12 SGLang `/generate` requests with concurrency 12 |
| `r1-identical-arithmetic-16x128-v1` | 16 copies of the original arithmetic chat, with identical prompt/input IDs | 128 | One BatchGen batch; native mode also supports 16 concurrent requests |
| `r1-identical-arithmetic-16x128-context12288-v1` | Same 16 arithmetic copies, independently named IDs | 128 | One BatchGen batch with pool context metadata 12288 |

Both use greedy temperature 0/top-p 1/top-k 1, allow EOS, and declare protocol
seed 20261023. SGLang receives `sampling_seed`; BatchGen has no request seed
field and uses greedy argmax. Teacher text/IDs remain in the frozen requests
as references; **no teacher-forced requests are sent**. A 2048-token cap may
still truncate thinking. Empty completions and finish metadata are retained.

Prepare and plan are CPU-only and make no HTTP calls. Invoking the script
without a subcommand prints help and performs no work. Use a new directory:

```sh
python scripts/r1_reproduction/longer_chat_probe.py prepare \
  --fixture /path/correctness-fixture-v1.json \
  --probe longer-chat --output /path/longer-chat-2048-v1

python scripts/r1_reproduction/longer_chat_probe.py prepare \
  --fixture /path/correctness-fixture-v1.json \
  --probe identical-arithmetic --output /path/identical-arithmetic-16-v1

python scripts/r1_reproduction/longer_chat_probe.py prepare \
  --fixture /path/correctness-fixture-v1.json \
  --probe identical-arithmetic-wide-pool --output /path/identical-arithmetic-wide-pool-v1
```

Prepare prints the manifest SHA. The new directory contains the original
fixture's **unmodified bytes**, a request file, and a manifest pinning these
files, generation settings, source helper hashes and all request IDs. Loading
also reconstructs the exact expected requests from the pinned source fixture;
rehashing a modified prompt/input-ID/teacher/order file does not make it valid.

Plan requires a backend, the explicit expected manifest SHA, one endpoint, and
a deployment-source JSON plus its expected SHA. The source may be an existing
deployment JSON or a run JSON with a `deployment` object. It must satisfy the
original fixed diagnostic's precision/tokenizer/source declarations, including
the audited SGLang commit `7b2fb3d47c9dc43367c6fd3bf9396f7ba45e6883`.
This is a pinned operator declaration, not remote verification of the server.
The full source JSON is copied verbatim when collecting a run.

```sh
python scripts/r1_reproduction/longer_chat_probe.py plan \
  --backend batchgen \
  --manifest /path/longer-chat-2048-v1/manifest.json \
  --expected-manifest-sha256 MANIFEST_SHA \
  --deployment-source /path/observed-bg-deployment.json \
  --expected-deployment-sha256 DEPLOYMENT_SHA \
  --endpoint http://node0:8000 --output /path/bg-longer-plan.json
```

Only the explicit `run` subcommand sends requests. Replace `plan` with `run`
and set `--output` to a new run directory after the deployment is admitted by
the coordinating operator. For native SGLang, use `--backend sglang`, its
deployment-source JSON, and `--endpoint http://node0:30000`. The identical-16
protocol is independently planned/run by supplying its separate manifest.
No service is launched or stopped. There are no discovery requests, warmups,
HTTP retries, automatic cancellation, API-key discovery, or GPU operations.
An external bounded guardian should own any actual client process.

BatchGen receives one files upload and one batch creation for the entire
request list. Polling and result downloads use the existing batch client with
a 1200-second phase deadline, 60-second per-HTTP timeout and EOS allowed.
The two original protocols retain batch context metadata 4096. The separate
wide-pool variant sets it to 12288 for an explicitly declared first-wave pool
intended to continue with longer requests, including the 8192+2048 QA sanity.
This changes neither arithmetic input IDs nor its 128-token generation cap.
The scheduler retains the first pool context metadata; worker model context
checks use the model's native window, so this metadata alone does not establish
a hard cap or prove actual KV capacity. This is not a formal KV-budget comparison.
Original downloaded result files and the complete parsed batch results are
retained. BatchGen API text is never re-tokenized into purported native IDs.
Its run records explicitly mark API native IDs unavailable.

SGLang receives the exact frozen `input_ids` through `/generate`, with
`return_logprob=true`. All 12 long-chat HTTP calls can be in flight together;
this is closed-loop concurrency 12 with exactly 12 total calls and a 1200-second
per-call timeout. The identical-16 variant uses concurrency 16. HTTP concurrency
does not guarantee identical internal scheduling or rank ownership across
engines. Every received HTTP body is saved byte-for-byte as `*.http-body`, and
the parsed response object, native `output_ids` if returned, output logprobs if
returned, usage and finish metadata are retained. Responses are saved as each
call finishes. Malformed/nonfinite responses remain collection failures with
raw bytes preserved. No extra retry is sent after a failed call.

The resulting `run.json` reports collection status and artifact hashes. It
does **not** define a quality gate or compare native outputs across engines.
The two long-chat copies share a mixed batch and may have different rank or
batch histories. Their differences must not be presented as determinism results.
Auxiliary client elapsed/rate fields are diagnostics and are not formal BCT.

For the identical arithmetic probe, an operator may separately enable the
opt-in `R1_NATIVE_TOKENS` worker trace in a fresh admitted engine. See
[R1_NATIVE_TOKEN_TRACE.md](R1_NATIVE_TOKEN_TRACE.md). Preserve the exact
service-run logs and associate records with the actual batch/UUID/global index
and original prompt hash. Compare native IDs only for matching prompt hashes,
complete traces and observed ownership. **Sixteen submitted requests do not
prove all sixteen ranks were covered.** Actual `rank`/`assigned_rank` records
must demonstrate coverage; missing/error/truncated traces remain missing
evidence. The helper neither enables tracing nor assumes a rank mapping.
To investigate fresh-engine first-wave ownership, this batch must be the first
submitted wave after the operator's admitted launch. The helper records that
intention but cannot prove freshness or prior endpoint activity.

CPU verification, including fake transports and a twelve-worker barrier:

```sh
python tests/test_r1_longer_chat_probe.py
```

These tests exercise exact freezing, source/deployment pins, rehashed input
tampering, single-batch submission, native response retention, empty EOS text,
nonfinite/error handling, no retries and twelve in-flight mocked native calls.
They do not load a model, call a real endpoint, or qualify GPU correctness.

Changing helper source changes its manifest source pin. Keep previously frozen
manifests and source snapshots for audit; prepare new directories for future
runs under the new source. Old manifest files must not be edited to substitute
a new source hash. The two original protocols preserve their request/teacher
bytes when freshly prepared with the new helper; only the new wide-pool
protocol has the new context metadata and independent request IDs/tag.

## Paired twelve-request observation, 2026-10-24 CST

BatchGen full-model cohort v3 and SGLang formal candidate A v2 both completed
the same six cases twice, using the native frozen input IDs and allowing EOS
within2048 output tokens. The shared manifest SHA256 is
`8704b23c1f0afd057ac8de65fdf3fd0789a8263a2afd90aef840fe8f0a772268`;
the effective payload hash is
`525ac820cbdb2c93236750da02fd993632fe11359fb82b78b50c21ef1a370843`.

Both produced12 successful, nonempty final answers with natural stop token1.
Manual review of each final answer after `</think>` found the expected answers:
323, x=5,60%, `silver maple 47`,14 and Yes, in both copies on both engines.
SGLang returned native output IDs and finite output-token log probabilities
for every request. Its10763 output tokens differ from BatchGen's11532;
this EOS-allowed diagnostic does not impose equal work or measure speedup.

The SGLang run SHA256 is
`c718f1e94b2100cb02f8f2726530626d3dfd05b067027e483f241b6e2267d557`;
its final guard is
`81211ff8f0b2917837b9c964edee9f54acb728f42044c04bfd075bf32bf41e6a`.
The original CPU guard/client exited normally, cleanup required no signals,
and the guard recorded no remaining service requests. This is a small
functional check, not a general quality, deterministic-output or performance
qualification. Earlier numerical-envelope failures and the separate QA64
failures remain unchanged; this result does not replace them.

## Graph-enabled A-v5 regression

The same twelve native payloads completed on the graph-enabled A-v5 candidate
at approximately20:57:55 UTC on2026-10-23. All12 returned successful responses,
nonempty correct final answers under the same six-case rubric, and natural
stop token1. Completion lengths ranged from194 to1666, below the2048 cap;
all native output IDs were available and no malformed/nonfinite output-token
log probabilities were recorded. Final-answer review found no degenerate
repetition in the answer suffixes; it did not assess every reasoning trace.

Run SHA256 is
`707005c4cc183d4c699492541cd37e0fd1a1ff1a5b4ddf2be3897045f8d9ee26`;
final guard SHA256 is
`9accebacd182ed5d16e40686fe2360346f627047e3cacf1ce0ac1c878076626f`.
Both original CPU identities were absent after natural exit0 with no cleanup
signals. All67 raw files were exported, with envelope SHA256
`424b3b5dbc76460c7635fc97ceefb207f8bfd6182b8f423b24e6088069665cc2`.
The final-answer review SHA256 is
`80ce75ea602762cd09e96b050ff3c0ebe24169227b550f03c625155a6cba62d1`.
This checks a small functional regression with graphs enabled; it does not
establish general accuracy, all-rank coverage or erase the prior failed gates.
