# R1 full-model correctness evidence and independent semantic regression

Status: tools and CPU contracts implemented; **no service/GPU request was run
by this task**. Current source base is upstream `33ce112`. This qualification
does not override earlier failed operator/layer checks or permit performance
experiments on its own.

## Actual API audit

Main BatchGen uses files/batches. `/v1/inference` explicitly returns 410.
Completion prompts are strings, not native token vectors. The worker's
`_gather_completed_tokens` decodes the native CPU token trajectory to text;
the completion queue otherwise retains text and lengths. The `logprobs` schema
field does not supply a matching teacher/full-vocabulary output path.

SGLang v0.5.5.post3 [GenerateReqInput](https://github.com/sgl-project/sglang/blob/v0.5.5.post3/python/sglang/srt/managers/io_struct.py)
accepts native IDs, requested-token logprobs and top logprobs. Its
[tokenizer manager](https://github.com/sgl-project/sglang/blob/v0.5.5.post3/python/sglang/srt/managers/tokenizer_manager.py)
returns native output IDs and selected logprobs. These subsets cannot establish
full-vocabulary divergence; the
[logits processor](https://github.com/sgl-project/sglang/blob/v0.5.5.post3/python/sglang/srt/layers/logits_processor.py)
provides the actual next-token logits inside the engine. No additional global
TP gather may mix different DP requests.

Raw official source SHA256 checked during this task:

| v0.5.5.post3 source | SHA256 |
|---|---|
| `managers/io_struct.py` | `2f6f2318df448a3577c4f9800ef09b3209a3e04fef29e11872a526088d3f416f` |
| `managers/tokenizer_manager.py` | `4d47295efe88ca894e4f0e1f232c542146e59c658fe2e0da7fbeb86017c49b51` |
| `sampling/sampling_params.py` | `50b85946103d0b5a0adf84f771bc2a68e0d5de2317f072f554e968fe07b0cd9c` |

The previous branch's `full_model_correctness.py` and native-token diagnostic
were consulted for contracts. Their explicitly diagnostic results are not
imported as a pass, and no old performance implementation was copied.

## Capture helper and minimal integration points

`r1_full_model_trace.py` imports no Torch/model library on its disabled path.
It never replaces a token, changes an activation or sanitizes nonfinite values.
Enable only for qualification and disable before performance collection.

Required environment:

```text
R1_FULL_MODEL_TRACE=1
R1_FULL_MODEL_TRACE_DIR=/absolute/fresh/service-trace-directory
R1_FULL_MODEL_TRACE_RUN_ID=explicit-service-run-id
R1_FULL_MODEL_TRACE_SUITE=/absolute/frozen-suite.json
R1_FULL_MODEL_TRACE_SUITE_SHA256=...
R1_FULL_MODEL_TRACE_DEPLOYMENT=/absolute/deployment.json
R1_FULL_MODEL_TRACE_DEPLOYMENT_SHA256=...
```

The deployment schema is `r1-full-model-deployment-v1`, with `backend`,
`run_id`, `model_id`, `endpoint`, `checkpoint_identity`, `world_size=16`,
`num_hidden_layers=61`, `vocab_size=129280`, `tokenizer_json_sha256`,
`graphs=false`, `prefix_cache=false`, `speculative_decoding=false`, and
`source_files={role:{path,sha256}}`. Source files are actual loaded-code files,
not mutable fixtures/launch logs. Include this helper itself as a source role.
Every rank reads and verifies the source files; final-request logits record
their hashes again. Freeze the same model/source identity for the smoke and
independent-semantic stages. A changed code identity requires new qualification.

Parent integration points, with no core edits made by this task:

1. At the end of R1 PSM `configure_prefill` / `configure_decoding`, after the
   actual wrappers/model exist, call
   `register_model_hooks(model, rank, world_size, phase)`.
2. At `batchgen_worker._select_tokens` entry call
   `trace_bg_logits(worker, logits, batch_sequences)`. It reads actual
   `model.config.phase`, including re-entry prefill, rather than guessing from
   output count. It captures the same logits the real greedy selection uses.
3. In `_gather_completed_tokens`, after the nonempty-completion check and
   **before owner filtering**, call `flush_layer_stats(rank, world_size)`.
   In the owner branch after obtaining `decoded_tokens`, before release, call
   `trace_bg_completion(worker, seq, decoded_tokens)`.
4. At intake token-book creation call
   `record_input_tokens(request_id, input_ids, rank, world_size)`; the first
   actual logits also confirm the native prefix. BG input needs no new HTTP
   API: preparation requires manifest IDs -> text -> production-style encode
   to be exactly equal before submission, then checks the observed native IDs.

The helper attaches hooks to actual 61 layers and their attention/MLP modules.
Packed BG prefill can bypass `layer.forward`, so actual attention and MLP child
events are required. Registering hooks is not execution coverage. Inputs choose
the floating hidden tensor, skipping SG's leading integer positions.

SG uses one model for both modes: register once with `phase='dynamic'`, call
`set_execution_phase(rank,world_size,phase)` before each true ForwardBatch,
then `capture_logits(rids, exact_prefix_ids, logits, rank, world_size, phase)`.
Use global forward mode for idle DP ranks. The SG bridge must preserve the
actual row/request/prefix mapping and flush statistics on all participating
ranks at completion boundaries. Two fixed phase registrations on one model
are rejected because they would fabricate phase coverage.

Every actual hidden input/output is checked for finite values. Journaling is
aggregated by rank/phase/layer/component, retaining checked totals, actual
forward counts, shapes and last nonempty execution timestamps; failures are
logged immediately. Repeated unchanged summaries are omitted. Every prediction
checks the complete vocabulary and records native prefix hash, context length,
greedy token and top-2 margin. Teacher rows additionally save all 129280 FP32
logits in a hashed little-endian binary file.

## Fixed startup, boundary, teacher and long stages

`qualify_r1_full_model.py` commands are `prepare`, `collect`, `compare`.
Every command is plan-only unless `--execute` is explicit. `collect` contacts
only the declared already-running endpoint, never starts/restarts services,
never retries failed requests, and never submits 512/6000 performance work.

Preparation requires `--tokenizer-json`, `--tokenizer-sha256`, `--output`.
It uses the pinned local tokenizer without loading a model or remote code.
The fixed suite contains:

- Six toy QA problems, each twice, natural EOS and max 384 output tokens;
  these are a startup sub-gate, not independent model quality qualification.
- Exact native prompt lengths 63/64/65/129 and 2048/8192, fixed decode129,
  covering prefill page edges and incremental decode page crossings.
- Sixteen short coverage probes dispatched together. Their count does not
  prove all ranks ran: all ranks/layers must actually appear in the trace.
- Up to four fixed teacher tokens per case. Each query conditions on the exact
  same known prefix and emits one actual next-token distribution. This is
  **independent prefix re-prefill scoring**, not incremental teacher forcing.
  Incremental decode is separately exercised by the generation jobs.
- Both final 8192->2048 and 2048->8192 correctness jobs, enabled only by
  `--long-correctness`. Both exact output lengths must finish successfully.

Collection requires suite and deployment paths plus explicit hashes, backend,
endpoint, existing trace directory and fresh output. Compare requires both
`run.json` paths and hashes, suite path/hash and the pinned tokenizer JSON.
It decodes **native** output IDs to score semantic answers; text re-encoding
is not native-token evidence. Supply `--long-correctness` consistently to
both captures and comparison for the final long smoke stage.

The predeclared numeric smoke contract compares full-vocabulary CPU-FP64
log-softmax distributions, allowing additive raw-logit offsets and ordinary
reduction differences. Sup logprob error <=0.5, logprob RMS <=0.1, JS divergence
<=0.001 nats, teacher-token error <=0.1 per token and <=0.03 mean are required.
A differing top-1 is allowed only when both observed top-2 margins are <=0.2;
such disagreements may occupy at most 5% of teacher positions. These are
**engineering smoke thresholds**, not paper quality thresholds, a derived
whole-model FP8 error budget, or statistical non-inferiority. They are frozen
before runs; failures are localized rather than followed by threshold relaxation.

Any nonfinite activation/logit, invalid native token/context index, output
length mismatch, failed semantic startup answer or changed source/input fails.
Missing native evidence/teacher matrices/actual rank coverage is incomplete.
All 16 ranks x 2 phases x 61 layers x attention+MLP require actual nonempty
execution in the capture window. Operator receipts cannot fill these cells.

Short success is `passed_full_model_short_stage_LONG_CORRECTNESS_PENDING`.
Long smoke success is
`passed_full_model_long_smoke_INDEPENDENT_SEMANTIC_PENDING`.
Both leave `full_model_qualified=false` and performance permission false.

## Independent full-context 48x2 semantic gate

`r1_independent_semantic.py import` reads the already frozen independent
payload. Its question, reference, selection, BG-request and SG-request files
are SHA-pinned in the script. The source contains 48 unique full-context QA
items across six LongBench task families, two repeats each, native inputs
1616..17636 tokens and answer aliases. Contexts, questions, raw completion
prompts and native IDs are preserved exactly; no clipped 8192-token proxy is
substituted. Old model results are not imported.

The original source sampling was temperature0.6/top_p0.95; SG used repeat
seeds 20261024/25 and BG lacked corresponding per-request seeds. The new
derived protocol is explicitly greedy temperature0/top_p1/top_k1, natural
EOS, max8192. This difference remains in provenance; it is not silently
described as replaying the old sampling configuration.

Import options: `--payload DIRECTORY --tokenizer-json FILE
--tokenizer-sha256 SHA --output semantic-suite.json [--execute]`.
Collection options match the smoke tool, plus explicit `--run-48x2` and
`--execute`. It requires an already-running service configured with this
semantic trace suite/hash; it does not change the service's environment.
The first 16 items provide a rank coverage opportunity, then items are sent
sequentially. The shared context contract is 32768, preserving each complete
source input and its output allowance. These are quality requests, not BCT
measurements.

`score` requires both new native-ID runs and hashes, semantic suite/hash,
the tokenizer, and the successful long-smoke comparison/hash. It rechecks
actual source/rank/layer/finite/native-token evidence on the semantic runs
themselves. Their model/source identity must match the qualified smoke runs.

Report both raw-completion scores and the observable final suffix after the
last `</think>`; no-think direct answers are allowed. Unclosed thinking has
no extractable final answer and scores zero. Normalize with lowercase,
ASCII punctuation deletion, article removal and whitespace collapse; take
best alias EM and token-overlap F1 separately. Keep all 96 items in the
denominator, retain every failure/truncation/unextractable answer, and report
per-item plus per-question paired differences, dataset and length bin.

Freeze the following **gross-regression rule**, averaging the two repeats
within each of the 48 questions before paired statistics. Fail if BG-minus-SG
mean EM <=-0.05 or mean F1 <=-0.05; also fail if the one-sided exact F1 sign
test is p<0.01 with mean F1<-0.02. Missing/runtime-invalid evidence does not
pass. If SG's mean F1 is zero, comparison is inconclusive. No paper-unreported
absolute accuracy target is invented, and failure to detect gross regression
does not establish non-inferiority or paper quality equivalence.

Only successful numeric/long smoke plus the independently observed paired
semantic stage can emit
`passed_full_model_fixed_correctness_and_paired_semantic_NOT_paper_quality_equivalent`.
Its `scope='full_model'` describes the observed complete model; the limited
dataset and engineering acceptance remain explicit. Kernel/layout/admission,
memory safety and all other alignment gates remain separate; this artifact
alone never launches or permits formal performance testing.

CPU contracts:

```text
python3 -m unittest discover -s tests -p test_r1_full_model_qualification_contract.py -v
```
