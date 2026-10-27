# Explicit fixed-suite v2: bounded R1 semantic generation budget

This revision changes the **generation budget**, not the answer, natural-EOS or numerical acceptance standard. It does not release formal performance testing. The old suite and observed run artifacts remain immutable.

## Observed old-budget limitation

The retained private snapshot `.repro-private/r1-alignment-20261026/sg-first-responses.json` has SHA256 `f2aea3488e5e4c51e627e1ef16b35acd9ded264731a16d7af34a2b2bc1a12d3e`. Its two `gen-arithmetic-r0/r1` responses each contain23 prompt tokens and384 completion tokens, `finish_reason=length`, an unclosed `<think>` section and no final `FINAL:` answer. The two native output-ID sequences are identical. Although the reasoning text includes323, it does **not** satisfy the frozen final-answer or natural-EOS gate.

These observations are retained as failures under the old384 budget; an incomplete old84 collection is not relabeled as a complete passed run. The truncation alone does not establish an arithmetic/kernel error: it establishes that the observed responses reached the old budget without completing their final answer. Available prefix/teacher/layer evidence can remain diagnostic, but no old success is inherited into the new suite. The separate96-question/repeat semantic qualification cannot retroactively convert an old fixed84 failure into a pass.

## Exact versioned change

| Contract | Legacy v1 | Explicit v2 |
|---|---|---|
| Schema |`r1-full-model-fixed-suite-v1`|`r1-full-model-fixed-suite-v2`|
|12 short semantic generations |384 max tokens, natural EOS |8192 max tokens, natural EOS |
|6 short nonsemantic boundary generations |129 tokens, ignore EOS |unchanged |
|48 teacher-prefix jobs |Four fixed one-token targets per case, independent prefix prefill |unchanged |
|16 coverage jobs |63-token inputs,3 generated tokens, actual16-rank wave |unchanged |
|2 long-correctness jobs |8192→2048 and2048→8192, exact output/ignore EOS |unchanged |
|Sampling, input/native IDs, answers, teacher targets, numerical thresholds |Existing frozen conditions |unchanged |

The new8192 cap reuses the already predeclared independent96-suite budget. It is one fixed cap for all12 semantic generation jobs on both engines; it is not selected per question/backend by trying successively larger values. `ignore_eos=false` and `finish_reason=stop` remain required for those12 jobs. A correct number inside unclosed reasoning still fails. If8192 is exhausted, the result remains a failure under v2; there is no automatic further increase.

`NUMERIC` and its contract version remain unchanged. The v2 suite adds an explicit `budget_protocol` record and locks the full84 job IDs/case/kind/stage roles, including all12×4 reviewed teacher positions. A missing teacher cannot be replaced by an unknown case merely to retain84 rows. Actual preparation regression tests compare v1/v2 job dictionaries and require exactly12 `max_new_tokens` changes, with all72 other jobs byte-for-byte identical as data objects.

The default `prepare` remains v1 for compatibility. Select **`--suite-version v2` explicitly**. Existing v1 manifests and runs remain valid inputs to the legacy validator/comparator; a natural-EOS failure remains a hard failure. New v2 collections record `suite_schema` and require it to match. Both versions continue to require the exact suite SHA in run/trace bindings, preventing old384 captures from being mixed into the new8192 run. Existing independent96 legacy metadata compatibility is preserved; trace/runtime/production source is not changed by this revision.

## Preparation and rerun protocol

The caller must first close the old collector/service safely and archive its source, suite, deployment, request responses and trace. Do not replace a file used by a running collector. Prepare a fresh suite file, bind its new SHA and the new qualifier source SHA into new deployment/run identities, and repeat **both** BG and SG under the same new suite. The successful short+long+teacher smoke remains a prerequisite for the separate independent96 quality gate. Neither suite is a performance measurement.

Example CPU fixture preparation using the current original bundled assets:

```bash
/home/admin/batchgen-r1-alignment-20261026/env/bin/python \
/home/admin/batchgen-r1-alignment-20261026/source-main-33ce112/scripts/r1_reproduction/qualify_r1_full_model.py \
  prepare --suite-version v2 \
  --tokenizer-json /home/admin/batchgen-r1-alignment-20261026/source-main-33ce112/batchgen/models/deepseek/deepseekv3/tokenizer.json \
  --tokenizer-sha256 ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d \
  --output /home/admin/batchgen-r1-alignment-20261026/full-model-fixed-v2-semantic8192.json \
  --execute
```

Omit `--execute` for plan-only. The output must be fresh. The adjacent original `tokenizer_config.json` identity remains required. The new schema version is distinct from historical filenames such as `full-model-suite-v2-bos.json`, whose stored schema may still be v1; read the file's schema and hash, not its filename.

This local implementation/review ran CPU contract tests and a prepare plan only. It did not prepare the real new fixture, deploy remotely, send HTTP, run GPU inference or modify any retained private artifact. The tests cover old384 truncation remaining hard failure, new8192 retaining natural-EOS requirements, numerical/answer preservation, exact12/72 job differences, and version/SHA/teacher substitution rejection. `full_model_qualified` and `formal_performance_permitted` are not granted by preparation.
