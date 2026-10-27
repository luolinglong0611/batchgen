# Current128-request candidate: source and derivation contract

This is an offline source review and a CPU preparation tool. It does not release formal testing, run a model, send HTTP or change the old dataset. Current scope is **128 submitted requests per length profile, the same batch on BG/SG, three repetitions per engine/profile, median**. No automatic expansion to512/6000 is included. Correctness, independent quality, actual runtime alignment, full-rank shape warmup and output-length verification remain separate gates.

## Frozen original source

The old workload is LongBench-v1-derived, dataset revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, raw task templates, seed20261023, with-replacement selection of6000 per profile, identical source-sample order across profiles. Its original preparation already middle-truncated context to exact token lengths and excluded short/non-fitting rows; no filler was inserted. This is an existing campaign convention, not a recovered paper request artifact and not the full-context independent semantic suite.

Original bundle manifest SHA256:

`8dc0c57dbe130c496b85cbe5563ca71df7d58638d7ad9ae5b7044d9e3945baa7`

| Profile | Original requests SHA256 | Original provenance SHA256 | Original first128 effective body hash |
|---|---|---|---|
|8192→2048|`c52bec745989c2d170b1681bc474605a39c3b534bdc89330155164303259a4ed`|`edccb1acbb7fd3358ce03d521258d1acf0c11f9f8e15bb9dbe9f61f545ec9b3f`|`b706b4e5d37f7630a57308cb85d9eecfc4d2608abe129dd5a18a71fdfd9340e9`|
|2048→8192|`e5f907b18a65f3580ab43563bf893c38f74e6e30cd5b9b67fdf884926340fe9a`|`398528c8c26bc6667dadb3f7d1497509da56f2104c4a345356f7821c6642f4ed`|`50f2778508a7a087b28dd56048767cc0040d31124098eac7bfd204c0eb1ad024`|

The effective hash is the historical client algorithm: SHA256 over each ordered `custom_id + NUL + canonical(body) + newline`. The new candidate explicitly adds top_k1/repetition_penalty1 and freezes the requested generation policy; therefore its **new** body/file hash is recorded separately and must not be mislabeled as the unchanged historical effective hash.

Remote original source location:

`/home/admin/batchgen-r1-repro-20261023/workload-formal-6000-v1/manifest.json`

A published source-bundle recovery location exists in the old persistence script:

`/mnt/fuse/oss/luolinglong.lll/batchgen-r1-repro-20261023/resumed-r1/workload-formal-6000-v1.tar.gz`

Its archive receipt should be `/home/admin/batchgen-r1-repro-20261023/workload-formal-published.json`. These are recovery leads, not a claim that this review opened or verified the remote archive. Any recovery must still match the manifest and four artifact hashes above.

## Locally verified512-prefix provenance

Real submitted prompt envelopes, not response records, are available locally at:

`/Users/lilele/Desktop/batchgen/.repro-private/index64-resume-20261024/closed-mmap-prefix512-{profile}/files/bg-prefix-r1-fast-diagnostic-v10-{profile}-n512-v3/client/measurement.submitted.jsonl`

For the two profiles, those complete512-line files hash to `9721250843e81f9f98303c79f53d3b5102c5ff978619a22502c70a9c98f70683` and `34c192aa117769e2acfe5dd59f3557ab80989e04efb576fced67775850490e88`, respectively. They contain full `body.prompt` and ordered IDs `r1-000000` through `r1-000511`. Recomputed first128 effective hashes exactly match the historical6000 source inspection above. First512 hashes are respectively `a613d76c5e1ca351f46cb904d5eb77962a7126c7a1ba632b2f5d0cb8864d1ba8` and `d98aefa558fec60dc754807ba9d3d0c79c5b875a8bcd331cc1c5a4bba2928f7a`.

The original first128 includes126 distinct source samples and two repeated selections; those repetitions must remain. The local response metadata supports a17-task selection, but is not used to reconstruct native input IDs. All locally inspected original prompts lack a literal leading BOS. This does **not** prove their lengths after the current automatic BOS rule. Full original6000 provenance/native-ID files were not recovered locally; `usage.prompt_tokens`, decoded output text, or response metadata cannot substitute for them.

## CPU derivation tool

`prepare_r1_formal128.py` accepts the complete pinned original bundle. Default is plan-only: no tokenizer imports or payload writes. With explicit CPU guard and `--execute`, it:

1. Hashes all four original files and checks6000 rows, paired IDs/order, original native IDs/hash/length and source-row provenance. It retains only deterministic prefix128 in memory; it does not search for better-fitting samples or deduplicate.
2. Verifies original `tokenizer.json` and `tokenizer_config.json` hashes in both the original model path and BG's actual bundled assets. It loads current local HF tokenizer and the actual BG tokenizer class sources, with unrelated registry imports disabled, without loading engines/models.
3. Encodes every unchanged original prompt through current HF, BG single encode, and the actual unpadded/non-truncated BG worker batch call. All three must equal the original native IDs, with exact8192/2048 lengths. Prompt UTF8 hashes and original body/ID hashes are recorded. Any BOS+1, same-length ID difference or worker-only discrepancy blocks preparation.
4. Only after every selected row in both profiles passes, writes byte-identical BG/SG completion envelope files. Policy is temperature0, top_k1, top_p1, repetition_penalty1, ignore_eos=true, output2048/8192, non-streaming n1; no extra stop strings are accepted. Native-input sidecars preserve original source-row provenance, selected order, duplicates, native IDs, prompt/body hashes and all generation-field changes.
5. Leaves `formal_test_permitted=false`, `model_quality_qualified=false`, actual prefix/spec controls unverified and actual output lengths unqualified. Prefix reuse/speculation must be OFF in both actual instances. The three-repetition plan reuses exactly the same frozen request bytes; this tool does not launch repetitions.

If current BOS changes an input to8193/2049 or the original text cannot match the intended native IDs, the tool emits diagnostic rows and explicit alternatives, with **no prepared request payloads**. It never trims text, removes a token, substitutes synthetic filler or changes the selection. Separately reviewed alternatives are retaining the actual different lengths, using a genuinely validated common native-ID path, or explicitly regenerating a new payload from the same original raw source rows; none is automatic.

Suggested remote CPU invocation after source recovery/deployment (root supplies the frozen current tokenizer source SHA and actual hostname):

```bash
CUDA_VISIBLE_DEVICES='' R1_FORMAL128_CPU_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/admin/batchgen-r1-alignment-20261026/env/bin/python \
/home/admin/batchgen-r1-alignment-20261026/source-main-33ce112/scripts/r1_reproduction/prepare_r1_formal128.py \
  --source-manifest /home/admin/batchgen-r1-repro-20261023/workload-formal-6000-v1/manifest.json \
  --tokenizer-dir /mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8 \
  --repo /home/admin/batchgen-r1-alignment-20261026/source-main-33ce112 \
  --bg-tokenizer-source-sha256 CURRENT_FROZEN_SOURCE_SHA256 \
  --expected-hostname ALLOCATED_HOSTNAME \
  --output /home/admin/batchgen-r1-alignment-20261026/formal128-candidate-v1 \
  --execute
```

Omit `--execute` for a read-only plan. Output must be fresh and outside original workload/tokenizer asset directories. CPU tests use tiny independent fixtures and tokenizer stand-ins; their pass does not claim the real128 source has already passed current tokenization. No real source tokenization, GPU or HTTP execution was performed by this source-review task.
