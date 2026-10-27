# Frozen original workload: explicit ordered prefix-N CPU derivation

`prepare_r1_formal_workload.py` prepares a new input candidate after the common capacity and submitted-request count are selected. It does not select capacity, run services, send HTTP, measure performance or release formal testing. The previous128-only tool remains unchanged at SHA256 `3f8d312f216215cd9a6cadce05efdc5a414828cf5d9de6906fdddd7c53a13aef`; the new tool reads/hashes/compiles those exact helper bytes and reuses its source, provenance, tokenizer and policy checks.

`--count N` is required. N must be a positive multiple of16 and at most6000. Selection is always the original firstN rows, preserving order and repeated source samples. There is no filtering, deduplication, replacement, fitting, text truncation, token deletion or filler insertion. The original dataset was already a LongBench-derived campaign workload with its recorded context truncation and sampling; the new derivation preserves those historical facts and does not claim the paper's original request artifact or an unmodified LongBench quality benchmark.

The source bundle is pinned by original manifest SHA `8dc0c57dbe130c496b85cbe5563ca71df7d58638d7ad9ae5b7044d9e3945baa7` plus all four original request/provenance file hashes. Every original file is completely read/hashed with6000-row/ID/order checks before the firstN prefix is accepted. All four artifacts are rehashed after tokenization and again after output. Manifest, original model/BG bundled tokenizer assets, executed BG tokenizer source bytes, frozen helper and derivation-tool identity are also checked for change.

For **all N rows in both length profiles**, the original unchanged prompt is encoded through the current local original HF tokenizer, current production BG single encode, and production BG unpadded/non-truncated worker batch path. All three must strictly equal the original sidecar native IDs and exact8192/2048 lengths. A historical first128 pass cannot substitute for rows129..N. A BOS+1 or any worker-only/late-row disagreement blocks the whole derived candidate; diagnostics and the existing explicit remediation choices are emitted without prepared request files.

On a successful CPU derivation, BG/SG completion envelope files are byte-identical, with output targets2048/8192, temperature0, top_k1, top_p1, repetition_penalty1, ignore_eos=true, non-streaming n1. No extra stop strings are inherited. Native-input sidecars retain the original full source-row provenance, order, duplicates, prompt UTF8 hash, old body hash, exact native IDs and every policy-field change. Prefix-N's original effective-body hash and its **new-policy** effective-body hash are computed afresh from all N rows; the old128 hash is used only as a separately named nested consistency check when N≥128, never as N's hash. The new derived manifest binds N, source hashes, selected IDs, selection hash, every output file hash and libraries/source identities.

Actual prefix reuse/speculation must remain OFF in both runtime instances. Actual input acceptance, output length enforcement, resource/shape qualification, independent quality and formal timing/repetitions are separate. Every derived manifest keeps `formal_test_permitted=false`, `model_quality_qualified=false` and `output_length_runtime_qualified=false` even after CPU tokenization succeeds.

The original bundle has been restored on node0 under `/home/admin/batchgen-r1-alignment-20261026/recovered-original-workload-v1`. Node1 may restore the same frozen OSS bundle and verify the same pins. Root selects N from the final common C/N plan and runs the following only as a CPU preparation step:

```bash
CUDA_VISIBLE_DEVICES='' R1_FORMAL_WORKLOAD_CPU_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/admin/batchgen-r1-alignment-20261026/env/bin/python \
/home/admin/batchgen-r1-alignment-20261026/source-main-33ce112/scripts/r1_reproduction/prepare_r1_formal_workload.py \
  --count SELECTED_MULTIPLE_OF_16 \
  --source-manifest /home/admin/batchgen-r1-alignment-20261026/recovered-original-workload-v1/manifest.json \
  --tokenizer-dir /mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8 \
  --repo /home/admin/batchgen-r1-alignment-20261026/source-main-33ce112 \
  --bg-tokenizer-source-sha256 86f7297714f2bb3d848b3b91ce6f02c1367a1beae1d037ed48bbf5f83a7a1584 \
  --expected-hostname ALLOCATED_HOSTNAME \
  --output /home/admin/batchgen-r1-alignment-20261026/formal-prefix-N-candidate-FRESH \
  --execute
```

Omit `--execute` for plan-only; the tool does not import Torch/tokenizers or write an output directory in that mode. `--output` must be fresh and outside original source/tokenizer/production-code directories. A failed final recheck leaves an explicitly unusable report and never a released input manifest. This implementation was tested locally with independent tiny tokenization stand-ins and full6000-row tiny-source fixtures, including N176 and a mismatch after row128. It did not execute real remote tokenization, GPU, HTTP or model work.
