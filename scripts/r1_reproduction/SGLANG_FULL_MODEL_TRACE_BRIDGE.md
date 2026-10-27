# SG actual request-row full-model trace bridge

Status: CPU contracts and the exact installed-stock source plan pass. No remote
operation, model startup, inference request or GPU qualification was executed
by this implementation task. All collection/numeric thresholds remain the
previously frozen engineering smoke contract.

`prepare_sglang_r1_full_model_trace.py` first admits all five frozen installed
SG source files. It calls the existing local-shared/common-kernel generator
without changing its implementation, then appends only an opt-in trace
bootstrap to the resulting model file. It emits the stock-to-combined diff
and the four-line alignment-to-trace diff separately.

Current exact model identities:

| Model source | SHA256 |
|---|---|
| Installed stock | `620cd09545404fd62952e2239b905cfb26bc7565b361ac76bdf8f6d89ca80f49` |
| Existing alignment candidate | `a00712388404cf3a22a70f770f8bbaf1adc2ed7127182dda7fd47fb94c4897cf` |
| Alignment plus trace bootstrap | `b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd` |

Plan or prepare an isolated artifact:

```text
python3 scripts/r1_reproduction/prepare_sglang_r1_full_model_trace.py \
  --source-root /absolute/stock/sglang/srt

python3 scripts/r1_reproduction/prepare_sglang_r1_full_model_trace.py \
  --source-root /absolute/stock/sglang/srt \
  --output-dir /absolute/fresh/sg-trace-candidate --execute
```

The second command writes a candidate directory only. It does not replace the
installed package. Parent orchestration must deploy the reviewed model file,
this bridge and the common trace helper on both nodes, preserving the existing
alignment adapter files and environment. The other four SG files remain stock;
runtime admission rechecks their exact SHA256 and the four method signatures.
Unknown source changes or an unknown composed model are rejected.

Additional environment, alongside the complete suite/deployment trace
environment documented in `FULL_MODEL_QUALIFICATION.md`:

```text
R1_FULL_MODEL_TRACE=1
R1_SG_TRACE_MODEL_SHA256=b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd
```

The source bootstrap invokes `install()` while importing the model. With the
flag off it imports no SG/Torch dependency and installs no wrapper. With the
flag on, runtime requires SG0.5.5.post3, TP16/PP1, a non-draft model, disabled
CUDA Graphs/overlap/piecewise Graph/TBO/SBO/speculation, and the local DP LM
head. No arithmetic, sampling or collective is added or replaced.

The exact evidence chain is:

1. `ScheduleBatch.get_model_worker_batch` copies the actual `Req.rid`, original
   IDs, generated IDs and consumed prefix into immutable tuples. It checks the
   real `seq_lens_cpu`, request-pool row indices and flattened model input IDs.
   Extend rows additionally match their actual prefix/extend partition and
   `fill_ids`. Overlap is rejected, so unresolved asynchronous token futures
   cannot be mistaken for native IDs.
2. `ForwardBatch.init_new` transfers that exact metadata and input snapshot
   across the worker boundary. The runner rechecks the actual native input,
   sequence lengths and row order before model execution.
3. `ModelRunner.forward` registers actual 61-layer dynamic hooks and sets the
   execution phase **before** the original model call. Active local mode is
   used; an idle DP rank uses the actual global mode. Mixed/speculative/split
   modes are rejected rather than assigned guessed identities.
4. `LogitsProcessor.forward` marks its actual output with the current exact
   ForwardBatch identity. The bridge waits for stock
   `post_forward_mlp_sync_batch` to finish and captures the actual returned
   logits that the sampler will consume. This avoids interpreting DP padding
   as a request row. A source marker, row count and full 129280 vocabulary are
   all required. There is no extra global-TP all-gather or guessed DP slice.
5. Intermediate prefill chunks are not counted as sampled predictions. Only
   rows whose complete actual original/generated prefix has been consumed are
   emitted. Native completion IDs remain the original `/generate` API output.

The trace helper still inspects every actual hidden input/output for finite
values. Owner terminal predictions flush their accumulated statistics; the
bridge also flushes all local statistics every 64 actual forwards and on an
all-idle boundary. Missing final/phase/rank evidence remains incomplete, never
a fabricated coverage pass. Source/import profiling before the first scheduled
request is not assigned a synthetic request ID.

The common helper uses weak model identity: a recreated PSM model can reuse a
Python `id`, so stale handles are removed when the weak referent differs.
The registry does not retain a model or its GPU weights. CPU tests cover this
lifecycle, true pre-forward phase selection, native ledger mutation, DP padded
versus actual returned rows, partial prefill, idle ranks, source/signature drift
and default-off behavior.

```text
python3 -m unittest discover -s tests -p test_sglang_r1_full_model_trace.py -v
python3 -m unittest discover -s tests -p test_r1_full_model_qualification_contract.py -v
```

These are CPU contracts, not measured full-model correctness. Actual native
trace, numeric, complete long profiles and the independent semantic gate must
still be collected on the deployed services.
