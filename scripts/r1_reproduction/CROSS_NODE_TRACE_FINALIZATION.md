# Offline completion of full-model trace evidence

`qualify_r1_full_model.snapshot_trace` reads one local directory. Identical
absolute paths on two nodes do not share files. A node0 collector may therefore
save complete HTTP results but only ranks0..7 traces. Preserve that original
collection unchanged; it is not a16-rank correctness pass.

The new CPU-only `finalize_r1_cross_node_trace.py` seals the two raw snapshots
and creates a **new derived run**. It does not send requests, modify services,
install software, rewrite raw events or grant full-model/performance approval.
Both subcommands default to a read-only plan; writing requires `--execute` and
a fresh absolute output directory.

`--mode fixed` (the default for existing callers) accepts only the original
fixed correctness suite through `qualify_r1_full_model.validate_suite`.
`--mode independent` accepts the separate 96-request full-context semantic
suite through the pinned `r1_independent_semantic.validate_suite`. Both the
original semantic source SHA and its common qualifier source SHA are required;
the semantic module's adjacent `qualify_r1_full_model.py` must match that exact
qualifier SHA. The original run's collector SHA, independent-semantic flag and
suite's semantic source SHA must match. Neither suite is relabelled as the other.

1. After the collector closes, preserve its exact `run.json`, suite, frozen
   deployment and the exact qualification-tool source used for that capture.
   Copy the complete stable raw trace directories from both actual nodes,
   including `rank-*-pid-*.jsonl` and every referenced native `*.f32` file.
   Parent-owned transport/archive receipts establish where the snapshots came
   from. Do not copy only node0's already-filtered `events.json`.
2. Run `seal` for node0 and node1, either at the origin or against their copied
   snapshots. Pass the actual origin hostname explicitly; `sealing_hostname`
   separately records where this CPU operation occurs. The same exact original
   collector run and input/source pins must be used for both nodes.
3. Keep each `sealed-node*/manifest.json` and complete `raw/` directory, and
   pass their exact manifest hashes to `finalize`. It writes a fresh union trace
   and derived `run.json`, preserving every original field except the trace
   location/identity and adding explicit derivation provenance.
4. Archive both sealed snapshots, the original collection, derived directory,
   all suite/deployment/source/launch/guard receipts and their SHA manifest.
   The existing paired correctness comparison consumes the new derived
   `run.json`; the original partial run stays unchanged.

Each `seal` invocation:

```text
python scripts/r1_reproduction/finalize_r1_cross_node_trace.py seal --mode fixed --execute
  --original-run ORIGINAL_COLLECTOR_RUN_JSON --original-run-sha256 ORIGINAL_RUN_SHA
  --suite FROZEN_SUITE_JSON --suite-sha256 SUITE_SHA
  --deployment FROZEN_DEPLOYMENT_JSON --deployment-sha256 DEPLOYMENT_SHA
  --qualification-tool ORIGINAL_QUALIFY_R1_FULL_MODEL_PY
  --qualification-tool-sha256 ORIGINAL_QUALIFIER_SHA
  --trace-directory COPIED_OR_ORIGIN_RAW_NODE_TRACE_DIRECTORY
  --node-rank NODE_RANK --origin-hostname ACTUAL_ORIGIN_HOSTNAME
  --output FRESH_ABSOLUTE_SEALED_NODE_DIRECTORY
```

The printed plan/receipt is not the manifest file hash. Hash the resulting
`manifest.json` bytes for the following explicit pins. Finalization:

```text
python scripts/r1_reproduction/finalize_r1_cross_node_trace.py finalize --mode fixed --execute
  --original-run ORIGINAL_COLLECTOR_RUN_JSON --original-run-sha256 ORIGINAL_RUN_SHA
  --suite FROZEN_SUITE_JSON --suite-sha256 SUITE_SHA
  --deployment FROZEN_DEPLOYMENT_JSON --deployment-sha256 DEPLOYMENT_SHA
  --qualification-tool ORIGINAL_QUALIFY_R1_FULL_MODEL_PY
  --qualification-tool-sha256 ORIGINAL_QUALIFIER_SHA
  --node-manifest SEALED_NODE0_MANIFEST --node-manifest-sha256 NODE0_MANIFEST_SHA
  --node-manifest SEALED_NODE1_MANIFEST --node-manifest-sha256 NODE1_MANIFEST_SHA
  --output FRESH_ABSOLUTE_DERIVED_DIRECTORY
```

For the independent 48x2 collection, use its original semantic run/suite/
deployment and replace `--mode fixed` with the following in **both** commands:

```text
  --mode independent
  --semantic-tool ORIGINAL_R1_INDEPENDENT_SEMANTIC_PY
  --semantic-tool-sha256 ORIGINAL_SEMANTIC_SOURCE_SHA
```

Keep `--qualification-tool` and `--qualification-tool-sha256` as well. Mode,
suite schema and both source identities are bound into each seal and derived
receipt; manifests from different modes cannot be merged. Seals also bind the
exact finalizer source SHA, so use one version for sealing and finalizing.
The derived provenance directory includes the independent source bytes when
that mode is selected.

Optional repeated `--external-evidence ACTUAL_LAUNCH_OR_RUNTIME_RECEIPT` plus
matching `--external-evidence-sha256 SHA` archive those original bytes in
`provenance/`. This tool preserves arbitrary extra frozen deployment fields,
including actual argv if present, but does **not** infer actual argv from a
plan or validate unknown external-receipt schemas. Actual launch/runtime
parameter review remains explicitly pending.

Hard storage/identity gates include both actual node partitions, distinct
origin hostnames, all16 ranks, exact suite/deployment/run/source identities,
one PID file per rank, complete newline-terminated JSONL and consecutive real
worker ordinals. Restarted/mixed workers, path traversal, symlinks, collisions,
tampering and changing snapshot files are rejected. The original node0 saved
events must all survive byte-equivalent JSON values, without deduplication.

The derived union uses the **original** collector start/finish timestamps.
`trace_start`/model-registration events can precede collection but cannot
postdate it. All other events stay inside that original interval. Clock skew
does not authorize expanding the interval or rewriting observations: missing
node1 coverage remains missing. Original native request/prefix/step/phase data
are preserved. Every selected teacher vector must be the exact native129280
FP32 little-endian values, correct size/hash and entirely finite; the original
`read_logits` and `validate_evidence` implement the existing substantive gates.
Independent mode uses exactly the independent scorer's existing native-evidence
call: `long_stage=False, semantic_smoke=False, context_limit=32768`. Its paired
answer-scoring rule and separate successful long-smoke prerequisite remain in
`r1_independent_semantic.score`; this utility does not run, relax or replace
either. Fixed mode retains the original fixed-suite evidence call unchanged.

`finalization.json` reports any missing actual16-rank ×61-layer ×prefill/decode
×attention/MLP coverage, missing native output/logit steps, source mismatch or
other existing native-evidence failures. An incomplete/failed HTTP capture
cannot be repaired by offline union. Even native-evidence success is labelled
`PARAMETERS_AND_PAIR_PENDING`; full-model, independent semantic, long-correctness
and formal performance gates remain separate. No thresholds are altered.
