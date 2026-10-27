# Cross-node correctness clock evidence

This is an additional collection check. It does not modify the qualifier,
semantic collector, trace hook, model code, original run, or any timestamps.
The CPU-only helper is `r1_clock_evidence.py`; network commands default to
plan-only and require `--execute`.

## Gap in the existing evidence

`snapshot_trace` and the cross-node finalizer select ordinary events by the
client's original Unix start/finish. `validate_collection_evidence` compares a
worker's `observed_unix_ns` directly to the client's per-request or cohort
window. Node1's wall clock has no recorded relation to node0's wall clock.
An offset can both discard a valid event before validation and make an event
appear on the wrong side of a barrier. Native IDs, ordinals, source hashes and
full vector hashes remain necessary but do not establish a clock relation.

The current archived BGv2 and SGv3 runs have no such contemporaneous clock
samples. Today's samples cannot certify yesterday's offsets. Keep their raw
JSONL, complete FP32 files, original HTTP records, timestamps, failed gates and
three-phase windows unchanged. BGv2 arithmetic r1's repetition finish and
unclosed reasoning remain hard failures. Clock evidence cannot fix semantic,
numerical or dense FP8/BF16 alignment failures.

## Minimal new-run procedure

1. Freeze this helper on both nodes and bind its SHA and the clock policy to a
   new launch receipt. Use the exact model run ID for the clock session ID.
   Record collector hostname, Linux boot ID and node0 IP in the launch receipt;
   it must agree with every clock sample's client identity. Bind rank0..7 to
   node0 and rank8..15 to node1 through the existing deployment/host records.
2. Run the clock-only endpoint on node1 under the existing owned monitor. It
   binds one explicit IP/port and accepts the one node0 peer, one GET path and
   no body; it cannot read files, browse directories or execute commands.
   It exposes only current clock/host/source identity. Each socket has a
   three-second timeout. The monitor remains the hard lifetime bound.
3. Start probing **from the actual collector host node0**, not from the Mac or
   a third orchestration host. Preserve at least two complete exchanges before
   model HTTP collection starts, then continuous exchanges throughout it and
   at least two after it finishes. A one-second interval is an initial sampling
   choice; failures stop the probe, preserve its partial journal and leave a
   coverage gap. Do not silently restart the journal or concatenate sessions.
4. Predeclare an upper bound on continuous relative clock drift, a maximum
   acceptable probe gap, and the wall-versus-monotonic discrepancy threshold.
   Save the actual time-service status on both nodes before/after and inspect
   clock-step/reboot records. This helper **does not measure an upper bound on
   future drift and cannot prove there were no hidden clock steps between
   samples**. A finite certificate is conditional on that explicit clock
   assumption; do not invent a small drift value from a low RTT. If the
   required assumption is unsupported, keep the gate incomplete.
5. Stop the sampler only after the post-run samples; seal journal bytes, helper
   source, policy/host receipts and raw traces. For this check, supply an events
   JSON array formed from **all sealed raw JSONL before time filtering**, not
   just the existing finalizer's selected events. Preserve every event object;
   bind the array SHA and its raw-seal provenance. The helper checks input
   hashes and event run/suite/deployment identities but does not itself prove
   that an externally supplied array contains every raw event. The finalizer
   and caller retain that obligation.
6. Run the supplementary audit, then the unchanged original evidence gates.
   Both must support the result. If timestamp uncertainty overlaps a boundary,
   retain `uncertain_at_original_window_boundary`; if the entire interval lies
   outside, retain `outside_original_window`. Missing brackets, excessive gaps,
   incompatible offsets, boot/source changes and observed clock steps are not
   certifiable. Never widen a model window, translate an original timestamp,
   substitute a midpoint for an uncertainty interval, or promote an original
   failed time-window gate. Fix clock conditions before a fresh run if needed.

For the node0 worker events and node0 collector, one shared host wall-clock
domain removes interhost offset. The wrapper must still bind that host/boot;
wall-clock steps on node0 can invalidate Unix chronology. Adjacent sample
wall/monotonic deltas detect observed discontinuities on both nodes. Performance
latency and throughput remain measured by the same client's monotonic clock;
these wall-clock samples do not change performance measurements.

## Commands for root integration

Example endpoint on node1 (no command has been run remotely by the tool author):

```sh
python r1_clock_evidence.py serve --execute \
  --server-ip 10.66.9.166 --client-ip 10.66.9.163 --port 30203 \
  --session-id NEW_MODEL_RUN_ID --duration-seconds 7500
```

Example continuous probe on node0, inside an owned monitor:

```sh
python r1_clock_evidence.py probe --execute \
  --server-ip 10.66.9.166 --client-ip 10.66.9.163 --port 30203 \
  --session-id NEW_MODEL_RUN_ID --count 7300 --interval-seconds 1 \
  --timeout-seconds 3 --output /absolute/new-clock-journal.jsonl
```

The count is an upper limit; normal per-exchange work adds to elapsed time.
Parent integration may terminate the owned probe after enough post-run complete
lines. A truncated final JSON line must be retained and treated as incomplete;
do not truncate it away to make a journal pass. Output files are exclusive.

Audit placeholders are deliberately not default acceptance thresholds:

```sh
python r1_clock_evidence.py audit --execute \
  --samples /absolute/clock.jsonl --samples-sha256 CLOCK_SHA \
  --run /absolute/run.json --run-sha256 ORIGINAL_RUN_SHA \
  --events /absolute/all-raw-events.json --events-sha256 RAW_UNION_SHA \
  --relative-drift-ppb DECLARED_RELATIVE_DRIFT_BOUND \
  --max-probe-gap-ns DECLARED_MAX_GAP \
  --step-tolerance-ns DECLARED_STEP_THRESHOLD \
  --output /absolute/new-clock-audit.json
```

## Bounds and interpretation

Each exchange records client send/receive and server receive/send Unix times,
each surrounded by local monotonic reads. Client/server identities include
boot ID, hostname, IP and helper SHA. No symmetry of outbound/inbound network
delay is assumed. With constant offset O = server clock minus client clock:

```
server_send - client_receive <= O <= server_receive - client_send
event_client_time is in [event_server_time - offset_upper,
                         event_server_time - offset_lower]
```

The implementation expands the bounds for the declared relative drift,
including drift during each exchange, intersects independent observations, and
requires close samples strictly bracketing the entire event interval. All
arithmetic uses integer nanoseconds. Support requires the whole event interval
inside the **original** request/cohort window and the original unadjusted gate
to pass. The output always says `NOT_qualified`/`NOT_supported` and forbids
formal-performance approval. This is stricter corroboration of original
evidence, not a mechanism for rehabilitating old failures.

## Teacher-prefix raw records

The archived original paths are:

```
/home/admin/batchgen-r1-alignment-20261026/bg-r1-full-correctness-v2-aborted-archive-v1/bg-r1-full-correctness-v2/trace
/home/admin/batchgen-r1-alignment-20261026/sg-r1-full-correctness-v3-aborted-archive-v1/sg-r1-full-correctness-v3/trace
```

Find `event == "logits"` and `request_id == "tf-arithmetic-p0"` through `p3`
in `rank-*-pid-*.jsonl` on both nodes. Use `generated_before_step == 0`,
`phase == "prefill"`, `context_length`, `input_ids_sha256`, `native_request_id`,
rank/ordinal/run/suite/deployment and the complete `full_logits` object from
each raw event. `full_logits.file` is relative to that trace directory, has
129280 little-endian FP32 values / 517120 bytes, and must match its recorded SHA.
Do not match records by array position or presume rank0 owns the request.
Prefix equality is by exact input IDs/hash, not the suite filename/version.
Root already extracted these four prefixes (actual ranks2..5) for the separate
numerical diagnostic; this helper does not rerun or alter that diagnostic.
