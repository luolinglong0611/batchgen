# SG LocalSharedTP1 distributed qualification candidate

This is a prepared CPU-tested probe, not a GPU result. The owner must run it
before the whole-model gate and retain every rank result. Only the aggregation
of all 16 ranks may set `final16_layout_qualified=true`; 2/4-rank runs remain
prechecks. No result permits a performance experiment or qualifies the full model.

## Frozen scope and oracle

The executable imports the installed generated SG methods, binds the real
`DeepseekV2MoE.forward/forward_normal/_forward_shared_experts` and
`DeepseekV2DecoderLayer.forward`, and uses the real `LayerCommunicator.prepare_mlp`
and `postprocess_layer`, native SG DP gather/scatter and routed TP SUM.
`prepare_attn`, attention and normalization are identities in this layer-only
probe. Inputs represent already-normalized local rows. Routed experts are an
explicit synthetic dyadic contribution, not a routed kernel qualification.

The shared branch is a real TP1 `DeepseekV2MLP` with the six original layer-3
checkpoint tensors: gate/up/down FP8 E4M3 weights and their original FP32
block128 scales. The merged gate/up layout is copied without re-quantization.
Each fixture first calls this same native shared module separately; the actual
adapter invocation must call it once and return bitwise the independently
constructed local routed value plus the separate shared result. Routed SUM
inputs and global gathered rows must match independently generated, unique
rank/token dyadic rows. Thus an extra pre-SUM shared contribution, duplicate
shared call, misplaced rows, or wrong scatter cannot satisfy the composition
oracle. Original parameter hashes must remain unchanged.

Static review v2 corrected the unrun fixture: every nonzero input has the fixed
power-of-two magnitude `0.0625`; six feature-sign bits encode a unique
`rank*4+token`. Unlike arbitrary integer-scaled amplitudes, all 1..16 equal
dyadic partial contributions are exactly BF16-representable, so the bitwise SUM
oracle cannot falsely fail from intermediate reduction rounding. Global server
arguments/MoE config are installed before distributed initialization. The
shared MLP is constructed under the real BF16 default, runs native post-load
processing, and must still match all six original tensor hashes afterward.

This does **not** independently validate the shared MLP's arithmetic: a bug
common to both invocations can pass. The coordinator clarified this scope
before any distributed run. The initial unrun mandatory FP64 draft is therefore
replaced by a composition-only gate, with
`independent_shared_MLP_numerical_qualified=false`. No numerical tolerance was
loosened after seeing results. An independent FP64 gate is optional only when a
separately frozen contract JSON and exact SHA are explicitly supplied; its
values are never invented by this utility. Ordinary FP8 primitive qualification
and full-model logits/semantic qualification remain separate gates.

All ranks run equal `[2]*world`, unequal/empty `[rank%4]`, and last-owner-only
`[0,...,0,3]` under both native `SUM_LEN` and `MAX_LEN`. MAX_LEN includes padded
zero rows; their outputs must remain zero and all outputs must be finite.

## Bound source identities

The probe refuses unknown generated model sources, including a caller-supplied
hash that is not in this exact admission list:

| Source | SHA256 |
|---|---|
| Alignment-only model | `a00712388404cf3a22a70f770f8bbaf1adc2ed7127182dda7fd47fb94c4897cf` |
| Installed alignment + disabled diagnostic bootstrap | `b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd` |
| Installed `distributed/parallel_state.py` | `274ba74d9a9ee5f4f182de0be11c30fbcc4862c146fd9f917c526b4619478fed` |
| Installed `layers/dp_attention.py` | `ea7fa7d5035aa8f9ada8049ff055a67dd87115b376bba6c0b13046b9eefd14d0` |
| Installed `layers/communicator.py` | `e3f911a97cb960c875a06014ab149d75bff8736ddebdf4a6ad0a3654003c6407` |
| Installed `server_args.py` | `34dfebfe793e7e7409ee69a71c77d3550039e20ed71bff0e954e23c0042bdfdf` |
| Installed `layers/linear.py` | `e985537aa15162f49747c99753ea84aad74cebff614013933fad61246a4809d3` |
| Installed `layers/quantization/fp8.py` | `31faac58174b3ee8298504c140e00647f502db2205b7c0e6156a8c0a407595d2` |
| Installed `layers/moe/utils.py` | `8fbe57a0b6021720145a3abbbf8d69c11031b9d5c3a32845d006c46b8df0b47a` |

For the combined model, both the exact bootstrap suffix and the SHA of the
remaining alignment-only body are checked. Trace stays disabled. Source paths,
actual common linear/SG FP8 source identities, tensor hashes, TP membership,
attention TP/DP sizes, EP size and actual shared parameter shapes are recorded.

## CPU weight preparation

Use the common environment's Python. Substitute only owner-observed paths and
original config/index hashes; the following names denote explicit argv values:

```text
python scripts/r1_reproduction/prepare_sglang_r1_shared_weights.py
  --execute --checkpoint-dir CHECKPOINT_DIR
  --config-sha256 ORIGINAL_CONFIG_SHA --index-sha256 ORIGINAL_INDEX_SHA
  --layer 3 --output-dir FRESH_ABSOLUTE_BUNDLE_DIR
```

This reads six tensors on CPU, hides CUDA before importing the reader, creates
an exclusive directory, and emits the small `shared-original.safetensors` plus
`shared-original.manifest.json`. The printed receipt contains both hashes for
the distributed argv. Each node must use identical bundle and manifest hashes.
Optional `--shard-sha-manifest` takes a JSON object mapping each selected
original shard filename to its full-file SHA. Without it, config/index are
pinned and logical tensor hashes observed; this utility explicitly does not
claim to have verified full source-shard hashes. The coordinator retains the
original checkpoint lineage/manifest separately.

## Launch contract for the owner

Preserve the already qualified common environment's toolkit/library paths and
the deployed repository on `PYTHONPATH`. Additional exact environment:

```text
R1_SG_SHARED_GUARDED=1
BATCHGEN_R1_SM103_FP8_COMPAT=1
DISABLE_MMA_V5=1
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_ENABLE_FLASHINFER_FP8_GEMM=0
SGLANG_R1_LOCAL_SHARED_TP1=1
SGLANG_R1_COMMON_FI_FA2=0
R1_FULL_MODEL_TRACE=0
```

FI attention is disabled only in this no-attention probe. Required software is
the existing SG 0.5.5.post3 / Torch 2.9.0+cu130 common environment on SM103. This
tool does not install anything or start a server. Parent creates a fresh
per-node absolute result directory and launches its own guarded process with:

```text
python -m torch.distributed.run
  --nnodes=2 --nproc-per-node=8 --node-rank=NODE_RANK
  --master-addr=NODE0_ADDRESS --master-port=OWNER_FREE_PORT
  --max-restarts=0 --monitor-interval=1
  scripts/r1_reproduction/qualify_sglang_r1_local_shared.py
  --execute --world-size=16
  --allowed-hostname=ACTUAL_NODE0_HOSTNAME --allowed-hostname=ACTUAL_NODE1_HOSTNAME
  --adapter-source-sha256=b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd
  --weights-bundle=ABSOLUTE_SHARED_BUNDLE --weights-sha256=BUNDLE_SHA
  --tensor-manifest=ABSOLUTE_SHARED_MANIFEST --tensor-manifest-sha256=MANIFEST_SHA
  --layer=3 --output-dir=FRESH_ABSOLUTE_NODE_RESULT_DIR
```

The two launchers must start together under the owner's process-identity guard:
wall 900s, Torch allocation cap 3 GiB/rank, external GPU cap 8 GiB/rank and RSS
cap 8 GiB/rank. The probe itself limits Torch memory and collective init to
180s; external caps and process-group termination are the owner's responsibility.
There is no old-PID stop logic. A single-node import/precheck can use
`--nnodes=1 --nproc-per-node=2` and `--world-size=2` with matching hostname,
but never substitutes for the real 16-rank result.

Resource-budget update after the owner's real v4 attempt: the original external
4 GiB/GPU guard stopped distributed initialization before any fixture or rank
qualification receipt; observed sampled GPU memory reached about4.23 GiB,
including NCCL/driver allocations outside Torch's allocator. The owner closed
all proven descendant ranks and retained that failed attempt. The next attempt
uses the identical `503f08d857a0f8ee0fa3ed13004c289210e9be73b3908f027134f742d85fc0f1`
probe, unchanged bitwise composition criteria, unchanged native NCCL defaults
and unchanged Torch3 GiB limit; only the **external** GPU guard becomes8 GiB.
The frozen probe's printed plan still contains the earlier external4 GiB draft
budget: retain it as history and use the external guard receipt for the actual
8 GiB resource condition. This resource-budget failure is not a numerical
correctness result and supplies no layout pass.

After both launchers close, copy their `rank-0.json` through `rank-15.json`
without overwriting into one fresh local directory and run the read-only CPU
aggregation with the **same** frozen probe source:

```text
python scripts/r1_reproduction/qualify_sglang_r1_local_shared.py
  --aggregate --world-size=16 --output-dir=ALL_CLOSED_RANK_RESULTS
```

Aggregation fails for any missing/duplicate rank or case, failed rank, source,
topology, weight, oracle or optional numeric-contract mismatch. Archive stdout,
exit statuses, all rank JSON, bundle/manifest, source hashes and the external
guard receipt. A pass is communication/layout/one-time-composition evidence;
whole-model trace and numerical/semantic gates must still pass separately.

## Optional bounded launcher with explicit two-node closure

`run_sglang_r1_shared_probe.py` defaults to a launch plan. Its `launch --execute`
mode starts only the explicit probe via a new owned `Popen` session, waits up to
840s, then terminates only that still-owned unreaped child process group on
timeout (TERM, 10s grace, KILL, 10s grace). Retain the owner's outer 900s and
GPU/RSS guards: native SG Gloo subgroups have a 120-minute default, so the
180s initial process-group timeout alone is insufficient. This helper never
controls preexisting PIDs.

For each node, launch under the exact same environment above:

```text
python scripts/r1_reproduction/run_sglang_r1_shared_probe.py launch --execute
  --python ABSOLUTE_COMMON_PYTHON --repo-root ABSOLUTE_DEPLOYED_REPO
  --job-dir FRESH_ABSOLUTE_NODE_JOB_DIR --expected-hostname THIS_NODE_HOSTNAME
  --nnodes=2 --nproc-per-node=8 --node-rank=NODE_RANK
  --master-addr=NODE0_ADDRESS --master-port=OWNER_FREE_PORT --wall-seconds=840
  -- --allowed-hostname=NODE0_HOSTNAME --allowed-hostname=NODE1_HOSTNAME
  --adapter-source-sha256=b37416df137fdb5b6e7a90f0a294c097cb5100db6508982529b55e05fc1406dd
  --weights-bundle=ABSOLUTE_SHARED_BUNDLE --weights-sha256=BUNDLE_SHA
  --tensor-manifest=ABSOLUTE_SHARED_MANIFEST --tensor-manifest-sha256=MANIFEST_SHA
  --layer=3
```

The helper owns `--execute/--world-size/--output-dir`, creates `ranks/`, saves
`torchrun.log`, and writes `launcher-result.json` only after the child is closed
or explicitly records that closure failed. A local pass requires exit0,
no timeout, and all eight expected rank receipts with the current probe hash.
Copy each complete closed node job directory, then aggregate on CPU:

```text
python scripts/r1_reproduction/run_sglang_r1_shared_probe.py aggregate
  --job-dir COPIED_NODE0_JOB_DIR --job-dir COPIED_NODE1_JOB_DIR
  --output FRESH_AGGREGATE_JSON
```

This requires distinct node ranks0/1, two closed exit0 launchers, frozen helper
and probe source hashes, and unchanged hashes for all16 rank receipts before
calling the same substantive probe aggregation. Only this stronger aggregate
adds `all16_rank_receipts_closed=true` and
`both_torchrun_launchers_closed=true`. Existing v1 weight bundle/manifest remain
valid: their original tensors and extraction contract are unchanged; historical
`probe_source_sha256` is provenance of extraction, not an execution-version pin.
