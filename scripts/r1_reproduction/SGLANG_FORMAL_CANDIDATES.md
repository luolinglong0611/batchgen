# SGLang formal candidate plans for 16 allocated GPUs

`plan_sglang_formal_candidate.py` creates local JSON plans only. It never starts
services, contacts the containers, queries GPUs or sends requests. It leaves
the diagnostic launcher/executor and frozen QA/6000-request workload untouched.
No generated candidate is admitted for a performance comparison yet.

The identity remains `sglang0.5.5.post3+R1-routing-compat`, using the original
R1 FP8 weights and F32 block scales. The R1 routing compatibility patch and the
four audited dense-kernel switches remain explicit. This is not stock SGLang.

| Candidate | Physical allocation | Independent services | Per service parallelism | Client endpoints |
| --- | --- | --- | --- | --- |
| A | Two nodes, eight GPUs each | One spanning both nodes | TP16, DP attention16, EP1 | Node0 HTTP only |
| B | Two nodes, eight GPUs each | Two, one on each node | TP8, DP attention8, EP1 | Both nodes' HTTP endpoints |

Both use sixteen GPUs total. B sets `nnodes=1,node-rank=0` on each physical
node and uses that node's own container IP for initialization. Its physical
node1 still uses node1's routing-patch receipt. Both services may reuse HTTP
30000 and rendezvous33155 in separate containers. Admission must also check
known ZMQ ports33156–33160, DP handshake port33168 and runtime-selected sockets; this planner
does not claim to reserve ports or inspect their availability.

## Generate a reviewable plan

From the repository, with the previously collected fixed source audit present:

```bash
python3 scripts/r1_reproduction/plan_sglang_formal_candidate.py \
  --topology A --per-dp-running 16 --per-dp-chunk 4096 \
  --memory-fraction 0.85 --output /tmp/r1-sglang-A-plan.json

python3 scripts/r1_reproduction/plan_sglang_formal_candidate.py \
  --topology B --per-dp-running 32 --per-dp-chunk 4096 \
  --memory-fraction 0.90 --output /tmp/r1-sglang-B-plan.json
```

Output creation is exclusive. Without `--output`, JSON goes to stdout. The
default audit location is `.repro-private/baseline-recovery/formal-source-audit`;
`--source-audit` can point to another byte-identical local export. The combined
receipt, five semantic source/help files, diagnostic plan provider and routing
pins must match their frozen SHA256 values. This checks a historical local
export, not the current remote installation. Command strings are for review;
do not bypass parent admission or process supervision by running them directly.

The 48 combinations are A/B × running16/32/64/128 per DP rank ×
chunk2048/4096/8192 per DP rank × static memory fraction0.85/0.90. The tool emits
one selected candidate at a time. CLI max-running and chunk are multiplied by
the DP count of that service: A multiplies by16; each B replica multiplies by8.

## Fixed-version semantics

These conclusions come from the pinned installed 0.5.5.post3 source export,
whose combined receipt SHA is
`7732753365d15c53297f88a6752bbb36bd1a9a9ac57c4d39a0a7a167f802ae1b`:

- `server_args.py:1383` multiplies schedule conservativeness by0.3, and `:1385`
  divides the CLI chunk by DP. The default effective conservativeness is0.3.
- `tp_worker.py:281` divides CLI max-running by DP, then caps it by the request
  pool. A target of128 is not a guarantee of128 simultaneously resident requests.
- `model_runner.py:483` passes the undivided CLI max-running to request-pool
  allocation. Equal per-DP concurrency therefore does not imply equal A/B
  request-metadata memory. Record actual pool slots and memory per rank.
- `server_args.py:1267` rejects explicit `bf16`/`bfloat16` for `trtllm_mla`, even
  though general CLI help lists them. The plan keeps `--kv-cache-dtype auto`
  with `--dtype bfloat16`; `model_runner.py:1547` selects model dtype unless a
  KV FP8 quantization mode overrides it. **Every rank must report effective
  `torch.bfloat16` before accepting the candidate.**
- `model_runner.py:1355` budgets remaining KV memory after model loading as
  available memory minus total capacity times `(1 - mem_fraction_static)`.
  Fraction0.90 does not dedicate90% to KV alone. No `--max-total-tokens` is
  emitted: the old diagnostic32768-token cap must not carry into formal tuning.
  Capacity is profiled and page-rounded, not known at planning time. BF16 R1
  MLA costs70272 bytes per token per rank before other allocations.
- Default max-prefill tokens16384 is not divided by DP and exceeds every
  proposed effective chunk. Graphs are explicitly disabled initially; piecewise
  graphs, torch compile and speculative decoding remain disabled by defaults.
  Radix/prefix caching and shared-expert fusion are disabled explicitly.

The generated environment is an exact allowlist, rendered using `env -i`, so
inherited credentials or `SGLANG_CUTLASS_MOE` cannot alter the plan. It includes
`NCCL_RAS_ENABLE=0` as the hardware stability adaptation documented in
[NCCL_RAS_RECOVERY.md](NCCL_RAS_RECOVERY.md). This does not upgrade NCCL. The
expected SGLang package version is recorded, but whether the **loaded** NCCL
library matches BatchGen remains unknown here and needs runtime provenance.

## Admission and fair measurement still required

The existing diagnostic executor assumes one two-node service and small
capacity. Its receipts and readiness logic do not authorize either new plan;
B requires a new coordinator aware of two independent services. Before any
execution, the parent must establish resource ownership on both containers,
fresh weight/source/library/patch checks, exact candidate communication and
kernel qualification, lifecycle supervision, runtime BF16 KV evidence and
actual memory/concurrency capacity. Prior short/empty-EOS quality observations
must remain visible; successful HTTP or source-level validation is insufficient.

After functional and capacity admission, tune on separate32→128→512 request
subsets and extend if saturation is not reached. Use eager as the control;
qualify graphs separately before selecting an optimized baseline. Freeze the
winning settings before the unchanged6000-request profiles and at least three
warm repetitions. This script does not choose a winner.

For A, one client sends6000 requests to node0. For B, the existing multi-endpoint
client sends the same6000 requests round-robin,3000 to each service. Measure one
cluster BCT from the first network attempt to the last complete response body;
do not add or average per-replica BCTs. Formal runs require fixed output length
with `ignore_eos=true` and exact input/output count validation. Record BatchGen's
batch API versus SGLang's async-burst transport. Loading, compilation, warmup and
offline scoring are outside BCT. No formal speedup is established by these plans.

CPU tests include actual AST-extracted methods from the pinned source for DP
division, request-pool capping and TRTLLM BF16 rejection, plus the48 topology
combinations, provenance rejection, deterministic exclusive output and no
process/network calls. Fixed-source tests skip only if that private export is
absent; report skips explicitly when validating elsewhere.
