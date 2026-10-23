# R1 two-node NCCL RAS failure and recovery

The first `r1-fast-diagnostic-v1` attempt failed before completing any request.
Do not use its empty client JSON files or stale `guard.json` as successful
collection or cleanup evidence. The later launcher shutdown audits and all-16
GPU release observations establish cleanup separately.

## Observed failure

The node0 worker exited with SIGSEGV (`-11`). Its core names worker PID 1151022
and crashing thread 1151573. The exact ELF notes show R9=0, fault address 0x48,
and a program counter whose file offset is 0x10ca98 in `libnccl.so.2`.
Both containers' library SHA256 is
`51c81bed3832ebf251d26f9366dac57d5ce3e5b4b1cf8dc29e7d60d665213db4`
(NCCL 2.27.7, 210652104 bytes). DWARF/objdump identify
`rasMsgHandlePeersUpdate(rasMsg*, rasSocket*)`, `src/ras/peers.cc:549`,
`mov 0x48(%r9),%rcx`.

The core's timestamp precedes submission of the first16 probe. The peer node's
first fatal report identifies this node0 worker; later NCCL abort messages
belong to cohort teardown. This is evidence of a RAS null-pointer crash, not
an attention-kernel failure or a successful model-quality test.

The platform pipe collector attempted a roughly 760 GiB core, but the container
quota stopped it after 336117125120 bytes (about 313 GiB). The exhausted quota
also prevented client summary/run JSON and guardian updates from completing.
The full memory payload was truncated; gdb could not read thread stacks beyond
EOF. The exact ELF header, all program headers and notes (310928 bytes,
SHA256 `8002c448425c582390a0d6ccfaa60be2ba71789be7a80796c0272205efc54b83`),
registers, mappings, gdb output and file identity were preserved. Only this
identified truncated core was removed. Its entire memory payload was not
archived. Model weights and experiment logs were retained.

## Explicit adaptation

The R1 diagnostic planner now sets `NCCL_RAS_ENABLE=0` before any fresh worker
initializes NCCL. NVIDIA's v2.27.7 source gates RAS initialization, rank updates
and split-communicator RAS handling on this setting. It disables NCCL RAS health
telemetry, not the collective operations or RDMA transport. PyTorch's separate
`TORCH_NCCL_ENABLE_MONITORING=0` does not disable RAS.

The launcher also applies its own inherited core mapping filter and checks
local output disk writes; see [R1_LAUNCH_SAFETY.md](R1_LAUNCH_SAFETY.md).
No host sysctl or platform collector setting is changed. The pre-workaround
weight-validation record remains usable only with its exact frozen launcher
SHA and otherwise exact command/environment, followed by unchanged-file and
current receipt/converter checks. Its old runtime settings are never replayed.

## Validation and limits

A fresh smoke run in the BatchGen environment (Torch 2.9.0+cu130, NCCL 2.27.7)
passed on all 16 ranks with RAS explicitly disabled. It covered three NCCL
process groups, exact all-reduce and all-gather results, and a ring send/receive.
The default group covered 4 KiB, 8 MiB and 64 MiB all-reduces. Every rank reported
an inherited zero core mapping filter. Both launchers exited zero with no
remaining owned processes; all GPUs returned to 4 MiB/0% observations.

This confirms basic communication with the workaround. It does not reproduce
the original crash deterministically, establish that every NCCL fault is fixed,
or qualify full-model outputs, attention numerics, or paper speedup. Existing
strict attention and NaN-padding failures remain recorded and unresolved.

Primary references:

- [NVIDIA NCCL RAS_ENABLE documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ras-enable)
- [NVIDIA v2.27.7 bootstrap implementation](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/bootstrap.cc)
- [NVIDIA issue 1996: same NCCL RAS crash location](https://github.com/NVIDIA/nccl/issues/1996)

The upstream issue is supporting context, not proof that another NCCL release
fixes this deployment; no library upgrade is claimed here.
