# R1 launch core policy and local disk preflight

The failed cohort produced a large native crash core before its first user
batch. The resulting project-quota exhaustion also prevented client/guardian
reports from being saved. The independent `r1_launch_safety.py` helper prepares
the next launcher; it does not qualify the model or fix the native crash.

The Linux core manual says pipe collectors may ignore RLIMIT_CORE, and that
coredump_filter is inherited at fork and preserved across exec.
[Linux core(5)](https://man7.org/linux/man-pages/man5/core.5.html)

The kernel proc documentation describes memory-type filter bits and notes that
vDSO remains included regardless of the mask. Zero excludes the normal anonymous,
shared, file, hugetlb and DAX memory categories.
[Kernel proc documentation](https://docs.kernel.org/filesystems/proc.html#proc-pid-coredump-filter-core-dump-filtering-settings)

ELF notes are emitted separately from memory payloads in the kernel ELF dumper.
Setting filter zero does not mean enabling the filter's mapped-ELF-header bit.
This preserves the kernel notes path while excluding ordinary weight/heap/stack
pages. Stack memory and values can therefore be unavailable to a debugger.
[Linux v6.6 ELF dumper](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux.git/+/refs/tags/v6.6/fs/binfmt_elf.c#2132)

The helper writes **only `/proc/self/coredump_filter`**, reads zero back, sets its
own soft RLIMIT_CORE to zero without changing the hard limit, and verifies an
isolated standard-library CPU exec child's actual inherited filter/soft limit.
The child only reads its own settings. No host sysctl or other-process proc
entry is modified. The zero filter remains in the launcher for subsequent
HTTP/worker descendants. A read/write/readback/CPU-probe failure rejects launch.
The receipt records the observed core_pattern, whether it is a pipe, original
settings, actual own readback and CPU child readback.

This controls the kernel's mapping payload. **It is not an exact upper bound
on bytes written by an independent host collector.** Notes, mandatory mappings,
collector policy and later descendant setting changes remain outside that
claim. The helper does not configure or signal the collector. It does not
deliberately crash a process to test core creation.

Local disk preflight requires the already existing local ROOT directory,
16 GiB filesystem availability and 1024 available inodes. It then creates a
unique private file in that directory, actually writes 1 MiB, fsyncs, closes and
removes only that new file. EDQUOT/ENOSPC/IO failure rejects launch, even if
statvfs reported plenty of free space. Both lexical and resolved `/mnt/fuse`
paths are rejected before statvfs or writes. Existing files are never deleted.

Filesystem availability does not report project-quota headroom. The small probe
proves present writeability only; it is not a reservation or future quota
guarantee. The parent still checks the platform quota. Runtime periodic disk
supervision is a separate follow-up, not implemented here.

Suggested integration in `launch_r1_diagnostic.py`, **execute mode only**, after
the allocated-container/Python/toolkit checks and before model/worker creation:

```python
from r1_launch_safety import (
    LaunchSafetyError, prepare_launch_safety,
    assert_self_core_policy, preflight_local_disk,
)

try:
    launch_safety = prepare_launch_safety(args.root)
except LaunchSafetyError as error:
    # Surface the partial receipt even when the target disk cannot save JSON.
    print(json.dumps({'launch_safety': error.record}), file=sys.stderr, flush=True)
    raise

# Include this in the launch state before saving it:
state['launch_safety'] = launch_safety

# After weight validation, immediately before the HTTP Popen:
state['launch_safety_pre_popen'] = {
    'core_policy': assert_self_core_policy(),
    'local_disk': preflight_local_disk(args.root),
}
```

Also add `r1_launch_safety.py` to launcher_source_sha256. Preserve normal model
preflight, source/weight checks and sole-child supervision. Do not run this
helper in plan mode: plan must remain free of proc writes and resource-limit
changes. The integration is proposed separately so the parent can merge it
with its NCCL environment change.

CPU checks:

```sh
python tests/test_r1_launch_safety.py
```

On macOS: nine tests pass and the real Linux inheritance test is skipped.
Tests cover zero-mask readback, inherited-policy refusal, own-only writes,
EDQUOT during write/fsync despite ample statvfs, byte/inode floors, private-file
cleanup, OSS rejection, and structured failure receipts. On Linux the extra
test uses only CPU children to verify real fork/exec inheritance while leaving
the test parent's filter/limits and core_pattern unchanged. No remote test,
deployment, GPU, HTTP or service action was performed for this change.
