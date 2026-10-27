"""Preflight only this launcher's inherited core policy and local output disk.

No torch/GPU import, service operation, host sysctl write, other-process /proc
write, or OSS statfs. filter=0 excludes normal memory mappings from kernel ELF
cores; mandatory mappings/notes remain. A pipe collector has no RLIMIT_CORE
byte guarantee and can have behavior outside this process's control.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

SELF_FILTER = Path('/proc/self/coredump_filter')
CORE_PATTERN = Path('/proc/sys/kernel/core_pattern')  # Read only.
DISK_MIN_FREE_BYTES = 16 * 1024**3
DISK_MIN_FREE_INODES = 1024
DISK_PROBE_BYTES = 1024**2
FORBIDDEN_DISK_ROOT = Path('/mnt/fuse')
INHERITANCE_PROBE = (
    "import json,os,resource;from pathlib import Path;"
    "print(json.dumps({'pid':os.getpid(),"
    "'coredump_filter':Path('/proc/self/coredump_filter').read_text().strip(),"
    "'rlimit_core_soft':resource.getrlimit(resource.RLIMIT_CORE)[0]}))"
)


class LaunchSafetyError(RuntimeError):
    """Fail closed before model launch; caller can retain the partial receipt."""
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def _error(record, error):
    record.update(status='failed_NOT_admitted', error=f'{type(error).__name__}: {error}')
    if isinstance(error, OSError):
        record['errno'] = error.errno
    return LaunchSafetyError(record['error'], record)


def _filter_value(text):
    value = text.strip()
    if not value or len(value) > 32 or any(c not in '0123456789abcdefABCDEFxX' for c in value):
        raise RuntimeError('Invalid coredump_filter readback')
    return int(value, 16)


def assert_self_core_policy():
    """Read only, immediately before Popen; do not silently repair drift."""
    observed = SELF_FILTER.read_text().strip()
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if _filter_value(observed) != 0 or soft != 0:
        raise RuntimeError('Own inherited core policy changed before model launch')
    return {'pid': os.getpid(), 'coredump_filter_observed': observed,
            'rlimit_core_soft': soft, 'rlimit_core_hard': hard}


def configure_self_core_policy():
    """Write only /proc/self/coredump_filter and this process's soft rlimit.

    A CPU Python fork/exec probe reads its OWN settings with -I -S, preventing
    sitecustomize/import side effects. No child filter or sysctl is written.
    The zero filter deliberately remains installed for all later descendants.
    """
    record = {'schema': 'r1-inherited-core-policy-v1', 'status': 'checking',
        'pid': os.getpid(), 'write_scope': '/proc/self/coredump_filter and own RLIMIT_CORE soft only',
        'requested_filter': '0x0', 'host_sysctl_modified': False,
        'hard_core_byte_limit_guaranteed': False,
        'scope': 'kernel mapping filtering; ELF notes and mandatory vDSO may remain; collector behavior outside this process is not controlled'}
    try:
        before = SELF_FILTER.read_text().strip()
        _filter_value(before)
        record['coredump_filter_before'] = before
        pattern = CORE_PATTERN.read_text().strip()
        if not pattern or len(pattern) > 4096:
            raise RuntimeError('Missing/invalid core_pattern readback')
        record['core_pattern_observed_read_only'] = pattern
        record['core_pattern_is_pipe'] = pattern.startswith('|')
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        record['rlimit_core_before'] = {'soft': soft, 'hard': hard}
        SELF_FILTER.write_text('0x0\n')
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
        record['coredump_filter_after_write'] = SELF_FILTER.read_text().strip()
        if _filter_value(record['coredump_filter_after_write']) != 0:
            raise RuntimeError('Own coredump_filter readback changed before model launch')
        record['self_readback'] = assert_self_core_policy()
        result = subprocess.run([sys.executable, '-I', '-S', '-c', INHERITANCE_PROBE],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10, check=True)
        inherited = json.loads(result.stdout)
        record['cpu_exec_probe'] = inherited
        if (not isinstance(inherited, dict) or type(inherited.get('pid')) is not int
                or inherited['pid'] <= 0 or inherited['pid'] == os.getpid()
                or _filter_value(inherited.get('coredump_filter', '')) != 0
                or inherited.get('rlimit_core_soft') != 0):
            raise RuntimeError('CPU fork/exec child did not inherit the zero core policy')
        record['cpu_exec_probe_interpreter'] = sys.executable
        record['rlimit_core_note'] = ('Pipe collectors may ignore RLIMIT_CORE; filter=0 is the mapping control'
                                     if pattern.startswith('|') else 'Soft RLIMIT_CORE=0 also disables ordinary file core generation')
        record['status'] = 'passed_NOT_model_qualified'
        return record
    except Exception as error:
        raise _error(record, error) from error


def _local_disk_path(root):
    root = Path(root).absolute()
    if root == FORBIDDEN_DISK_ROOT or FORBIDDEN_DISK_ROOT in root.parents:
        raise ValueError('Never statfs or write a quota probe on /mnt/fuse/OSS')
    resolved = root.resolve(strict=True)
    if resolved == FORBIDDEN_DISK_ROOT or FORBIDDEN_DISK_ROOT in resolved.parents:
        raise ValueError('Local output path resolves into /mnt/fuse/OSS')
    if not resolved.is_dir():
        raise ValueError('Local output root must already be a directory')
    return resolved


def preflight_local_disk(root):
    """Small same-directory real write/fsync probe, plus filesystem headroom.

    statvfs is not a project-quota API. A successful 1 MiB probe only proves
    present writeability, not future quota headroom or an exact free quota.
    Only the newly created private probe is removed; no existing file is edited.
    """
    record = {'schema': 'r1-local-disk-preflight-v1', 'status': 'checking',
        'min_free_bytes': DISK_MIN_FREE_BYTES, 'min_free_inodes': DISK_MIN_FREE_INODES,
        'probe_bytes': DISK_PROBE_BYTES, 'project_quota_headroom_proven': False,
        'scope': 'local output filesystem and point-in-time actual write/fsync; not future quota admission'}
    fd = None
    probe = None
    failure = None
    try:
        local = _local_disk_path(root)
        record['directory'] = str(local)
        filesystem = os.statvfs(local)
        record['filesystem_available_bytes'] = filesystem.f_bavail * filesystem.f_frsize
        record['filesystem_available_inodes'] = filesystem.f_favail
        if (record['filesystem_available_bytes'] < DISK_MIN_FREE_BYTES
                or record['filesystem_available_inodes'] < DISK_MIN_FREE_INODES):
            raise RuntimeError('Insufficient local filesystem byte/inode headroom')
        fd, name = tempfile.mkstemp(prefix='.r1-launch-write-probe-', dir=local)
        probe = Path(name)
        record['probe_path'] = name
        chunk = b'R1-local-quota-probe\n' * 2048
        count = 0
        while count < DISK_PROBE_BYTES:
            written = os.write(fd, chunk[:min(len(chunk), DISK_PROBE_BYTES - count)])
            if written <= 0:
                raise OSError('Short/zero local quota-probe write')
            count += written
        os.fsync(fd)
        info = os.fstat(fd)
        if info.st_size != DISK_PROBE_BYTES:
            raise RuntimeError('Local quota-probe size differs after fsync')
        record['probe_written_bytes'] = count
        record['probe_allocated_bytes_observed'] = info.st_blocks * 512
        record['status'] = 'passed_point_in_time_NOT_future_quota_guarantee'
    except Exception as error:
        failure = error
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as error:
                failure = failure or error
        if probe is not None:
            try:
                probe.unlink()
                record['probe_removed'] = True
            except OSError as error:
                record['probe_removed'] = False
                record['probe_cleanup_error'] = f'{type(error).__name__}: {error}'
                failure = failure or error
    if failure is not None:
        raise _error(record, failure) from failure
    return record


def prepare_launch_safety(root):
    """Call inside the verified launcher before any model/worker is created."""
    record = {'schema': 'r1-launch-safety-preflight-v1', 'status': 'checking'}
    try:
        record['core_policy'] = configure_self_core_policy()
        record['local_disk'] = preflight_local_disk(root)
        record['status'] = 'passed_prelaunch_NOT_model_qualified'
        return record
    except LaunchSafetyError as error:
        record.update(status='failed_NOT_admitted', failed_check=error.record)
        raise LaunchSafetyError(str(error), record) from error
