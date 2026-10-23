"""Linux supervision for one direct, new-session model-server child.

This module never starts a process, queries a GPU, or calls HTTP. The caller
must keep sole wait/reap ownership of its Popen child. WNOWAIT keeps that child
unreaped until the last primary-process-group signal and pre-reap audit.
Other process groups are audited, never signalled. This cannot reclaim a child
that deliberately escapes its session; the operator must investigate leftovers.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import signal
import threading
import time
import traceback

MIN_RELEASE_GRACE_SECONDS = 180
DEFAULT_RELEASE_GRACE_SECONDS = 300


def process_identity(pid):
    """Read identity only; never expose command-line/environment arguments."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text(errors='replace').rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'session': int(fields[3]), 'start_ticks': int(fields[19]),
                'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_identity(current, expected):
    return current is not None and all(current[key] == expected[key]
        for key in ('pid', 'ppid', 'pgid', 'session', 'start_ticks'))


def session_members(session):
    rows = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            row = process_identity(int(path.name))
            if row is not None and row['session'] == session:
                rows.append(row)
    return sorted(rows, key=lambda row: row['pid'])


def child_exited(child):
    """Observe exit without poll(), wait(), waitpid(), or reaping the child."""
    result = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return result is not None and result.si_pid == child.pid


def assert_owned(child, identity):
    if child.returncode is not None:
        raise RuntimeError('Child was reaped before the last signal/audit')
    current = process_identity(child.pid)
    if (not same_identity(current, identity) or identity['ppid'] != os.getpid()
            or identity['pid'] != identity['pgid'] or identity['pid'] != identity['session']):
        raise RuntimeError('Direct new-session child identity no longer matches')
    # ECHILD also rejects a competing reaper even if the numeric PID now exists.
    child_exited(child)
    return current


def snapshot(child, identity):
    assert_owned(child, identity)
    members = session_members(identity['session'])
    return {'leader_exited_unreaped': child_exited(child),
            'primary_group_live': [row for row in members
                if row['pgid'] == identity['pgid'] and row['state'] not in ('Z', 'X')],
            'other_group_live_audit_only': [row for row in members
                if row['pgid'] != identity['pgid'] and row['state'] not in ('Z', 'X')],
            'session_members_including_zombies': members}


def signal_leader(child, identity, sig):
    current = assert_owned(child, identity)
    if current['state'] not in ('Z', 'X'):
        # Original unreaped direct-child ownership pins this PID. No poll or
        # other reaper may run concurrently. Group fallback is separate below.
        os.kill(identity['pid'], sig)
        return True
    return False


def signal_primary_group(child, identity, sig):
    assert_owned(child, identity)
    # Only this primary PGID is pinned by the unreaped new-session leader.
    # Its SID does not pin an additional compiler/worker PGID in that session.
    try:
        os.killpg(identity['pgid'], sig)
    except ProcessLookupError:
        # A last live member can finish between the snapshot and this signal.
        # Continue to the fresh audit; never substitute another numeric group.
        pass


def wait_for_release(child, identity, seconds, interval, update):
    deadline = time.monotonic() + seconds
    last_update = float('-inf')
    while True:
        current = snapshot(child, identity)
        now = time.monotonic()
        if now - last_update >= 5:
            update(current)
            last_update = now
        if current['leader_exited_unreaped'] and not current['primary_group_live']:
            return current
        if now >= deadline:
            return current
        time.sleep(min(interval, deadline - now))


def supervise_child(child, *, release_grace_seconds=DEFAULT_RELEASE_GRACE_SECONDS,
                    group_term_grace_seconds=30, kill_release_grace_seconds=300,
                    poll_seconds=0.2, on_update=None):
    """Wait indefinitely, or stop one owned cohort on SIGTERM/SIGINT.

    A first stop request becomes one TERM to the HTTP leader, allowing normal
    lifespan/worker release for >=180 s (default 300). If needed, signal only
    the pinned primary PGID TERM, then KILL, with bounded grace after each.
    Normal leader exit also enters audit/cleanup before reaping. Repeated
    TERM/INT are ignored throughout cleanup. No numeric signal follows reap.

    Returns a JSON-compatible audit including errors and parent_cleanup_required.
    on_update is an optional persistence callback; callback errors are recorded
    but cannot interrupt cleanup. This must run in the main thread with default
    SIGCHLD, on a Linux /proc filesystem, before anybody polls/waits the child.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Supervision requires the main thread for signal shielding')
    if (not Path('/proc/self/stat').is_file() or not hasattr(os, 'WNOWAIT')
            or not hasattr(os, 'waitid') or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
        raise RuntimeError('Linux /proc, WNOWAIT and default SIGCHLD are required')
    limits = (release_grace_seconds, group_term_grace_seconds, kill_release_grace_seconds, poll_seconds)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in limits):
        raise ValueError('Grace periods and polling interval must be finite numbers')
    if release_grace_seconds < MIN_RELEASE_GRACE_SECONDS:
        raise ValueError('Normal native release requires at least 180 seconds')
    if min(group_term_grace_seconds, kill_release_grace_seconds) < 0 or poll_seconds <= 0:
        raise ValueError('Invalid group/reap grace or polling interval')
    record = {'schema': 'r1-launch-process-audit-v1', 'status': 'admitting',
              'child_pid': child.pid, 'requested_stop_signal': None, 'signals': [],
              'parent_cleanup_required': False, 'child_reaped': False,
              'release_grace_seconds': release_grace_seconds,
              'group_term_grace_seconds': group_term_grace_seconds,
              'kill_release_grace_seconds': kill_release_grace_seconds,
              'policy': 'sole direct-child WNOWAIT ownership; primary PGID only; audit before reap; no later numeric signal',
              'visibility_scope': 'visible /proc session members; escaped sessions are not discoverable here',
              'callback_errors': []}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    installed = False
    identity = None

    def persist():
        if on_update is not None:
            try:
                on_update(record)
            except BaseException as error:
                if len(record['callback_errors']) < 16:
                    record['callback_errors'].append(f'{type(error).__name__}: {error}')

    def requested(signum, _frame):
        if record['requested_stop_signal'] is None:
            record['requested_stop_signal'] = signum

    def observed(value):
        record['latest_snapshot'] = value
        persist()

    try:
        identity = process_identity(child.pid)
        if identity is None:
            raise RuntimeError('Could not capture direct child identity')
        assert_owned(child, identity)
        record['child_identity'] = identity
        for sig in old_handlers:
            signal.signal(sig, requested)
        installed = True
        record['status'] = 'waiting'
        persist()
        try:
            while record['requested_stop_signal'] is None and not child_exited(child):
                time.sleep(poll_seconds)
        except BaseException:
            record['supervision_error'] = traceback.format_exc()
        finally:
            # Shield the complete sequence, including audit and wait. A second
            # stop cannot interrupt the native release or force a stale killpg.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            record['status'] = 'cleaning'
            persist()
            try:
                current = snapshot(child, identity)
                record['cleanup_entry_snapshot'] = current
                if not current['leader_exited_unreaped']:
                    if signal_leader(child, identity, signal.SIGTERM):
                        record['signals'].append({'target': 'leader', 'signal': int(signal.SIGTERM)})
                    persist()
                current = wait_for_release(child, identity, release_grace_seconds, poll_seconds, observed)
                for sig, grace in ((signal.SIGTERM, group_term_grace_seconds),
                                   (signal.SIGKILL, kill_release_grace_seconds)):
                    if current['leader_exited_unreaped'] and not current['primary_group_live']:
                        break
                    signal_primary_group(child, identity, sig)
                    record['signals'].append({'target': 'primary_group', 'signal': int(sig)})
                    persist()
                    current = wait_for_release(child, identity, grace, poll_seconds, observed)
                # The final ownership check/audit occurs while leader is still
                # unreaped. All signal sites precede this point.
                record['pre_reap_snapshot'] = snapshot(child, identity)
                persist()
                if record['pre_reap_snapshot']['leader_exited_unreaped']:
                    record['exit_code'] = child.wait(timeout=1)
                    record['child_reaped'] = True
                else:
                    record['leader_not_reaped_still_live'] = True
                remains = record['pre_reap_snapshot']
                record['parent_cleanup_required'] = bool(remains['primary_group_live']
                    or remains['other_group_live_audit_only'] or not record['child_reaped'])
                record['status'] = ('parent_cleanup_required' if record['parent_cleanup_required']
                                    else 'supervision_failed_cleanup_completed' if 'supervision_error' in record
                                    else 'stopped' if record['requested_stop_signal'] is not None else 'exited')
            except BaseException:
                record['cleanup_error'] = traceback.format_exc()
                record['parent_cleanup_required'] = True
                record['status'] = 'cleanup_failed_parent_required'
    except BaseException:
        record['admission_error'] = traceback.format_exc()
        record['status'] = 'admission_failed_parent_required'
        record['parent_cleanup_required'] = True
    finally:
        persist()
        if installed:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return record
