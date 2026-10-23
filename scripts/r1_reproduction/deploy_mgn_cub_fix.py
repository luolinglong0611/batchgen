#!/usr/bin/env python3
"""Plan an MGN-only binary deployment; apply/rollback are explicit and CPU-only.

Never installs the qualification wheel. Only the two pinned import locations
are replaced, individually with os.replace; the pair is a recoverable transaction,
not an atomic multi-file operation. Keep BatchGen launchers stopped throughout.
A SIGKILL/power loss requires --rollback-from the durable transaction directory.
No GPU inventory, CUDA device query, model load, or kernel execution is performed.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import tempfile
import time
import zipfile

ROOT = Path('/home/admin/batchgen-r1-repro-20261023')
HOSTS = ('serverless-010066009163.wm226', 'serverless-010066009166.wm226')
SOURCE = '17b9fcbc2a4f9cfb9a5d31c6cb8b7db99f6331bd'
OLD_SHA = 'ff1ed9bff0688373c02a26c23242a7ec6c1534a49b3e19d9cd6360d2d1359fd6'
NEW_SHA = '0f535cbd3c3fda1f4678a4bdec316d083f6383c3ed7fefb1777b935c9f075b2a'
NEW_SIZE = 11883584
WHEEL_SHA = '7df577f7d4cb0e0944806fb0e38275b92f8626589e167729bb39321632bfb015'
CORE_SHA = 'f0f0e5af58fb7c6f404e1e390b626556c5355497179b18af133bc729ed8a6a29'
NAME = '_C_mgn_ops.cpython-311-x86_64-linux-gnu.so'
MEMBER = 'batchgen_kernels/common/' + NAME
WRAPPERS = {'__init__.py': '7cbfa006d1cb4006c9684379e632a93b3074a01d37031d77c7ab76422e3c1b82',
            'common/mgn.py': '0cf006c8c328a6c9272d9e296f897e87cf962cdaaef7b2004216b584334e6b14'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def regular(path):
    require(path.resolve(strict=True) == path.absolute(), f'Symlink path refused: {path}')
    require(stat.S_ISREG(path.lstat().st_mode), f'Not a regular file: {path}')
    return path


def layout(root):
    site = root / 'env/lib/python3.11/site-packages'
    packages = {'repo': root / 'repo/batchgen_kernels', 'env': site / 'batchgen_kernels'}
    return site, packages, {key: value / 'common' / NAME for key, value in packages.items()}


def plan(root, rank, wheel):
    _, _, targets = layout(root)
    return {'status': 'plan_only', 'root': str(root), 'node_rank': rank,
            'hostname': HOSTS[rank], 'source_commit': SOURCE, 'wheel': str(wheel),
            'wheel_sha256': WHEEL_SHA, 'old_sha256': OLD_SHA, 'new_sha256': NEW_SHA,
            'targets': {key: str(value) for key, value in targets.items()},
            'preserve': 'All other native extensions, build copies, wrappers, and core_engine',
            'verification': 'Fresh -I -B CPU-hidden AOT imports from repo-first and environment-only paths',
            'precondition': 'Keep all BatchGen launchers stopped; apply rechecks /proc before each replacement',
            'rollback': 'Only verified per-target backups from transaction.json; no wheel/package reinstall',
            'package_record': 'dist-info/RECORD and package version remain original; deployment SHA receipt is the binary provenance',
            'scope': 'No GPU query or kernel; individual atomic replacements, recoverable two-file transaction'}


def protected_inventory(root):
    _, packages, targets = layout(root)
    result = {}
    for package in packages.values():
        for relative, expected in WRAPPERS.items():
            path = regular(package / relative)
            require(digest(path) == expected, f'Unqualified loader/wrapper: {path}')
        paths = list(package.rglob('*.so')) + [package / p for p in WRAPPERS]
        for path in sorted(set(paths)):
            if path in targets.values():
                continue
            regular(path)
            result[str(path)] = digest(path)
    cores = sorted((root / 'repo/batchgen').glob('core_engine*.so'))
    require(len(cores) == 1 and digest(regular(cores[0])) == CORE_SHA, 'Qualified core_engine missing/changed')
    result[str(cores[0])] = CORE_SHA
    return result


def candidate_bytes(wheel):
    regular(wheel)
    require(digest(wheel) == WHEEL_SHA, 'Candidate wheel SHA mismatch')
    with zipfile.ZipFile(wheel) as archive:
        require(archive.namelist().count(MEMBER) == 1, 'Candidate wheel has missing/duplicate MGN member')
        member = archive.getinfo(MEMBER)
        require(member.file_size == NEW_SIZE, 'Candidate MGN byte count mismatch')
        data = archive.read(member)  # No extraction/install of any other member.
    require(hashlib.sha256(data).hexdigest() == NEW_SHA, 'Candidate MGN SHA mismatch')
    return data


def process_reasons(argv, maps, cwd, root):
    reasons = []
    if any(re.match(r'^batchgen(?:[./]|$)', arg) for arg in argv):
        reasons.append('BatchGen command/module')
    if any('/batchgen/' in arg and arg.endswith('.py') for arg in argv):
        reasons.append('BatchGen script')
    if any(Path(arg).name == 'launch_r1_diagnostic.py' for arg in argv) and '--execute' in argv:
        reasons.append('BatchGen model launcher')
    if '_C_mgn_ops' in maps or re.search(r'/batchgen/core_engine[^/\n ]*\.so', maps):
        reasons.append('Mapped MGN/core_engine library')
    if cwd == str(root / 'repo') and any('multiprocessing.' in arg for arg in argv):
        reasons.append('Multiprocessing worker/tracker in BatchGen checkout')
    return reasons


def audit_no_batchgen(root, proc=Path('/proc')):
    """Audit our real/effective UID; explicitly retain other-UID unknowns.

    Shared containers expose platform processes whose maps cannot be read by
    this user. Those processes are outside this audit, not evidence of absence
    of BatchGen in the full PID namespace. Their commands are never inspected.
    Our UID remains fail-closed; the real admission lock is held for mutations.
    """
    blockers, scanned, other_uid = [], 0, []
    own_uid = os.geteuid()
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            raw_stat = (directory / 'stat').read_text()
            fields = raw_stat[raw_stat.rfind(')') + 2:].split()
            if fields[0] in ('Z', 'X'):
                continue
            status = (directory / 'status').read_text()
            uids = next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1:]
            require(len(uids) == 4, f'Invalid UID fields for process {directory.name}')
            real_uid, effective_uid = int(uids[0]), int(uids[1])
            if own_uid not in (real_uid, effective_uid):
                item = {'pid': int(directory.name), 'real_uid': real_uid, 'effective_uid': effective_uid,
                        'scope': 'Other UID; command and mappings not audited'}
                try:
                    with (directory / 'maps').open('r') as handle:
                        handle.read(1)
                    item['map_access'] = 'readable_but_outside_uid_scope'
                except PermissionError as exc:
                    item.update(map_access='permission_denied', errno=exc.errno)
                other_uid.append(item)
                continue
            argv = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            maps = (directory / 'maps').read_text()
            try:
                cwd = os.readlink(directory / 'cwd')
            except FileNotFoundError:
                cwd = ''
            reasons = process_reasons(argv, maps, cwd, root)
            scanned += 1
            if reasons:
                blockers.append({'pid': int(directory.name), 'starttime_ticks': int(fields[19]),
                                 'real_uid': real_uid, 'effective_uid': effective_uid, 'reasons': reasons})
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during the snapshot.
        except OSError as exc:
            if directory.exists():
                raise RuntimeError(f'Cannot audit live process {directory.name}: {exc}') from exc
    require(not blockers, f'Active BatchGen process(es): {json.dumps(blockers)}')
    return {'time_unix': time.time(), 'audit_uid': own_uid, 'processes_scanned': scanned, 'blockers': [],
            'same_uid_audit_complete': True, 'uninspected_other_uid': other_uid,
            'namespace_complete': not other_uid,
            'scope': 'Our real/effective UID: command/module, mapped MGN/core, checkout multiprocessing; '
                     'other UIDs explicitly uninspected; not a service-start lock'}


PROBE = '''import os,sys,json,hashlib,pathlib
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['BATCHGEN_KERNELS_DEV']=='0'
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; expected=sys.argv[3]
if mode=='repo': sys.path.insert(0,str(root/'repo'))
import torch
assert not torch.cuda.is_initialized()
import batchgen_kernels
from batchgen_kernels.common import mgn
ext=mgn._get_mgn()
assert not torch.cuda.is_initialized()
base=root/'repo' if mode=='repo' else root/'env/lib/python3.11/site-packages'
pkg=base/'batchgen_kernels'
assert pathlib.Path(batchgen_kernels.__file__).resolve()==pkg/'__init__.py'
assert pathlib.Path(mgn.__file__).resolve()==pkg/'common/mgn.py'
assert pathlib.Path(ext.__file__).resolve()==pkg/'common'/sys.argv[4]
actual=hashlib.sha256(pathlib.Path(ext.__file__).read_bytes()).hexdigest()
assert actual==expected
print(json.dumps({'mode':mode,'package':batchgen_kernels.__file__,'wrapper':mgn.__file__,
 'extension':ext.__file__,'sha256':actual,'cuda_initialized':torch.cuda.is_initialized()}))
'''


def verify_imports(root, expected_sha):
    site, _, _ = layout(root)
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'TORCH_EXTENSIONS_DIR'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', BATCHGEN_KERNELS_DEV='0', PYTHONDONTWRITEBYTECODE='1',
               LD_LIBRARY_PATH=str(site / 'torch/lib') + ':' + str(site / 'nvidia/cu13/lib'))
    reports = []
    for mode in ('repo', 'env'):
        command = [str(root / 'env/bin/python'), '-I', '-B', '-c', PROBE,
                   str(root), mode, expected_sha, NAME]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=45)
        require(result.returncode == 0, f'{mode} CPU import failed: {result.stderr[-4000:]}')
        report = json.loads(result.stdout)
        require(report['sha256'] == expected_sha and report['cuda_initialized'] is False,
                f'{mode} invalid CPU probe report')
        reports.append(report)
    return reports


def write_json(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def atomic_replace(path, data, mode):
    regular(path)
    fd, name = tempfile.mkstemp(prefix='.' + NAME + '.deploy-', dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            os.fchmod(handle.fileno(), mode)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        staged.unlink(missing_ok=True)


@contextlib.contextmanager
def deployment_lock(root):
    import fcntl
    admission = Path('/tmp/batchgen-runtime-locks')
    admission.mkdir(mode=0o700, exist_ok=True)
    require(admission.resolve(strict=True) == admission and admission.is_dir(), 'Unsafe BatchGen admission lock root')
    require(admission.stat().st_uid == os.geteuid() and not admission.stat().st_mode & 0o077,
            'BatchGen admission lock root ownership/permissions differ')
    descriptors = []
    try:
        # Hold the real server admission lock as well as this tool's transaction
        # lock. Preflight can run before admission, so /proc audits still apply.
        for path in (root / '.mgn-cub-deployment.lock', admission / 'host.lock'):
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def validate_backups(root, transaction):
    manifest = json.loads((transaction / 'transaction.json').read_text())
    require(manifest['root'] == str(root) and manifest['source_commit'] == SOURCE,
            'Rollback manifest belongs to another deployment')
    _, _, targets = layout(root)
    require(manifest['targets'] == {key: str(path) for key, path in targets.items()},
            'Rollback targets differ from pinned import locations')
    require(manifest['old_sha256'] == OLD_SHA and manifest['new_sha256'] == NEW_SHA,
            'Rollback SHA pins differ')
    backups = {}
    for key, target in targets.items():
        backup = regular(transaction / (key + '.original.so'))
        require(digest(backup) == OLD_SHA, f'Rollback backup SHA mismatch: {key}')
        require(digest(regular(target)) in (OLD_SHA, NEW_SHA), f'Unexpected current target: {target}')
        backups[key] = backup.read_bytes()
    return manifest, backups


def restore(root, transaction):
    manifest, backups = validate_backups(root, transaction)
    _, _, targets = layout(root)
    for key, target in targets.items():
        audit_no_batchgen(root)
        require(digest(regular(target)) in (OLD_SHA, NEW_SHA), f'Target changed before rollback: {target}')
        atomic_replace(target, backups[key], manifest['modes'][key])
    require(protected_inventory(root) == manifest['protected'], 'Protected native/loader files changed')
    return {'status': 'rolled_back', 'rollback_from': str(transaction),
            'probes': verify_imports(root, OLD_SHA), 'protected_unchanged': True}


def apply(root, rank, wheel, output):
    data = candidate_bytes(wheel)
    _, _, targets = layout(root)
    protected = protected_inventory(root)
    for target in targets.values():
        require(digest(regular(target)) == OLD_SHA, f'Expected original MGN: {target}')
    audit = audit_no_batchgen(root)
    before = verify_imports(root, OLD_SHA)
    output.mkdir(mode=0o700)  # Fresh transaction only; never overwrite evidence.
    manifest = dict(plan(root, rank, wheel), status='prepared', protected=protected,
                    modes={key: stat.S_IMODE(path.stat().st_mode) for key, path in targets.items()},
                    process_audit=audit, before_probes=before, prepared_unix=time.time())
    for key, target in targets.items():
        backup = output / (key + '.original.so')
        with backup.open('xb') as handle:
            handle.write(target.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        require(digest(backup) == OLD_SHA, 'Backup changed before deployment')
    write_json(output / 'transaction.json', manifest)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # The new directory entry also needs durability in its parent before any
    # production binary is replaced; fsync(output) alone is insufficient.
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    changed = []
    previous = {}
    def interrupted(signum, frame):
        raise RuntimeError(f'Deployment interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        for key, target in targets.items():
            audit_no_batchgen(root)
            require(digest(regular(target)) == OLD_SHA, f'Target changed before replace: {target}')
            atomic_replace(target, data, manifest['modes'][key])
            changed.append(key)
        require(protected_inventory(root) == protected, 'Protected native/loader files changed')
        result = {'status': 'applied_cpu_import_verified', 'changed': changed,
                  'probes': verify_imports(root, NEW_SHA), 'protected_unchanged': True,
                  'gpu_qualification': 'Previously archived isolated sanitizer; no GPU used by deployment'}
    except BaseException as exc:
        # Do not let another interrupt prevent a recoverable rollback receipt.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        result = {'status': 'apply_failed', 'error': repr(exc), 'changed': changed}
        try:
            result['rollback'] = restore(root, output)
        except BaseException as rollback_exc:
            result['rollback_error'] = repr(rollback_exc)
            result['parent_cleanup_required'] = True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--candidate-wheel', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect', action='store_true', help='Read-only filesystem/process audit; no import')
    modes.add_argument('--apply', action='store_true')
    modes.add_argument('--rollback-from', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh transaction/report directory for mutation')
    parser.add_argument('--confirm-no-active-batchgen', action='store_true',
                        help='Operator keeps BatchGen launchers stopped throughout; /proc audit is still mandatory')
    args = parser.parse_args()
    root = args.root.absolute()
    wheel = args.candidate_wheel or root / ('mgn-cub-17b9fcb-v1/dist/'
        'batchgen_mgn_qualification-0.0.0+17b9fcb-cp311-cp311-linux_x86_64.whl')
    if not (args.inspect or args.apply or args.rollback_from):
        print(json.dumps(plan(root, args.node_rank, wheel), indent=2))
        return 0
    require(root == ROOT and root.resolve(strict=True) == root, 'Only the verified container experiment root is allowed')
    require(socket.gethostname() == HOSTS[args.node_rank] and Path('/.dockerenv').exists(),
            'Expected allocated container hostname/marker')
    if args.inspect:
        print(json.dumps({'status': 'inspection_only', 'protected': protected_inventory(root),
                          'process_audit': audit_no_batchgen(root), 'plan': plan(root, args.node_rank, wheel)}, indent=2))
        return 0
    require(args.confirm_no_active_batchgen and args.output is not None,
            'Mutation requires --confirm-no-active-batchgen and a fresh --output')
    output = args.output.absolute()
    require(output.parent.resolve(strict=True) == root and not output.exists(),
            'Output must be a fresh direct child of the experiment root')
    with deployment_lock(root):
        if args.apply:
            result = apply(root, args.node_rank, wheel, output)
        else:
            transaction = args.rollback_from.absolute()
            require(transaction.parent.resolve(strict=True) == root and transaction.resolve(strict=True) == transaction,
                    'Rollback source must be a direct, nonsymlink transaction directory')
            audit_no_batchgen(root)
            # Validate every backup and target before creating a new report or replacing anything.
            validate_backups(root, transaction)
            output.mkdir(mode=0o700)
            result = restore(root, transaction)
            write_json(output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('applied_cpu_import_verified', 'rolled_back') else 1


if __name__ == '__main__':
    raise SystemExit(main())
