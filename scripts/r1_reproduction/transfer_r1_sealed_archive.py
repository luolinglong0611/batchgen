#!/usr/bin/env python3
"""Transfer exactly one already-sealed file; no archive extraction or execution.

Default is plan-only. Source and receiver use literal IPv4 addresses and a
private token file. Tokens are never printed; one authenticated attempt consumes
the source token even if the connection fails. No model qualification is granted.
"""
import argparse
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time


BLOCK = 1024 * 1024
ROUTE = '/archive'


def ipv4(value):
    address = ipaddress.ip_address(value)
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError('An explicit unicast IPv4 address is required')
    return str(address)


def check_contract(expected_sha, expected_bytes):
    if not isinstance(expected_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise ValueError('Explicit lowercase SHA256 required')
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError('Explicit positive byte length required')


def check_destination(path):
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError('Destination must be absolute with an existing parent directory')
    if os.path.lexists(path):
        raise FileExistsError('Destination already exists; replacement is forbidden')


def create_token(path):
    check_destination(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        stream.write(secrets.token_hex(32) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_token(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size != 65:
            raise ValueError('Token must be a private ordinary file containing one generated token')
        raw = stream.read(66)
    if not re.fullmatch(rb'[0-9a-f]{64}\n', raw):
        raise ValueError('Invalid token file format')
    return raw[:-1].decode('ascii')


def token_marker(path):
    return path.with_name(path.name + '.used')


def consume_token(path):
    # Persist the claim before sending bytes. A restart cannot reuse this token.
    with os.fdopen(os.open(token_marker(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        stream.write(json.dumps({'consumed_unix_ns': time.time_ns()}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(info):
    return {name: getattr(info, name) for name in
            ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


class SealedFile:
    def __init__(self, path, expected_sha, expected_bytes):
        check_contract(expected_sha, expected_bytes)
        if not path.is_absolute():
            raise ValueError('The one sealed source path must be absolute')
        self.path, self.sha256, self.bytes = path, expected_sha, expected_bytes
        self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
                raise ValueError('Sealed source is not an ordinary file of the pinned length')
            self.before = identity(info)
            self.verify_stat()
            digest = hashlib.sha256()
            while block := os.read(self.fd, BLOCK):
                digest.update(block)
            self.verify_stat()
            if digest.hexdigest() != expected_sha:
                raise ValueError('Sealed source SHA differs before serving')
            os.lseek(self.fd, 0, os.SEEK_SET)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def verify_stat(self):
        # fstat protects the open inode; lstat also detects replacement of the
        # original pathname, including a replacement symlink.
        if identity(os.fstat(self.fd)) != self.before or identity(os.lstat(self.path)) != self.before:
            raise ValueError('Sealed source identity/stat changed')

    def send(self, output, deadline=None, set_timeout=None):
        self.verify_stat()
        os.lseek(self.fd, 0, os.SEEK_SET)
        count, digest = 0, hashlib.sha256()
        while block := os.read(self.fd, BLOCK):
            if count + len(block) > self.bytes:
                raise ValueError('Sealed source grew beyond pinned byte length')
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Sealed archive transfer deadline exceeded')
                if set_timeout is not None:
                    set_timeout(remaining)
            written = output.write(block)
            if written != len(block):
                raise OSError('Short archive socket write')
            digest.update(block)
            count += len(block)
        self.verify_stat()
        if count != self.bytes or digest.hexdigest() != self.sha256:
            raise ValueError('Sealed source changed during streaming')
        return {'bytes': count, 'sha256': digest.hexdigest(), 'source_stat_before': self.before,
                'source_stat_after': identity(os.fstat(self.fd))}

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class OneShot:
    def __init__(self, source, receiver_ip, token_path):
        self.source, self.receiver_ip, self.token_path = source, ipv4(receiver_ip), token_path
        self._token = read_token(token_path)
        if os.path.lexists(token_marker(token_path)):
            raise ValueError('Source token has already been consumed')
        self.consumed = False
        self.result = None

    def claim(self, method, path, peer_ip, authorization, body_length='0', transfer_encoding=None):
        if method != 'GET':
            return 405
        if path != ROUTE:
            return 404
        if peer_ip != self.receiver_ip:
            return 403
        if self.consumed:
            return 410
        if not isinstance(authorization, str) or not hmac.compare_digest(authorization, 'Bearer ' + self._token):
            return 403
        if body_length != '0' or transfer_encoding is not None:
            return 400
        self.source.verify_stat()
        consume_token(self.token_path)
        self.consumed = True
        self._token = ''
        return 200


class ArchiveHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass  # Never log request paths, headers or tokens.

    def handle_one_request(self):
        remaining = self.server.deadline - time.monotonic()
        if remaining <= 0:
            self.close_connection = True
            return
        self.connection.settimeout(min(remaining, 15.0))
        super().handle_one_request()

    def do_GET(self):
        self.close_connection = True
        state = self.server.state
        try:
            auth = self.headers.get_all('Authorization', [])
            lengths = self.headers.get_all('Content-Length', [])
            code = state.claim(self.command, self.path, self.client_address[0],
                               auth[0] if len(auth) == 1 else None,
                               lengths[0] if len(lengths) == 1 else '0' if not lengths else 'invalid',
                               self.headers.get('Transfer-Encoding'))
            self.send_response(code)
            self.send_header('Connection', 'close')
            self.send_header('Content-Length', str(state.source.bytes) if code == 200 else '0')
            if code == 200:
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('X-Archive-SHA256', state.source.sha256)
            self.end_headers()
            if code == 200:
                evidence = state.source.send(self.wfile, self.server.deadline, self.connection.settimeout)
                state.result = {'status': 'archive_bytes_sent_NOT_receiver_verified', **evidence}
        except Exception as exc:
            # Static exception type only: no raw request/token/response content.
            state.result = {'status': 'archive_transfer_failed', 'error_type': type(exc).__name__}

    do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_PATCH = do_GET


def serve_archive(source, source_ip, receiver_ip, port, token_path, timeout):
    state = OneShot(source, receiver_ip, token_path)
    server = HTTPServer((ipv4(source_ip), port), ArchiveHandler)
    server.state = state
    server.deadline = time.monotonic() + timeout
    try:
        while not state.consumed and state.result is None and time.monotonic() < server.deadline:
            server.timeout = min(1.0, max(0.001, server.deadline - time.monotonic()))
            server.handle_request()
        return state.result or {'status': 'archive_transfer_timeout', 'token_consumed': state.consumed}
    finally:
        server.server_close()


def publish_new(temp_path, destination):
    # Same-directory hard-link publication is atomic and fails if destination
    # exists. Unlike replace/ordinary rename, it can never overwrite a racer.
    os.link(temp_path, destination, follow_symlinks=False)
    temp_path.unlink()
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def receive_archive(destination, source_ip, receiver_ip, port, token_path, expected_sha,
                    expected_bytes, timeout, connection_factory=http.client.HTTPConnection):
    check_contract(expected_sha, expected_bytes)
    check_destination(destination)
    source_ip, receiver_ip = ipv4(source_ip), ipv4(receiver_ip)
    token = read_token(token_path)
    connection = connection_factory(source_ip, port, timeout=timeout, source_address=(receiver_ip, 0))
    temp_path = destination.with_name('.' + destination.name + '.part-' + secrets.token_hex(12))
    deadline = time.monotonic() + timeout
    count, digest = 0, hashlib.sha256()
    try:
        connection.request('GET', ROUTE, headers={'Authorization': 'Bearer ' + token, 'Connection': 'close'})
        response = connection.getresponse()
        headers = response.getheaders()
        lengths = [value for name, value in headers if name.lower() == 'content-length']
        hashes = [value for name, value in headers if name.lower() == 'x-archive-sha256']
        if (response.status != 200 or lengths != [str(expected_bytes)] or hashes != [expected_sha]
                or any(name.lower() == 'transfer-encoding' for name, _ in headers)):
            raise ValueError('Source response status/length/SHA contract differs')
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as output:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Sealed archive receive deadline exceeded')
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                block = response.read(min(BLOCK, expected_bytes - count + 1))
                if not block:
                    break
                count += len(block)
                if count > expected_bytes:
                    raise ValueError('Source exceeded pinned byte length')
                output.write(block)
                digest.update(block)
            if count != expected_bytes or digest.hexdigest() != expected_sha:
                raise ValueError('Received archive length/SHA verification failed')
            output.flush()
            os.fsync(output.fileno())
        publish_new(temp_path, destination)
        return {'status': 'sealed_archive_received_verified_NOT_extracted_or_qualified',
                'path': str(destination), 'source_ip': source_ip, 'receiver_ip': receiver_ip,
                'sha256': digest.hexdigest(), 'bytes': count, 'full_model_qualified': False,
                'formal_performance_permitted': False}
    finally:
        connection.close()
        if temp_path.exists():
            temp_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    token = sub.add_parser('token')
    token.add_argument('--output', type=Path, required=True)
    token.add_argument('--execute', action='store_true')
    for name in ('serve', 'receive'):
        command = sub.add_parser(name)
        command.add_argument('--execute', action='store_true')
        command.add_argument('--source-ip', required=True)
        command.add_argument('--receiver-ip', required=True)
        command.add_argument('--port', type=int, required=True)
        command.add_argument('--token-file', type=Path, required=True)
        command.add_argument('--sha256', required=True)
        command.add_argument('--bytes', type=int, required=True)
        command.add_argument('--timeout', type=float, default=600)
        command.add_argument('--archive' if name == 'serve' else '--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'token':
        check_destination(args.output)
        if args.execute:
            create_token(args.output)
        result = {'status': 'private_token_created_NOT_printed' if args.execute else 'plan_only_NOT_executed'}
    else:
        ipv4(args.source_ip)
        ipv4(args.receiver_ip)
        check_contract(args.sha256, args.bytes)
        if not 1 <= args.port <= 65535 or not 0 < args.timeout <= 86400:
            raise ValueError('Valid port and finite timeout up to 86400 seconds required')
        read_token(args.token_file)
        if args.command == 'serve':
            source = SealedFile(args.archive, args.sha256, args.bytes)
            try:
                result = serve_archive(source, args.source_ip, args.receiver_ip, args.port,
                                       args.token_file, args.timeout) if args.execute else {'status': 'plan_only_NOT_executed'}
            finally:
                source.close()
        else:
            check_destination(args.output)
            result = receive_archive(args.output, args.source_ip, args.receiver_ip, args.port,
                                     args.token_file, args.sha256, args.bytes, args.timeout) if args.execute else {'status': 'plan_only_NOT_executed'}
    print(json.dumps(result, sort_keys=True))
    return 1 if result['status'] in ('archive_transfer_failed', 'archive_transfer_timeout') else 0


if __name__ == '__main__':
    raise SystemExit(main())
