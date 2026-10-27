"""CPU/filesystem-only transfer tests: no listener, socket bind or remote calls."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r1_sealed_transfer', ROOT / 'scripts/r1_reproduction/transfer_r1_sealed_archive.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Response:
    def __init__(self, payload, digest, length=None, status=200, extra=None):
        self.stream = io.BytesIO(payload)
        self.status = status
        self.headers = [('Content-Length', str(len(payload) if length is None else length)),
                        ('X-Archive-SHA256', digest)] + (extra or [])

    def getheaders(self):
        return self.headers

    def read(self, size):
        return self.stream.read(size)


class Connection:
    sock = None

    def __init__(self, response):
        self.response, self.closed, self.request_args = response, False, None

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class SealedArchiveTransfer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b'sealed archive bytes\x00' * 100
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.archive = self.root / 'source.tar'
        self.archive.write_bytes(self.payload)
        self.token_path = self.root / 'private-token'
        M.create_token(self.token_path)
        self.destination = self.root / 'received.tar'

    def tearDown(self):
        self.temp.cleanup()

    def source(self):
        return M.SealedFile(self.archive, self.sha, len(self.payload))

    def receive(self, response, timeout=60, destination=None):
        connection = Connection(response)
        calls = []
        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return connection
        self.connection, self.connection_calls = connection, calls
        return M.receive_archive(destination or self.destination, '10.66.9.163', '10.66.9.166', 30201,
                                 self.token_path, self.sha, len(self.payload), timeout,
                                 connection_factory=factory)

    def test_source_stream_uses_pinned_file_hash_length_and_unchanged_stat(self):
        source = self.source()
        try:
            output = io.BytesIO()
            result = source.send(output)
            self.assertEqual(output.getvalue(), self.payload)
            self.assertEqual(result['sha256'], self.sha)
            self.assertEqual(result['bytes'], len(self.payload))
            self.assertEqual(result['source_stat_before'], result['source_stat_after'])
        finally:
            source.close()

    def test_source_rejects_bad_initial_hash_length_and_symlink(self):
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            M.SealedFile(self.archive, '0' * 64, len(self.payload))
        with self.assertRaisesRegex(ValueError, 'pinned length'):
            M.SealedFile(self.archive, self.sha, len(self.payload) + 1)
        symlink = self.root / 'link.tar'
        symlink.symlink_to(self.archive)
        with self.assertRaises(OSError):
            M.SealedFile(symlink, self.sha, len(self.payload))

    def test_source_detects_path_replacement_and_mutation_during_stream(self):
        source = self.source()
        try:
            replaced = self.root / 'replacement.tar'
            replaced.write_bytes(self.payload)
            replaced.replace(self.archive)
            with self.assertRaisesRegex(ValueError, 'identity/stat changed'):
                source.send(io.BytesIO())
        finally:
            source.close()
        source = self.source()
        archive = self.archive
        class MutatingOutput(io.BytesIO):
            def write(self, block):
                with archive.open('r+b') as stream:
                    stream.write(b'X')
                return super().write(block)
        try:
            with self.assertRaisesRegex(ValueError, 'identity/stat changed'):
                source.send(MutatingOutput())
        finally:
            source.close()

    def test_token_is_private_and_authorization_is_single_file_single_receiver_single_use(self):
        self.assertEqual(os.stat(self.token_path).st_mode & 0o077, 0)
        source = self.source()
        try:
            state = M.OneShot(source, '10.66.9.166', self.token_path)
            auth = 'Bearer ' + M.read_token(self.token_path)
            self.assertEqual(state.claim('GET', '/', '10.66.9.166', auth), 404)
            self.assertEqual(state.claim('GET', '/archive?file=other', '10.66.9.166', auth), 404)
            self.assertEqual(state.claim('POST', '/archive', '10.66.9.166', auth), 405)
            self.assertEqual(state.claim('GET', '/archive', '10.66.9.1', auth), 403)
            self.assertEqual(state.claim('GET', '/archive', '10.66.9.166', 'Bearer incorrect'), 403)
            self.assertEqual(state.claim('GET', '/archive', '10.66.9.166', auth, '1'), 400)
            self.assertFalse(M.token_marker(self.token_path).exists())
            self.assertEqual(state.claim('GET', '/archive', '10.66.9.166', auth), 200)
            self.assertTrue(M.token_marker(self.token_path).exists())
            self.assertEqual(state.claim('GET', '/archive', '10.66.9.166', auth), 410)
            with self.assertRaisesRegex(ValueError, 'already been consumed'):
                M.OneShot(source, '10.66.9.166', self.token_path)
        finally:
            source.close()
        self.token_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'private ordinary file'):
            M.read_token(self.token_path)

    def test_receiver_streams_hashes_and_publishes_once_without_token_receipt(self):
        with patch.object(M, 'BLOCK', 17):
            result = self.receive(Response(self.payload, self.sha))
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(result['sha256'], self.sha)
        self.assertFalse(result['full_model_qualified'])
        self.assertFalse(result['formal_performance_permitted'])
        self.assertNotIn(M.read_token(self.token_path), json.dumps(result))
        self.assertEqual(self.connection_calls, [(('10.66.9.163', 30201),
            {'timeout': 60, 'source_address': ('10.66.9.166', 0)})])
        self.assertEqual(self.connection.request_args[0], ('GET', '/archive'))
        self.assertTrue(self.connection.closed)
        self.assertEqual(list(self.root.glob('*.part-*')), [])

    def test_http_handler_wires_only_authorized_fixed_file_without_opening_a_socket(self):
        from email.message import Message
        source = self.source()
        try:
            state = M.OneShot(source, '10.66.9.166', self.token_path)
            handler = object.__new__(M.ArchiveHandler)
            handler.server = SimpleNamespace(state=state, deadline=M.time.monotonic() + 60)
            handler.command, handler.path = 'GET', '/archive'
            handler.client_address = ('10.66.9.166', 12345)
            handler.headers = Message()
            handler.headers['Authorization'] = 'Bearer ' + M.read_token(self.token_path)
            handler.connection = SimpleNamespace(settimeout=lambda value: None)
            handler.wfile = io.BytesIO()
            statuses, headers = [], []
            handler.send_response = statuses.append
            handler.send_header = lambda key, value: headers.append((key, value))
            handler.end_headers = lambda: None
            handler.do_GET()
            self.assertEqual(statuses, [200])
            self.assertEqual(handler.wfile.getvalue(), self.payload)
            self.assertIn(('Content-Length', str(len(self.payload))), headers)
            self.assertIn(('X-Archive-SHA256', self.sha), headers)
            self.assertEqual(state.result['status'], 'archive_bytes_sent_NOT_receiver_verified')
            self.assertTrue(state.consumed)
            self.assertNotIn(M.read_token(self.token_path), json.dumps(headers) + json.dumps(state.result))
        finally:
            source.close()

    def test_receiver_rejects_truncation_overflow_and_wrong_bytes_without_publishing(self):
        for payload in (self.payload[:-1], self.payload + b'X', b'X' + self.payload[1:]):
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    self.receive(Response(payload, self.sha, length=len(self.payload)))
                self.assertFalse(self.destination.exists())
                self.assertEqual(list(self.root.glob('*.part-*')), [])
                self.assertTrue(self.connection.closed)

    def test_receiver_rejects_response_identity_header_and_status_changes(self):
        responses = [Response(self.payload, '0' * 64), Response(self.payload, self.sha, status=403),
            Response(self.payload, self.sha, length=len(self.payload) + 1),
            Response(self.payload, self.sha, extra=[('Content-Length', str(len(self.payload)))]),
            Response(self.payload, self.sha, extra=[('Transfer-Encoding', 'chunked')])]
        for response in responses:
            with self.assertRaisesRegex(ValueError, 'response status/length/SHA contract'):
                self.receive(response)
            self.assertFalse(self.destination.exists())
            self.assertEqual(list(self.root.glob('*.part-*')), [])

    def test_existing_or_racing_destination_is_never_replaced(self):
        self.destination.write_bytes(b'previous result')
        with self.assertRaises(FileExistsError):
            self.receive(Response(self.payload, self.sha))
        self.assertEqual(self.destination.read_bytes(), b'previous result')
        self.destination.unlink()
        original_publish = M.publish_new
        def racer(temp_path, destination):
            destination.write_bytes(b'concurrent result')
            original_publish(temp_path, destination)
        with patch.object(M, 'publish_new', side_effect=racer), self.assertRaises(FileExistsError):
            self.receive(Response(self.payload, self.sha))
        self.assertEqual(self.destination.read_bytes(), b'concurrent result')
        self.assertEqual(list(self.root.glob('*.part-*')), [])

    def test_receive_and_source_deadlines_fail_without_published_result(self):
        with patch.object(M.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                self.receive(Response(self.payload, self.sha), timeout=1)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob('*.part-*')), [])
        source = self.source()
        try:
            with patch.object(M.time, 'monotonic', return_value=2), self.assertRaises(TimeoutError):
                source.send(io.BytesIO(), deadline=1)
        finally:
            source.close()

    def test_cli_plans_do_not_bind_connect_write_or_print_tokens(self):
        common = ['--source-ip', '10.66.9.163', '--receiver-ip', '10.66.9.166', '--port', '30201',
                  '--token-file', str(self.token_path), '--sha256', self.sha, '--bytes', str(len(self.payload))]
        output = io.StringIO()
        with patch.object(M, 'HTTPServer', side_effect=AssertionError('must not bind')), \
             patch.object(M, 'receive_archive', side_effect=AssertionError('must not connect')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(M.main(['serve', *common, '--archive', str(self.archive)]), 0)
            self.assertEqual(M.main(['receive', *common, '--output', str(self.destination)]), 0)
            self.assertEqual(M.main(['token', '--output', str(self.root / 'not-created')]), 0)
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / 'not-created').exists())
        self.assertNotIn(M.read_token(self.token_path), output.getvalue())
        self.assertEqual(output.getvalue().count('plan_only_NOT_executed'), 3)


if __name__ == '__main__':
    unittest.main()
