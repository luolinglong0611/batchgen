"""Transient OSS receipt reads must not weaken pinned-source validation."""
import errno
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/convert_verified_shards.py'
SPEC = importlib.util.spec_from_file_location('r1_conversion_publication', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Publication(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.entry = {'Path': 'model.safetensors', 'Size': 3, 'Sha256': 'a'*64, 'Revision': 'r'}
        (self.root/'model.safetensors').write_bytes(b'abc')
        (self.root/'model.safetensors.verified.json').write_text(json.dumps(
            {'size': 3, 'sha256': 'a'*64, 'revision': 'r'}))

    def test_transient_receipt_error_defers_then_rechecks(self):
        real_read = MODULE.read_json
        for code in (errno.EIO, errno.ENOENT, errno.ESTALE):
            with self.subTest(errno=code), patch.object(MODULE, 'read_json', side_effect=OSError(code, 'transient')):
                self.assertFalse(MODULE.download_ready(self.root, self.entry))
        self.assertIs(MODULE.read_json, real_read)
        self.assertTrue(MODULE.download_ready(self.root, self.entry))

    def test_permission_errors_remain_fatal(self):
        with patch.object(MODULE, 'read_json', side_effect=PermissionError(errno.EACCES, 'denied')):
            with self.assertRaises(PermissionError):
                MODULE.download_ready(self.root, self.entry)

    def test_mismatched_receipt_remains_fatal(self):
        self.entry['Sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'does not match'):
            MODULE.download_ready(self.root, self.entry)

    def test_partial_json_defers_but_wrong_size_remains_fatal(self):
        receipt = self.root/'model.safetensors.verified.json'
        original = receipt.read_text()
        receipt.write_text('{')
        self.assertFalse(MODULE.download_ready(self.root, self.entry))
        receipt.write_text(original)
        (self.root/'model.safetensors').write_bytes(b'ab')
        with self.assertRaisesRegex(ValueError, 'wrong size'):
            MODULE.download_ready(self.root, self.entry)


if __name__ == '__main__':
    unittest.main()
