"""CPU checks for the native allocator qualification's independent oracle."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/qualify_sglang_r1_page64_allocator.py'
SPEC = importlib.util.spec_from_file_location('sg_allocator_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class NativeAllocatorOracleTests(unittest.TestCase):
    def test_frozen_uniform_oracle_bound_rejects_finite_zero_outputs(self):
        self.assertEqual(M.bf16_ulp_reference(1), 1 / 128)
        self.assertEqual(M.bf16_ulp_reference(-2), 1 / 64)
        self.assertEqual(M.bf16_ulp_reference(0), 2.0**-133)
        for oracle in (0.96875, 2, 8.5):
            self.assertGreater(abs(0 - oracle), M.bf16_ulp_reference(oracle))
        with self.assertRaises(ValueError):
            M.bf16_ulp_reference(float('nan'))
    def test_noncontiguous_physical_pages_preserve_exact_token_addresses(self):
        rows = [list(range(64, 128)) + [7 * 64], list(range(3 * 64, 3 * 64 + 63)), []]
        result = M.address_contract(rows, [65, 63, 0])
        self.assertEqual(result['unique_valid_token_addresses'], 128)
        self.assertEqual(result['physical_pages'], 3)
        self.assertEqual(result['allocator_page_size'], 64)
        self.assertEqual(result['kernel_page_size'], 1)

    def test_duplicate_or_offset_page_addresses_are_rejected(self):
        for rows, lengths in (([[64], [64]], [1, 1]), ([[65]], [1]),
                              ([[64, 66]], [2]), ([[64]], [2]), ([[0]], [1])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                M.address_contract(rows, lengths)

    def test_plan_never_imports_gpu_or_touches_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'status.json'
            with patch.object(M.importlib, 'import_module') as imported, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(M.main(['--output', str(output)]), 0)
            imported.assert_not_called()
            self.assertFalse(output.exists())

    def test_missing_external_guard_fails_before_import(self):
        with patch.dict(M.os.environ, {}, clear=True), patch.object(M.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(RuntimeError, 'parent single-GPU guard'):
                M.execute(type('Args', (), {'expected_hostname': 'unused'})())
            imported.assert_not_called()


if __name__ == '__main__':
    unittest.main()
