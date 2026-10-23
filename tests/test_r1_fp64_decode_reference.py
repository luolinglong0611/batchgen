"""CPU numerical contract for the explicitly selected diagnostic fallback."""
import ast
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

SOURCE = Path(__file__).resolve().parents[1] / 'batchgen/attention/mla/sm103_backend.py'


@unittest.skipIf(torch is None, 'PyTorch is required for CPU numerical checks')
class FP64DecodeReference(unittest.TestCase):
    def test_bounded_tiles_match_dense_and_ignore_nan_padding(self):
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_torch_fp64_reference')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        torch.manual_seed(6013)
        lengths = torch.tensor([1, 65, 257], dtype=torch.int32)
        heads = 17  # crosses head tile; 257 crosses history tile and page boundary
        query = torch.randn(3, 1, heads, 576).bfloat16()
        cache = torch.full((10, 64, 1, 576), float('nan'), dtype=torch.bfloat16)
        table = torch.full((3, 5), -1, dtype=torch.int32)
        permutation = [7, 2, 8, 1, 4, 6, 0, 3]
        next_page = 0
        expected = []
        for row, length in enumerate(lengths.tolist()):
            pages = (length + 63) // 64
            table[row, :pages] = torch.tensor(permutation[next_page:next_page+pages])
            next_page += pages
            history = torch.randn(length, 576).bfloat16()
            positions = torch.arange(length)
            cache[table[row, positions // 64].long(), positions % 64, 0] = history
            probabilities = ((query[row, 0].double() @ history.double().T) * 0.125).softmax(-1)
            expected.append((probabilities @ history[:, :512].double()).bfloat16())
        actual = namespace['_decode_torch_fp64_reference'](query, cache, table, lengths, 0.125)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual[:, 0], torch.stack(expected), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
