"""CPU-only contracts for checkpoint selection and bounded long-context oracle."""
import ast
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/attention_layer_qualification.py'
TREE = ast.parse(SOURCE.read_text())


def definitions(*names, torch_module=None):
    selected = [node for node in TREE.body if
                (isinstance(node, ast.FunctionDef) and node.name in names) or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SHAPES' for t in node.targets))]
    namespace = {'torch': torch_module}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class ShardSelection(unittest.TestCase):
    def setUp(self):
        self.functions = definitions('attention_tensor_names', 'select_attention_shards')
        self.names = self.functions['attention_tensor_names'](42)
        self.index = {'weight_map': {name: 'part-a.safetensors' if i < 6 else 'part-b.safetensors'
                                   for i, name in enumerate(self.names)}}
        self.entries = {'part-a.safetensors': {}, 'part-b.safetensors': {}}

    def test_exact_twelve_tensors_from_only_requested_layer(self):
        self.index['weight_map']['model.layers.43.self_attn.q_a_proj.weight'] = 'unavailable.safetensors'
        result = self.functions['select_attention_shards'](self.index, self.entries, 42)
        self.assertEqual(len(self.names), 12)
        self.assertEqual({n for names in result.values() for n in names}, set(self.names))
        self.assertEqual(set(result), set(self.entries))

    def test_missing_tensor_rejected(self):
        del self.index['weight_map'][self.names[-1]]
        with self.assertRaises(KeyError):
            self.functions['select_attention_shards'](self.index, self.entries, 42)

    def test_unpinned_and_nonweight_files_rejected(self):
        for filename in ('../untrusted.safetensors', 'model.safetensors.index.json'):
            with self.subTest(filename=filename):
                self.index['weight_map'][self.names[0]] = filename
                with self.assertRaises(ValueError):
                    self.functions['select_attention_shards'](self.index, self.entries, 42)

    def test_invalid_layers_rejected(self):
        for layer in (-1, 61):
            with self.assertRaises(ValueError):
                self.functions['attention_tensor_names'](layer)


@unittest.skipIf(torch is None, 'PyTorch required for CPU numerical checks')
class ChunkedReference(unittest.TestCase):
    def test_dense_equivalence_and_chunk_tails(self):
        function = definitions('reference_causal_attention_chunked', torch_module=torch)['reference_causal_attention_chunked']
        torch.manual_seed(4242)
        for length in (1, 65, 257):
            with self.subTest(length=length):
                q = torch.randn(length, 9, 24).bfloat16()
                k = torch.randn_like(q)
                v = torch.randn(length, 9, 17).bfloat16()
                scores = q.float().transpose(0, 1) @ k.float().transpose(0, 1).transpose(-1, -2) * 0.125
                causal = torch.ones(length, length, dtype=torch.bool).triu(1)
                dense = (scores.masked_fill(causal, -torch.inf).softmax(-1) @ v.float().transpose(0, 1)).transpose(0, 1).bfloat16()
                actual = function(q, k, v, 0.125)
                torch.testing.assert_close(actual, dense, rtol=0, atol=0)

    def test_invalid_tile_rejected(self):
        function = definitions('reference_causal_attention_chunked', torch_module=torch)['reference_causal_attention_chunked']
        x = torch.zeros(1, 1, 8).bfloat16()
        with self.assertRaises(ValueError):
            function(x, x, x, 1.0, query_chunk=0)


if __name__ == '__main__':
    unittest.main()
