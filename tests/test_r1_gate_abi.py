"""Exercise the real R1 call and Python wrapper without CUDA dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py'
WRAPPER = ROOT / 'batchgen_kernels/common/mgn.py'


def extract(path, name, namespace, class_name=None):
    body = ast.parse(path.read_text()).body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Scores:
    def __init__(self, rows):
        self.rows = rows

    def size(self, dim):
        return len(self.rows) if dim == 0 else len(self.rows[0])


class Hidden:
    shape = (2, 1, 7168)

    def view(self, *shape):
        return self

    def float(self):
        return self


class GateABI(unittest.TestCase):
    def test_r1_routes_eight_of_256_and_normalizes_unbiased_scores(self):
        # Descending scores normally favor the high expert IDs. Bias forces
        # low IDs to win instead, while weights must still use original scores.
        scores = Scores([[0.1 + i / 512 for i in range(256)],
                         [0.2 + i / 1024 for i in range(256)]])
        bias = [4.0 - i / 256 for i in range(256)]
        calls = []

        def native(scores, bias, groups, selected_groups, experts, topk, scale):
            calls.append((groups, selected_groups, experts, topk, scale))
            ids, weights = [], []
            width = experts // groups
            for row in scores.rows:
                choice = [a + b for a, b in zip(row, bias)]
                group_scores = [sum(sorted(choice[g * width:(g + 1) * width], reverse=True)[:2])
                                for g in range(groups)]
                selected = sorted(range(groups), key=lambda g: group_scores[g], reverse=True)[:selected_groups]
                eligible = [i for i in range(experts) if i // width in selected]
                chosen = sorted(eligible, key=lambda i: choice[i], reverse=True)[:topk]
                total = sum(row[i] for i in chosen)
                ids.append(chosen)
                weights.append([row[i] * scale / total for i in chosen])
            return ids, weights

        wrapper = extract(WRAPPER, 'moe_fused_gate', {
            'torch': SimpleNamespace(Tensor=object), 'Tuple': Tuple,
            '_get_mgn': lambda: SimpleNamespace(moe_fused_gate=native),
        })
        hybrid = extract(MODEL, 'moe_gate_forward_hybrid', {
            'torch': SimpleNamespace(sigmoid=lambda x: x),
            'F': SimpleNamespace(linear=lambda *args: scores),
            'moe_fused_gate': wrapper,
        }, class_name='MoEGate')
        gate = SimpleNamespace(weight=Hidden(), e_score_correction_bias=bias,
                               n_group=8, topk_group=4, n_routed_experts=256,
                               top_k=8, routed_scaling_factor=2.5)
        ids, weights = hybrid(gate, Hidden())
        self.assertEqual(calls, [(8, 4, 256, 8, 2.5)])
        self.assertEqual(ids, [list(range(8)), list(range(8))])
        for row, output in zip(scores.rows, weights):
            self.assertEqual(len(output), 8)
            self.assertAlmostEqual(sum(output), 2.5)
            for i, value in enumerate(output):
                self.assertAlmostEqual(value, row[i] / sum(row[:8]) * 2.5)
            # Bias affects the chosen IDs, not the weights being normalized.
            self.assertLess(output[0], output[-1])

    def test_all_r1_native_gate_calls_go_through_one_reviewed_hybrid_call(self):
        tree = ast.parse(MODEL.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'moe_fused_gate']
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].args)
        self.assertEqual({k.arg for k in calls[0].keywords}, {
            'input_tensor', 'bias', 'num_expert_group', 'topk_group', 'topk',
            'num_fused_shared_experts', 'routed_scaling_factor',
        })


if __name__ == '__main__':
    unittest.main()
