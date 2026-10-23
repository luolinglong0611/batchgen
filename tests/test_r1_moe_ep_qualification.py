"""CPU audit of the EP qualification workload and fixed error contract."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r1_ep_gate', ROOT / 'scripts/r1_reproduction/moe_ep_qualification.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EPQualificationContract(unittest.TestCase):
    def test_unmodified_existing_composition_gate(self):
        tree = ast.parse((ROOT / 'scripts/r1_reproduction/moe_gemm_qualification.py').read_text())
        gates = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'GATES' for t in n.targets))
        self.assertEqual(module.GATE, gates['composition'])

    def test_routes_respect_real_grouped_top8_and_cross_nodes(self):
        for name, counts, selected in module.CASES:
            self.assertEqual(len(counts), 16)
            self.assertEqual(len(set(selected)), 8)
            self.assertTrue(all(0 <= e < 256 for e in selected))
            groups = {e // 32 for e in selected}
            self.assertEqual(len(groups), 4)
            self.assertTrue(all(sum(e // 32 == group for e in selected) == 2 for group in groups))
            self.assertEqual({e // 128 for e in selected}, {0, 1})
        uneven = module.CASES[1][1]
        self.assertEqual(set(uneven), {0, 1, 2, 3})
        self.assertIn(0, uneven[:8])
        self.assertIn(0, uneven[8:])
        self.assertEqual(sum(module.CASES[-1][1]), 0)
        self.assertEqual({e // 16 for _, counts, selected in module.CASES
                          if sum(counts) for e in selected}, set(range(16)))


if __name__ == '__main__':
    unittest.main()
