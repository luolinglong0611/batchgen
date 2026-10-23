"""CPU-only checks for the predeclared MoE gate, without CUDA dependencies."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('moe_qualification', ROOT / 'scripts/r1_reproduction/moe_gemm_qualification.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GateContract(unittest.TestCase):
    def metrics(self, **changes):
        metrics = dict(finite=True, relative_l2=0, max_abs_over_reference_rms=0, cosine=1)
        metrics.update(changes)
        return metrics

    def test_stage_budgets_fixed(self):
        self.assertEqual(MODULE.GATES['stage'], dict(relative_l2_max=0.005, max_abs_over_reference_rms_max=0.05))
        self.assertTrue(MODULE.accepts(self.metrics(relative_l2=0.005, max_abs_over_reference_rms=0.05), 'stage'))

    def test_each_stage_limit_matters(self):
        for changes in (dict(finite=False), dict(relative_l2=0.00501), dict(max_abs_over_reference_rms=0.05001)):
            with self.subTest(changes=changes):
                self.assertFalse(MODULE.accepts(self.metrics(**changes), 'stage'))

    def test_composition_budgets_fixed(self):
        self.assertEqual(MODULE.GATES['composition'], dict(relative_l2_max=0.02, max_abs_over_reference_rms_max=0.15, cosine_min=0.9998))
        self.assertTrue(MODULE.accepts(self.metrics(relative_l2=0.02, max_abs_over_reference_rms=0.15, cosine=0.9998), 'composition'))

    def test_each_composition_limit_matters(self):
        for changes in (dict(finite=False), dict(relative_l2=0.02001), dict(max_abs_over_reference_rms=0.15001), dict(cosine=0.99979)):
            with self.subTest(changes=changes):
                self.assertFalse(MODULE.accepts(self.metrics(**changes), 'composition'))

    def test_nan_cannot_pass(self):
        for key in ('relative_l2', 'max_abs_over_reference_rms', 'cosine'):
            with self.subTest(key=key):
                self.assertFalse(MODULE.accepts(self.metrics(**{key: float('nan')}), 'composition'))

    def test_real_dimensions_and_sparse_slots(self):
        self.assertEqual((MODULE.HIDDEN, MODULE.INTERMEDIATE, MODULE.TOPK, MODULE.LOCAL_EXPERTS), (7168, 2048, 8, 16))
        self.assertEqual(len(set(MODULE.ACTIVE)), 8)
        self.assertLess(max(MODULE.ACTIVE), 16)
        self.assertGreater(max(MODULE.ACTIVE), 7)


if __name__ == '__main__':
    unittest.main()
