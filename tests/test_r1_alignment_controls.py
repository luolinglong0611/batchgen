import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/r1_alignment_controls.py"
SPEC = importlib.util.spec_from_file_location("r1_alignment_controls", SOURCE)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


class AlignmentControls(unittest.TestCase):
    def test_default_keeps_native_context(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(controls.effective_context_limit(163840, "DeepSeek-R1"), 163840)

    def test_cap_is_separate_from_model_configuration(self):
        model = {"max_position_embeddings": 163840, "rope_scaling": {"factor": 40}}
        env = {controls.CONTEXT_ENV: "32768", "BATCHGEN_R1_SM103_MLA_COMPAT": "flashinfer-fa2",
               "BATCHGEN_R1_SM103_FP8_COMPAT": "1"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(controls.effective_context_limit(model["max_position_embeddings"], "deepseek-ai/DeepSeek-R1"), 32768)
            self.assertEqual(model, {"max_position_embeddings": 163840, "rope_scaling": {"factor": 40}})
            for value in ("0", "-1", "163841", "32768.0", " 32768", "01", ""):
                os.environ[controls.CONTEXT_ENV] = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    controls.effective_context_limit(163840, "DeepSeek-R1")

    def test_other_models_and_partial_profiles_cannot_use_the_override(self):
        env = {controls.CONTEXT_ENV: "32768", "BATCHGEN_R1_SM103_MLA_COMPAT": "flashinfer-fa2",
               "BATCHGEN_R1_SM103_FP8_COMPAT": "1"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                controls.effective_context_limit(163840, "DeepSeek-V3")
            os.environ.pop("BATCHGEN_R1_SM103_FP8_COMPAT")
            with self.assertRaises(ValueError):
                controls.effective_context_limit(163840, "DeepSeek-R1")

    def test_exact_querybook_budget_and_partial_context_page(self):
        self.assertEqual(controls.trajectory_capacity_bytes(32768, 1024), 128 * 1024**2)
        self.assertEqual(controls.trajectory_capacity_bytes(65, 16), 128 * 4 * 16)
        for values in ((0, 16), (64, True), (64, 0)):
            with self.assertRaises(ValueError):
                controls.trajectory_capacity_bytes(*values)


if __name__ == "__main__":
    unittest.main()
