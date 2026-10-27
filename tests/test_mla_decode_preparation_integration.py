"""Exercise the real eager worker hook without loading a CUDA model."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class WorkerPreparationIntegration(unittest.TestCase):
    def run_worker(self, *, model_type="deepseek_v3", paged=True, fa2=True,
                   fail=False, empty=False):
        source = ROOT / "batchgen/batchgen_worker.py"
        method = next(n for n in ast.walk(ast.parse(source.read_text()))
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "_glm5_decode_model_forward")
        tree = ast.Module(body=[*ast.parse("from __future__ import annotations").body,
                                method], type_ignores=[])
        events = []
        manager, mask = object(), object()
        class Positions:
            def numel(self):
                return 0 if empty else 131

            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter")

        positions = Positions()

        @contextlib.contextmanager
        def position_scope(value):
            self.assertIs(value, positions)
            events.append("position-enter")
            try:
                yield
            finally:
                events.append("position-exit")

        @contextlib.contextmanager
        def fa2_scope(value):
            self.assertIs(value, manager)
            events.append("fa2-enter")
            try:
                yield
            finally:
                events.append("fa2-exit")

        class Model:
            config = SimpleNamespace(model_type=model_type)

            def __call__(self, tokens, **kwargs):
                if kwargs != {"attention_mask": mask, "position_ids": positions,
                              "use_cache": False}:
                    raise AssertionError("Model arguments changed")
                events.append("model")
                if fail:
                    raise RuntimeError("model failure")
                return tokens

        namespace = {
            "AttnWrapperBase": SimpleNamespace(gpu_paged_kv_manager=manager if paged else None),
            "Attn_Wrapper": SimpleNamespace(position_ids=positions, attention_mask=mask),
        }
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
        modules = {
            "batchgen.attention.mla.compat_selection": SimpleNamespace(common_attention_enabled=lambda: fa2),
            "batchgen.attention.mla.common_flashinfer_backend": SimpleNamespace(decode_forward_scope=fa2_scope),
            "batchgen.attention.mla.decode_position_validation": SimpleNamespace(mla_decode_position_scope=position_scope),
        }
        worker = SimpleNamespace(model=Model(), model_name="DeepSeek-R1")
        with patch.dict(sys.modules, modules):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    namespace[method.name](worker, "tokens")
            else:
                self.assertEqual(namespace[method.name](worker, "tokens"), "tokens")
        return events

    def test_position_read_precedes_existing_fa2_scope(self):
        self.assertEqual(self.run_worker(), ["position-enter", "fa2-enter", "model",
                                             "fa2-exit", "position-exit"])

    def test_native_paged_deepseek_also_reuses_position_max(self):
        self.assertEqual(self.run_worker(fa2=False), ["position-enter", "model", "position-exit"])

    def test_unrelated_models_and_nonpaged_path_do_not_read_positions(self):
        self.assertEqual(self.run_worker(model_type="other", fa2=False), ["model"])
        self.assertEqual(self.run_worker(paged=False, fa2=False), ["model"])

    def test_exception_unwinds_both_scopes(self):
        self.assertEqual(self.run_worker(fail=True), ["position-enter", "fa2-enter", "model",
                                                      "fa2-exit", "position-exit"])

    def test_empty_inference_positions_skip_scalar_scope(self):
        self.assertEqual(self.run_worker(empty=True),
                         ["fa2-enter", "model", "fa2-exit"])


if __name__ == "__main__":
    unittest.main()
