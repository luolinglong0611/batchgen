"""Ownership and exact write-set contracts; GPU arithmetic is qualified separately."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/r1_reproduction/mla_tail_safety.py"
spec = importlib.util.spec_from_file_location("r1_tail_safety_contract", SOURCE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
qualification_spec = importlib.util.spec_from_file_location(
    "r1_tail_qualification_contract", SOURCE.with_name("qualify_r1_mla_tail_safety.py"))
qualification = importlib.util.module_from_spec(qualification_spec)
with patch.dict(sys.modules, {"mla_tail_safety": tool}):
    qualification_spec.loader.exec_module(qualification)


class TailOwnershipContract(unittest.TestCase):
    def fixture(self):
        return {"sequence_ids": [11, 22, 33], "lengths": [63, 64, 65],
                "table_rows": [[4, -1], [2, -1], [6, 1]],
                "all_sequence_pages": {11: [4], 22: [2], 33: [6, 1]},
                "page_refcounts": {4: 1, 2: 1, 6: 1, 1: 1}, "page_count": 8}

    def test_boundary_ranges_are_only_unused_last_page_tokens(self):
        original = self.fixture()
        before = copy.deepcopy(original)
        result = tool.plan_tail_initialization(**original)
        self.assertEqual(result["ranges"], [
            {"row": 0, "sequence_id": 11, "physical_page": 4, "start_token": 63, "stop_token": 64},
            {"row": 2, "sequence_id": 33, "physical_page": 1, "start_token": 1, "stop_token": 64},
        ])
        self.assertEqual(result["elements_written"], 64 * 576)
        self.assertEqual(original, before)

    def test_complete_ownership_rejects_shared_tail_outside_microbatch(self):
        data = self.fixture()
        data["all_sequence_pages"][99] = [1]
        data["page_refcounts"][1] = 2
        with self.assertRaisesRegex(ValueError, "shared last page"):
            tool.plan_tail_initialization(**data)

    def test_refcount_one_cannot_hide_another_sequence_reference(self):
        data = self.fixture()
        data["all_sequence_pages"][99] = [1]
        with self.assertRaisesRegex(ValueError, "reference counts disagree"):
            tool.plan_tail_initialization(**data)

    def test_wrong_sequence_page_even_if_exclusively_owned_is_rejected(self):
        data = self.fixture()
        data["table_rows"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            tool.plan_tail_initialization(**data)

    def test_unknown_or_out_of_range_ownership_and_duplicate_rows_rejected(self):
        fixtures = []
        data = self.fixture(); data["page_refcounts"].pop(1); fixtures.append(data)
        data = self.fixture(); data["sequence_ids"][0] = 999; fixtures.append(data)
        data = self.fixture(); data["page_count"] = 4; fixtures.append(data)
        data = self.fixture(); data["sequence_ids"][0] = 22; fixtures.append(data)
        data = self.fixture(); data["lengths"][0] = 0; fixtures.append(data)
        data = self.fixture(); data["lengths"][0] = True; fixtures.append(data)
        data = self.fixture(); data["table_rows"][2] = [6]; fixtures.append(data)
        data = self.fixture(); data["table_rows"][0][0] = -1; fixtures.append(data)
        for data in fixtures:
            with self.subTest(data=data), self.assertRaises(ValueError):
                tool.plan_tail_initialization(**data)

    def test_shared_aligned_page_is_never_written(self):
        data = self.fixture()
        data["all_sequence_pages"][99] = [2]
        data["page_refcounts"][2] = 2
        result = tool.plan_tail_initialization(**data)
        self.assertNotIn(2, [r["physical_page"] for r in result["ranges"]])

    def test_reuse_and_microbatch_mapping_have_exact_dense_write_set(self):
        generator = random.Random(261026)
        for _ in range(60):
            lengths = [generator.randint(1, 513) for _ in range(generator.randint(1, 9))]
            required = [(n + 63) // 64 for n in lengths]
            pages = list(range(sum(required) + 7)); generator.shuffle(pages)
            allocations, cursor = {}, 0
            for sequence, count in enumerate(required):
                allocations[sequence] = pages[cursor:cursor + count]; cursor += count
            active = list(range(len(lengths))); generator.shuffle(active)
            active = active[:generator.randint(1, len(active))]
            data = {"sequence_ids": active, "lengths": [lengths[s] for s in active],
                    "table_rows": [allocations[s] + [-1, -1] for s in active],
                    "all_sequence_pages": allocations,
                    "page_refcounts": {p: 1 for p in pages[:cursor]}, "page_count": len(pages)}
            result = tool.plan_tail_initialization(**data)
            actual = {(r["physical_page"], token) for r in result["ranges"]
                      for token in range(r["start_token"], r["stop_token"])}
            expected = {(allocations[s][-1], token) for s in active if lengths[s] % 64
                        for token in range(lengths[s] % 64, 64)}
            valid = {(page, token) for s in range(len(lengths))
                     for logical, page in enumerate(allocations[s])
                     for token in range(min(64, lengths[s] - logical * 64))}
            self.assertEqual(actual, expected)
            self.assertFalse(actual & valid)
            self.assertLessEqual(result["elements_written"], len(active) * 63 * 576)


class QualificationAdmission(unittest.TestCase):
    def command(self, output):
        return [sys.executable, str(SOURCE.with_name("qualify_r1_mla_tail_safety.py")),
                "--expected-hostname", "future-container", "--flashinfer-version", "0.6.13",
                "--output", str(output)]

    def test_plan_does_not_import_accelerators_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "sitecustomize.py").write_text(
                "import sys\n"
                "class RejectAccelerators:\n"
                " def find_spec(self,fullname,path=None,target=None):\n"
                "  if fullname.split('.')[0] in ('torch','flashinfer','triton'):\n"
                "   raise RuntimeError('Plan imported an accelerator package')\n"
                "sys.meta_path.insert(0,RejectAccelerators())\n")
            output = directory / "must-not-exist"
            result = subprocess.run(self.command(output), capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(directory)}, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "plan_only_NOT_executed")
            self.assertEqual(report["lengths"], [1, 63, 64, 65, 2047, 2048, 2049, 8191, 8192, 8193])
            self.assertFalse(output.exists())

    def test_execute_without_parent_guard_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            env = {k: v for k, v in os.environ.items() if k != "R1_MLA_TAIL_GUARDED"}
            result = subprocess.run(self.command(output) + ["--execute"], capture_output=True,
                                    text=True, env=env, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parent-owned bounded single-GPU guard", result.stderr)
            self.assertFalse(output.exists())

    def test_each_failed_or_missing_contract_prevents_a_pass(self):
        good = {"mode": "zero", "initialized": {"finite": True},
                "full_cache_exact_expected_write_set": True, "write_plan_equal": True,
                "initialized_repeat_bitwise_equal": True,
                "initialized_bitwise_equal_to_zero_tail": True,
                "kernel_inputs_unchanged": True, "raw_finite_tail_invariance": True}
        self.assertTrue(qualification.case_contract_passed(good))
        for field in good.keys() - {"mode", "initialized"}:
            for invalid in (False, None, "true"):
                changed = copy.deepcopy(good); changed[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    self.assertFalse(qualification.case_contract_passed(changed))
            changed = copy.deepcopy(good); changed.pop(field)
            self.assertFalse(qualification.case_contract_passed(changed))
        changed = copy.deepcopy(good); changed["initialized"]["finite"] = False
        self.assertFalse(qualification.case_contract_passed(changed))
        changed = copy.deepcopy(good); changed["mode"] = "not-a-known-case"
        self.assertFalse(qualification.case_contract_passed(changed))


if __name__ == "__main__":
    unittest.main()
