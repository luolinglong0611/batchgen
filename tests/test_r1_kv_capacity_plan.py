"""CPU arithmetic independent checks, including native reservation boundaries."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_capacity", ROOT / "scripts/r1_reproduction/plan_r1_kv_capacity.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class CapacityPlanTests(unittest.TestCase):
    def test_exact_geometry_capacity_brackets_and_default_rank128_overflow(self):
        result = tool.calculate(list(tool.DEFAULT_TOKENS))
        self.assertEqual(result["geometry"]["BF16_KV_bytes_per_token_all61layers"], 70272)
        self.assertEqual([row["target_bracketing_global_caps"] for row in result["candidates"]],
                         [[160, 176], [336, 352], [688, 704], [1392, 1408], [2784, 2800]])
        self.assertEqual([row["logical_KV_data_gib_per_rank"] for row in result["candidates"]],
                         [8.578125, 17.15625, 34.3125, 68.625, 137.25])
        self.assertEqual(result["candidates"][0]["closest_global_cap_arithmetic_ONLY"], 176)
        self.assertEqual(result["candidates"][-1]["closest_global_cap_arithmetic_ONLY"], 2784)
        self.assertTrue(all(row["requires_raising_BG_default128_rank_limit"] for row in result["candidates"][-1]["options"]))
        self.assertFalse(result["formal_test_permitted"])

    def test_per_node_host_pool_does_not_multiply_by_local8_but_trajectory_is_global(self):
        report = tool.calculate([131072])
        row = next(row for row in report["candidates"][0]["options"] if row["global_active_cap"] == 176)
        self.assertEqual(row["BG_hostKV_balanced_full_sequence_pages_per_node"], 88 * 160)
        self.assertEqual(row["BG_hostKV_balanced_minimum_data_gib_per_node"], 59)
        self.assertEqual(row["BG_hostKV_data_budget_gib_with_planned_page_headroom_per_node"], 70)
        self.assertEqual(row["BG_input_trajectory_bytes_ON_EACH_NODE"], 176 * 32768 * 4)
        self.assertEqual(row["BG_input_trajectory_bytes_ON_EACH_NODE"], 22 * 1024**2)
        self.assertGreaterEqual(row["BG_hostKV_resulting_free_page_percent"], 15)
        self.assertGreater(row["BG_hostKV_all_global_owners_on_one_node_minimum_data_gib"], 59)

    def test_native_host_reservation_plus1_and32_runway_are_capped_not_added_at_EOS(self):
        row = tool.calculate([131072])["candidates"][0]["options"][1]
        a, b = row["profile_reservations"]
        self.assertEqual((a["full_reservation_pages_per_sequence"], b["full_reservation_pages_per_sequence"]), (160, 160))
        self.assertEqual((a["BG_initial_host_reservation_pages_at_candidate_default_chunk"], b["BG_initial_host_reservation_pages_at_candidate_default_chunk"]), (160, 160))
        self.assertEqual((a["BG_initial_GPU_runway_pages_per_sequence"], b["BG_initial_GPU_runway_pages_per_sequence"]), (160, 65))
        self.assertLess(b["SG_prompt_only_occupied_KV_percent"], row["predicted_terminal_KV_pool_occupancy_percent"])

    def test_every_matching_field_is_integral_and_common_not_a_launch_or_occupancy_pass(self):
        result = tool.calculate(list(tool.DEFAULT_TOKENS))
        for candidate in result["candidates"]:
            for row in candidate["options"]:
                fields = row["matching_candidate_fields_NOT_launch_authorization"]
                self.assertEqual(row["global_active_cap"] % 16, 0)
                self.assertEqual(row["per_DP_rank_active_cap"], fields["SG_max_running_requests_global"] // 16)
                self.assertEqual(fields["BG_BATCHGEN_MAX_DECODE_RANK_BSZ"], fields["BG_BATCHGEN_MAX_RANK_BSZ"])
                self.assertEqual(int(fields["BG_BATCHGEN_MAX_RANK_BSZ"]), row["per_DP_rank_active_cap"])
                self.assertEqual(fields["SG_max_total_tokens_per_rank"], int(fields["BG_BATCHGEN_R1_GPU_KV_TOKENS"]))
                self.assertTrue(row["GPU_memory_headroom_NOT_established"])
                self.assertFalse(row["allocation_qualified"])
                self.assertFalse(row["formal_test_permitted"])

    def test_assumed_host_and_GPU_leftovers_are_not_real_free_memory(self):
        result = tool.calculate([2097152], host_ram_gib=1024)
        self.assertTrue(all(not row["named_host_reservations_leave_assumed_reserve"] for row in result["candidates"][0]["options"]))
        row = result["candidates"][0]["options"][0]
        expected = Fraction(275040 * 1024**2 * 85, 100) - (2097152 + 64) * 70272
        self.assertEqual(row["maximum_nonKV_residency_gib_under_fraction_NOT_measured"], float(expected / 1024**3))
        self.assertTrue(row["GPU_memory_headroom_NOT_established"])

    def test_invalid_tokens_or_fraction_cannot_produce_a_candidate(self):
        for values in ([0], [True], [131073], [2**31], [131072, 131072]):
            with self.subTest(values=values), self.assertRaises(ValueError): tool.calculate(values)
        for target in ("0", "1", "NaN", True, .85):
            with self.subTest(target=target), self.assertRaises((ValueError, ZeroDivisionError)): tool.calculate([131072], target=target)


if __name__ == "__main__":
    unittest.main()
