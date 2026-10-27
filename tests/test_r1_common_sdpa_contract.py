"""CPU metadata contracts; optional actual CPU Torch math checks are explicit."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/attention/mla/common_sdpa_backend.py"
SPEC = importlib.util.spec_from_file_location("r1_common_sdpa_contract", SOURCE)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
HAS_TORCH = importlib.util.find_spec("torch") is not None


class MetadataContracts(unittest.TestCase):
    def test_module_import_and_contract_do_not_require_torch_or_select_runtime(self):
        contract = backend.contract()
        self.assertEqual(contract["status"], "explicit_compatibility_candidate")
        self.assertEqual(contract["graph_scope"], "off")
        self.assertFalse(contract["decode"]["cache_writes"])
        self.assertIn("NOT_full_model", contract["qualification_scope"])

    def test_packed_spans_and_empty_rank(self):
        self.assertEqual(backend.plan_packed_prefill(
            (7, 128, 192), (7, 128, 192), (7, 128, 128), [0, 3, 7], 4), [(0, 3), (3, 7)])
        self.assertEqual(backend.plan_packed_prefill(
            (0, 128, 192), (0, 128, 192), (0, 128, 128), [0], 4), [])

    def test_packed_invalid_offsets_dimensions_and_limits_rejected(self):
        bad = [([1, 4], 4), ([0, 0, 4], 4), ([0, 3], 4), ([0, 4], 3), ([0, True, 4], 4)]
        for cu, bound in bad:
            with self.subTest(cu=cu, bound=bound), self.assertRaises(ValueError):
                backend.plan_packed_prefill((4, 128, 192), (4, 128, 192), (4, 128, 128), cu, bound)
        with self.assertRaises(ValueError):
            backend.plan_packed_prefill((4, 128, 192), (4, 1, 192), (4, 128, 128), [0, 4], 4)

    def test_chunk_masks_use_absolute_query_offsets(self):
        self.assertEqual(backend.causal_query_chunks(65), [(0, 32), (32, 64), (64, 65)])
        for first, last in backend.causal_query_chunks(65):
            for query in range(first, last):
                self.assertEqual([key for key in range(65) if key <= query], list(range(query + 1)))

    def test_partial_page_and_padding_have_only_valid_token_indices(self):
        plan = backend.plan_paged_decode((2, 1, 128, 576), (8, 64, 1, 576),
                                          [[3, 1, -1], [4, 6, -999]], [65, 127], 127)
        indices = backend.token_page_indices(plan)
        self.assertEqual(indices["kv_indptr"], [0, 65, 192])
        self.assertEqual(indices["kv_indices"][:65], list(range(192, 256)) + [64])
        self.assertEqual(indices["kv_indices"][65:], list(range(256, 320)) + list(range(384, 447)))
        self.assertNotIn(65, indices["kv_indices"][:65])
        self.assertEqual(indices["allocator_page_size"], 64)
        self.assertEqual(indices["kernel_page_size"], 1)

    def test_exact_boundary_and_empty_batch(self):
        plan = backend.plan_paged_decode((1, 1, 128, 576), (2, 64, 1, 576), [[1, -1]], [64], 64)
        self.assertEqual(backend.token_page_indices(plan)["kv_indices"], list(range(64, 128)))
        self.assertEqual(backend.plan_paged_decode((0, 1, 128, 576), (2, 64, 1, 576), [], [], 64), [])

    def test_shared_pages_across_rows_are_read_only_and_supported(self):
        tables, lengths = [[2], [2]], [3, 5]
        plan = backend.plan_paged_decode((2, 1, 128, 576), (3, 64, 1, 576), tables, lengths, 5)
        self.assertEqual(backend.token_page_indices(plan)["kv_indices"], [128, 129, 130, 128, 129, 130, 131, 132])
        self.assertEqual(tables, [[2], [2]])
        self.assertEqual(lengths, [3, 5])

    def test_invalid_used_pages_lengths_and_duplicate_pages_rejected(self):
        bad = [([[-1]], [1]), ([[3]], [1]), ([[0]], [65]), ([[0, 0]], [65]),
               ([[True]], [1]), ([[0]], [0]), ([[0]], [66])]
        for table, lengths in bad:
            with self.subTest(table=table, lengths=lengths), self.assertRaises(ValueError):
                backend.plan_paged_decode((1, 1, 128, 576), (3, 64, 1, 576), table, lengths, 65)

    def test_page1_int32_overflow_is_rejected(self):
        plan = [{"row": 0, "length": 1, "physical_pages": [2**31 // 64]}]
        with self.assertRaises(ValueError):
            backend.token_page_indices(plan)

    def test_scale_and_query_chunk_domains(self):
        for value in (None, True, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend._scale(value)
        for value in (True, 0, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.causal_query_chunks(10, value)


@unittest.skipUnless(HAS_TORCH, "Torch is unavailable; actual CPU numerical checks not executed")
class TorchCPUContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(20261026)

    def test_packed_causal_head_mapping_matches_independent_fp64_reference(self):
        torch = self.torch
        query = torch.randn(7, 2, 192).bfloat16()
        key = torch.randn_like(query)
        value = torch.randn(7, 2, 128).bfloat16()
        output = backend.packed_prefill_bf16(query, key, value, torch.tensor([0, 3, 7]), 4,
                                             softmax_scale=0.173, query_chunk_size=2)
        for start, stop in ((0, 3), (3, 7)):
            for head in range(2):
                scores = query[start:stop, head].double() @ key[start:stop, head].double().T * 0.173
                mask = torch.ones(stop-start, stop-start, dtype=torch.bool).triu(1)
                scores.masked_fill_(mask, -float("inf"))
                expected = (scores.softmax(-1) @ value[start:stop, head].double()).bfloat16()
                torch.testing.assert_close(output[start:stop, head], expected, rtol=0.008, atol=0.008)

    def test_decode_shuffled_pages_ignores_nan_tails_and_does_not_modify_cache(self):
        torch = self.torch
        query = torch.randn(2, 1, 2, 576).bfloat16()
        cache = torch.full((5, 64, 1, 576), float("nan"), dtype=torch.bfloat16)
        histories = [torch.randn(65, 576).bfloat16(), torch.randn(3, 576).bfloat16()]
        cache[3, :, 0] = histories[0][:64]
        cache[1, 0, 0] = histories[0][64]
        cache[4, :3, 0] = histories[1]
        before = cache.view(torch.int16).clone()
        output = backend.paged_decode_bf16(query, cache, torch.tensor([[3, 1, -1], [4, -1, -1]]),
                                           torch.tensor([65, 3]), 65, softmax_scale=0.091)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(cache.view(torch.int16), before))
        for row, history in enumerate(histories):
            for head in range(2):
                scores = query[row, 0, head].double() @ history.double().T * 0.091
                expected = (scores.softmax(-1) @ history[:, :512].double()).bfloat16()
                torch.testing.assert_close(output[row, 0, head], expected, rtol=0.008, atol=0.008)

    def test_valid_nan_is_not_sanitized(self):
        torch = self.torch
        cache = torch.zeros((1, 64, 1, 576), dtype=torch.bfloat16)
        cache[0, 0, 0, 0] = float("nan")
        query = torch.ones(1, 1, 2, 576).bfloat16()
        output = backend.paged_decode_bf16(query, cache, torch.tensor([[0]]), torch.tensor([1]),
                                           1, softmax_scale=0.1)
        self.assertFalse(torch.isfinite(output).all())

    def test_empty_batches_return_correct_shapes_without_sdpa_work(self):
        torch = self.torch
        prefill = backend.packed_prefill_bf16(torch.empty(0, 2, 192).bfloat16(),
            torch.empty(0, 2, 192).bfloat16(), torch.empty(0, 2, 128).bfloat16(),
            torch.tensor([0]), 1, softmax_scale=0.1)
        decode = backend.paged_decode_bf16(torch.empty(0, 1, 2, 576).bfloat16(),
            torch.empty(1, 64, 1, 576).bfloat16(), torch.empty(0, 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64), 1, softmax_scale=0.1)
        self.assertEqual(tuple(prefill.shape), (0, 2, 128))
        self.assertEqual(tuple(decode.shape), (0, 1, 2, 512))


if __name__ == "__main__":
    unittest.main()
