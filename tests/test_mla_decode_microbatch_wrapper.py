"""Check the actual wrapper's row order, KV callback and MoE handoff on CPU.

The CUDA backend and scope are replaced at the boundary; GPU validation remains
necessary for numerical parity and real per-slice preparation reuse.
"""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/models/deepseek/deepseekv3/wrappers.py"


def actual_functions(state):
    tree = ast.parse(SOURCE.read_text())
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "DeepSeekAttnWrapper")
    method = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_forward_decode")
    parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_read_decode_attention_micro_batch_size")
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body,
                             method, parser], type_ignores=[])
    ns = {"torch": torch, "os": os, "AttnWrapperBase": state}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns


class DecodeMicrobatchWrapper(unittest.TestCase):
    def execute(self, *, batch_size=131, chunk=64, paged=True,
                fail_at=None, bad_output=False, bad_metadata=False):
        events = []
        # Deliberately unsorted row IDs and unequal lengths: ordering must come
        # from the full manager table, not sorting or per-chunk row numbering.
        ids = torch.arange(batch_size - 1, -1, -1)
        positions = (ids % 17 + 8192).reshape(-1, 1)
        lengths = positions.flatten() + 1
        hidden = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 1, 4)
        manager = SimpleNamespace(row_ids=ids, writes=torch.zeros(batch_size, dtype=torch.int64))
        callbacks = []
        state = SimpleNamespace(
            past_key_states=None, attention_mask=None,
            position_ids=positions, scale=None,
            cache_seqlens=lengths[:-1] if bad_metadata else lengths,
            max_seqlen=9000, gpu_paged_kv_manager=manager if paged else None,
            kv_append_callback=lambda layer, k, v: callbacks.append((layer, k.clone(), v)),
        )

        @contextmanager
        def scope(gpu_manager, full_positions, full_lengths, start, end):
            self.assertIs(gpu_manager, manager)
            self.assertIs(full_positions, positions)
            self.assertIs(full_lengths, lengths)
            events.append(("scope-enter", start, end))
            try:
                yield full_positions[start:end], full_lengths[start:end]
            finally:
                events.append(("scope-exit", start, end))

        calls = []
        def backend(h, p, c, max_len, scales, gm, layer, batch_slice):
            start, end = batch_slice if batch_slice is not None else (0, batch_size)
            calls.append((end - start, batch_slice))
            events.append(("attention", start, end))
            if fail_at == start:
                raise RuntimeError("injected backend failure")
            self.assertIs(gm, manager)
            self.assertEqual(max_len, 9000)
            self.assertEqual(layer, 7)
            self.assertTrue(torch.equal(h, hidden[start:end]))
            self.assertTrue(torch.equal(p, positions[start:end]))
            self.assertTrue(torch.equal(c, lengths[start:end]))
            manager.writes[start:end] += 1
            output = h + p.unsqueeze(-1)
            k = ids[start:end].reshape(-1, 1, 1, 1).float()
            if bad_output:
                output = output[:-1]
            return output, k

        def fp8_backend(*args):
            events.append(("fp8",))
            return hidden, None, None

        obj = SimpleNamespace(
            module=SimpleNamespace(decoding_attn_mode_3_bf16=backend,
                                   decoding_attn_mode_3_fp8=fp8_backend),
            layer_idx=7, weight_dequant_scale={}, decode_attn_micro_batch_size=chunk,
        )
        ns = actual_functions(state)
        stub = SimpleNamespace(decode_attention_microbatch_scope=scope)
        with patch.dict(sys.modules, {"batchgen.attention.mla.decode_microbatch_scope": stub}):
            try:
                output = ns["_forward_decode"](obj, hidden)
                # Decoder layer only receives one return and thus makes one
                # full-batch MoE call, after every attention slice has finished.
                events.append(("moe-ingress", output[0].shape[0]))
                error = None
            except Exception as exc:
                output, error = None, exc
        return SimpleNamespace(output=output, error=error, events=events,
                               callbacks=callbacks, calls=calls, manager=manager,
                               hidden=hidden, positions=positions, ids=ids)

    def test_131_rows_chunk64_tail_order_and_single_moe_handoff(self):
        x = self.execute()
        self.assertIsNone(x.error)
        self.assertEqual(x.calls, [(64, (0, 64)), (64, (64, 128)), (3, (128, 131))])
        self.assertTrue(torch.equal(x.output[0], x.hidden + x.positions.unsqueeze(-1)))
        self.assertTrue(torch.equal(x.manager.writes, torch.ones(131, dtype=torch.int64)))
        self.assertEqual(len(x.callbacks), 1)
        self.assertEqual(x.callbacks[0][0], 7)
        self.assertTrue(torch.equal(x.callbacks[0][1].flatten(), x.ids.float()))
        self.assertEqual(x.events[-1], ("moe-ingress", 131))

    def test_chunk32_and_exact_division(self):
        for batch_size, sizes in [(131, [32, 32, 32, 32, 3]), (128, [32] * 4)]:
            with self.subTest(batch_size=batch_size):
                x = self.execute(batch_size=batch_size, chunk=32)
                self.assertIsNone(x.error)
                self.assertEqual([b for b, _ in x.calls], sizes)
                self.assertEqual(len(x.callbacks), 1)

    def test_disabled_and_non_split_cases_preserve_unsliced_backend_call(self):
        for batch_size, chunk in [(131, 0), (32, 32), (7, 64), (0, 64)]:
            with self.subTest(batch_size=batch_size, chunk=chunk):
                x = self.execute(batch_size=batch_size, chunk=chunk)
                self.assertIsNone(x.error)
                self.assertEqual(x.calls, [(batch_size, None)])
                self.assertFalse(any(e[0] == "scope-enter" for e in x.events))
                self.assertEqual(len(x.callbacks), 1)

    def test_backend_failure_unwinds_scope_without_publishing_partial_kv(self):
        x = self.execute(fail_at=64)
        self.assertIsInstance(x.error, RuntimeError)
        self.assertEqual(x.events[-1], ("scope-exit", 64, 128))
        self.assertEqual(x.callbacks, [])
        self.assertFalse(any(e[0] == "moe-ingress" for e in x.events))

    def test_malformed_backend_output_rejected_before_callback(self):
        x = self.execute(bad_output=True)
        self.assertIsInstance(x.error, ValueError)
        self.assertEqual(x.callbacks, [])

    def test_full_batch_metadata_mismatch_rejected_before_any_write(self):
        x = self.execute(bad_metadata=True)
        self.assertIsInstance(x.error, ValueError)
        self.assertEqual(x.calls, [])
        self.assertEqual(x.callbacks, [])

    def test_explicit_option_does_not_silently_enable_unsupported_nonpaged_path(self):
        x = self.execute(paged=False)
        self.assertIsInstance(x.error, ValueError)
        self.assertEqual(x.events, [])
        old = self.execute(paged=False, chunk=0)
        self.assertIsNone(old.error)
        self.assertEqual(old.events[0], ("fp8",))

    def test_option_parsing_and_invalid_values(self):
        parse = actual_functions(SimpleNamespace())["_read_decode_attention_micro_batch_size"]
        name = "BATCHGEN_DECODE_ATTN_MICRO_BATCH_SIZE"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse(), 0)
        for text, expected in [("0", 0), ("32", 32), ("64", 64), ("1", 1)]:
            with patch.dict(os.environ, {name: text}):
                self.assertEqual(parse(), expected)
        for text in ["", "-1", "1.5", "true", " 64", "+64", "６４"]:
            with self.subTest(text=text), patch.dict(os.environ, {name: text}):
                with self.assertRaisesRegex(ValueError, name):
                    parse()

    def test_opt_in_split_rejects_cuda_graph_before_metadata_or_backend_work(self):
        state = SimpleNamespace(
            past_key_states=None, attention_mask=None, position_ids=None,
            scale=None, cache_seqlens=None, max_seqlen=9000,
            gpu_paged_kv_manager=object(), kv_append_callback=None,
        )
        ns = actual_functions(state)
        hidden = SimpleNamespace(shape=(131, 1, 4), ndim=3, is_cuda=True)
        obj = SimpleNamespace(decode_attn_micro_batch_size=64)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA Graph OFF"):
                ns["_forward_decode"](obj, hidden)


if __name__ == "__main__":
    unittest.main()
