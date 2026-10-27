"""CPU contracts for real-method wrappers; no SG/Torch/GPU is imported."""
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/r1_reproduction/sglang_r1_full_model_trace.py"
spec = importlib.util.spec_from_file_location("r1_sg_trace_contract", PATH)
bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)


class Mode:
    def __init__(self, name): self.name = name
    def is_idle(self): return self.name == "IDLE"
    def is_decode(self): return self.name == "DECODE"


class Logits:
    def __init__(self, rows, marker="returned"): self.shape = (rows, 129280); self.marker = marker
    def dim(self): return 2
    def __getitem__(self, indices): return Logits(len(indices), "eligible-from-" + self.marker)


def schedule_data(name="DECODE"):
    if name == "DECODE":
        reqs = [types.SimpleNamespace(rid="one", origin_input_ids=[11, 12], output_ids=[13], req_pool_idx=7),
                types.SimpleNamespace(rid="two", origin_input_ids=[21, 22], output_ids=[23, 24], req_pool_idx=3)]
        mb = types.SimpleNamespace(forward_mode=Mode(name), global_forward_mode=Mode(name),
                                  seq_lens_cpu=[3, 4], req_pool_indices=[7, 3], input_ids=[13, 24])
    elif name == "EXTEND":
        reqs = [types.SimpleNamespace(rid="one", origin_input_ids=[11, 12, 13, 14], output_ids=[],
                                     fill_ids=[11, 12], req_pool_idx=7)]
        mb = types.SimpleNamespace(forward_mode=Mode(name), global_forward_mode=Mode(name), seq_lens_cpu=[2],
                                  req_pool_indices=[7], input_ids=[11, 12], extend_prefix_lens=[0], extend_seq_lens=[2])
    else:
        reqs = []
        mb = types.SimpleNamespace(forward_mode=Mode("IDLE"), global_forward_mode=Mode("DECODE"),
                                  seq_lens_cpu=[], req_pool_indices=[], input_ids=[])
    return types.SimpleNamespace(reqs=reqs, enable_overlap=False, forward_mode=mb.forward_mode), mb


class BridgeContract(unittest.TestCase):
    def setUp(self):
        bridge._SEEN_SCHEDULE = False
        self.events = []
        self.trace = types.SimpleNamespace(
            register_model_hooks=lambda *args: self.events.append(("register", args)),
            set_execution_phase=lambda *args: self.events.append(("phase", args)),
            capture_logits=lambda *args: self.events.append(("capture", args)),
            flush_layer_stats=lambda *args: self.events.append(("flush", args)))

    def classes(self):
        outer = self
        class Schedule:
            def get_model_worker_batch(self, seq_lens_cpu_cache=None): return self.result
        class Forward:
            @classmethod
            def init_new(cls, batch, model_runner):
                result = cls(); result.__dict__.update(batch.__dict__)
                result.batch_size = len(batch.seq_lens_cpu)
                return result
        class Processor:
            def forward(self, input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states=None):
                outer.events.append(("actual_logits", logits_metadata))
                return types.SimpleNamespace(next_token_logits=Logits(logits_metadata.batch_size + 3, "pre-DP-trim"))
        class Runner:
            def forward(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None,
                        reinit_attn_backend=False, split_forward_count=1):
                outer.events.append(("actual_model", forward_batch))
                if getattr(self, "bypass_processor", False):
                    result = types.SimpleNamespace(next_token_logits=Logits(forward_batch.batch_size))
                else:
                    result = self.processor.forward(forward_batch.input_ids, None, None, forward_batch)
                    # Simulate the real stock post_forward_mlp_sync_batch;
                    # the bridge must capture this actual return, not slice
                    # padded processor output using a guessed DP offset.
                    result.next_token_logits = Logits(forward_batch.batch_size, "actual-post-DP-trim")
                return result, getattr(self, "graph_used", False)
        return Schedule, Forward, Runner, Processor

    def installed(self, name="DECODE"):
        Schedule, Forward, Runner, Processor = self.classes()
        bridge.install_wrappers(Schedule, Forward, Runner, Processor, self.trace)
        data, mb = schedule_data(name)
        schedule = Schedule(); schedule.__dict__.update(data.__dict__); schedule.result = mb
        class Model: pass
        runner = Runner(); runner.model = Model(); runner.tp_rank = 5; runner.tp_size = 16; runner.pp_size = 1; runner.is_draft_worker = False
        runner.processor = Processor()
        runner.server_args = types.SimpleNamespace(disable_cuda_graph=True, disable_overlap_schedule=True,
            enable_piecewise_cuda_graph=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False,
            enable_dp_lm_head=True, speculative_algorithm=None)
        worker_batch = schedule.get_model_worker_batch()
        forward = Forward.init_new(worker_batch, runner)
        return schedule, runner, forward

    def test_default_disabled_installer_never_loads_sg_or_accelerators(self):
        with patch.dict(os.environ, {"R1_FULL_MODEL_TRACE": "0"}), patch.object(bridge.importlib, "import_module", side_effect=AssertionError("unexpected SG import")):
            self.assertEqual(bridge.install()["status"], "disabled")

    def test_runtime_audit_is_opt_in_and_observes_real_return_once_per_model_phase(self):
        _, runner, forward = self.installed()
        with patch.dict(os.environ, {"R1_RUNTIME_AUDIT_DIR": ""}), patch.object(bridge.importlib, "import_module", side_effect=AssertionError("disabled collector import")):
            runner.forward(forward)
        calls = []
        collector = types.SimpleNamespace(record_sg=lambda *args: calls.append(args) or {"observed": True})
        with patch.dict(os.environ, {"R1_RUNTIME_AUDIT_DIR": "/unused-test", "R1_RUNTIME_RUN_ID": "test", "R1_RUNTIME_DEPLOYMENT_SHA": "a"*64}), patch.object(bridge.importlib, "import_module", return_value=collector):
            runner.forward(forward)
            runner.forward(forward)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], (runner.model, runner, runner.server_args, "decode"))
            bridge.record_runtime_once(runner, "prefill")
            self.assertEqual(len(calls), 2)
            old_model = runner.model
            runner.model = type(old_model)()
            runner.forward(forward)
            self.assertEqual(len(calls), 3)
        self.assertIsNot(calls[0][0], calls[-1][0])

    def test_native_ledger_is_immutable_and_exact_not_decoded_text(self):
        schedule, batch = schedule_data()
        rows = bridge.freeze_schedule_rows(schedule, batch)
        self.assertEqual(rows[0], ("one", (11, 12, 13), 3, 7, True, "decode"))
        self.assertEqual(rows[1][1], (21, 22, 23, 24))
        schedule.reqs[0].output_ids.append(999)
        self.assertEqual(rows[0][1], (11, 12, 13))

    def test_native_input_pool_sequence_and_overlap_mismatches_fail_closed(self):
        for mutation in ("pool", "length", "input", "overlap", "bounds"):
            schedule, batch = schedule_data()
            if mutation == "pool": batch.req_pool_indices.reverse()
            elif mutation == "length": batch.seq_lens_cpu[0] -= 1
            elif mutation == "input": batch.input_ids[0] = 100
            elif mutation == "overlap": schedule.enable_overlap = True
            else: schedule.reqs[0].output_ids[-1] = 129280
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bridge.freeze_schedule_rows(schedule, batch)

    def test_actual_forward_mode_set_before_model_and_actual_post_trim_logits_captured(self):
        _, runner, forward = self.installed()
        result = runner.forward(forward)
        names = [e[0] for e in self.events]
        self.assertLess(names.index("phase"), names.index("actual_model"))
        self.assertLess(names.index("register"), names.index("actual_model"))
        capture = next(e[1] for e in self.events if e[0] == "capture")
        self.assertEqual(capture[0], ["one", "two"])
        self.assertEqual(capture[1], [(11, 12, 13), (21, 22, 23, 24)])
        self.assertIs(capture[2], result[0].next_token_logits)
        self.assertEqual(capture[2].marker, "actual-post-DP-trim")
        self.assertEqual(capture[3:], (5, 16, "decode"))

    def test_forward_batch_mapping_mutation_cannot_reach_the_model(self):
        for field in ("req_pool_indices", "input_ids", "seq_lens_cpu"):
            _, runner, forward = self.installed()
            getattr(forward, field)[0] = 999
            self.events.clear()
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.forward(forward)
            self.assertNotIn("actual_model", [e[0] for e in self.events])

    def test_graph_or_missing_actual_logits_processor_witness_cannot_qualify(self):
        for field in ("graph_used", "bypass_processor"):
            _, runner, forward = self.installed(); setattr(runner, field, True)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.forward(forward)

    def test_partial_prefill_is_not_fabricated_as_a_sampled_native_token(self):
        _, runner, forward = self.installed("EXTEND")
        runner.forward(forward)
        self.assertFalse(any(e[0] == "capture" for e in self.events))
        self.assertIn("actual_model", [e[0] for e in self.events])

    def test_idle_rank_uses_global_phase_without_inventing_request_rows(self):
        _, runner, forward = self.installed("IDLE")
        runner.forward(forward)
        mode = next(e[1] for e in self.events if e[0] == "phase")
        self.assertEqual(mode, (5, 16, "decode"))
        self.assertFalse(any(e[0] == "capture" for e in self.events))

    def test_mixed_unknown_modes_and_signature_drift_are_rejected(self):
        for name in ("MIXED", "DRAFT_EXTEND", "SPLIT_PREFILL"):
            with self.assertRaises(ValueError): bridge.phase(Mode(name))
        Schedule, Forward, Runner, Processor = self.classes()
        Runner.forward = lambda self, unknown: None
        with self.assertRaisesRegex(RuntimeError, "signature"):
            bridge.install_wrappers(Schedule, Forward, Runner, Processor, self.trace)


if __name__ == "__main__":
    unittest.main()
