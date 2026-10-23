"""Actual CPU tensor and safety checks; these never qualify CUDA candidates."""

import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/r1_reproduction"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load("attention_only_cpu_test", SCRIPTS / "attention_only_qualification.py")
guard = load("attention_only_cpu_guard_test", SCRIPTS / "attention_only_guard.py")
HAS_TORCH = importlib.util.find_spec("torch") is not None


class PlanAndGuard(unittest.TestCase):
    def test_plan_cli_never_imports_cuda_packages_or_queries_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marker = directory / "nvidia-query"
            binary = directory / "nvidia-smi"
            binary.write_text('#!/bin/sh\ntouch "$QUERY_MARKER"\nexit 99\n')
            binary.chmod(0o755)
            (directory / "sitecustomize.py").write_text(
                "import sys\n"
                "class RejectAccelerators:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                '        if fullname.split(".")[0] in ("torch", "triton", "flashinfer", "flash_attn"):\n'
                '            raise RuntimeError("Plan imported accelerator package: " + fullname)\n'
                "sys.meta_path.insert(0, RejectAccelerators())\n"
            )
            output = directory / "must-not-exist"
            command = [
                sys.executable,
                str(SCRIPTS / "attention_only_qualification.py"),
                "--stage",
                "decode8192",
                "--gpu",
                "1",
                "--expected-hostname",
                "future-container",
                "--config",
                "/future/config.json",
                "--weights-root",
                "/future/weights",
                "--weights-manifest",
                "/future/manifest.json",
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
                env={
                    **os.environ,
                    "PYTHONPATH": str(directory),
                    "QUERY_MARKER": str(marker),
                    "PATH": str(directory) + os.pathsep + os.environ.get("PATH", ""),
                },
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "plan_only_NOT_executed")
            self.assertEqual(plan["lengths_including_current_decode_token"], [8191, 8192, 8193])
            self.assertEqual(plan["layer_index"], 42)
            self.assertEqual(plan["limits"]["session_RSS_bytes"], 8 * 1024**3)
            self.assertEqual(plan["limits"]["target_GPU_bytes"], 8 * 1024**3)
            self.assertEqual(plan["limits"]["seconds"], 900)
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())
            for name, digest in plan["source_sha256"].items():
                self.assertEqual(digest, tool.sha256_file(ROOT / name))

    def test_source_identity_reuse_is_rejected(self):
        original = {"pid": 7, "pgid": 7, "sid": 7, "start_ticks": 13}
        self.assertTrue(guard.same_identity(dict(original), original))
        for field in original:
            changed = {**original, field: original[field] + 1}
            self.assertFalse(guard.same_identity(changed, original))
        self.assertFalse(guard.same_identity(None, original))

    def test_each_external_resource_limit_and_foreign_gpu_pid(self):
        limits = tool.LIMITS
        normal = {
            "elapsed_seconds": 1,
            "session_RSS_bytes": 1,
            "GPU_bytes": 1,
            "MemAvailable_bytes": limits["MemAvailable_min_bytes"],
            "unexpected_GPU_pids": [],
        }
        self.assertIsNone(guard.violation(normal, limits))
        checks = [
            ("elapsed_seconds", limits["seconds"], "wall_time_limit"),
            ("session_RSS_bytes", limits["session_RSS_bytes"], "session_RSS_limit"),
            ("GPU_bytes", limits["target_GPU_bytes"], "assigned_GPU_memory_limit"),
            (
                "MemAvailable_bytes",
                limits["MemAvailable_min_bytes"] - 1,
                "host_memory_floor",
            ),
            ("unexpected_GPU_pids", [123], "unexpected_process_on_assigned_GPU"),
        ]
        for field, value, reason in checks:
            with self.subTest(field=field):
                self.assertEqual(guard.violation({**normal, field: value}, limits), reason)

    def test_signal_requires_unreaped_session_leader_identity(self):
        bad = {"pid": 2, "pgid": 3, "sid": 4, "start_ticks": 5}
        with patch.object(guard, "session_members", return_value=[]), patch.object(
            guard.os, "killpg"
        ) as kill:
            with self.assertRaisesRegex(RuntimeError, "fresh session leader"):
                guard.signal_primary_group(bad, 15)
            kill.assert_not_called()

    def test_extra_group_is_never_signalled_and_requires_parent_cleanup(self):
        leader = {"pid": 7, "pgid": 7, "sid": 7, "start_ticks": 13, "state": "Z"}
        extra = {"pid": 8, "pgid": 8, "sid": 7, "start_ticks": 15, "state": "S"}
        from unittest.mock import Mock

        child = Mock(pid=7)
        child.wait.return_value = 0
        with patch.object(guard, "session_members", return_value=[leader, extra]), patch.object(
            guard, "child_exited", return_value=True
        ), patch.object(guard.os, "killpg") as kill:
            receipt = guard.cleanup(child, leader)
        kill.assert_not_called()
        self.assertFalse(receipt["cleanup_complete"])
        self.assertTrue(receipt["parent_cleanup_required"])
        self.assertEqual(receipt["observed_other_pgid_members"], [extra])
        self.assertEqual(receipt["remaining_live_other_pgid_members"], [extra])

    def test_cleanup_shield_survives_repeated_stop_signals_and_restores_handlers(self):
        def forbidden(signum, frame):
            raise AssertionError("Stop signal interrupted final audit/save")

        previous = {sig: signal.signal(sig, forbidden) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with guard.cleanup_signal_shield():
                for _ in range(4):
                    for sig in (signal.SIGTERM, signal.SIGINT):
                        os.kill(os.getpid(), sig)
                with guard.cleanup_signal_shield():
                    os.kill(os.getpid(), signal.SIGTERM)
                self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            for sig in previous:
                self.assertIs(signal.getsignal(sig), forbidden)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_supervisor_final_source_audit_and_report_save_shield_repeated_stop(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config, manifest = directory / "config.json", directory / "manifest.json"
            config.write_text("{}")
            manifest.write_text("{}")
            args = SimpleNamespace(
                expected_hostname="cpu-only-fixture",
                gpu=0,
                output=directory / "output",
                config=config,
                weights_manifest=manifest,
                weights_root=directory / "unused",
                stage="prefill",
                seed=42,
                repeats=2,
            )
            stopped = False
            injections = []
            original_sha, original_write = tool.sha256_file, Path.write_text
            original_exists = Path.exists

            def inject(where):
                if stopped:
                    injections.append(where)
                    for sig in (signal.SIGTERM, signal.SIGINT):
                        self.assertEqual(signal.getsignal(sig), signal.SIG_IGN)
                        os.kill(os.getpid(), sig)

            def fail_launch(*args, **kwargs):
                nonlocal stopped
                stopped = True
                raise RuntimeError("CPU fixture stops before launching any child")

            def audited_sha(path):
                inject("source-or-report-audit")
                return original_sha(path)

            def audited_write(path, *args, **kwargs):
                if path.name == "guard.json.tmp":
                    inject("final-save")
                return original_write(path, *args, **kwargs)

            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            with patch.dict(sys.modules, {"attention_only_qualification": tool}), patch.object(
                guard.socket, "gethostname", return_value=args.expected_hostname
            ), patch.object(guard, "gpu_apps", return_value=[]), patch.object(
                guard, "memory_available", return_value=128 * 1024**3
            ), patch.object(
                guard.subprocess, "Popen", side_effect=fail_launch
            ), patch.object(
                tool, "sha256_file", side_effect=audited_sha
            ), patch.object(
                Path, "write_text", audited_write
            ), patch.object(
                Path,
                "exists",
                lambda path: True if str(path) == "/.dockerenv" else original_exists(path),
            ):
                self.assertEqual(
                    guard.supervise(args, SCRIPTS / "attention_only_qualification.py"), 1
                )
            receipt = json.loads((args.output / "guard.json").read_text())
            self.assertEqual(receipt["status"], "failed_or_stopped_NOT_qualified")
            self.assertTrue(receipt["sources_unchanged"])
            self.assertIn("source-or-report-audit", injections)
            self.assertIn("final-save", injections)
            for sig, handler in previous.items():
                self.assertEqual(signal.getsignal(sig), handler)


@unittest.skipUnless(
    sys.platform == "linux" and hasattr(os, "WNOWAIT"),
    "Linux WNOWAIT lifecycle test; no CUDA needed",
)
class LinuxGuardLifecycle(unittest.TestCase):
    def test_cleanup_signals_only_primary_worker_which_reaps_its_extra_group(self):
        # A real CPU child imitates Ninja's separate PGID, without compiling or
        # accessing a GPU. The worker reaps that child before it exits.
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready.json"
            code = (
                "import json,os,signal,subprocess,sys,time\n"
                "stopping=False\n"
                "def stop(signum,frame):\n"
                "    global stopping\n"
                "    stopping=True\n"
                "signal.signal(signal.SIGTERM,stop)\n"
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],preexec_fn=os.setpgrp)\n"
                f"with open({str(ready)+'.tmp'!r},'x') as handle: json.dump({{'child_pid':child.pid}},handle)\n"
                f"os.replace({str(ready)+'.tmp'!r},{str(ready)!r})\n"
                "while not stopping: time.sleep(.05)\n"
                "child.terminate()\n"
                "child.wait(timeout=5)\n"
            )
            child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            saved = guard.process_identity(child.pid)
            cleaned = False
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists())
                compiler_pid = json.loads(ready.read_text())["child_pid"]
                members = guard.session_members(saved)
                groups = {row["pgid"] for row in members}
                self.assertIn(child.pid, groups)
                self.assertIn(compiler_pid, groups)
                with patch.object(
                    guard, "gpu_apps", side_effect=AssertionError("GPU query forbidden")
                ):
                    receipt = guard.cleanup(child, saved)
                cleaned = True
                self.assertTrue(receipt["cleanup_complete"])
                self.assertFalse(receipt["parent_cleanup_required"])
                self.assertEqual(receipt["exit_code"], 0)
                self.assertEqual(
                    {row["pid"] for row in receipt["observed_other_pgid_members"]},
                    {compiler_pid},
                )
                self.assertEqual(receipt["remaining_live_other_pgid_members"], [])
                self.assertTrue(receipt["signals"])
                for event in receipt["signals"]:
                    self.assertEqual(event["pgids"], [child.pid])
                self.assertEqual(receipt["remaining_live_session_members"], [])
                self.assertFalse(Path(f"/proc/{child.pid}").exists())
                self.assertFalse(Path(f"/proc/{compiler_pid}").exists())
                # Repeating stop after reap must not signal a numeric PID/PGID,
                # including when that PID is later reused by another process.
                with patch.object(guard.os, "killpg") as kill:
                    with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                        guard.cleanup(child, saved)
                    reused = {**saved, "start_ticks": saved["start_ticks"] + 1}
                    with patch.object(guard, "process_identity", return_value=reused):
                        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                            guard.cleanup(child, saved)
                    kill.assert_not_called()
            finally:
                if not cleaned:
                    guard.cleanup(child, saved)


@unittest.skipUnless(HAS_TORCH, "CPU Torch required")
class TensorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise unittest.SkipTest("Set CUDA_VISIBLE_DEVICES='' explicitly for CPU tests")
        import torch

        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")
        sys.path.insert(0, str(SCRIPTS))
        cls.addClassCleanup(lambda: sys.path.remove(str(SCRIPTS)))
        cls.torch = torch
        cls.primitive = load("attention_only_cpu_reference", SCRIPTS / "attention_qualification.py")
        cls.layer = load("attention_only_cpu_layer", SCRIPTS / "attention_layer_qualification.py")

    def histories(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(4242)
        return [torch.randn(n, 576, generator=generator).bfloat16() for n in (63, 64, 65)]

    def test_real_l42_tensor_selector_excludes_other_layers(self):
        names = self.layer.attention_tensor_names(42)
        index = {
            "weight_map": {
                name: ("a.safetensors" if i < 5 else "b.safetensors")
                for i, name in enumerate(names)
            }
        }
        index["weight_map"]["model.layers.41.self_attn.q_a_proj.weight"] = "missing.safetensors"
        selected = self.layer.select_attention_shards(
            index, {"a.safetensors": {}, "b.safetensors": {}}, 42
        )
        self.assertEqual(len(names), 12)
        self.assertEqual(set(sum(selected.values(), [])), set(names))
        del index["weight_map"][names[0]]
        with self.assertRaises(KeyError):
            self.layer.select_attention_shards(
                index, {"a.safetensors": {}, "b.safetensors": {}}, 42
            )

    def test_page_layout_round_trip_and_actual_adapter_agree(self):
        torch = self.torch
        histories = self.histories()
        layout = tool.build_paged_capture(histories, 123)
        self.assertEqual(int(layout["tail_mask"].sum()), 64)  # 1 + 0 + 63
        self.assertEqual(layout["raw_table"].shape[1] % 2, 1)
        self.assertEqual(layout["table"].shape[1] % 2, 0)
        backend = load(
            "attention_only_cpu_sm103_adapter",
            ROOT / "batchgen/attention/mla/sm103_backend.py",
        )
        before = layout["raw_table"].clone()
        expected = backend._validated_block_table(
            layout["cache"],
            layout["raw_table"],
            layout["lengths"],
            3,
            torch.device("cpu"),
        )
        torch.testing.assert_close(layout["table"], expected, rtol=0, atol=0)
        torch.testing.assert_close(layout["raw_table"], before, rtol=0, atol=0)
        for row, history in enumerate(histories):
            positions = torch.arange(history.shape[0])
            actual = layout["cache"][
                layout["table"][row, positions // 64].long(), positions % 64, 0
            ]
            torch.testing.assert_close(actual, history, rtol=0, atol=0)
        again = tool.build_paged_capture(histories, 123)
        torch.testing.assert_close(layout["table"], again["table"], rtol=0, atol=0)

    def test_tail_fills_preserve_every_valid_byte_and_base_cache(self):
        torch = self.torch
        histories = self.histories()
        layout = tool.build_paged_capture(histories, 123)
        original = tool.tensor_record(layout["cache"])
        for mode in ("zero", "finite_poison", "nan_latent"):
            with self.subTest(mode=mode):
                cache = tool.fill_tail(layout["cache"], layout["tail_mask"], mode)
                self.assertEqual(tool.tensor_record(layout["cache"]), original)
                for row, history in enumerate(histories):
                    positions = torch.arange(history.shape[0])
                    actual = cache[layout["table"][row, positions // 64].long(), positions % 64, 0]
                    self.assertEqual(tool.tensor_record(actual), tool.tensor_record(history))
                if mode == "finite_poison":
                    self.assertTrue(bool(torch.isfinite(cache).all()))
                    self.assertTrue(bool((cache[:, :, 0][layout["tail_mask"]].abs() == 64).all()))
                elif mode == "nan_latent":
                    self.assertEqual(int(torch.isnan(cache).sum()), 64 * 512)
                    self.assertTrue(bool(torch.isfinite(cache[:, :, 0, 512:]).all()))

    def test_same_input_reference_matches_independent_cpu_dense_math(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(88)
        q = torch.randn(5, 4, 192, generator=generator).bfloat16()
        k = torch.randn_like(q)
        v = torch.randn(5, 4, 128, generator=generator).bfloat16()
        actual = self.primitive.prefill_reference(q, k, v, [0, 2, 5], 0.125)
        expected = torch.empty_like(actual)
        for start, end in ((0, 2), (2, 5)):
            for i in range(start, end):
                for h in range(4):
                    scores = (k[start : i + 1, h].float() @ q[i, h].float()) * 0.125
                    expected[i, h] = scores.softmax(0) @ v[start : i + 1, h].float()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        histories = self.histories()
        query = torch.randn(3, 1, 4, 576, generator=generator).bfloat16()
        decoded = self.primitive.decode_reference(query, histories, 0.125)
        for row, history in enumerate(histories):
            for head in range(4):
                scores = history.float() @ query[row, 0, head].float() * 0.125
                oracle = scores.softmax(0) @ history[:, :512].float()
                torch.testing.assert_close(decoded[row, 0, head], oracle, rtol=1e-5, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_gate_failures_remain_failed_and_nan_is_explicit(self):
        torch = self.torch
        gate_path = SCRIPTS / "attention_qualification.py"
        before = tool.sha256_file(gate_path)
        reference = torch.ones(256)
        exact = reference.bfloat16()
        self.assertEqual(tool.historical_gate(exact, reference, "exact")["status"], "passed")
        changed = exact.clone()
        changed[0] = 2
        failed = tool.historical_gate(changed, reference, "different")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure_count"], 1)
        changed[0] = float("nan")
        nan = tool.historical_gate(changed, reference, "nan-tail-diagnostic")
        self.assertEqual(nan["status"], "failed")
        self.assertIn("nonfinite", nan["reason"])
        self.assertEqual(tool.sha256_file(gate_path), before)
        metrics = tool.independent_metrics(changed, reference)
        self.assertEqual(metrics["actual_nonfinite"], 1)
        self.assertNotIn("relative_l2", metrics)

    def test_exact_artifacts_and_repeat_consistency_are_byte_based(self):
        torch = self.torch
        payload = {
            "bf16": torch.arange(20).bfloat16().reshape(4, 5).t(),
            "fp8": torch.arange(12).to(torch.float8_e4m3fn),
            "positions": torch.tensor([62, 63, 64]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.pt"
            receipt = tool.save_tensors(path, payload)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(receipt["sha256"], tool.sha256_file(path))
            self.assertEqual(
                receipt["tensors"],
                {key: tool.tensor_record(value) for key, value in restored.items()},
            )
            with self.assertRaises(FileExistsError):
                tool.save_tensors(path, payload)
        original = torch.ones(4, dtype=torch.bfloat16)
        self.assertTrue(tool.repeat_metrics(original.clone(), original)["same_raw_bytes"])
        different = original.clone()
        different[0] = 2
        comparison = tool.repeat_metrics(different, original)
        self.assertFalse(comparison["same_raw_bytes"])
        self.assertGreater(comparison["relative_l2"], 0)

    def test_candidate_input_writes_are_fatal_before_another_candidate(self):
        torch = self.torch
        captured = {
            "q": torch.arange(8).bfloat16(),
            "table": torch.tensor([[1, 2]], dtype=torch.int32),
        }
        fixed = {name: tool.tensor_record(value) for name, value in captured.items()}
        cache = torch.zeros(2, 64, 1, 576, dtype=torch.bfloat16)
        before = tool.tensor_record(cache)
        tool.assert_inputs_unchanged(captured, fixed, cache, before)
        captured["table"][0, 0] = 0
        with self.assertRaises(tool.CapturedInputMutation):
            tool.assert_inputs_unchanged(captured, fixed, cache, before)
        captured["table"][0, 0] = 1
        cache[0, 0, 0, 0] = 1
        with self.assertRaises(tool.CapturedInputMutation):
            tool.assert_inputs_unchanged(captured, fixed, cache, before)

    def test_mutation_then_exception_is_fatal_and_synchronization_runs(self):
        from unittest.mock import Mock

        torch = self.torch
        captured = {"q": torch.zeros(4, dtype=torch.bfloat16)}
        fixed = {name: tool.tensor_record(value) for name, value in captured.items()}
        cache = torch.zeros(4, dtype=torch.bfloat16)
        before = tool.tensor_record(cache)
        for target in (captured["q"], cache):
            with self.subTest(target="capture" if target is captured["q"] else "cache"):
                synchronize = Mock()

                def mutate_then_raise(_cache):
                    target[0] = 1
                    raise ValueError("Unrelated candidate error")

                with self.assertRaises(tool.CapturedInputMutation):
                    tool.invoke_immutable_candidate(
                        mutate_then_raise, captured, fixed, cache, before, synchronize
                    )
                synchronize.assert_called_once_with()
                target.zero_()
        synchronize = Mock()

        def ordinary_error(_cache):
            raise ValueError("Unchanged candidate error")

        with self.assertRaisesRegex(ValueError, "Unchanged candidate error"):
            tool.invoke_immutable_candidate(
                ordinary_error, captured, fixed, cache, before, synchronize
            )
        synchronize.assert_called_once_with()

    def test_changed_inputs_are_rejected_before_candidate_call(self):
        from unittest.mock import Mock

        torch = self.torch
        captured = {"q": torch.zeros(4, dtype=torch.bfloat16)}
        fixed = {name: tool.tensor_record(value) for name, value in captured.items()}
        captured["q"][0] = 1
        candidate, synchronize = Mock(), Mock()
        with self.assertRaises(tool.CapturedInputMutation):
            tool.invoke_immutable_candidate(candidate, captured, fixed, None, None, synchronize)
        candidate.assert_not_called()
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
