"""Read-only main-declaration capture and CPU-only execution guard contracts."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r1_minimal_aot_build_tests", ROOT / "scripts/r1_reproduction/build_r1_compat_extensions.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def arguments(folder):
    return SimpleNamespace(repo=ROOT, output=Path(folder) / "fresh-build", expected_hostname="allocated-host",
        cuda_home=build.CUDA_HOME, max_jobs=2, nvcc_threads=4, timeout_seconds=1800,
        expected_source_manifest_sha256=None, extra_include_dir=[], extra_library_dir=[])


class MinimalAOTBuildContracts(unittest.TestCase):
    def test_plan_captures_only_two_main_extensions_without_native_imports_processes_or_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            before_env = dict(os.environ)
            before_native = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
            with patch.object(build.subprocess, "Popen", side_effect=AssertionError("No process in plan-only")), \
                 patch.object(build.subprocess, "run", side_effect=AssertionError("No process in plan-only")):
                result = build.make_plan(args)
            self.assertEqual([extension["name"] for extension in result["extensions"]], list(build.NAMES))
            self.assertEqual(result["status"], "plan_only_NOT_built")
            self.assertFalse(result["formal_test_permitted"])
            self.assertFalse(result["runtime_JIT_enabled"])
            self.assertFalse(result["core_engine_built"])
            self.assertFalse(args.output.exists())
            self.assertEqual(dict(os.environ), before_env)
            self.assertEqual({name for name in sys.modules if name == "torch" or name.startswith("torch.")}, before_native)
            json.dumps(result, allow_nan=False)

    def test_main_math_flags_and_source_order_are_retained_including_token_permutation_fast_math(self):
        captured = build.capture_plan_setup(ROOT)
        selected = build.selected_extensions(captured, ROOT)
        self.assertGreater(len(captured["ext_modules"]), 2)
        for extension in selected:
            original = next(ext for ext in captured["ext_modules"] if ext.name == extension.name)
            self.assertEqual(extension.extra_compile_args, original.extra_compile_args)
            self.assertTrue(all(Path(path).is_absolute() for path in extension.sources))
            self.assertEqual([Path(path).name for path in extension.sources], [Path(path).name for path in original.sources])
            self.assertIn("arch=compute_100,code=sm_100", extension.extra_compile_args["nvcc"])
            self.assertFalse(any("90" in flag or "100a" in flag for flag in extension.extra_compile_args["nvcc"]))
        self.assertNotIn("--use_fast_math", selected[0].extra_compile_args["nvcc"])
        self.assertIn("--use_fast_math", selected[1].extra_compile_args["nvcc"])

    def test_unknown_architecture_duplicate_or_missing_main_extension_is_rejected(self):
        captured = build.capture_plan_setup(ROOT)
        for mode in ("architecture", "duplicate", "missing"):
            changed = copy.deepcopy(captured)
            target = next(ext for ext in changed["ext_modules"] if ext.name == build.NAMES[0])
            if mode == "architecture":
                flags = target.extra_compile_args["nvcc"]
                flags[flags.index("arch=compute_100,code=sm_100")] = "arch=compute_103a,code=sm_103a"
            elif mode == "duplicate":
                changed["ext_modules"].append(target)
            else:
                changed["ext_modules"].remove(target)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                build.selected_extensions(changed, ROOT)

    def test_sources_cannot_escape_repo_or_be_missing(self):
        captured = build.capture_plan_setup(ROOT)
        target = next(ext for ext in captured["ext_modules"] if ext.name == build.NAMES[0])
        target.sources = ["/etc/passwd"]
        with self.assertRaisesRegex(ValueError, "outside repo"):
            build.selected_extensions(captured, ROOT)

    def test_header_manifest_and_explicit_extra_include_identity_bind_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            header = Path(folder) / "headers"
            header.mkdir()
            (header / "example.h").write_text("#define VALUE 1\n")
            args.extra_include_dir = [header]
            result = build.make_plan(args, full_manifest=True)
            paths = {entry["path"] for entry in result["source_manifest"]}
            self.assertIn("batchgen_kernels/src/moe/mgn/include/mgn_kernel_ops.h", paths)
            self.assertTrue(any(path.startswith("batchgen_kernels/3rd/cutlass/include/") for path in paths))
            old = result["explicit_extra_include_roots"][0]["tree_sha256"]
            (header / "example.h").write_text("#define VALUE 2\n")
            self.assertNotEqual(build.make_plan(args)["explicit_extra_include_roots"][0]["tree_sha256"], old)
            args.expected_source_manifest_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                build.make_plan(args)

    def test_default_cli_is_plan_only_and_output_exclusive_execution_is_not_attempted(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.object(build, "execute", side_effect=AssertionError("No build")):
                code = build.main(["--output", str(args.output), "--expected-hostname", args.expected_hostname])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["BUILD_ARCH"], "sm100")
            self.assertFalse(args.output.exists())

    def test_execution_requires_allocated_linux_and_explicit_gpu_hiding_and_reviewed_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            args.repo = Path(folder) / "repo"
            args.cuda_home = Path(folder) / "cuda"
            args.expected_source_manifest_sha256 = "a" * 64
            good = {build.GUARD: "1", "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none"}
            with patch.object(build, "ALLOCATED_ROOT", Path(folder)), \
                 patch.object(build.platform, "system", return_value="Linux"), \
                 patch.object(build.socket, "gethostname", return_value=args.expected_hostname), \
                 patch.object(Path, "is_file", return_value=True):
                with patch.dict(os.environ, good, clear=True):
                    build.validate_execution(args)
                for invalid in ({build.GUARD: "0"}, {"CUDA_VISIBLE_DEVICES": "0"},
                                {"NVIDIA_VISIBLE_DEVICES": "all"}, {"BATCHGEN_KERNELS_DEV": "1"}):
                    with self.subTest(invalid=invalid), patch.dict(os.environ, dict(good, **invalid), clear=True):
                        with self.assertRaises(RuntimeError):
                            build.validate_execution(args)
                with patch.dict(os.environ, good, clear=True):
                    args.expected_source_manifest_sha256 = None
                    with self.assertRaisesRegex(ValueError, "reviewed"):
                        build.validate_execution(args)

    def test_wrong_host_is_rejected_before_any_native_dependency_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            with patch.object(build.socket, "gethostname", return_value="other-host"), \
                 patch.object(build.platform, "system", return_value="Linux"), \
                 self.assertRaisesRegex(RuntimeError, "allocated Linux"):
                build.execute(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
