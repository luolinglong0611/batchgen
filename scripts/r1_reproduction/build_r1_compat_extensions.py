#!/usr/bin/env python3
"""Build only main's two R1 AOT dependencies in an allocated CPU-only container.

Default is a read-only plan: no Torch/setuptools import, subprocess, network or
filesystem writes. Execution re-captures the real main setup.py declarations,
retains their source lists and mathematical flags, and invokes that setup's
BuildExtension subclass with build_ext --inplace. It does not build core_engine,
enable runtime JIT, initialize CUDA, or qualify any GPU/model arithmetic.
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


ALLOCATED_ROOT = Path("/home/admin/batchgen-r1-alignment-20261026")
CUDA_HOME = ALLOCATED_ROOT / "compat-cuda130/nvidia/cu13"
NAMES = ("batchgen_kernels.common._C_mgn_ops",
         "batchgen_kernels.moe._C_fused_moe_token_permutation")
GUARD = "R1_MAIN_COMPAT_EXTENSIONS_GUARDED"
STATUS = "built_two_R1_AOT_extensions_CPU_imported_NOT_GPU_or_model_qualified"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ExtensionDeclaration:
    """No CUDAExtension construction or toolkit discovery in plan-only mode."""
    def __init__(self, name, sources, **kwargs):
        self.name, self.sources = name, sources
        for key, value in kwargs.items():
            setattr(self, key, value)


def capture_plan_setup(repo, nvcc_threads=4):
    """Execute main setup with narrow CPU stand-ins; do not import native deps.

    The setup and nested _version.py exec share an isolated environment. This
    intentionally captures the source's own declarations rather than copying
    its flags into a second handwritten build recipe.
    """
    repo = Path(repo).resolve()
    setup_path = repo / "batchgen_kernels/setup.py"
    calls = []
    environment = {"BUILD_ARCH": "sm100", "NVCC_THREADS": str(nvcc_threads),
                   "MAX_JOBS": "1", "CUDA_HOME": str(CUDA_HOME)}
    fake_os = SimpleNamespace(path=os.path, environ=environment,
                              cpu_count=os.cpu_count, getenv=environment.get)
    fake_torch = SimpleNamespace(_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
                                 version=SimpleNamespace(cuda="13.0"))
    fake_cpp = SimpleNamespace(BuildExtension=type("PlanBuildExtension", (), {}),
                               CUDAExtension=ExtensionDeclaration)
    fake_setuptools = SimpleNamespace(setup=lambda **kwargs: calls.append(kwargs))
    original_import, original_exec = builtins.__import__, builtins.exec
    isolated_builtins = dict(vars(builtins))

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        substitutes = {"os": fake_os, "torch": fake_torch, "setuptools": fake_setuptools,
                       "torch.utils.cpp_extension": fake_cpp,
                       "shutil": SimpleNamespace(which=lambda _: None)}
        if name in substitutes and level == 0:
            return substitutes[name]
        return original_import(name, globals, locals, fromlist, level)

    def nested_exec(code, globals=None, locals=None):
        namespace = globals if globals is not None else {}
        namespace.setdefault("__builtins__", isolated_builtins)
        return original_exec(code, namespace, locals)

    isolated_builtins.update(__import__=importing, exec=nested_exec)
    namespace = {"__file__": str(setup_path), "__name__": "r1_plan_main_setup",
                 "__builtins__": isolated_builtins}
    with contextlib.redirect_stdout(io.StringIO()):
        original_exec(compile(setup_path.read_bytes(), str(setup_path), "exec"), namespace)
    if len(calls) != 1:
        raise ValueError("main kernel setup must declare exactly one setup() call")
    return calls[0]


def selected_extensions(kwargs, repo, *, allow_injected_includes=False):
    """Filter only by exact name and absolutize paths; never rewrite flags."""
    repo = Path(repo).resolve()
    kernels = repo / "batchgen_kernels"
    selected = []
    for name in NAMES:
        matches = [extension for extension in kwargs["ext_modules"] if extension.name == name]
        if len(matches) != 1:
            raise ValueError("Missing/duplicate main extension declaration: " + name)
        extension = copy.deepcopy(matches[0])
        for attribute in ("sources", "include_dirs"):
            paths = []
            for value in getattr(extension, attribute, []):
                path = Path(value)
                path = path.resolve() if path.is_absolute() else (kernels / path).resolve()
                if not path.exists() or (not path.is_relative_to(repo)
                        and not (attribute == "include_dirs" and allow_injected_includes)):
                    raise ValueError("Main extension input is missing or outside repo: " + str(path))
                paths.append(str(path))
            setattr(extension, attribute, paths)
        flags = getattr(extension, "extra_compile_args", None)
        if not isinstance(flags, dict) or not isinstance(flags.get("nvcc"), list):
            raise ValueError("Unknown main CUDA compile flag contract")
        nvcc = flags["nvcc"]
        architectures = [flag for flag in nvcc if "arch=" in flag or flag.startswith("-arch")]
        if architectures != ["arch=compute_100,code=sm_100"]:
            raise ValueError("R1 compatibility build requires only ordinary SM100 code, no architecture-specific SM90a/SM100a")
        selected.append(extension)
    return selected


def declaration(extension, repo):
    root = Path(repo).resolve()
    return {"name": extension.name,
            "sources": [str(Path(path).relative_to(root)) for path in extension.sources],
            "include_dirs": [str(Path(path).relative_to(root)) for path in extension.include_dirs
                             if Path(path).is_relative_to(root)],
            "extra_compile_args": extension.extra_compile_args,
            "define_macros": getattr(extension, "define_macros", []),
            "undef_macros": getattr(extension, "undef_macros", [])}


def source_manifest(repo, extensions):
    repo = Path(repo).resolve()
    files = {repo / "batchgen_kernels/setup.py", repo / "batchgen_kernels/_version.py"}
    for extension in extensions:
        files.update(map(Path, extension.sources))
        for directory in extension.include_dirs:
            files.update(path for path in Path(directory).rglob("*") if path.is_file())
    entries = []
    for path in sorted(files):
        resolved = path.resolve()
        if not resolved.is_relative_to(repo):
            raise ValueError("Build source/header symlink escapes repo: " + str(path))
        entries.append({"path": str(path.relative_to(repo)), "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    return {"files": entries, "sha256": json_digest(entries)}


def directory_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Explicit compatibility include/library input must be a directory")
    entries = [{"path": str(file.relative_to(path)), "bytes": file.stat().st_size, "sha256": digest(file)}
               for file in sorted(path.rglob("*")) if file.is_file()]
    return {"path": str(path), "files": len(entries), "tree_sha256": json_digest(entries)}


def make_plan(args, full_manifest=False):
    repo = args.repo.resolve(strict=True)
    captured = capture_plan_setup(repo, args.nvcc_threads)
    extensions = selected_extensions(captured, repo)
    manifest = source_manifest(repo, extensions)
    if args.expected_source_manifest_sha256 and manifest["sha256"] != args.expected_source_manifest_sha256:
        raise ValueError("Reviewed main source/header manifest SHA256 mismatch")
    includes = [directory_identity(path) for path in args.extra_include_dir]
    libraries = [directory_identity(path) for path in args.extra_library_dir]
    result = {"schema": "r1-minimal-aot-build-v1", "status": "plan_only_NOT_built",
              "repo": str(repo), "output": str(args.output.resolve()),
              "expected_hostname": args.expected_hostname, "cuda_home": str(args.cuda_home.resolve()),
              "worker_sha256": digest(__file__), "BUILD_ARCH": "sm100",
              "extensions": [declaration(extension, repo) for extension in extensions],
              "source_manifest_sha256": manifest["sha256"], "source_file_count": len(manifest["files"]),
              "source_files": [entry for entry in manifest["files"] if entry["path"].endswith((".cu", ".cc", "setup.py", "_version.py"))
                               and "/3rd/" not in entry["path"]],
              "explicit_extra_include_roots": includes, "explicit_extra_library_roots": libraries,
              "max_jobs": args.max_jobs, "nvcc_threads": args.nvcc_threads,
              "timeout_seconds": args.timeout_seconds,
              "required_environment": {GUARD: "1", "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none"},
              "setup_arguments": ["-v", "build_ext", "--inplace", "--force", "--parallel=1",
                                  "--build-temp", str(args.output.resolve() / "build-temp")],
              "runtime_required": {"torch": "2.9.0+cu130", "CUDA_toolkit": "13.0.88", "CXX11_ABI": True},
              "core_engine_built": False, "runtime_JIT_enabled": False,
              "CUDA_initialization_permitted": False, "GPU_arithmetic_qualified": False,
              "full_model_qualified": False, "formal_test_permitted": False}
    if full_manifest:
        result["source_manifest"] = manifest["files"]
    return result


def validate_execution(args):
    if platform.system() != "Linux" or socket.gethostname() != args.expected_hostname or not Path("/.dockerenv").is_file():
        raise RuntimeError("Execution requires the named allocated Linux container")
    if os.environ.get(GUARD) != "1":
        raise RuntimeError("Execution requires the coordinator-owned " + GUARD + "=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise RuntimeError("CPU compilation requires explicit CUDA_VISIBLE_DEVICES='' and NVIDIA_VISIBLE_DEVICES=none")
    if os.environ.get("BATCHGEN_KERNELS_DEV") not in (None, "0"):
        raise RuntimeError("Runtime JIT must remain disabled")
    for path in (args.repo, args.output, args.cuda_home):
        if not path.is_absolute() or not path.resolve().is_relative_to(ALLOCATED_ROOT.resolve()):
            raise ValueError("Build paths must stay within the owned allocated root")
    if not args.expected_source_manifest_sha256:
        raise ValueError("Execution requires the reviewed --expected-source-manifest-sha256")
    for value in (args.max_jobs, args.nvcc_threads, args.timeout_seconds):
        if type(value) is not int or value <= 0:
            raise ValueError("Compilation parallelism/time limits must be positive integers")


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def child_environment(args):
    env = os.environ.copy()
    # Do not accidentally inherit another CUDA release's compile flags/paths.
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "CFLAGS", "CXXFLAGS",
                 "CPPFLAGS", "LDFLAGS", "NVCCFLAGS", "TORCH_NVCC_FLAGS", "CUDAHOSTCXX", "TORCH_CUDA_ARCH_LIST",
                 "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_PATH", "CUDACXX"):
        env.pop(name, None)
    cuda = args.cuda_home.resolve()
    for name in ("nvcc", "ptxas"):
        if not (cuda / "bin" / name).is_file():
            raise FileNotFoundError("Pinned CUDA13.0 compiler tool missing: " + name)
    cc, cxx = shutil.which("gcc"), shutil.which("g++")
    if not cc or not cxx:
        raise FileNotFoundError("Explicit GCC/G++ host compilers are required")
    env.update(CUDA_HOME=str(cuda), CC=str(Path(cc).resolve()), CXX=str(Path(cxx).resolve()),
               PYTORCH_NVCC=str(cuda / "bin/nvcc"), BUILD_ARCH="sm100",
               PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
               MAX_JOBS=str(args.max_jobs), NVCC_THREADS=str(args.nvcc_threads),
               CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    env.pop("BATCHGEN_KERNELS_DEV", None)
    return env


def compiler_identity(path):
    path = Path(path).resolve(strict=True)
    command = [str(path), "--version"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=True, timeout=30)
    return {"path": str(path), "sha256": digest(path), "argv": command, "version_output": result.stdout}


def build_child(plan_path):
    """Internal, source-bound subprocess; all execution guards still apply."""
    receipt = json.loads(Path(plan_path).read_text())
    args = SimpleNamespace(repo=Path(receipt["repo"]), output=Path(receipt["output"]),
        cuda_home=Path(receipt["cuda_home"]), expected_hostname=receipt["expected_hostname"],
        expected_source_manifest_sha256=receipt["source_manifest_sha256"],
        extra_include_dir=[Path(item["path"]) for item in receipt["explicit_extra_include_roots"]],
        extra_library_dir=[Path(item["path"]) for item in receipt["explicit_extra_library_roots"]],
        max_jobs=receipt["max_jobs"], nvcc_threads=receipt["nvcc_threads"], timeout_seconds=receipt["timeout_seconds"])
    validate_execution(args)
    if Path(plan_path).resolve() != (args.output / "plan.json").resolve() or digest(__file__) != receipt["worker_sha256"]:
        raise ValueError("Internal build plan/worker binding mismatch")
    current = make_plan(args)
    for key in ("extensions", "explicit_extra_include_roots", "explicit_extra_library_roots"):
        if current[key] != receipt[key]:
            raise ValueError("Build source/declaration/explicit extra inputs changed")
    import torch
    if str(torch.__version__) != "2.9.0+cu130" or torch.version.cuda != "13.0" or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("Pinned Torch2.9cu130 / CXX11 ABI required")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before CPU compilation")
    import setuptools
    import torch.utils.cpp_extension as cpp
    if Path(cpp.CUDA_HOME).resolve() != args.cuda_home.resolve():
        raise RuntimeError("Torch cpp_extension resolved a different CUDA toolkit")
    compiler = {name: compiler_identity(path) for name, path in {
        "nvcc": args.cuda_home / "bin/nvcc", "ptxas": args.cuda_home / "bin/ptxas",
        "CC": os.environ["CC"], "CXX": os.environ["CXX"], "python": sys.executable,
    }.items()}
    if not re.search(r"\bV13\.0\.88\b", compiler["nvcc"]["version_output"]):
        raise RuntimeError("The explicit private CUDA compiler must be V13.0.88")
    save_json(args.output / "compiler-identities.json", compiler)
    save_json(args.output / "torch-build-identity.json", {
        "observation_phase": "before_compilation",
        "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "CXX11_ABI": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torch_module": {"path": torch.__file__, "sha256": digest(torch.__file__)},
        "cpp_extension_source": {"path": cpp.__file__, "sha256": digest(cpp.__file__)},
        "CUDA_HOME": cpp.CUDA_HOME, "CUDA_initialized": torch.cuda.is_initialized(),
    })
    captured = []
    with patch.object(setuptools, "setup", side_effect=lambda **kwargs: captured.append(kwargs)):
        runpy.run_path(str(args.repo / "batchgen_kernels/setup.py"), run_name="r1_real_main_setup_capture")
    if len(captured) != 1:
        raise RuntimeError("main setup capture is ambiguous")
    kwargs = captured[0]
    # CUDAExtension adds the pinned Torch/toolkit include directories when
    # constructing a real Extension. Main's explicit repo include list still
    # must match; generated compiler commands record the injected paths.
    selected = selected_extensions(kwargs, args.repo, allow_injected_includes=True)
    if [declaration(extension, args.repo) for extension in selected] != receipt["extensions"]:
        raise RuntimeError("Real CUDAExtension declarations differ from the reviewed CPU plan")
    kwargs["ext_modules"] = selected
    kwargs["package_dir"] = {name: str((args.repo / "batchgen_kernels" / relative).resolve())
                             for name, relative in kwargs["package_dir"].items()}
    kwargs["script_args"] = receipt["setup_arguments"]
    for extension in selected:
        extension.include_dirs.extend(str(path.resolve()) for path in args.extra_include_dir)
        extension.library_dirs.extend(str(path.resolve()) for path in args.extra_library_dir)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while declaring CPU-only build")
    os.chdir(args.repo)
    setuptools.setup(**kwargs)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during AOT CPU compilation")
    imports = []
    from importlib.machinery import EXTENSION_SUFFIXES
    for name in NAMES:
        stem = args.repo / Path(*name.split("."))
        artifacts = [Path(str(stem) + suffix) for suffix in EXTENSION_SUFFIXES if Path(str(stem) + suffix).is_file()]
        if len(artifacts) != 1:
            raise RuntimeError("Expected one in-place AOT artifact: " + name)
        path = artifacts[0].resolve()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU import initialized CUDA: " + name)
        imports.append({"name": name, "path": str(path), "bytes": path.stat().st_size,
                        "sha256": digest(path), "CPU_import_ok": True, "CUDA_initialized": False})
    save_json(args.output / "artifact-imports.json", imports)
    return 0


def execute(args):
    validate_execution(args)
    plan = make_plan(args, full_manifest=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("A new exclusive build output directory is required")
    # Refuse to overwrite an older binary: the coordinator can first preserve
    # or remove its owned prior build explicitly, then prepare a fresh build.
    from importlib.machinery import EXTENSION_SUFFIXES
    for name in NAMES:
        stem = args.repo / Path(*name.split("."))
        if any(Path(str(stem) + suffix).exists() or Path(str(stem) + suffix).is_symlink()
               for suffix in EXTENSION_SUFFIXES):
            raise FileExistsError("Preserve the existing in-place AOT artifact before rebuilding: " + name)
    env = child_environment(args)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "plan.json", plan)
    command = [sys.executable, str(Path(__file__).resolve()), "--_build-child", str(args.output / "plan.json")]
    report = dict(plan, status="building_two_R1_AOT_extensions", argv=command,
                  started_unix=time.time(), environment={name: env.get(name) for name in (
                      "CUDA_HOME", "CC", "CXX", "PYTORCH_NVCC", "MAX_JOBS", "NVCC_THREADS", "BUILD_ARCH",
                      "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")})
    save_json(args.output / "report.json", report)
    started = time.monotonic()
    exit_code = 1
    process = None
    try:
        with (args.output / "build.stdout").open("xb") as stdout, (args.output / "build.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, env=env, cwd=args.repo, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            report["pid"] = process.pid
            save_json(args.output / "report.json", report)
            try:
                code = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise TimeoutError("Owned AOT build subprocess exceeded its deadline")
        report["build_exit_code"] = code
        if code:
            raise RuntimeError("AOT build/CPU import failed; original compiler logs preserved")
        if make_plan(args)["source_manifest_sha256"] != plan["source_manifest_sha256"]:
            raise RuntimeError("Main build sources/headers changed during compilation")
        report["artifacts"] = json.loads((args.output / "artifact-imports.json").read_text())
        if len(list((args.output / "build-temp").rglob("build.ninja"))) != 2:
            raise RuntimeError("Expected one recorded Ninja compile/link command manifest per extension")
        if digest(__file__) != plan["worker_sha256"]:
            raise RuntimeError("Build worker changed during execution")
        report.update(status=STATUS, sources_unchanged=True, worker_unchanged=True)
        exit_code = 0
    except BaseException as error:
        report.update(status="failed_R1_minimal_AOT_build", error_type=type(error).__name__, error=str(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        report["ninja_commands"] = [{"path": str(path), "sha256": digest(path), "text": path.read_text()}
                                    for path in sorted((args.output / "build-temp").rglob("build.ninja"))]
        for key, filename in (("compiler_identities", "compiler-identities.json"),
                              ("torch_build_identity", "torch-build-identity.json")):
            if (args.output / filename).is_file():
                report[key] = json.loads((args.output / filename).read_text())
        report["inplace_artifacts_including_partial_builds"] = []
        for name in NAMES:
            stem = args.repo / Path(*name.split("."))
            for suffix in EXTENSION_SUFFIXES:
                path = Path(str(stem) + suffix)
                if path.is_file():
                    report["inplace_artifacts_including_partial_builds"].append({
                        "name": name, "path": str(path.resolve()), "sha256": digest(path),
                        "CPU_import_qualified": any(item.get("name") == name for item in report.get("artifacts", [])),
                    })
        report["elapsed_seconds_NOT_benchmark"] = time.monotonic() - started
        save_json(args.output / "report.json", report)
    return exit_code


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) == 2 and values[0] == "--_build-child":
        return build_child(Path(values[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--cuda-home", type=Path, default=CUDA_HOME)
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--extra-include-dir", type=Path, action="append", default=[])
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--nvcc-threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(values)
    if args.execute:
        return execute(args)
    print(json.dumps(make_plan(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
