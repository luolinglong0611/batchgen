"""CPU-only helper tests; run Torch cases with CUDA_VISIBLE_DEVICES=''.

The Linux lifecycle test uses two spawned processes and a tiny real memfd.
It cannot qualify the 19 GiB skeleton, eight workers, or any GPU operation.
"""
import gc
import hashlib
import importlib.util
import io
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader = _load_module("r1_worker_skeleton_loader", ROOT / "batchgen/worker_skeleton.py")
qualification = _load_module(
    "r1_worker_skeleton_fixture", ROOT / "scripts/r1_reproduction/skeleton_mmap_qualification.py"
)
HAS_TORCH = importlib.util.find_spec("torch") is not None
FLAG = "BATCHGEN_R1_SKELETON_MMAP"
MODEL = "deepseek-ai/DeepSeek-R1"


def _mapped_child(connection, path, inode):
    """Exercise the production helper, not a second torch.load implementation."""
    try:
        os.environ[FLAG] = "1"
        import torch
        torch.set_num_threads(1)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert not torch.cuda.is_initialized()
        tensors = loader.load_worker_skeleton(path, MODEL)
        connection.send({"event": "loaded", "manifest": qualification.tensor_manifest(tensors),
                         "alias_preserved": qualification.alias_contract(tensors),
                         "mapping": qualification.mapping_stats(inode)})
        while True:
            command = connection.recv()
            if command == "mutate":
                tensors["fp32"][0, 0] = 31415.0
                assert tensors["alias"][0].item() == 31415.0
                assert tensors["noncontiguous"][0, 0].item() == 31415.0
            elif command == "owner_closed":
                try:
                    loader.load_worker_skeleton(path, MODEL)
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("New load succeeded after owner FD closed")
            elif command == "release":
                del tensors
                gc.collect()
                connection.send({"event": command, "mapping": qualification.mapping_stats(inode),
                                 "cuda_initialized": torch.cuda.is_initialized()})
                break
            elif command != "snapshot":
                raise ValueError(command)
            connection.send({"event": command, "manifest": qualification.tensor_manifest(tensors),
                             "mapping": qualification.mapping_stats(inode)})
    except BaseException:
        connection.send({"event": "error", "traceback": traceback.format_exc()})
    finally:
        connection.close()


class SelectionContract(unittest.TestCase):
    def test_disabled_unless_flag_is_exactly_one(self):
        for flag in (None, "", "0", "true", "01"):
            with self.subTest(flag=flag), patch.dict(os.environ, {}, clear=True):
                if flag is not None:
                    os.environ[FLAG] = flag
                self.assertFalse(loader.r1_skeleton_mmap_enabled(MODEL))
                self.assertFalse(loader.r1_skeleton_mmap_enabled("any-other-model"))

    def test_exact_r1_selector_and_case_insensitive_dispatch(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            self.assertTrue(loader.r1_skeleton_mmap_enabled(MODEL))
            self.assertTrue(loader.r1_skeleton_mmap_enabled(MODEL.upper()))
            for model in (None, "DeepSeek-R1", "deepseek-ai/DeepSeek-V3",
                          MODEL + "-0528", MODEL + "-Distill-Qwen-32B", "/models/DeepSeek-R1",
                          " " + MODEL):
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, "supports only"):
                    loader.r1_skeleton_mmap_enabled(model)

    def test_opt_in_requires_filename_and_zip_before_loading(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            for path in (io.BytesIO(b"content"), b"file", ""):
                with self.subTest(path=path), self.assertRaisesRegex(TypeError, "filename path"):
                    loader.load_worker_skeleton(path, MODEL)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "not-a-zip.pt"
                path.write_bytes(b"legacy or invalid data")
                with self.assertRaisesRegex(ValueError, "ZIP serialization"):
                    loader.load_worker_skeleton(path, MODEL)
                with self.assertRaises(FileNotFoundError):
                    loader.load_worker_skeleton(path.with_name("missing.pt"), MODEL)


@unittest.skipUnless(HAS_TORCH, "Requires a CPU-capable Torch environment")
class TensorContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise unittest.SkipTest("Set CUDA_VISIBLE_DEVICES='' for CPU-only helper tests")
        import torch
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only tests require CUDA to remain uninitialized")

    def test_ordinary_and_mapped_full_bytes_views_and_aliases_match(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.pt"
            original = qualification.fixture()
            expected = qualification.tensor_manifest(original)
            torch.save(original, path)
            for flag in ("0", "1"):
                with self.subTest(flag=flag), patch.dict(os.environ, {FLAG: flag}):
                    actual = loader.load_worker_skeleton(path, MODEL)
                    self.assertEqual(qualification.tensor_manifest(actual), expected)
                    self.assertTrue(qualification.alias_contract(actual))
                    self.assertTrue(all(t.device.type == "cpu" for t in actual.values()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_default_retains_file_object_and_legacy_format_support(self):
        import torch
        fixture = {"value": torch.arange(8, dtype=torch.bfloat16)}
        stream = io.BytesIO()
        torch.save(fixture, stream, _use_new_zipfile_serialization=False)
        stream.seek(0)
        with patch.dict(os.environ, {FLAG: "0"}):
            actual = loader.load_worker_skeleton(stream, "other-model")
            torch.testing.assert_close(actual["value"], fixture["value"], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            path.write_bytes(stream.getvalue())
            with patch.dict(os.environ, {FLAG: "1"}):
                with self.assertRaisesRegex(ValueError, "ZIP serialization"):
                    loader.load_worker_skeleton(path, MODEL)

    def test_map_shared_default_is_rejected_and_not_changed(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.pt"
            torch.save({"value": torch.arange(8)}, path)
            with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
                with patch.dict(os.environ, {FLAG: "1"}):
                    with self.assertRaisesRegex(RuntimeError, "MAP_PRIVATE"):
                        loader.load_worker_skeleton(path, MODEL)
                self.assertEqual(torch.serialization.get_default_mmap_options(), mmap.MAP_SHARED)

    def test_mmap_failure_propagates_without_ordinary_load_fallback(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.pt"
            torch.save({"value": torch.arange(8)}, path)
            with patch.dict(os.environ, {FLAG: "1"}):
                with patch.object(torch, "load", side_effect=RuntimeError("mmap failed")) as load:
                    with self.assertRaisesRegex(RuntimeError, "mmap failed"):
                        loader.load_worker_skeleton(path, MODEL)
                    self.assertEqual(load.call_count, 1)

    def test_private_mapping_write_preserves_sibling_and_serialized_bytes(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.pt"
            original = qualification.fixture()
            expected = qualification.tensor_manifest(original)
            torch.save(original, path)
            backing_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict(os.environ, {FLAG: "1"}):
                writer = loader.load_worker_skeleton(path, MODEL)
                reader = loader.load_worker_skeleton(path, MODEL)
                writer["fp32"][0, 0] = 31415.0
                self.assertEqual(writer["alias"][0].item(), 31415.0)
                self.assertEqual(writer["noncontiguous"][0, 0].item(), 31415.0)
                self.assertEqual(qualification.tensor_manifest(reader), expected)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), backing_sha)
                # Existing mappings survive removal of their filename. Linux's
                # owner-FD lifecycle is checked separately with real memfds.
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    loader.load_worker_skeleton(path, MODEL)
                self.assertEqual(qualification.tensor_manifest(reader), expected)
                self.assertEqual(writer["fp32"][0, 0].item(), 31415.0)
                del writer, reader
                gc.collect()
            self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(sys.platform == "linux", "Linux memfd and /proc/smaps required")
    def test_two_process_helper_cow_owner_close_and_release(self):
        import torch
        children, connections = [], []
        fd = qualification.create_memfd("batchgen_r1_worker_skeleton_test")
        path = f"/proc/{os.getpid()}/fd/{fd}"
        deadline = time.monotonic() + 90
        try:
            original = qualification.fixture()
            expected = qualification.tensor_manifest(original)
            with os.fdopen(os.dup(fd), "wb") as handle:
                torch.save(original, handle)
            inode = os.fstat(fd).st_ino
            backing_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            ctx = mp.get_context("spawn")
            for _ in range(2):
                connection, child_connection = ctx.Pipe()
                child = ctx.Process(target=_mapped_child, args=(child_connection, path, inode))
                child.start()
                child_connection.close()
                children.append(child)
                connections.append(connection)
            loaded = [qualification._receive(c, "loaded", deadline) for c in connections]
            for result in loaded:
                self.assertEqual(result["manifest"], expected)
                self.assertTrue(result["alias_preserved"])
                self.assertGreater(result["mapping"]["mappings"], 0)
            connections[0].send("mutate")
            mutated = qualification._receive(connections[0], "mutate", deadline)
            connections[1].send("snapshot")
            unaffected = qualification._receive(connections[1], "snapshot", deadline)
            self.assertNotEqual(mutated["manifest"]["fp32"], expected["fp32"])
            self.assertEqual(unaffected["manifest"], expected)
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), backing_sha)
            self.assertGreater(mutated["mapping"]["Private_Dirty_kib"],
                               loaded[0]["mapping"]["Private_Dirty_kib"])
            os.close(fd)
            fd = -1
            self.assertFalse(Path(path).exists())
            for connection in connections:
                connection.send("owner_closed")
            closed = [qualification._receive(c, "owner_closed", deadline) for c in connections]
            self.assertEqual(closed[0]["manifest"], mutated["manifest"])
            self.assertEqual(closed[1]["manifest"], expected)
            for connection in connections:
                connection.send("release")
            for connection in connections:
                result = qualification._receive(connection, "release", deadline)
                self.assertEqual(result["mapping"]["mappings"], 0)
                self.assertFalse(result["cuda_initialized"])
            for child in children:
                child.join(max(0, deadline - time.monotonic()))
                self.assertEqual(child.exitcode, 0)
                self.assertFalse(Path(f"/proc/{child.pid}").exists())
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            if fd >= 0:
                os.close(fd)
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(5)
            for connection in connections:
                connection.close()


if __name__ == "__main__":
    unittest.main()
