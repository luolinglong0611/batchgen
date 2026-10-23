"""Exercise the production Init method without native imports or allocations."""
import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = (Path(__file__).resolve().parents[1]
          / "batchgen/models/deepseek/deepseek_parameter_server.py")


class CheckpointPath:
    def __fspath__(self):
        return "/checkpoints/converted model"


def init_method(native, converter):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepSeek_Parameter_Server")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "Init")
    namespace = {
        "os": os, "logging": logging,
        "torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (1024, 2048))),
        "Parameter_Server": native,
        "ckpt_converter": converter,
        "get_model_byte_size": lambda name: 675 * 1024**3,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["Init"]


class PathBridgeTests(unittest.TestCase):
    def run_bridge(self, path, *, convert=False, architecture="DeepseekV3ForCausalLM"):
        calls = []

        def native_init(shm, meta, byte_size, directory, names):
            # Model the binding's std::string contract, preserving all other
            # arguments for assertions. No native object or memory is created.
            self.assertIs(type(directory), str)
            calls.append((shm, meta, byte_size, directory, names))

        native_instance = SimpleNamespace(Init=Mock(side_effect=native_init))
        native = Mock(return_value=native_instance)
        converter_instance = SimpleNamespace(convert_model_directory=Mock(return_value=path))
        converter = Mock(return_value=converter_instance)
        names = {"model.layers.0.self_attn.q_proj.weight": {"module_key": "attn_0"}}
        server = SimpleNamespace(
            _parse_state_dict=Mock(), enable_hugetlbfs=False, enable_thp=True,
            model_config=SimpleNamespace(architectures=[architecture]),
            huggingface_ckpt_name="deepseek-ai/DeepSeek-R1",
            cache_dir=Path("/checkpoints/source"),
            converted_ckpt_dir=None if convert else path,
            state_dict_name_map=names,
        )
        result = init_method(native, converter)(server)
        server._parse_state_dict.assert_called_once_with()
        native.assert_called_once_with(False, True)
        self.assertEqual(len(calls), 1)
        shm, meta, byte_size, directory, actual_names = calls[0]
        self.assertEqual(result, (shm, meta))
        self.assertTrue(shm.startswith("/shm_"))
        self.assertTrue(meta.startswith("/shm_"))
        self.assertNotEqual(shm, meta)
        self.assertEqual(byte_size, 675 * 1024**3)
        self.assertEqual(directory, "/checkpoints/converted model")
        self.assertIs(actual_names, names)
        self.assertIs(server.converted_ckpt_dir, path)
        if convert:
            converter.assert_called_once_with()
            converter_instance.convert_model_directory.assert_called_once_with(server.cache_dir)
        else:
            converter.assert_not_called()

    def test_preconverted_directory_accepts_string_and_pathlike_without_conversion(self):
        for path in ("/checkpoints/converted model", Path("/checkpoints/converted model"), CheckpointPath()):
            with self.subTest(path_type=type(path).__name__):
                self.run_bridge(path)

    def test_converter_result_accepts_string_and_pathlike_including_cached_return(self):
        # The converter's cached-valid and fresh-conversion paths both return
        # output_dir. Exercise each permitted return type at this common bridge.
        for path in ("/checkpoints/converted model", Path("/checkpoints/converted model"), CheckpointPath()):
            with self.subTest(path_type=type(path).__name__):
                self.run_bridge(path, convert=True)

    def test_v4_shared_bridge_keeps_same_path_contract(self):
        self.run_bridge(Path("/checkpoints/converted model"), architecture="DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
