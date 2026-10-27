"""Execute the real small Python boundaries with CPU stand-ins, no model load."""
import ast
import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    parent = tree if class_name is None else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = copy.deepcopy(next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name))
    function.decorator_list = []
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class MainPythonABITests(unittest.TestCase):
    def test_parameter_server_init_passes_text_path_to_real_cpp_boundary_shape(self):
        calls = []

        class Native:
            def __init__(self, *args):
                pass

            def Init(self, shm, meta, byte_size, directory, names):
                # pybind std::string accepts text, not pathlib.Path. Test the
                # actual Init method rather than a duplicate normalization helper.
                if type(directory) is not str:
                    raise TypeError("C++ std::string path required")
                calls.append((byte_size, directory, names))

        namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (1024, 2048))),
                     "logging": logging, "Parameter_Server": Native, "os": os,
                     "get_model_byte_size": lambda _: 671}
        initialize = method("batchgen/models/deepseek/deepseek_parameter_server.py", "DeepSeek_Parameter_Server", "Init", namespace)
        for directory in (Path("/converted/带空格 weights"), "/converted/text", b"/converted/bytes"):
            owner = SimpleNamespace(converted_ckpt_dir=directory, _parse_state_dict=lambda: None,
                enable_hugetlbfs=False, enable_thp=False, huggingface_ckpt_name="deepseek-ai/DeepSeek-R1",
                model_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]), state_dict_name_map={"x": "x"})
            initialize(owner)
            self.assertEqual(calls[-1], (671, os.fsdecode(directory), owner.state_dict_name_map))
            self.assertEqual(owner.converted_ckpt_dir, directory)

    def test_r1_hybrid_gate_preserves_topk8_group_and_scaling_through_actual_mgn_wrapper(self):
        calls = []

        class Tensor:
            shape = (2, 1, 7168)

            def view(self, *shape):
                return self

            def float(self):
                return self

            def size(self, dimension):
                return 256

        native = SimpleNamespace(moe_fused_gate=lambda *args: (calls.append(args) or ("indices", "weights")))
        wrapper = method("batchgen_kernels/common/mgn.py", None, "moe_fused_gate", {"_get_mgn": lambda: native})
        score = Tensor()
        gate = method("batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py", "MoEGate", "moe_gate_forward_hybrid", {
            "torch": SimpleNamespace(sigmoid=lambda value: value),
            "F": SimpleNamespace(linear=lambda *args: score), "moe_fused_gate": wrapper})
        bias = object()
        owner = SimpleNamespace(weight=Tensor(), e_score_correction_bias=bias, n_group=8,
                                topk_group=4, n_routed_experts=256, top_k=8, routed_scaling_factor=2.5)
        self.assertEqual(gate(owner, Tensor()), ("indices", "weights"))
        self.assertEqual(calls, [(score, bias, 8, 4, 256, 8, 2.5)])


if __name__ == "__main__":
    unittest.main()
