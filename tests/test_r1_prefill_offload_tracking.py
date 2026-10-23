"""Exercise real R1 enqueue + shared retirement methods without a GPU runtime.

AST extraction avoids importing native/CUDA dependencies; only tensors and
KVAsyncTask are stand-ins. This checks Python ownership/exception propagation,
not native DMA correctness or attention numerics.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "batchgen/models/deepseek/deepseekv3/wrappers.py"
BASE = ROOT / "batchgen/models/wrappers/attention.py"


def _methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [next(node for node in cls.body
                 if isinstance(node, ast.FunctionDef) and node.name == name)
            for name in names]


def _load_real_methods():
    methods = _methods(BASE, "AttnWrapperBase", [
        "retire_pending_prefill_offloads",
        "retire_pending_prefill_offloads_before_layer",
        "pin_prefill_offload_tensor", "track_prefill_offload_task",
    ])
    cls = ast.ClassDef(name="AttnWrapperBase", bases=[], keywords=[],
                       body=methods, decorator_list=[])
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        cls,
        *_methods(WRAPPERS, "DeepSeekAttnWrapper", [
            "_forward_prefill", "_offload_prepacked_kv",
        ]),
    ], type_ignores=[])
    namespace = {"logging": logging}
    exec(compile(ast.fix_missing_locations(tree), str(WRAPPERS), "exec"), namespace)
    base = namespace["AttnWrapperBase"]
    base.pending_prefill_offload_tasks = []
    base.pending_prefill_offload_tensors = []
    base.pending_prefill_offload_layer_idx = None
    return base, namespace["_forward_prefill"], namespace["_offload_prepacked_kv"]


class _Tensor:
    device = "cuda:0"

    def __init__(self, rows, shape=None, owner=None):
        self.rows = rows
        self.shape = shape or (len(rows), 576)
        self.owner = owner if owner is not None else self

    def __getitem__(self, index):
        return _Tensor(self.rows[index], owner=self.owner)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return _Tensor(self.rows, tuple(shape), self.owner)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Task:
    def __init__(self, log, index):
        self.log = log
        self.index = index
        self.error = None
        self.waits = 0

    def wait(self):
        self.waits += 1
        self.log.append(("wait", self.index))
        if self.error is not None:
            raise self.error


class PrefillOffloadTrackingTest(unittest.TestCase):
    def setUp(self):
        self.base, self.forward, self.offload = _load_real_methods()
        self.log, self.calls, self.tasks = [], [], []

        def issue(**kwargs):
            index = len(self.tasks)
            self.log.append(("issue", index))
            self.calls.append(kwargs)
            task = _Task(self.log, index)
            self.tasks.append(task)
            return task

        self.wrapper = SimpleNamespace(
            layer_idx=7, prepack_cu_seqlens=[_Scalar(v) for v in [0, 2, 5, 6]],
            prepack_num_sequences=3, cur_batch=[30, 32, 34],
            core_engine=SimpleNamespace(host_paged_kv_worker_view=SimpleNamespace(
                async_offload_layer_kv_to_host=issue)),
        )
        self.tensor = _Tensor(list(range(6)))

    def _enqueue(self):
        self.offload(self.wrapper, self.tensor)

    def _next_forward(self, layer=8):
        def attention(*args):
            self.log.append(("attention", layer))
            return "output", "kv"
        wrapper = SimpleNamespace(
            layer_idx=layer, prepack_mode=False, weight_dequant_scale={},
            module=SimpleNamespace(prefill_attn_w8a16=attention),
        )
        return self.forward(wrapper, self.tensor)

    def test_every_sequence_view_and_task_retained_then_retired(self):
        self._enqueue()
        self.assertEqual(self.base.pending_prefill_offload_tasks, self.tasks)
        self.assertEqual(self.base.pending_prefill_offload_layer_idx, 7)
        views = self.base.pending_prefill_offload_tensors
        self.assertEqual([v.rows for v in views], [[0, 1], [2, 3, 4], [5]])
        self.assertEqual([v.shape for v in views], [(1, 2, 1, 576), (1, 3, 1, 576), (1, 1, 1, 576)])
        self.assertTrue(all(v.owner is self.tensor for v in views))
        for call, view, seq_id, length in zip(self.calls, views, [30, 32, 34], [2, 3, 1]):
            self.assertEqual(call, dict(layer_idx=7, sequence_ids=[seq_id],
                                       k_tensor=view, v_tensor=None, sequence_lengths=[length]))
        self.assertEqual([task.waits for task in self.tasks], [0, 0, 0])
        self.assertEqual(self.base.retire_pending_prefill_offloads(reason="end prefill"), 3)
        self.assertEqual([task.waits for task in self.tasks], [1, 1, 1])
        self.assertEqual(self.base.pending_prefill_offload_tasks, [])
        self.assertEqual(self.base.pending_prefill_offload_tensors, [])
        self.assertIsNone(self.base.pending_prefill_offload_layer_idx)
        self.assertEqual(self.base.retire_pending_prefill_offloads(), 0)

    def test_next_layer_waits_before_attention_computation(self):
        self._enqueue()
        self.assertEqual(self._next_forward(), ("output", None, "kv"))
        self.assertEqual(self.log, [("issue", 0), ("issue", 1), ("issue", 2),
                                    ("wait", 0), ("wait", 1), ("wait", 2), ("attention", 8)])
        self.assertEqual(self.base.pending_prefill_offload_tensors, [])

    def test_same_layer_keeps_tasks_for_end_prefill_retirement(self):
        self._enqueue()
        self._next_forward(layer=7)
        self.assertEqual([task.waits for task in self.tasks], [0, 0, 0])
        self.assertEqual(self.base.retire_pending_prefill_offloads(), 3)

    def test_end_prefill_surfaces_copy_error_and_retains_ownership(self):
        self._enqueue()
        error = RuntimeError("injected asynchronous D2H failure")
        self.tasks[1].error = error
        with self.assertRaises(RuntimeError) as raised:
            self.base.retire_pending_prefill_offloads(reason="end prefill")
        self.assertIs(raised.exception, error)
        self.assertEqual(self.base.pending_prefill_offload_tasks, self.tasks)
        self.assertEqual(len(self.base.pending_prefill_offload_tensors), 3)
        self.assertEqual(self.base.pending_prefill_offload_layer_idx, 7)
        self.assertEqual([task.waits for task in self.tasks], [1, 1, 0])

    def test_failed_previous_layer_prevents_next_attention(self):
        self._enqueue()
        self.tasks[0].error = RuntimeError("injected asynchronous D2H failure")
        with self.assertRaisesRegex(RuntimeError, "asynchronous D2H failure"):
            self._next_forward()
        self.assertNotIn(("attention", 8), self.log)
        self.assertEqual(len(self.base.pending_prefill_offload_tensors), 3)


if __name__ == "__main__":
    unittest.main()
