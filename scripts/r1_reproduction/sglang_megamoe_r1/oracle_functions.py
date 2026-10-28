"""Load the pinned ORIGINAL numerical functions without importing their stacks.

This executes the unchanged AST functions from the actual old C1 source snapshot
and official FlashInfer source, rather than reimplementing their quantizers.
Only dependency-free numeric helpers are selected. No GPU work at import time.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

D = Path(__file__).resolve().parent


def checked_source(relative: str) -> Path:
    pins = json.loads((D / "REFERENCE_PINS.json").read_text())
    path = D / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != pins[relative]:
        raise RuntimeError("Pinned numerical source changed: " + relative)
    return path


def selected_functions(path: Path, names, scope):
    tree = ast.parse(path.read_text())
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in chosen} != set(names):
        raise RuntimeError("Original numerical source lacks expected functions")
    code = ast.Module(body=chosen, type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope


def c1_functions(torch):
    path = checked_source("reference/c1_fp8_utils.actual.py")
    return selected_functions(path, ("block_quant_dequant", "quant_weight_ue8m0",
        "per_block_cast_to_fp8", "ceil_to_ue8m0"), {
        "torch": torch, "List": List, "Tuple": Tuple,
        "ceil_div": lambda n, d: (n + d - 1) // d,
        "ceil_align": lambda n, d: ((n + d - 1) // d) * d,
    })


def fi_functions(torch):
    funcs = selected_functions(checked_source("reference/fi_host_utils.upstream.py"),
        ("mxfp8_quantize_per_block_32",), {
        "torch": torch, "Tuple": Tuple, "Mxfp8BlockSize": 32,
        "Fp8E4M3RcpLimit": 1.0 / 448, "Fp8E5M2RcpLimit": 1.0 / 57344,
        "_Mxfp8ScaleDtype": torch.float8_e8m0fnu,
    })
    return selected_functions(checked_source("reference/fi_runner_common.upstream.py"),
        ("_check_triton_flat_index", "_get_swiglu_pair_hw_match_triton_kernel",
         "_swiglu_pair_hw_match_cuda"), dict(funcs, functools=functools,
        _TRITON_FLAT_INDEX_LIMIT=2 ** 31))
