"""Opt-in CPU storage mapping for the original R1 worker skeleton.

This candidate changes only how serialized CPU storages are loaded. It does
not remove GPU transfers, make tensors read-only, or establish a full-model
memory saving. MAP_PRIVATE permits process-local writes through COW; such
writes can consume private RAM and must be measured before increasing host KV.
The owner must retain its memfd until every worker has opened the path.
"""

import logging
import mmap
import os
import zipfile


def r1_skeleton_mmap_enabled(model_name):
    """Use the exact R1 selector accepted by the worker's model dispatch."""
    if os.environ.get("BATCHGEN_R1_SKELETON_MMAP") != "1":
        return False
    if not isinstance(model_name, str) or model_name.lower() != "deepseek-ai/deepseek-r1":
        raise ValueError(
            "BATCHGEN_R1_SKELETON_MMAP=1 supports only deepseek-ai/DeepSeek-R1"
        )
    return True


def load_worker_skeleton(path, model_name):
    """Retain ordinary torch.load unless the R1 mmap candidate is explicit.

    The enabled path requires a filename, current torch.save ZIP format, and
    Torch's default MAP_PRIVATE option. A load failure propagates: allocating
    an ordinary private copy as a fallback would hide a capacity failure.
    """
    enabled = r1_skeleton_mmap_enabled(model_name)
    if enabled:
        try:
            filename = os.fspath(path)
        except TypeError as exc:
            raise TypeError("R1 skeleton mmap requires a filename path") from exc
        if not isinstance(filename, str) or not filename:
            raise TypeError("R1 skeleton mmap requires a nonempty string filename path")
        # Opening here preserves a useful FileNotFoundError for an owner whose
        # /proc/<pid>/fd/<fd> path has disappeared. No descriptor is retained.
        with open(filename, "rb") as handle:
            if not zipfile.is_zipfile(handle):
                raise ValueError("R1 skeleton mmap requires torch.save ZIP serialization")

    import torch

    if not enabled:
        return torch.load(path)
    if torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
        raise RuntimeError("R1 skeleton mmap requires the default MAP_PRIVATE option")
    logging.warning(
        "Experimental R1 skeleton MAP_PRIVATE load; CPU writes use COW. "
        "Full-model memory use and GPU transfer correctness remain unqualified."
    )
    return torch.load(filename, map_location="cpu", weights_only=True, mmap=True)
