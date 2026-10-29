"""Collect strided admission tokens without pickling each prompt's storage.

Only small shape/error metadata uses an object collective. Token bytes travel
as one padded int64 tensor per rank. Imports of Torch are confined to the call,
so selecting the legacy transport does not initialize a device or dependency.
"""
from __future__ import annotations


DEFAULT_MAX_DEVICE_BUFFER_BYTES = 2 * 1024**3
_SCHEMA = "batchgen-tokenized-int64-gather-v1"


def gather_tokenized_tensors(
    items,
    *,
    device,
    active_ranks,
    num_sequences,
    local_error=None,
    max_device_buffer_bytes=DEFAULT_MAX_DEVICE_BUFFER_BYTES,
    group=None,
):
    """Return the same rank-ordered ``list[list[dict]]`` as all_gather_object.

    Every process-group rank calls once per tokenizer wave, including inactive
    ranks. ``active_ranks`` describes ownership in that wave; index i still
    belongs exclusively to rank i % world_size. Shape/ownership and caller
    errors are exchanged before a token-data collective, so a malformed local
    partition produces the same exception on all ranks instead of stranding
    peers in all_gather.

    The live token-data device buffers total (world_size + 1) * padded_tokens
    * 8 bytes, matching the order of the old padded object collective. They
    are local variables, never cached. Exactly one bulk token-data .cpu()
    follows all_gather. Returned 1-D CPU views retain that one gathered CPU
    allocation until the caller consumes/drops the final view; no CUDA storage
    is retained by the result. Empty/short ranks contribute only zero padding.
    """
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    rows, tensors, error, active = [], [], None, ()
    try:
        if local_error is not None:
            raise ValueError("tokenizer: " + str(local_error)[:1000])
        if type(num_sequences) is not int or num_sequences < 0:
            raise ValueError("num_sequences must be a nonnegative integer")
        if type(max_device_buffer_bytes) is not int or max_device_buffer_bytes <= 0:
            raise ValueError("positive max_device_buffer_bytes required")
        active = tuple(active_ranks)
        if (len(set(active)) != len(active) or any(type(r) is not int or not 0 <= r < world_size for r in active)):
            raise ValueError("active_ranks must contain unique process-group ranks")
        if type(items) is not list:
            raise ValueError("tokenized partition must be a list")
        target = torch.device(device)
        backend = str(dist.get_backend(group)).lower()
        if ("nccl" in backend and target.type != "cuda") or ("gloo" in backend and target.type != "cpu"):
            raise ValueError("token transport device must match NCCL CUDA or Gloo CPU")
        if "nccl" in backend and target.index is not None and target.index != torch.cuda.current_device():
            raise ValueError("token transport must use the rank's current CUDA device")
        for item in items:
            if type(item) is not dict:
                raise ValueError("tokenized item must be a dict")
            index, length, tokens = item.get("idx"), item.get("length"), item.get("input_ids")
            if type(index) is not int or type(length) is not int or length < 0:
                raise ValueError("integer idx and nonnegative integer length required")
            if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.device.type != "cpu":
                raise ValueError("input_ids must be a CPU int64 tensor")
            if tokens.ndim != 1 or not tokens.is_contiguous() or tokens.numel() != length:
                raise ValueError("input_ids must be contiguous 1-D and match length")
            rows.append((index, length)); tensors.append(tokens)
        expected = list(range(rank, num_sequences, world_size)) if rank in active else []
        if [index for index, _ in rows] != expected:
            raise ValueError("strided ownership/order differs from the active tokenizer wave")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:1000]
        rows, tensors, active = [], [], ()

    # Never pickle token tensors. This message is O(number of prompts), with
    # two small integers per prompt and a bounded error string on failure.
    metadata = [None] * world_size
    dist.all_gather_object(metadata, {
        "schema": _SCHEMA, "num_sequences": num_sequences if type(num_sequences) is int else None,
        "active_ranks": active, "max_device_buffer_bytes": max_device_buffer_bytes if type(max_device_buffer_bytes) is int else None,
        "rows": rows, "error": error,
    }, group=group)
    errors = [(source, part.get("error")) for source, part in enumerate(metadata) if part.get("error")]
    if errors:
        raise ValueError("Compact tokenizer validation failed before token transfer: " + repr(errors))
    reference = {key: metadata[0][key] for key in ("schema", "num_sequences", "active_ranks", "max_device_buffer_bytes")}
    for part in metadata:
        if any(part.get(key) != value for key, value in reference.items()):
            raise ValueError("Compact tokenizer wave metadata disagrees across ranks")
    for source, part in enumerate(metadata):
        expected = list(range(source, reference["num_sequences"], world_size)) if source in reference["active_ranks"] else []
        if [index for index, _ in part["rows"]] != expected:
            raise ValueError("Compact tokenizer gathered ownership differs at rank " + str(source))
    sizes = [sum(length for _, length in part["rows"]) for part in metadata]
    padded_tokens = max(sizes, default=0)
    buffer_bytes = (world_size + 1) * padded_tokens * 8
    if buffer_bytes > max_device_buffer_bytes:
        raise ValueError("Compact tokenizer padded device buffers exceed the agreed byte budget: "
                         + str(buffer_bytes) + " > " + str(max_device_buffer_bytes))
    if not padded_tokens:
        # Keep zero-length prompts distinct even when every rank has zero bytes.
        return [[{"idx": index, "length": length, "input_ids": torch.empty(0, dtype=torch.int64)}
                 for index, length in part["rows"]] for part in metadata]

    flat_cpu = send = gathered_device = None
    preparation_error = None
    try:
        flat_cpu = torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.int64)
        send = torch.zeros(padded_tokens, dtype=torch.int64, device=target)
        gathered_device = torch.empty((world_size, padded_tokens), dtype=torch.int64, device=target)
        if flat_cpu.numel():
            send[:flat_cpu.numel()].copy_(flat_cpu)
    except Exception as exc:
        preparation_error = type(exc).__name__ + ": " + str(exc)[:1000]
        flat_cpu = send = gathered_device = None
    # Allocation/copy can fail on only one device. A second tiny status exchange
    # keeps peers out of the token collective in that case; it carries no IDs.
    preparation_errors = [None] * world_size
    dist.all_gather_object(preparation_errors, preparation_error, group=group)
    if any(x is not None for x in preparation_errors):
        raise RuntimeError("Compact tokenizer preparation failed before token transfer: " + repr(preparation_errors))
    outputs = list(gathered_device.unbind(0))
    dist.all_gather(outputs, send, group=group)
    gathered_cpu = gathered_device.cpu()
    # Releasing all CUDA references here is intentional: only CPU views escape.
    del outputs, gathered_device, send, flat_cpu, tensors
    result = []
    for source, part in enumerate(metadata):
        offset, rank_items = 0, []
        for index, length in part["rows"]:
            rank_items.append({"idx": index, "length": length,
                "input_ids": gathered_cpu[source, offset:offset + length]})
            offset += length
        result.append(rank_items)
    return result
