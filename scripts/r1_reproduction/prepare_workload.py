#!/usr/bin/env python3
"""Prepare audited, offline LongBench-derived R1 performance workloads.

This is a campaign convention, NOT the paper's original workload artifact.
Official LongBench v1 uses ``input``; the repository's dataset card describes a
``question`` variant. Select the schema explicitly. Supply local JSONL/parquet
and a local dataset2prompt.json (task -> template with {context}/{input}). Nothing
is downloaded. Only local tokenizer wrappers are loaded; model engines are not.

Example (requires transformers, tokenizers, torch; pyarrow only for parquet):
  python prepare_workload.py --data /data/LongBench --tasks qasper,hotpotqa \
    --input-field input --templates /data/dataset2prompt.json \
    --dataset-revision <source-commit> --tokenizer /models/DeepSeek-R1 \
    --tokenizer-revision <checkpoint-commit> --prompt-format chat \
    --sampling with-replacement --output /data/r1-workload

Each profile produces a /v1/completions batch JSONL, an aligned provenance JSONL,
and a manifest. Send the SAME request body and order to both engines. A SGLang
client must unpack these batch envelopes without reformatting their prompts.
Do not submit these already rendered strings to /v1/chat/completions.

Inputs have EXACT lengths after add_special_tokens=True. Only context is
middle-truncated; short or non-exactly-fitting examples are excluded, not padded.
This changes the original task distribution and can remove answer evidence.
Reference answers are preserved for paired quality checks, not an unmodified
LongBench leaderboard score. Output lengths are CAPS unless --ignore-eos is
explicitly chosen and both engines have passed equal length-enforcement checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import string
import sys
import types
from typing import Any


PROFILES = ((8192, 2048), (2048, 8192))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(canonical(ids).encode()).hexdigest()


def load_service_tokenizer(tokenizer_json: Path):
    """Instantiate the real DeepSeek service class without importing engines.

    Load the actual BaseTokenizer/FastTokenizer/DeepSeekV3Tokenizer source. Only
    the registration decorator is inert: normal registry startup imports all
    model families and native kernels, unnecessary for an offline tokenizer audit.
    Restore sys.modules afterwards so this helper cannot alter another runtime.
    """
    root = Path(__file__).resolve().parents[2]
    if tokenizer_json.name != "tokenizer.json" or not (tokenizer_json.parent / "tokenizer_config.json").is_file():
        raise ValueError("Service tokenizer needs adjacent tokenizer.json and tokenizer_config.json")
    sources = {
        "batchgen.config.base_tokenizer": root / "batchgen/config/base_tokenizer.py",
        "batchgen.config.fast_tokenizer": root / "batchgen/config/fast_tokenizer.py",
        "_r1_audit_deepseek_tokenizer": root / "batchgen/models/deepseek/deepseekv3/tokenizer.py",
    }
    names = ["batchgen", "batchgen.config", "batchgen.config.tokenizer_registry", *sources]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name, directory in (("batchgen", root / "batchgen"), ("batchgen.config", root / "batchgen/config")):
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
        registry = types.ModuleType("batchgen.config.tokenizer_registry")

        def register(kind):
            if kind != "deepseek_v3":
                raise ValueError(f"Unexpected service tokenizer family: {kind}")
            return lambda cls: cls

        registry.register_tokenizer = register
        sys.modules[registry.__name__] = registry
        module = None
        for name, source in sources.items():
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        # The flag selects an actual deployed bundle directory. It never
        # substitutes a raw tokenizer for the model's production wrapper.
        module.TOKENIZER_DIR = tokenizer_json.resolve().parent
        service = module.DeepSeekV3Tokenizer()
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    provenance = {
        "class": "DeepSeekV3Tokenizer",
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in sources.values()},
        "assets": {name: digest(tokenizer_json.parent / name) for name in ("tokenizer.json", "tokenizer_config.json")},
        "add_bos_token": service.add_bos_token, "add_eos_token": service.add_eos_token,
        "comparison_paths": ["encode(add_special_tokens=True)", "worker-style unpadded batch encoding"],
    }
    return service, provenance


def tokenizer_file_pins(files: list[Path], manifest_path: Path):
    """Record independently versioned assets, rather than calling all weights' revision theirs."""
    entries = {item["Path"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8"))}
    pins = {}
    for path in files:
        if path.name not in entries:
            pins[path.name] = {"sha256": digest(path), "revision": None}
            continue
        entry = entries[path.name]
        if digest(path) != entry["Sha256"] or path.stat().st_size != entry["Size"]:
            raise ValueError(f"Tokenizer asset does not match pinned file manifest: {path.name}")
        pins[path.name] = {"sha256": entry["Sha256"], "revision": entry["Revision"]}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if name not in pins or pins[name]["revision"] is None:
            raise ValueError(f"Pinned manifest lacks required tokenizer asset: {name}")
    return pins


def iter_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for row_number, line in enumerate(source, 1):
                if line.strip():
                    yield row_number, json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                offset += 1
                yield offset, row


def validate_template(template: str) -> None:
    fields = set()
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if field not in {"context", "input"} or spec or conversion:
            raise ValueError("Templates may only interpolate {context} and {input}")
        fields.add(field)
    if "context" not in fields:
        raise ValueError("Each template must include {context}")


def render_prompt(tokenizer, template: str, context: str, question: str, mode: str) -> str:
    content = template.format(context=context, input=question)
    if mode == "raw":
        return content
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # R1's template emits BOS, while completion tokenization also adds BOS.
    # Remove it ONLY if re-encoding proves that the intended token IDs are kept.
    desired = tokenizer.encode(rendered, add_special_tokens=False)
    candidates = [rendered]
    bos = tokenizer.bos_token
    if bos and rendered.startswith(bos):
        candidates.append(rendered[len(bos):])
    for candidate in candidates:
        if tokenizer.encode(candidate, add_special_tokens=True) == desired:
            return candidate
    raise ValueError("Chat template cannot be represented with completion add_special_tokens=True")


def fit_context(tokenizer, template: str, row: dict, target: int, mode: str):
    """Return exact prompt, token IDs and context truncation facts, or None.

    Token counts at text boundaries need not be monotonic. We first estimate a
    cut and then search a bounded local neighborhood, always re-encoding the
    complete rendered text. We never assume slicing token IDs preserves length.
    """
    context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
    original = render_prompt(tokenizer, template, row["context"], row["question"], mode)
    original_ids = tokenizer.encode(original, add_special_tokens=True)
    if len(original_ids) < target:
        return None, "short_input"
    if len(original_ids) == target:
        return (original, original_ids, len(context_ids), len(original_ids)), None
    tried = set()

    def attempt(keep: int):
        keep = max(0, min(keep, len(context_ids)))
        if keep in tried:
            return None, None
        tried.add(keep)
        left = (keep + 1) // 2
        right = keep // 2
        kept = context_ids[:left] + (context_ids[-right:] if right else [])
        context = tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Cutting a byte-level token may produce U+FFFD; do not manufacture text.
        if "\ufffd" in context and "\ufffd" not in row["context"]:
            return None, None
        prompt = render_prompt(tokenizer, template, context, row["question"], mode)
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = (prompt, ids, keep, len(original_ids)) if len(ids) == target else None
        return result, len(ids)

    estimate = max(0, len(context_ids) - (len(original_ids) - target))
    for _ in range(12):
        result, length = attempt(estimate)
        if result is not None:
            return result, None
        if length is None:
            break
        estimate += target - length
    # Do not silently approximate if decoding/re-tokenization skips the target.
    for distance in range(129):
        for keep in (estimate - distance, estimate + distance):
            result, _ = attempt(keep)
            if result is not None:
                return result, None
    return None, "no_exact_context_cut"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Local JSONL/parquet file or directory")
    parser.add_argument("--tasks", required=True, help="Comma-separated explicit dataset names; do not silently mix LongBench and LongBench-E")
    parser.add_argument("--input-field", required=True, choices=("input", "question"))
    parser.add_argument("--templates", required=True, type=Path, help="Local task-to-prompt-template JSON; use {input} even for question schema")
    parser.add_argument("--dataset-revision", required=True, help="Data source commit/version; also file-hashed below")
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local pinned R1 tokenizer directory")
    parser.add_argument("--tokenizer-revision", required=True, help="Operator's snapshot label; verified per-file revisions below are authoritative")
    parser.add_argument("--tokenizer-file-manifest", type=Path,
                        default=Path(__file__).resolve().with_name("r1_model_manifest.json"),
                        help="Pinned file manifest; tokenizer.json/config may have different revisions")
    parser.add_argument("--batchgen-tokenizer-json", type=Path,
                        default=Path(__file__).resolve().parents[2] / "batchgen/models/deepseek/deepseekv3/tokenizer.json",
                        help="Actual BatchGen bundled tokenizer, which is loaded independently of model weights")
    parser.add_argument("--prompt-format", required=True, choices=("raw", "chat"), help="Explicit raw task template or locally rendered user chat template")
    parser.add_argument("--sampling", required=True, choices=("without-replacement", "with-replacement"))
    parser.add_argument("--count", type=int, default=6000, help="Per-profile count; 6000 is a configurable convention, not verification that paper 6K means 6000")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1", help="Use identical served model alias on both engines")
    parser.add_argument("--ignore-eos", action="store_true", help="Opt-in fixed-output candidate; not qualified until both engines actually enforce the same length")
    parser.add_argument("--output", required=True, type=Path, help="New directory; refuses overwrite")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.output.exists():
        parser.error("--output already exists; use a new directory")
    from transformers import AutoTokenizer, __version__ as transformers_version
    from tokenizers import __version__ as tokenizers_version

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    service_tokenizer, service_provenance = load_service_tokenizer(args.batchgen_tokenizer_json)
    tokenizer_files = sorted(path for path in args.tokenizer.iterdir() if path.is_file() and (
        path.name.startswith(("tokenizer", "special_tokens", "added_tokens", "chat_template"))
        or path.suffix in {".model", ".jinja"}
    ))
    if not tokenizer_files:
        raise ValueError("No tokenizer assets found")
    file_pins = tokenizer_file_pins(tokenizer_files, args.tokenizer_file_manifest)
    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    tasks = sorted(set(part.strip() for part in args.tasks.split(",") if part.strip()))
    if not tasks:
        raise ValueError("At least one task is required")
    for task in tasks:
        validate_template(templates[task.removesuffix("_e")])
    files = [args.data] if args.data.is_file() else sorted(
        path for path in args.data.rglob("*") if path.suffix in {".jsonl", ".parquet"}
    )
    if not files:
        raise ValueError("No local JSONL/parquet data found")
    eligible, excluded, seen, source_files, selected_tasks = [], Counter(), {}, [], Counter()
    for path in files:
        source_files.append({"path": str(path.resolve()), "sha256": digest(path)})
        for row_number, raw in iter_rows(path):
            task = raw.get("dataset")
            # E subsets often retain the unsuffixed dataset field. Preserve the
            # explicit subset identity from the official file/directory name.
            for inferred in (path.stem, path.parent.name):
                if inferred.endswith("_e") and inferred.removesuffix("_e") == task:
                    task = inferred
            if task not in tasks:
                continue
            for field in ("context", args.input_field, "_id", "answers"):
                if field not in raw:
                    raise ValueError(f"{path}:{row_number}: missing {field}; select the correct schema")
            if not all(isinstance(raw[field], str) for field in ("context", args.input_field, "_id")):
                raise ValueError(f"{path}:{row_number}: context/input/_id must be strings")
            if not isinstance(raw["answers"], list) or not all(isinstance(item, str) for item in raw["answers"]):
                raise ValueError(f"{path}:{row_number}: answers must be a list of strings")
            key = (task, raw["_id"])
            row_hash = hashlib.sha256(canonical(raw).encode()).hexdigest()
            if key in seen:
                if seen[key] != row_hash:
                    raise ValueError(f"Conflicting duplicate sample {key}")
                excluded["duplicate_source_row"] += 1
                continue
            seen[key] = row_hash
            selected_tasks[task] += 1
            row = {"context": raw["context"], "question": raw[args.input_field]}
            fitted = []
            for input_length, _ in PROFILES:
                result, reason = fit_context(tokenizer, templates[task.removesuffix("_e")], row, input_length, args.prompt_format)
                if result is None:
                    excluded[f"{input_length}:{reason}"] += 1
                    break
                prompt, ids, _, _ = result
                single_ids = service_tokenizer.encode(prompt, add_special_tokens=True)
                worker_ids = service_tokenizer([prompt], return_tensors=None, padding=False,
                                               truncation=False, return_attention_mask=False)["input_ids"][0]
                if single_ids != ids or worker_ids != ids:
                    raise ValueError(f"Actual BatchGen service/HF tokenizer disagreement for {key}; do not benchmark")
                fitted.append(result)
            if len(fitted) != len(PROFILES):
                continue
            eligible.append({
                "dataset": task, "original_id": raw["_id"], "source_file": str(path.resolve()),
                "source_row": row_number, "source_row_sha256": row_hash,
                "answers": raw["answers"], "all_classes": raw.get("all_classes"),
                "language": raw.get("language"), "source_length": raw.get("length"),
                "original_context_tokens": len(tokenizer.encode(raw["context"], add_special_tokens=False)),
                "profiles": fitted,
            })
    missing = set(tasks) - set(selected_tasks)
    if missing:
        raise ValueError(f"No source rows found for requested tasks: {sorted(missing)}")
    if not eligible:
        raise ValueError(f"No samples fit both profiles: {dict(excluded)}")
    rng = random.Random(args.seed)
    if args.sampling == "without-replacement":
        if args.count > len(eligible):
            raise ValueError(f"Requested {args.count} unique samples, only {len(eligible)} eligible; lower count or explicitly choose with-replacement")
        selection = rng.sample(range(len(eligible)), args.count)
    else:
        selection = [rng.randrange(len(eligible)) for _ in range(args.count)]
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for profile_index, (input_length, output_length) in enumerate(PROFILES):
        prefix = f"input{input_length}-output{output_length}"
        requests_path = args.output / f"{prefix}.requests.jsonl"
        provenance_path = args.output / f"{prefix}.provenance.jsonl"
        with requests_path.open("w", encoding="utf-8") as requests, provenance_path.open("w", encoding="utf-8") as provenance:
            for ordinal, index in enumerate(selection):
                sample = eligible[index]
                prompt, ids, context_kept, original_length = sample["profiles"][profile_index]
                custom_id = f"r1-{ordinal:06d}"
                body = {"model": args.model, "prompt": prompt, "max_tokens": output_length,
                        "temperature": 0.0, "top_p": 1.0, "stream": False, "n": 1,
                        "ignore_eos": args.ignore_eos}
                requests.write(canonical({"custom_id": custom_id, "method": "POST", "url": "/v1/completions", "body": body}) + "\n")
                metadata = {key: value for key, value in sample.items() if key != "profiles"}
                metadata.update({"custom_id": custom_id, "input_tokens": len(ids), "input_token_ids": ids,
                                 "input_token_ids_sha256": ids_digest(ids), "max_output_tokens": output_length,
                                 "original_input_tokens": original_length, "context_tokens_kept": context_kept,
                                 "context_modified": original_length != len(ids)})
                provenance.write(canonical(metadata) + "\n")
        artifacts.append({"input_tokens": input_length, "max_output_tokens": output_length,
                          "requests": requests_path.name, "requests_sha256": digest(requests_path),
                          "provenance": provenance_path.name, "provenance_sha256": digest(provenance_path)})
    manifest = {
        "schema_version": 1, "generator_sha256": digest(Path(__file__)),
        "dataset_revision": args.dataset_revision, "source_files": source_files,
        "tasks": tasks, "schema_input_field": args.input_field,
        "templates_sha256": digest(args.templates), "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_files": {path.name: digest(path) for path in tokenizer_files},
        "tokenizer_file_pins": file_pins,
        "tokenizer_file_manifest_sha256": digest(args.tokenizer_file_manifest),
        "tokenizer_revision_semantics": "Operator snapshot label only; verified per-file revisions are authoritative",
        "batchgen_service_tokenizer": service_provenance,
        "batchgen_tokenizer_json": {"path": str(args.batchgen_tokenizer_json.resolve()),
                                     "sha256": digest(args.batchgen_tokenizer_json)},
        "libraries": {"transformers": transformers_version, "tokenizers": tokenizers_version},
        "prompt_format": args.prompt_format, "add_special_tokens": True,
        "chat_template_handling": "Rendered locally once; BOS deduplicated only with exact token-ID equality proof; completion endpoint only",
        "seed": args.seed, "count_per_profile": args.count, "sampling": args.sampling,
        "unique_selected_samples": len(set(selection)), "eligible_samples": len(eligible),
        "source_task_counts": dict(selected_tasks), "exclusions": dict(excluded),
        "selected_task_counts": dict(Counter(eligible[index]["dataset"] for index in selection)),
        "same_sample_order_across_profiles": True, "ignore_eos": args.ignore_eos,
        "output_length_qualified": False, "server_tokenization_qualified": False,
        "artifacts": artifacts,
        "limitations": [
            "Derived campaign workload, not a recovered paper artifact; 6000 and K=1024 are explicit conventions.",
            "Eligibility requires exact 8192 and 2048 inputs on the same source sample; shorter sources are excluded.",
            "Context is middle-truncated; answer evidence may be removed. Retained references support paired quality checks only.",
            "Source dataset length is words/characters, not model token count; measured tokens are recorded separately.",
            "Use the same pinned tokenizer files and completion bodies on both servers; assert actual usage.prompt_tokens equals input_tokens before timing.",
            "Actual service encode and worker-style batch encoding match HF offline; runtime overrides or prompt rewriting still require runtime verification.",
            "Output lengths are maxima by default; fixed-length claims require both engines to actually produce the requested length.",
            "Disable prefix reuse on both engines, especially if samples are repeated.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "count_per_profile": args.count,
                      "eligible_samples": len(eligible), "unique_selected_samples": len(set(selection)),
                      "manifest_sha256": digest(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
