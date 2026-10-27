#!/usr/bin/env python3
"""Build an offline R1 parameter comparison and a fail-closed measurement gate.

Input schema ``r1-alignment-observations-v1`` contains ``engines.BG`` and
``engines.SG``. Each engine has ``observed_at_utc``, a ``binding`` with
``source_sha256`` and ``input_manifest_sha256``, and ``parameters``. Parameter
records contain ``value``, ``evidence_level`` and nonempty ``evidence`` links.

``correctness[engine][qualification]`` records use the same evidence fields,
plus ``status``, ``scope``, ``observed_at_utc`` and ``binding``. Only an explicit
``scope: full_model`` can qualify the full-model formal-measurement gate. A
passed operator fixture retains its reported result without being promoted to
full-model qualification. By default a qualification
must bind to that engine's source and input. A separately frozen fixture can be
specified in ``qualification_expected_bindings[engine][qualification]``; its
source must still match the observed engine source. This lets semantic and
operator fixtures differ from the performance request list without accepting
an unbound qualification.

Historical, source-derived and planned values remain visible but never qualify
as current runtime evidence. No service is queried, started or benchmarked.
Evidence paths are references, not instructions, and this tool does not execute
or dereference them. A passed gate establishes the supplied evidence contract;
it does not independently attest that referenced external reports are genuine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


INPUT_SCHEMA = "r1-alignment-observations-v1"
OUTPUT_SCHEMA = "r1-alignment-audit-v1"
ENGINES = ("BG", "SG")
EVIDENCE_LEVELS = {"runtime", "historical_runtime", "source", "planned", "missing"}
REQUIRED_PARAMETERS = (
    "model_revision", "weights_tensor_hash", "tokenizer_hash",
    "input_manifest_hash", "weight_dtype", "kv_dtype", "actual_attention_tp",
    "attention_dp", "routed_ep", "expert_tp", "shared_expert_tp", "dense_tp",
    "lm_head_tp", "graph_scope", "attention_kernel", "moe_kernel",
    "context_cap", "active_cap", "prefill_budget", "host_memory_bytes",
    "nccl_version", "warmup_protocol", "timing_protocol",
    "gpu_uuids", "node_count", "gpu_count", "page_size", "gpu_memory_budget",
    "python_version", "torch_version", "triton_version", "cuda_runtime_version",
    "driver_version", "gemm_kernel", "quantization_policy", "generation_policy",
    "stop_policy", "prefix_reuse", "speculative_decoding",
)
QUALIFICATIONS = ("memory_safety", "nan_tail", "numerical", "semantic")
HASH_PARAMETERS = {"weights_tensor_hash", "tokenizer_hash", "input_manifest_hash"}
STRING_PARAMETERS = {
    "model_revision", "weight_dtype", "kv_dtype", "attention_kernel",
    "moe_kernel", "nccl_version",
    "python_version", "torch_version", "triton_version", "cuda_runtime_version",
    "driver_version", "gemm_kernel",
}
POSITIVE_INT_PARAMETERS = {
    "actual_attention_tp", "attention_dp", "routed_ep", "expert_tp",
    "shared_expert_tp", "dense_tp", "lm_head_tp", "context_cap", "active_cap",
    "host_memory_bytes",
    "node_count", "gpu_count", "page_size",
}
BOOLEAN_PARAMETERS = {"prefix_reuse", "speculative_decoding"}
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def _json_value(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def _strict_equal(left: Any, right: Any) -> bool:
    """Do not let Python's True == 1 or 1.0 == 1 establish alignment."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _utc(value: Any) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result.astimezone(timezone.utc)


def _timestamp_reasons(value: Any, as_of: datetime, max_age: int) -> list[str]:
    observed = _utc(value)
    if observed is None:
        return ["missing_or_invalid_timezone_aware_observation_time"]
    age = (as_of - observed).total_seconds()
    if age < 0:
        return ["observation_is_after_audit_time"]
    if age > max_age:
        return ["stale_runtime_observation"]
    return []


def _binding_reasons(binding: Any) -> list[str]:
    if type(binding) is not dict:
        return ["missing_source_and_input_binding"]
    return [
        "missing_or_invalid_" + field
        for field in ("source_sha256", "input_manifest_sha256")
        if type(binding.get(field)) is not str or not HASH_RE.fullmatch(binding[field])
    ]


def _evidence_reasons(record: dict) -> list[str]:
    reasons = []
    level = record.get("evidence_level", "missing")
    if type(level) is not str or level not in EVIDENCE_LEVELS:
        reasons.append("unknown_evidence_level")
    if level != "runtime":
        reasons.append("not_current_runtime_evidence")
    evidence = record.get("evidence")
    if type(evidence) is not list or not evidence or any(
        type(item) is not str or not item.strip() for item in evidence
    ):
        reasons.append("missing_evidence_references")
    return reasons


def _parameter_value_reasons(name: str, record: dict) -> list[str]:
    if "value" not in record or record["value"] is None:
        return ["missing_parameter_value"]
    value = record["value"]
    if name in HASH_PARAMETERS:
        valid = type(value) is str and bool(HASH_RE.fullmatch(value))
    elif name in POSITIVE_INT_PARAMETERS:
        valid = type(value) is int and value > 0
    elif name in BOOLEAN_PARAMETERS:
        valid = type(value) is bool
    elif name == "gpu_uuids":
        valid = type(value) is list and bool(value) and all(
            type(item) is str and bool(item.strip()) for item in value
        )
        if valid:
            valid = len(set(value)) == len(value)
    elif name == "gpu_memory_budget":
        valid = type(value) is dict and bool(value)
    elif name == "quantization_policy":
        valid = (type(value) is str and bool(value.strip())) or (
            type(value) is dict and bool(value)
        )
    elif name == "generation_policy":
        valid = type(value) is dict and bool(value)
        if valid:
            numeric = lambda number: type(number) in (int, float) and math.isfinite(number)
            valid = (
                numeric(value.get("temperature")) and value["temperature"] >= 0
                and numeric(value.get("top_p")) and 0 < value["top_p"] <= 1
                and type(value.get("top_k")) is int and value["top_k"] >= -1
                and type(value.get("seed")) is int and value["seed"] >= 0
            )
    elif name == "stop_policy":
        valid = type(value) is dict and type(value.get("ignore_eos")) is bool
        if valid:
            length = value.get("output_length")
            valid = type(length) is int and length > 0
            if type(length) is dict:
                valid = bool(length) and all(
                    type(key) is str and bool(key)
                    and type(item) is int and item > 0
                    for key, item in length.items()
                )
    elif name in {"graph_scope", "prefill_budget", "warmup_protocol", "timing_protocol"}:
        # Structured values must identify their meaning explicitly. Lists may
        # describe graph buckets, but an empty structure cannot prove a policy.
        valid = type(value) in (str, dict, list) and bool(value)
    elif name in STRING_PARAMETERS:
        valid = type(value) is str and bool(value.strip())
    else:
        # Additional contract fields may be booleans (prefix reuse), zero
        # (retry count), lists (GPU UUIDs), or structured execution policies.
        # They still require exact type-aware equality and runtime evidence.
        valid = _json_value(value)
    return [] if valid else ["invalid_parameter_value_type_or_domain"]


def _engine(data: dict, name: str, as_of: datetime, max_age: int) -> dict:
    raw = data.get("engines", {}).get(name, {})
    if type(raw) is not dict:
        raise ValueError(f"engines.{name} must be an object")
    parameters = raw.get("parameters", {})
    if type(parameters) is not dict:
        raise ValueError(f"engines.{name}.parameters must be an object")
    reasons = _timestamp_reasons(raw.get("observed_at_utc"), as_of, max_age)
    reasons.extend(_binding_reasons(raw.get("binding")))
    input_record = parameters.get("input_manifest_hash", {})
    binding = raw.get("binding", {})
    if type(input_record) is dict and type(binding) is dict:
        if input_record.get("value") != binding.get("input_manifest_sha256"):
            reasons.append("input_manifest_parameter_does_not_match_engine_binding")
    gpu_record = parameters.get("gpu_uuids", {})
    count_record = parameters.get("gpu_count", {})
    if type(gpu_record) is dict and type(count_record) is dict:
        uuids, count = gpu_record.get("value"), count_record.get("value")
        if type(uuids) is list and type(count) is int and len(uuids) != count:
            reasons.append("gpu_uuid_count_does_not_match_gpu_count")
    return {
        "observed_at_utc": raw.get("observed_at_utc"),
        "binding": raw.get("binding"),
        "current_runtime_qualified": not reasons,
        "reasons": reasons,
        "parameters": parameters,
    }


def _parameter_observation(engine: dict, name: str) -> dict:
    raw = engine["parameters"].get(name, {})
    if type(raw) is not dict:
        raise ValueError(f"Parameter {name} must be an observation object")
    reasons = list(engine["reasons"])
    reasons.extend(_evidence_reasons(raw))
    reasons.extend(_parameter_value_reasons(name, raw))
    return {
        "value": raw.get("value"),
        "evidence_level": raw.get("evidence_level", "missing"),
        "evidence": raw.get("evidence", []),
        "current_runtime_qualified": not reasons,
        "reasons": reasons,
    }


def _qualification(data: dict, engines: dict, engine_name: str, name: str,
                   as_of: datetime, max_age: int) -> dict:
    raw = data.get("correctness", {}).get(engine_name, {}).get(name, {})
    if type(raw) is not dict:
        raise ValueError(f"correctness.{engine_name}.{name} must be an object")
    engine = engines[engine_name]
    reasons = list(engine["reasons"])
    reasons.extend(_evidence_reasons(raw))
    reasons.extend(_timestamp_reasons(raw.get("observed_at_utc"), as_of, max_age))
    reasons.extend(_binding_reasons(raw.get("binding")))
    expected = data.get("qualification_expected_bindings", {}).get(engine_name, {}).get(
        name, engine["binding"]
    )
    reasons.extend("expected_" + reason for reason in _binding_reasons(expected))
    if type(expected) is dict and type(engine["binding"]) is dict:
        if expected.get("source_sha256") != engine["binding"].get("source_sha256"):
            reasons.append("expected_qualification_source_is_not_observed_engine_source")
    if not _strict_equal(raw.get("binding"), expected):
        reasons.append("qualification_source_or_input_binding_mismatch")
    if raw.get("status") != "passed":
        reasons.append("qualification_not_passed")
    if raw.get("scope") != "full_model":
        reasons.append("qualification_scope_is_not_full_model")
    return {
        "engine": engine_name,
        "qualification": name,
        "reported_status": raw.get("status", "missing"),
        "scope": raw.get("scope", "missing"),
        "qualified": not reasons,
        "binding": raw.get("binding"),
        "expected_binding": expected,
        "observed_at_utc": raw.get("observed_at_utc"),
        "evidence_level": raw.get("evidence_level", "missing"),
        "evidence": raw.get("evidence", []),
        "reasons": reasons,
    }


def audit(data: dict, *, as_of_utc: str | None = None,
          max_runtime_age_seconds: int = 3600) -> dict:
    if type(data) is not dict or data.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"Expected schema {INPUT_SCHEMA}")
    if not _json_value(data):
        raise ValueError("Observations must contain only finite JSON values")
    for key in ("engines", "correctness", "qualification_expected_bindings"):
        if type(data.get(key, {})) is not dict:
            raise ValueError(f"{key} must be an object")
    for key in ("correctness", "qualification_expected_bindings"):
        for engine in ENGINES:
            if type(data.get(key, {}).get(engine, {})) is not dict:
                raise ValueError(f"{key}.{engine} must be an object")
    if type(max_runtime_age_seconds) is not int or max_runtime_age_seconds <= 0:
        raise ValueError("max_runtime_age_seconds must be a positive integer")
    as_of = datetime.now(timezone.utc) if as_of_utc is None else _utc(as_of_utc)
    if as_of is None:
        raise ValueError("as_of_utc must be a timezone-aware ISO timestamp")
    extra = data.get("additional_required_parameters", [])
    if type(extra) is not list or any(type(name) is not str or not name for name in extra):
        raise ValueError("additional_required_parameters must be a list of names")
    names = list(dict.fromkeys((*REQUIRED_PARAMETERS, *extra)))
    engines = {name: _engine(data, name, as_of, max_runtime_age_seconds) for name in ENGINES}
    rows = []
    blockers = []
    for name in names:
        bg, sg = (_parameter_observation(engines[engine], name) for engine in ENGINES)
        comparable = bg["value"] is not None and sg["value"] is not None
        differs = comparable and not _strict_equal(bg["value"], sg["value"])
        if differs:
            status = "different"
        elif bg["current_runtime_qualified"] and sg["current_runtime_qualified"]:
            status = "aligned"
        else:
            status = "unverified"
        rows.append({"parameter": name, "required": True, "BG": bg, "SG": sg,
                     "status": status, "reported_values_differ": differs})
        if status != "aligned":
            blockers.append({"kind": "parameter", "parameter": name, "status": status})
    qualifications = [
        _qualification(data, engines, engine, name, as_of, max_runtime_age_seconds)
        for engine in ENGINES for name in QUALIFICATIONS
    ]
    for row in qualifications:
        if not row["qualified"]:
            blockers.append({"kind": "correctness", "engine": row["engine"],
                             "qualification": row["qualification"], "reasons": row["reasons"]})
    return {
        "schema": OUTPUT_SCHEMA,
        "as_of_utc": as_of.isoformat().replace("+00:00", "Z"),
        "max_runtime_age_seconds": max_runtime_age_seconds,
        "engines": {name: {key: value for key, value in engine.items() if key != "parameters"}
                    for name, engine in engines.items()},
        "parameters": rows,
        "correctness": qualifications,
        "gate": {"formal_test_permitted": not blockers,
                 "status": "ready" if not blockers else "blocked",
                 "required_parameter_count": len(names),
                 "aligned_parameter_count": sum(row["status"] == "aligned" for row in rows),
                 "blockers": blockers},
        "scope": [
            "Offline evidence-contract audit; no network, service, GPU or benchmark action.",
            "Different means reported values differ; evidence level and freshness remain explicit.",
            "Historical, source and planned observations cannot establish current runtime alignment.",
            "Returned text/usage consistency is not a substitute for the four correctness qualifications.",
            "Referenced reports are not independently dereferenced or attested by this tool.",
        ],
    }


def _cell(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(result: dict) -> str:
    gate = result["gate"]
    lines = [
        "# R1 parameter alignment and correctness audit", "",
        f"Audit UTC: `{result['as_of_utc']}`. Runtime freshness limit: "
        f"{result['max_runtime_age_seconds']} seconds.", "",
        f"**Formal measurement permitted: {str(gate['formal_test_permitted']).lower()}**. "
        f"{gate['aligned_parameter_count']}/{gate['required_parameter_count']} required parameters aligned.",
        "", "Reported differences retain their evidence levels; historical or planned values are not current observations.",
        "", "| Parameter | BG value | BG evidence | SG value | SG evidence | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["parameters"]:
        lines.append("| " + " | ".join((row["parameter"], _cell(row["BG"]["value"]),
                      _cell(row["BG"]["evidence_level"]), _cell(row["SG"]["value"]),
                      _cell(row["SG"]["evidence_level"]), row["status"])) + " |")
    lines.extend(("", "## Evidence and unresolved conditions", ""))
    for row in result["parameters"]:
        lines.append(f"- **{row['parameter']}** ({row['status']}):")
        for engine in ENGINES:
            observation = row[engine]
            evidence = observation["evidence"]
            references = "; ".join(evidence) if type(evidence) is list and all(
                type(item) is str for item in evidence
            ) else _cell(evidence)
            references = references or "none"
            reasons = ", ".join(observation["reasons"]) or "current runtime evidence qualified"
            lines.append(f"  - {engine}: {references}. {reasons}.")
    lines.extend(("", "## Correctness qualifications", "",
                  "| Engine | Qualification | Reported status | Scope | Qualified | Unresolved conditions |",
                  "|---|---|---|---|---|---|"))
    for row in result["correctness"]:
        lines.append("| " + " | ".join((row["engine"], row["qualification"],
                      _cell(row["reported_status"]), _cell(row["scope"]),
                      str(row["qualified"]).lower(),
                      ", ".join(row["reasons"]) or "none")) + " |")
    lines.extend(("", "Qualification evidence:", ""))
    for row in result["correctness"]:
        evidence = row["evidence"]
        references = "; ".join(evidence) if type(evidence) is list and all(
            type(item) is str for item in evidence
        ) else _cell(evidence)
        lines.append(f"- {row['engine']} {row['qualification']}: "
                     + (references or "none"))
    lines.extend(("", "## Scope", ""))
    lines.extend("- " + item for item in result["scope"])
    return "\n".join(lines) + "\n"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_observations(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def write_reports(result: dict, json_output: Path, markdown_output: Path) -> None:
    if json_output.resolve() == markdown_output.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    # Check both before creating either; mode x also prevents a later overwrite.
    for path in (json_output, markdown_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to replace {path}")
        if not path.parent.is_dir():
            raise ValueError(f"Output parent does not exist: {path.parent}")
    json_text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                           allow_nan=False) + "\n"
    markdown_text = markdown(result)
    with json_output.open("x") as json_handle:
        with markdown_output.open("x") as markdown_handle:
            json_handle.write(json_text)
            markdown_handle.write(markdown_text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--as-of-utc", help="Explicit replay time; default is current UTC")
    parser.add_argument("--max-runtime-age-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    result = audit(load_observations(args.observations), as_of_utc=args.as_of_utc,
                   max_runtime_age_seconds=args.max_runtime_age_seconds)
    write_reports(result, args.json_output, args.markdown_output)
    print(json.dumps({"status": result["gate"]["status"],
                      "formal_test_permitted": result["gate"]["formal_test_permitted"],
                      "aligned_parameters": result["gate"]["aligned_parameter_count"],
                      "required_parameters": result["gate"]["required_parameter_count"]}))
    return 0 if result["gate"]["formal_test_permitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
