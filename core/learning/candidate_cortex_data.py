"""Deterministic, candidate-bound persona and CRSM dataset preparation.

The legacy builders write shared training paths and use process-global random
state.  This module keeps their CRSM parsing and safety gate, but surrounds it
with an immutable, content-addressed transaction.  Source corpora are read-only;
only a complete candidate generation is published.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway

RECEIPT_SCHEMA = "aura.candidate_cortex_data.receipt.v1"
INTEGRATION_MANIFEST_SCHEMA = "aura.candidate_cortex_data.crsm_integration.v1"
DELTA_MANIFEST_SCHEMA = "aura.candidate_cortex_data.crsm_delta.v1"
DEFAULT_VALID_FRACTION = 0.1
DEFAULT_MAX_CRSM_EXAMPLES = 600
DEFAULT_RETENTION_EXAMPLES = 512
DEFAULT_SPLIT_SEED = 20260822
MAX_JSON_LINE_BYTES = 16 * 1024 * 1024
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


class CandidateCortexDataError(ValueError):
    """A stable candidate-data contract failure."""


def _fail(code: str) -> None:
    raise CandidateCortexDataError(code)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CandidateCortexDataError("canonical_json_invalid") from exc


def document_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateCortexDataError("input_unreadable") from exc
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    lines = 0
    try:
        with path.open("rb") as handle:
            for _ in handle:
                lines += 1
    except OSError as exc:
        raise CandidateCortexDataError("input_unreadable") from exc
    return lines


def file_binding(path: Path, *, include_lines: bool = True) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise CandidateCortexDataError("input_unreadable") from exc
    if not resolved.is_file():
        _fail("input_not_file")
    binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": stat.st_size,
    }
    if include_lines:
        binding["lines"] = _line_count(resolved)
    return binding


def _output_binding(path: Path, generation_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = generation_root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise CandidateCortexDataError("output_path_escape") from exc
    binding = file_binding(resolved)
    binding.pop("path")
    return {"path": relative.as_posix(), **binding}


def _strict_json(raw: bytes, *, role: str) -> Any:
    if not raw or len(raw) > MAX_JSON_LINE_BYTES:
        _fail(f"{role}_size_invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_number_invalid")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except CandidateCortexDataError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CandidateCortexDataError(f"{role}_json_invalid") from exc


def _normalized_display_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _normalized_identity_text(value: str) -> str:
    return " ".join(_normalized_display_text(value).split()).casefold()


def _canonical_conversation(record: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"messages"}:
        _fail(f"{role}_record_schema_invalid")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        _fail(f"{role}_messages_invalid")
    canonical: list[dict[str, str]] = []
    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            _fail(f"{role}_message_{index}_schema_invalid")
        message_role = message.get("role")
        content = message.get("content")
        if message_role not in _ALLOWED_ROLES or not isinstance(content, str):
            _fail(f"{role}_message_{index}_value_invalid")
        normalized_content = _normalized_display_text(content)
        if not normalized_content:
            _fail(f"{role}_message_{index}_empty")
        roles.append(message_role)
        canonical.append({"role": message_role, "content": normalized_content})
    if "user" not in roles or roles[-1] != "assistant":
        _fail(f"{role}_conversation_shape_invalid")
    return {"messages": canonical}


def conversation_digest(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        _fail("conversation_messages_invalid")
    identity = [
        {
            "role": str(message["role"]).strip().casefold(),
            "content": _normalized_identity_text(str(message["content"])),
        }
        for message in messages
    ]
    return document_sha256(identity)


def _prefer_record(
    existing: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if existing is None:
        return candidate
    return min((existing, candidate), key=canonical_json_bytes)


def _read_conversations(path: Path, *, role: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    total = 0
    duplicates = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                total += 1
                value = _strict_json(raw, role=f"{role}_line_{line_number}")
                record = _canonical_conversation(value, role=f"{role}_line_{line_number}")
                digest = conversation_digest(record)
                duplicates += int(digest in records)
                records[digest] = _prefer_record(records.get(digest), record)
    except OSError as exc:
        raise CandidateCortexDataError(f"{role}_unreadable") from exc
    if not records:
        _fail(f"{role}_empty")
    return records, {
        "records": total,
        "unique_records": len(records),
        "duplicate_records": duplicates,
        "content_identity_sha256": document_sha256(sorted(records)),
    }


def _validate_crsm_source(path: Path) -> dict[str, Any]:
    total = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                total += 1
                value = _strict_json(raw, role=f"crsm_source_line_{line_number}")
                if not isinstance(value, dict) or not set(value).issubset({"text", "_quality"}):
                    _fail(f"crsm_source_line_{line_number}_schema_invalid")
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    _fail(f"crsm_source_line_{line_number}_text_invalid")
                quality = value.get("_quality")
                if quality is not None and (
                    isinstance(quality, bool)
                    or not isinstance(quality, (int, float))
                    or not math.isfinite(float(quality))
                    or not 0.0 <= float(quality) <= 1.0
                ):
                    _fail(f"crsm_source_line_{line_number}_quality_invalid")
    except OSError as exc:
        raise CandidateCortexDataError("crsm_source_unreadable") from exc
    if total <= 0:
        _fail("crsm_source_empty")
    return {"records": total}


def _validate_descriptor(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _strict_json(path.read_bytes(), role="candidate_descriptor")
    if not isinstance(descriptor, dict):
        _fail("candidate_descriptor_schema_invalid")
    material = dict(descriptor)
    claimed = material.pop("descriptor_sha256", None)
    if claimed != document_sha256(material):
        _fail("candidate_descriptor_digest_invalid")
    if claimed != expected_sha256:
        _fail("candidate_descriptor_not_admitted")
    return descriptor, file_binding(path)


def _merge_record_maps(*record_maps: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for records in record_maps:
        for digest, record in records.items():
            merged[digest] = _prefer_record(merged.get(digest), record)
    return merged


def _split_records(
    records: Mapping[str, dict[str, Any]],
    *,
    valid_fraction: float,
    seed: int,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < valid_fraction < 1.0:
        _fail("valid_fraction_invalid")
    if len(records) < 2:
        _fail(f"{scope}_split_too_small")
    valid_count = max(1, min(len(records) - 1, round(len(records) * valid_fraction)))
    ranking = sorted(
        records,
        key=lambda digest: hashlib.sha256(
            f"{seed}:{scope}:{digest}".encode("ascii")
        ).hexdigest(),
    )
    valid_digests = set(ranking[:valid_count])
    train = [records[digest] for digest in sorted(records) if digest not in valid_digests]
    valid = [records[digest] for digest in sorted(valid_digests)]
    return train, valid


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    with local_internal_governed_scope(
        "candidate_cortex_data.write_jsonl", domain="file_write"
    ):
        get_file_write_gateway().write_bytes(
            path,
            payload,
            source="candidate_cortex_data.write_jsonl",
        )


def _write_document(path: Path, value: Mapping[str, Any]) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_data.write_document", domain="file_write"
    ):
        get_file_write_gateway().write_bytes(
            path,
            canonical_json_bytes(value) + b"\n",
            source="candidate_cortex_data.write_document",
        )


def _ensure_directory(path: Path) -> Path:
    with local_internal_governed_scope(
        "candidate_cortex_data.ensure_directory", domain="file_write"
    ):
        created = get_file_write_gateway().ensure_directory(
            path,
            source="candidate_cortex_data.ensure_directory",
        )
    return Path(created)


def _publish_directory(staging: Path, destination: Path) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_data.publish", domain="file_write"
    ):
        get_file_write_gateway().move_path(
            staging,
            destination,
            source="candidate_cortex_data.publish",
        )


def _discard_staging(path: Path) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_data.cleanup", domain="file_write"
    ):
        get_file_write_gateway().delete_path(
            path,
            recursive=True,
            source="candidate_cortex_data.cleanup",
        )


def _manifest_with_digest(material: dict[str, Any]) -> dict[str, Any]:
    return {**material, "manifest_sha256": document_sha256(material)}


def _receipt_with_digest(material: dict[str, Any]) -> dict[str, Any]:
    return {**material, "receipt_sha256": document_sha256(material)}


def _manifest_output_binding(path: Path, generation_root: Path) -> dict[str, Any]:
    binding = _output_binding(path, generation_root)
    return {
        "path": binding["path"],
        "sha256": binding["sha256"],
        "size": binding["size_bytes"],
        "lines": binding["lines"],
    }


def _source_bindings(
    *,
    descriptor_path: Path,
    persona_train: Path,
    persona_valid: Path,
    crsm_source: Path,
    source_repo_root: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "candidate_descriptor": file_binding(descriptor_path),
        "persona_train": file_binding(persona_train),
        "persona_valid": file_binding(persona_valid),
        "crsm_source": file_binding(crsm_source),
        "crsm_builder": file_binding(source_repo_root / "training/build_dataset_v3.py"),
        "personality_spec": file_binding(source_repo_root / "training/personality_spec_v2.py"),
    }


def _semantic_input_identity(
    *,
    persona_train: Mapping[str, Any],
    persona_valid: Mapping[str, Any],
    crsm_source_binding: Mapping[str, Any],
) -> str:
    return document_sha256(
        {
            "persona_train": sorted(persona_train),
            "persona_valid": sorted(persona_valid),
            "crsm_source_sha256": crsm_source_binding["sha256"],
        }
    )


def _build_crsm_examples(
    source: Path,
    *,
    max_examples: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # This import is intentionally local: the preparation module itself stays
    # model-free, while the production CRSM parser and safety gate remain the
    # single source of truth for eligible captures.
    from training.build_dataset_v3 import (
        SYSTEM_VARIANTS,
        build_crsm_experience_examples,
    )

    if not SYSTEM_VARIANTS:
        _fail("crsm_system_variant_missing")
    examples, builder_manifest = build_crsm_experience_examples(
        source,
        max_examples=max_examples,
        system_variants=[SYSTEM_VARIANTS[0]],
    )
    records: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for index, example in enumerate(examples):
        record = _canonical_conversation(example, role=f"crsm_builder_record_{index}")
        digest = conversation_digest(record)
        duplicates += int(digest in records)
        records[digest] = _prefer_record(records.get(digest), record)
    if not records:
        _fail("crsm_no_eligible_records")
    stable_manifest = {
        key: builder_manifest.get(key)
        for key in (
            "source_lines",
            "source_size",
            "source_sha256",
            "accepted",
            "deduplicated",
            "max_examples",
            "rejected_by_reason",
        )
    }
    stable_manifest["post_builder_duplicate_records"] = duplicates
    return records, stable_manifest


def _retention_records(
    persona: Mapping[str, dict[str, Any]],
    *,
    excluded: set[str],
    count: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    eligible = [digest for digest in persona if digest not in excluded]
    ranked = sorted(
        eligible,
        key=lambda digest: hashlib.sha256(
            f"{seed}:crsm-retention:{digest}".encode("ascii")
        ).hexdigest(),
    )
    return {digest: persona[digest] for digest in ranked[: max(0, count)]}


def _assert_source_preserved(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in before:
        for field in ("path", "sha256", "size_bytes", "lines"):
            if before[name].get(field) != after[name].get(field):
                _fail(f"source_changed_during_preparation:{name}:{field}")


def _validate_output_dataset(path: Path, *, role: str) -> tuple[set[str], int]:
    records, stats = _read_conversations(path, role=role)
    if stats["duplicate_records"]:
        _fail(f"{role}_duplicates_present")
    return set(records), int(stats["records"])


def _validate_hashed_document(path: Path, *, schema: str) -> dict[str, Any]:
    value = _strict_json(path.read_bytes(), role=path.stem)
    if not isinstance(value, dict) or value.get("schema") != schema:
        _fail(f"{path.stem}_schema_invalid")
    material = dict(value)
    claimed = material.pop("manifest_sha256", None)
    if claimed != document_sha256(material):
        _fail(f"{path.stem}_digest_invalid")
    return value


def validate_candidate_cortex_data_receipt(
    receipt_path: Path,
    *,
    expected_descriptor_sha256: str | None = None,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    """Verify a published generation and return its receipt."""

    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt = _strict_json(receipt_path.read_bytes(), role="candidate_data_receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        _fail("receipt_schema_invalid")
    material = dict(receipt)
    claimed = material.pop("receipt_sha256", None)
    if claimed != document_sha256(material):
        _fail("receipt_digest_invalid")
    candidate = receipt.get("candidate")
    descriptor_sha256 = candidate.get("descriptor_sha256") if isinstance(candidate, dict) else None
    if expected_descriptor_sha256 and descriptor_sha256 != expected_descriptor_sha256:
        _fail("receipt_candidate_mismatch")

    generation_root = receipt_path.parent.resolve(strict=True)
    if receipt.get("generation_root") != str(generation_root):
        _fail("receipt_generation_root_mismatch")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        _fail("receipt_bindings_invalid")
    if verify_inputs:
        for name, binding in inputs.items():
            if not isinstance(binding, dict):
                _fail(f"receipt_input_{name}_invalid")
            current = file_binding(Path(str(binding.get("path"))))
            for field in ("sha256", "size_bytes", "lines"):
                if current.get(field) != binding.get(field):
                    _fail(f"receipt_input_{name}_{field}_mismatch")
    for name, binding in outputs.items():
        if not isinstance(binding, dict):
            _fail(f"receipt_output_{name}_invalid")
        relative = Path(str(binding.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"receipt_output_{name}_path_invalid")
        output = (generation_root / relative).resolve(strict=True)
        if not output.is_relative_to(generation_root):
            _fail(f"receipt_output_{name}_path_escape")
        current = _output_binding(output, generation_root)
        if current != binding:
            _fail(f"receipt_output_{name}_mismatch")

    persona_train = generation_root / "data/train.jsonl"
    persona_valid = generation_root / "data/valid.jsonl"
    delta_train = generation_root / "data/crsm_delta/train.jsonl"
    delta_valid = generation_root / "data/crsm_delta/valid.jsonl"
    persona_train_keys, _ = _validate_output_dataset(persona_train, role="output_persona_train")
    persona_valid_keys, _ = _validate_output_dataset(persona_valid, role="output_persona_valid")
    delta_train_keys, _ = _validate_output_dataset(delta_train, role="output_crsm_train")
    delta_valid_keys, _ = _validate_output_dataset(delta_valid, role="output_crsm_valid")
    if persona_train_keys & persona_valid_keys:
        _fail("persona_split_leakage")
    if delta_train_keys & delta_valid_keys:
        _fail("crsm_split_leakage")

    integration = _validate_hashed_document(
        generation_root / "data/crsm_integration_manifest.json",
        schema=INTEGRATION_MANIFEST_SCHEMA,
    )
    delta = _validate_hashed_document(
        generation_root / "data/crsm_delta_manifest.json",
        schema=DELTA_MANIFEST_SCHEMA,
    )
    for manifest, train_path, valid_path in (
        (integration, persona_train, persona_valid),
        (delta, delta_train, delta_valid),
    ):
        if manifest.get("model_descriptor_sha256") != descriptor_sha256:
            _fail("manifest_candidate_mismatch")
        output = manifest.get("output")
        if not isinstance(output, dict):
            _fail("manifest_output_invalid")
        for split, path in (("train", train_path), ("valid", valid_path)):
            expected = _manifest_output_binding(path, generation_root)
            if output.get(split) != expected:
                _fail(f"manifest_{split}_stale")
    return receipt


def prepare_candidate_cortex_data(
    *,
    descriptor_path: Path,
    expected_descriptor_sha256: str,
    persona_train: Path,
    persona_valid: Path,
    crsm_source: Path,
    output_root: Path,
    source_repo_root: Path,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    max_crsm_examples: int = DEFAULT_MAX_CRSM_EXAMPLES,
    retention_examples: int = DEFAULT_RETENTION_EXAMPLES,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Prepare and atomically publish one candidate-scoped data generation."""

    if max_crsm_examples <= 0 or retention_examples < 0 or split_seed < 0:
        _fail("preparation_parameter_invalid")
    descriptor_path = descriptor_path.expanduser().resolve(strict=True)
    persona_train = persona_train.expanduser().resolve(strict=True)
    persona_valid = persona_valid.expanduser().resolve(strict=True)
    crsm_source = crsm_source.expanduser().resolve(strict=True)
    source_repo_root = source_repo_root.expanduser().resolve(strict=True)
    descriptor, _ = _validate_descriptor(
        descriptor_path,
        expected_sha256=expected_descriptor_sha256,
    )
    _validate_crsm_source(crsm_source)
    before = _source_bindings(
        descriptor_path=descriptor_path,
        persona_train=persona_train,
        persona_valid=persona_valid,
        crsm_source=crsm_source,
        source_repo_root=source_repo_root,
    )
    train_records, train_stats = _read_conversations(persona_train, role="persona_train")
    valid_records, valid_stats = _read_conversations(persona_valid, role="persona_valid")
    overlap = set(train_records) & set(valid_records)
    crsm_records, crsm_builder_manifest = _build_crsm_examples(
        crsm_source,
        max_examples=max_crsm_examples,
    )
    persona_records = _merge_record_maps(train_records, valid_records, crsm_records)
    persona_train_rows, persona_valid_rows = _split_records(
        persona_records,
        valid_fraction=valid_fraction,
        seed=split_seed,
        scope=expected_descriptor_sha256 + ":persona",
    )
    retention = _retention_records(
        persona_records,
        excluded=set(crsm_records),
        count=retention_examples,
        seed=split_seed,
    )
    delta_records = _merge_record_maps(crsm_records, retention)
    delta_train_rows, delta_valid_rows = _split_records(
        delta_records,
        valid_fraction=valid_fraction,
        seed=split_seed,
        scope=expected_descriptor_sha256 + ":crsm-delta",
    )

    parameters = {
        "valid_fraction": valid_fraction,
        "max_crsm_examples": max_crsm_examples,
        "retention_examples": retention_examples,
        "split_seed": split_seed,
    }
    run_material = {
        "model_descriptor_sha256": expected_descriptor_sha256,
        "inputs": {name: binding["sha256"] for name, binding in before.items()},
        "semantic_input_sha256": _semantic_input_identity(
            persona_train=train_records,
            persona_valid=valid_records,
            crsm_source_binding=before["crsm_source"],
        ),
        "parameters": parameters,
    }
    run_id = document_sha256(run_material)[:24]
    root = _ensure_directory(output_root.expanduser().resolve(strict=False))
    candidate_root = _ensure_directory(root / expected_descriptor_sha256[:16])
    generation_root = candidate_root / run_id
    lock_path = candidate_root / ".candidate_cortex_data.lock"

    with interprocess_file_lock(lock_path):
        receipt_path = generation_root / "candidate_cortex_data_receipt.json"
        if receipt_path.is_file():
            receipt = validate_candidate_cortex_data_receipt(
                receipt_path,
                expected_descriptor_sha256=expected_descriptor_sha256,
            )
            return {**receipt, "resumed": True}
        if generation_root.exists():
            _fail("existing_generation_incomplete")

        staging = candidate_root / f".{run_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        if staging.exists():
            _fail("staging_path_collision")
        try:
            data_dir = staging / "data"
            delta_dir = data_dir / "crsm_delta"
            _ensure_directory(delta_dir)
            persona_train_path = data_dir / "train.jsonl"
            persona_valid_path = data_dir / "valid.jsonl"
            delta_train_path = delta_dir / "train.jsonl"
            delta_valid_path = delta_dir / "valid.jsonl"
            _write_jsonl(persona_train_path, persona_train_rows)
            _write_jsonl(persona_valid_path, persona_valid_rows)
            _write_jsonl(delta_train_path, delta_train_rows)
            _write_jsonl(delta_valid_path, delta_valid_rows)

            integration_material = {
                "schema": INTEGRATION_MANIFEST_SCHEMA,
                "model_descriptor_sha256": expected_descriptor_sha256,
                "source_sha256": before["crsm_source"]["sha256"],
                "source_lines": before["crsm_source"]["lines"],
                "source_size": before["crsm_source"]["size_bytes"],
                "builder_sha256": before["crsm_builder"]["sha256"],
                "accepted": len(crsm_records),
                "deduplicated": int(crsm_builder_manifest.get("deduplicated") or 0),
                "rejected_by_reason": dict(
                    sorted((crsm_builder_manifest.get("rejected_by_reason") or {}).items())
                ),
                "input_persona": {
                    "train_sha256": before["persona_train"]["sha256"],
                    "valid_sha256": before["persona_valid"]["sha256"],
                },
                "output": {
                    "builder": "core.learning.candidate_cortex_data",
                    "total_examples": len(persona_records),
                    "crsm_examples": len(crsm_records),
                    "train": _manifest_output_binding(persona_train_path, staging),
                    "valid": _manifest_output_binding(persona_valid_path, staging),
                },
            }
            integration_path = data_dir / "crsm_integration_manifest.json"
            _write_document(integration_path, _manifest_with_digest(integration_material))

            delta_material = {
                "schema": DELTA_MANIFEST_SCHEMA,
                "model_descriptor_sha256": expected_descriptor_sha256,
                "source_sha256": before["crsm_source"]["sha256"],
                "source_lines": before["crsm_source"]["lines"],
                "source_size": before["crsm_source"]["size_bytes"],
                "builder_sha256": before["crsm_builder"]["sha256"],
                "delta_mode": True,
                "seed": split_seed,
                "retention_examples": len(retention),
                "selection_sha256": document_sha256(sorted(delta_records)),
                "output": {
                    "builder": "core.learning.candidate_cortex_data",
                    "total_examples": len(delta_records),
                    "crsm_examples": len(crsm_records),
                    "retention_examples": len(retention),
                    "train": _manifest_output_binding(delta_train_path, staging),
                    "valid": _manifest_output_binding(delta_valid_path, staging),
                },
            }
            delta_manifest_path = data_dir / "crsm_delta_manifest.json"
            _write_document(delta_manifest_path, _manifest_with_digest(delta_material))

            after = _source_bindings(
                descriptor_path=descriptor_path,
                persona_train=persona_train,
                persona_valid=persona_valid,
                crsm_source=crsm_source,
                source_repo_root=source_repo_root,
            )
            _assert_source_preserved(before, after)
            output_paths = {
                "persona_train": persona_train_path,
                "persona_valid": persona_valid_path,
                "crsm_integration_manifest": integration_path,
                "crsm_delta_train": delta_train_path,
                "crsm_delta_valid": delta_valid_path,
                "crsm_delta_manifest": delta_manifest_path,
            }
            receipt_material: dict[str, Any] = {
                "schema": RECEIPT_SCHEMA,
                "candidate": {
                    "descriptor_sha256": expected_descriptor_sha256,
                    "repository_id": descriptor.get("repository_id"),
                    "revision": descriptor.get("revision"),
                },
                "run_id": run_id,
                "generation_root": str(generation_root),
                "parameters": parameters,
                "semantic_input_sha256": run_material["semantic_input_sha256"],
                "inputs": before,
                "outputs": {
                    name: _output_binding(path, staging)
                    for name, path in sorted(output_paths.items())
                },
                "repair": {
                    "persona_train_duplicates_removed": train_stats["duplicate_records"],
                    "persona_valid_duplicates_removed": valid_stats["duplicate_records"],
                    "persona_split_overlaps_removed": len(overlap),
                    "crsm_manifest_source_sha256": before["crsm_source"]["sha256"],
                    "crsm_builder_accepted": len(crsm_records),
                    "crsm_retention_examples": len(retention),
                },
                "invariants": {
                    "source_datasets_preserved": True,
                    "persona_within_split_duplicates": 0,
                    "persona_cross_split_overlap": 0,
                    "crsm_within_split_duplicates": 0,
                    "crsm_cross_split_overlap": 0,
                    "candidate_bound": True,
                    "atomic_generation": True,
                },
            }
            receipt = _receipt_with_digest(receipt_material)
            _write_document(staging / "candidate_cortex_data_receipt.json", receipt)
            _publish_directory(staging, generation_root)
            validated = validate_candidate_cortex_data_receipt(
                generation_root / "candidate_cortex_data_receipt.json",
                expected_descriptor_sha256=expected_descriptor_sha256,
            )
            return {**validated, "resumed": False}
        except BaseException:
            _discard_staging(staging)
            raise


__all__ = [
    "CandidateCortexDataError",
    "DEFAULT_MAX_CRSM_EXAMPLES",
    "DEFAULT_RETENTION_EXAMPLES",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VALID_FRACTION",
    "DELTA_MANIFEST_SCHEMA",
    "INTEGRATION_MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "canonical_json_bytes",
    "conversation_digest",
    "document_sha256",
    "prepare_candidate_cortex_data",
    "validate_candidate_cortex_data_receipt",
]
