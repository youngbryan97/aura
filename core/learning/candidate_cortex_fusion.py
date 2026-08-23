"""Exact post-training fusion contract for a candidate cortex generation."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.brain.llm.model_artifact_profile import validate_model_artifact_descriptor
from core.governance_context import local_internal_governed_scope
from core.learning.candidate_cortex_training import (
    admitted_adaptive_checkpoint,
    document_sha256,
    file_sha256,
    load_and_verify_plan,
    read_authenticated_journal,
)
from core.runtime.file_write_gateway import get_file_write_gateway

FUSION_PLAN_SCHEMA: Final = "aura.candidate_cortex_fusion.plan.v1"
FUSION_RECEIPT_SCHEMA: Final = "aura.candidate_cortex_fusion.receipt.v1"
FUSION_PROVENANCE_SCHEMA: Final = "aura.candidate_cortex_fusion.provenance.v1"
FUSION_PLAN_FILE: Final = "fusion_plan.json"
FUSION_RECEIPT_FILE: Final = "fusion_receipt.json"
FUSION_PROVENANCE_FILE: Final = "aura_fusion_provenance.json"
MAX_DOCUMENT_BYTES: Final = 16 * 1024 * 1024


class CandidateCortexFusionError(ValueError):
    """A stable post-training fusion contract failure."""


def _fail(code: str) -> None:
    raise CandidateCortexFusionError(code)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("fusion_json_duplicate_key")
        result[key] = value
    return result


def _strict_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES:
            _fail(f"{role}_invalid")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail("fusion_json_non_finite"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CandidateCortexFusionError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _file_binding(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        _fail("bound_file_invalid")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        _fail("bound_file_invalid")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": stat.st_size,
    }


def _validate_file_binding(raw: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if set(raw) != {"path", "sha256", "size_bytes"}:
        _fail(f"{role}_binding_invalid")
    try:
        observed = _file_binding(Path(str(raw["path"])))
    except (CandidateCortexFusionError, OSError, ValueError) as exc:
        raise CandidateCortexFusionError(f"{role}_binding_invalid") from exc
    if observed != dict(raw):
        _fail(f"{role}_binding_drift")
    return observed


def _key(path: Path) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        _fail("journal_key_invalid")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        _fail("journal_key_invalid")
    payload = resolved.read_bytes()
    if len(payload) < 32:
        _fail("journal_key_invalid")
    return payload


def _adaptive_authority(
    run_root: Path,
    *,
    journal_key: bytes,
    verify_full_model: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = load_and_verify_plan(run_root, verify_full_model=verify_full_model)
    result = _strict_json(run_root / "adaptive_result.json", role="adaptive_result")
    events = read_authenticated_journal(
        Path(str(plan["paths"]["journal"])),
        key=journal_key,
    )
    authority = admitted_adaptive_checkpoint(
        plan,
        authenticated_events=events,
        adaptive_result=result,
    )
    return plan, result, authority


def prepare_fusion_plan(
    *,
    run_root: Path,
    journal_key_path: Path,
    fusion_root: Path,
    output_root: Path,
    target_source: Path,
    verifier_source: Path,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    """Build a content-addressed fusion plan from one admitted final stage."""

    run_root = run_root.expanduser().resolve(strict=True)
    key = _key(journal_key_path)
    training, result, authority = _adaptive_authority(
        run_root,
        journal_key=key,
        verify_full_model=verify_full_model,
    )
    adapter_root = Path(str(training["paths"]["adapter_root"])).resolve(strict=True)
    adapter_config = adapter_root / "adapter_config.json"
    config_binding = _file_binding(adapter_config)
    checkpoint = dict(authority["checkpoint"])
    if Path(str(checkpoint["path"])).parent != adapter_root:
        _fail("adaptive_checkpoint_path_escape")

    base_descriptor_path = Path(
        str(training["model"]["descriptor_path"])
    ).resolve(strict=True)
    descriptor = _strict_json(base_descriptor_path, role="base_descriptor")
    fusion_root = fusion_root.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    identity_material = {
        "base_descriptor_sha256": training["model"]["descriptor_sha256"],
        "adaptive_result_sha256": result["result_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "adapter_config_sha256": config_binding["sha256"],
    }
    generation_id = document_sha256(identity_material)[:20]
    fused_revision = document_sha256({**identity_material, "kind": "fused-cortex"})[:40]
    output_path = output_root / f"Aura-Qwen3.8-27B-persona-crsm-{generation_id}"
    fused_descriptor_path = fusion_root / "fused_model_descriptor.json"
    source_bindings = {
        "contract": _file_binding(Path(__file__)),
        "target": _file_binding(target_source),
        "verifier": _file_binding(verifier_source),
    }
    profile = descriptor.get("artifact_profile")
    if not isinstance(profile, Mapping):
        _fail("base_descriptor_profile_invalid")
    weight_bytes = profile.get("weight_bytes")
    if isinstance(weight_bytes, bool) or not isinstance(weight_bytes, int) or weight_bytes <= 0:
        _fail("base_descriptor_weight_bytes_invalid")

    body: dict[str, Any] = {
        "schema": FUSION_PLAN_SCHEMA,
        "training_plan_sha256": training["plan_sha256"],
        "python": training["python"],
        "python_binding": training["python_binding"],
        "base_model": {
            "canonical_path": training["model"]["canonical_path"],
            "descriptor": _file_binding(base_descriptor_path),
            "descriptor_sha256": training["model"]["descriptor_sha256"],
            "repository_id": training["model"]["repository_id"],
            "revision": training["model"]["revision"],
            "weight_bytes": weight_bytes,
        },
        "adaptive": {
            "run_root": str(run_root),
            "result": _file_binding(run_root / "adaptive_result.json"),
            "result_sha256": result["result_sha256"],
            "stage_index": authority["stage_index"],
            "cumulative_iterations": authority["cumulative_iterations"],
            "checkpoint": checkpoint,
            "adapter_config": config_binding,
        },
        "output": {
            "generation_id": generation_id,
            "repository_id": "aura/Qwen3.8-27B-persona-crsm",
            "revision": fused_revision,
            "fusion_root": str(fusion_root),
            "root": str(output_root),
            "path": str(output_path),
            "staging_path": str(output_root / f".{output_path.name}.partial"),
            "descriptor_path": str(fused_descriptor_path),
            "receipt_path": str(fusion_root / FUSION_RECEIPT_FILE),
            "minimum_free_bytes": int(weight_bytes * 1.25) + 4 * 1024**3,
        },
        "sources": source_bindings,
    }
    return {**body, "fusion_plan_sha256": document_sha256(body)}


def validate_fusion_plan(
    raw: Mapping[str, Any],
    *,
    journal_key_path: Path,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    required = {
        "schema",
        "training_plan_sha256",
        "python",
        "python_binding",
        "base_model",
        "adaptive",
        "output",
        "sources",
        "fusion_plan_sha256",
    }
    if set(raw) != required or raw.get("schema") != FUSION_PLAN_SCHEMA:
        _fail("fusion_plan_invalid")
    material = dict(raw)
    claimed = material.pop("fusion_plan_sha256", None)
    if claimed != document_sha256(material):
        _fail("fusion_plan_digest_invalid")
    base = raw.get("base_model")
    adaptive = raw.get("adaptive")
    output = raw.get("output")
    sources = raw.get("sources")
    if not all(isinstance(value, Mapping) for value in (base, adaptive, output, sources)):
        _fail("fusion_plan_invalid")
    assert isinstance(base, Mapping)
    assert isinstance(adaptive, Mapping)
    assert isinstance(output, Mapping)
    assert isinstance(sources, Mapping)
    if set(base) != {
        "canonical_path",
        "descriptor",
        "descriptor_sha256",
        "repository_id",
        "revision",
        "weight_bytes",
    }:
        _fail("fusion_base_model_invalid")
    if set(adaptive) != {
        "run_root",
        "result",
        "result_sha256",
        "stage_index",
        "cumulative_iterations",
        "checkpoint",
        "adapter_config",
    }:
        _fail("fusion_adaptive_invalid")
    if set(output) != {
        "generation_id",
        "repository_id",
        "revision",
        "fusion_root",
        "root",
        "path",
        "staging_path",
        "descriptor_path",
        "receipt_path",
        "minimum_free_bytes",
    }:
        _fail("fusion_output_invalid")
    if set(sources) != {"contract", "target", "verifier"}:
        _fail("fusion_source_manifest_invalid")
    for name, binding in sources.items():
        if not isinstance(binding, Mapping):
            _fail("fusion_source_manifest_invalid")
        _validate_file_binding(binding, role=f"fusion_source_{name}")

    run_root = Path(str(adaptive.get("run_root"))).expanduser().resolve(strict=True)
    training, result, authority = _adaptive_authority(
        run_root,
        journal_key=_key(journal_key_path),
        verify_full_model=verify_full_model,
    )
    if raw.get("training_plan_sha256") != training["plan_sha256"]:
        _fail("fusion_training_plan_mismatch")
    if (
        raw.get("python") != training["python"]
        or raw.get("python_binding") != training["python_binding"]
    ):
        _fail("fusion_python_identity_mismatch")
    if base.get("descriptor_sha256") != training["model"]["descriptor_sha256"]:
        _fail("fusion_base_model_mismatch")
    if base.get("canonical_path") != training["model"]["canonical_path"]:
        _fail("fusion_base_model_mismatch")
    if (
        base.get("repository_id") != training["model"]["repository_id"]
        or base.get("revision") != training["model"]["revision"]
    ):
        _fail("fusion_base_model_mismatch")
    if adaptive.get("result_sha256") != result["result_sha256"]:
        _fail("fusion_adaptive_result_mismatch")
    if adaptive.get("stage_index") != authority["stage_index"]:
        _fail("fusion_adaptive_stage_mismatch")
    if adaptive.get("cumulative_iterations") != authority["cumulative_iterations"]:
        _fail("fusion_adaptive_stage_mismatch")
    if adaptive.get("checkpoint") != authority["checkpoint"]:
        _fail("fusion_adaptive_checkpoint_mismatch")
    for name in ("result", "adapter_config"):
        binding = adaptive.get(name)
        if not isinstance(binding, Mapping):
            _fail(f"fusion_{name}_binding_invalid")
        _validate_file_binding(binding, role=f"fusion_{name}")
    if adaptive["result"]["path"] != str(run_root / "adaptive_result.json"):
        _fail("fusion_result_path_mismatch")
    adapter_root = Path(str(training["paths"]["adapter_root"])).resolve(strict=True)
    if Path(str(adaptive["adapter_config"]["path"])).parent != adapter_root:
        _fail("fusion_adapter_config_path_mismatch")
    descriptor_binding = base.get("descriptor")
    if not isinstance(descriptor_binding, Mapping):
        _fail("fusion_base_descriptor_binding_invalid")
    _validate_file_binding(descriptor_binding, role="fusion_base_descriptor")

    output_root = Path(str(output.get("root"))).expanduser().resolve(strict=False)
    fusion_root = Path(str(output.get("fusion_root"))).expanduser().resolve(strict=False)
    output_path = Path(str(output.get("path"))).expanduser().resolve(strict=False)
    staging_path = Path(str(output.get("staging_path"))).expanduser().resolve(strict=False)
    descriptor_path = Path(str(output.get("descriptor_path"))).expanduser().resolve(strict=False)
    receipt_path = Path(str(output.get("receipt_path"))).expanduser().resolve(strict=False)
    if output_path.parent != output_root or staging_path.parent != output_root:
        _fail("fusion_output_path_escape")
    if descriptor_path.parent != fusion_root or receipt_path.parent != fusion_root:
        _fail("fusion_control_path_escape")
    if descriptor_path.name != "fused_model_descriptor.json":
        _fail("fusion_descriptor_path_invalid")
    if receipt_path.name != FUSION_RECEIPT_FILE:
        _fail("fusion_receipt_path_invalid")
    generation_id = output.get("generation_id")
    revision = output.get("revision")
    identity_material = {
        "base_descriptor_sha256": training["model"]["descriptor_sha256"],
        "adaptive_result_sha256": result["result_sha256"],
        "checkpoint_sha256": authority["checkpoint"]["sha256"],
        "adapter_config_sha256": adaptive["adapter_config"]["sha256"],
    }
    expected_generation = document_sha256(identity_material)[:20]
    expected_revision = document_sha256(
        {**identity_material, "kind": "fused-cortex"}
    )[:40]
    if (
        not isinstance(generation_id, str)
        or len(generation_id) != 20
        or any(character not in "0123456789abcdef" for character in generation_id)
        or output_path.name != f"Aura-Qwen3.8-27B-persona-crsm-{generation_id}"
        or staging_path.name != f".{output_path.name}.partial"
        or output.get("repository_id") != "aura/Qwen3.8-27B-persona-crsm"
        or generation_id != expected_generation
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or revision != expected_revision
    ):
        _fail("fusion_output_identity_invalid")
    weight_bytes = base.get("weight_bytes")
    if isinstance(weight_bytes, bool) or not isinstance(weight_bytes, int) or weight_bytes <= 0:
        _fail("fusion_base_model_invalid")
    minimum_free = output.get("minimum_free_bytes")
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or minimum_free != int(weight_bytes * 1.25) + 4 * 1024**3
    ):
        _fail("fusion_disk_budget_invalid")
    return dict(raw)


def build_fusion_provenance(
    plan: Mapping[str, Any], *, fused_module_count: int
) -> dict[str, Any]:
    """Describe the exact base, adapter, and authority inside fused bytes."""

    if (
        isinstance(fused_module_count, bool)
        or not isinstance(fused_module_count, int)
        or fused_module_count <= 0
    ):
        _fail("fusion_module_count_invalid")
    adaptive = plan["adaptive"]
    base = plan["base_model"]
    output = plan["output"]
    body = {
        "schema": FUSION_PROVENANCE_SCHEMA,
        "fusion_plan_sha256": plan["fusion_plan_sha256"],
        "training_plan_sha256": plan["training_plan_sha256"],
        "generation_id": output["generation_id"],
        "base_descriptor_sha256": base["descriptor_sha256"],
        "adaptive_result_sha256": adaptive["result_sha256"],
        "adapter_checkpoint_sha256": adaptive["checkpoint"]["sha256"],
        "adapter_config_sha256": adaptive["adapter_config"]["sha256"],
        "cumulative_iterations": adaptive["cumulative_iterations"],
        "fused_module_count": fused_module_count,
        "representation_boundary": (
            "fused weights define a new model identity; prior steering and "
            "recurrent tensors are not representation-compatible"
        ),
    }
    return {**body, "provenance_sha256": document_sha256(body)}


def validate_fusion_provenance(
    plan: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    expected = build_fusion_provenance(
        plan,
        fused_module_count=raw.get("fused_module_count"),
    )
    if dict(raw) != expected:
        _fail("fusion_provenance_mismatch")
    return expected


def validate_fusion_receipt(
    plan: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    verify_full_model: bool,
) -> dict[str, Any]:
    """Validate the terminal fused artifact without trusting its filenames."""

    required = {
        "schema",
        "fusion_plan_sha256",
        "generation_id",
        "model_path",
        "descriptor",
        "descriptor_sha256",
        "provenance",
        "receipt_sha256",
    }
    if set(raw) != required or raw.get("schema") != FUSION_RECEIPT_SCHEMA:
        _fail("fusion_receipt_invalid")
    material = dict(raw)
    claimed = material.pop("receipt_sha256", None)
    if claimed != document_sha256(material):
        _fail("fusion_receipt_digest_invalid")
    output = plan["output"]
    if (
        raw.get("fusion_plan_sha256") != plan["fusion_plan_sha256"]
        or raw.get("generation_id") != output["generation_id"]
        or raw.get("model_path") != output["path"]
    ):
        _fail("fusion_receipt_plan_mismatch")
    descriptor_binding = raw.get("descriptor")
    provenance_binding = raw.get("provenance")
    if not isinstance(descriptor_binding, Mapping) or not isinstance(
        provenance_binding, Mapping
    ):
        _fail("fusion_receipt_binding_invalid")
    descriptor_binding = _validate_file_binding(
        descriptor_binding, role="fusion_receipt_descriptor"
    )
    provenance_binding = _validate_file_binding(
        provenance_binding, role="fusion_receipt_provenance"
    )
    if descriptor_binding["path"] != output["descriptor_path"]:
        _fail("fusion_receipt_descriptor_path_mismatch")
    expected_provenance = str(
        Path(str(output["path"])) / FUSION_PROVENANCE_FILE
    )
    if provenance_binding["path"] != expected_provenance:
        _fail("fusion_receipt_provenance_path_mismatch")
    provenance = _strict_json(Path(expected_provenance), role="fusion_provenance")
    validate_fusion_provenance(plan, provenance)
    descriptor = _strict_json(
        Path(str(output["descriptor_path"])), role="fused_descriptor"
    )
    try:
        descriptor = validate_model_artifact_descriptor(
            descriptor,
            model_path=output["path"],
            verify_full_hash=verify_full_model,
        )
    except ValueError as exc:
        raise CandidateCortexFusionError("fused_descriptor_invalid") from exc
    if raw.get("descriptor_sha256") != descriptor["descriptor_sha256"]:
        _fail("fusion_receipt_descriptor_mismatch")
    return dict(raw)


def load_and_validate_fusion_plan(
    plan_path: Path,
    *,
    journal_key_path: Path,
    verify_full_model: bool = True,
) -> dict[str, Any]:
    return validate_fusion_plan(
        _strict_json(plan_path.expanduser().resolve(strict=True), role="fusion_plan"),
        journal_key_path=journal_key_path,
        verify_full_model=verify_full_model,
    )


def publish_json(path: Path, payload: Mapping[str, Any], *, source: str) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with local_internal_governed_scope(source, domain="file_write"):
        get_file_write_gateway().write_bytes_if_absent(
            path,
            data.encode("ascii"),
            mode=0o600,
            source=source,
        )


def ensure_clean_directory(path: Path, *, source: str) -> Path:
    with local_internal_governed_scope(source, domain="file_write"):
        gateway = get_file_write_gateway()
        if os.path.lexists(path):
            gateway.delete_path(path, recursive=True, source=source)
        return Path(gateway.ensure_directory(path, source=source))


def copy_file(source_path: Path, destination: Path, *, source: str) -> Path:
    with local_internal_governed_scope(source, domain="file_write"):
        return Path(
            get_file_write_gateway().copy_path(
                source_path,
                destination,
                source=source,
            )
        )


def move_directory(source_path: Path, destination: Path, *, source: str) -> Path:
    with local_internal_governed_scope(source, domain="file_write"):
        return Path(
            get_file_write_gateway().move_path(
                source_path,
                destination,
                source=source,
            )
        )


def available_bytes(path: Path) -> int:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        _fail("fusion_disk_root_invalid")
    return int(shutil.disk_usage(resolved).free)


__all__ = [
    "CandidateCortexFusionError",
    "FUSION_PLAN_FILE",
    "FUSION_PLAN_SCHEMA",
    "FUSION_PROVENANCE_FILE",
    "FUSION_PROVENANCE_SCHEMA",
    "FUSION_RECEIPT_FILE",
    "FUSION_RECEIPT_SCHEMA",
    "available_bytes",
    "build_fusion_provenance",
    "copy_file",
    "ensure_clean_directory",
    "load_and_validate_fusion_plan",
    "move_directory",
    "prepare_fusion_plan",
    "publish_json",
    "validate_fusion_plan",
    "validate_fusion_provenance",
    "validate_fusion_receipt",
]
