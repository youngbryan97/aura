"""Materialize qualified steering bytes from cortex migration authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.learning.cortex_migration_authority import validate_component_authority
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import get_file_write_gateway


class ModelBoundSteeringError(ValueError):
    """Qualified steering evidence cannot be reopened or materialized."""


#: Expected during a model migration: the tissue is sound and was built for a
#: checkpoint that is no longer resident. Distinct from ``invalid``, which
#: means the authority itself does not hold up.
CHECKPOINT_INCOMPATIBLE = "checkpoint_incompatible"

#: Statuses that mean "correctly not attached", as opposed to "something is
#: wrong". A caller separating expected from unexpected should key on this
#: rather than on a list it maintains itself.
EXPECTED_DETACHED_STATUSES = frozenset({"deferred", "retired", CHECKPOINT_INCOMPATIBLE})


@dataclass(frozen=True)
class SteeringGenerationResolution:
    """Disposition of steering tissue for one exact active cortex."""

    status: str
    cache_dir: Path | None = None
    reason: str = ""

    @property
    def migration_pending(self) -> bool:
        """True when the tissue is fine and the checkpoint moved under it."""
        return self.status == CHECKPOINT_INCOMPATIBLE

    @property
    def expected_detachment(self) -> bool:
        """True when not attaching is the correct outcome, not a fault."""
        return self.status in EXPECTED_DETACHED_STATUSES


def _strict_json(payload: bytes, *, role: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ModelBoundSteeringError(f"{role}_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ModelBoundSteeringError(f"{role}_non_finite")

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelBoundSteeringError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        raise ModelBoundSteeringError(f"{role}_invalid")
    return value


def _binding_bytes(binding: Mapping[str, Any], *, role: str) -> bytes:
    path = binding.get("path")
    size = binding.get("size_bytes")
    digest = binding.get("sha256")
    if (
        not isinstance(path, str)
        or type(size) is not int
        or size <= 0
        or not isinstance(digest, str)
    ):
        raise ModelBoundSteeringError(f"{role}_binding_invalid")
    payload = read_stable_bytes(Path(path), max_bytes=16 * 1024 * 1024)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise ModelBoundSteeringError(f"{role}_binding_drift")
    return payload


def materialize_qualified_generation(
    authority: Mapping[str, Any],
    *,
    descriptor_sha256: str,
    model_cache_root: Path,
    authority_key_path: Path | None = None,
) -> Path:
    """Publish only the signed generation's exact bytes into its model cache."""

    validated = validate_component_authority(
        authority,
        component="steering",
        descriptor_sha256=descriptor_sha256,
        authority_key_path=authority_key_path,
    )
    if validated.get("status") != "qualified":
        raise ModelBoundSteeringError("steering_generation_not_qualified")
    evidence = validated.get("evidence")
    claims = validated.get("claims")
    if not isinstance(evidence, Mapping) or not isinstance(claims, Mapping):
        raise ModelBoundSteeringError("steering_authority_invalid")
    metadata_binding = evidence.get("metadata")
    if not isinstance(metadata_binding, Mapping):
        raise ModelBoundSteeringError("steering_metadata_binding_missing")
    metadata_payload = _binding_bytes(metadata_binding, role="steering_metadata")
    metadata = _strict_json(metadata_payload, role="steering_metadata")
    generation = str(metadata.get("generation_sha256") or "")
    if generation != claims.get("generation_sha256"):
        raise ModelBoundSteeringError("steering_generation_claim_mismatch")
    vectors = metadata.get("vector_files")
    if not isinstance(vectors, list) or not vectors:
        raise ModelBoundSteeringError("steering_vector_manifest_invalid")

    target = model_cache_root.expanduser().absolute() / f"qualified_{generation[:16]}"
    gateway = get_file_write_gateway()
    with local_internal_governed_scope("caa.materialize_qualified_generation", domain="file_write"):
        gateway.ensure_directory(
            target,
            source="caa.materialize_qualified_generation",
        )
        for vector in vectors:
            if not isinstance(vector, Mapping) or not isinstance(vector.get("name"), str):
                raise ModelBoundSteeringError("steering_vector_manifest_invalid")
            name = vector["name"]
            if Path(name).name != name:
                raise ModelBoundSteeringError("steering_vector_manifest_invalid")
            binding = evidence.get(f"vector:{name}")
            if not isinstance(binding, Mapping):
                raise ModelBoundSteeringError("steering_vector_binding_missing")
            payload = _binding_bytes(binding, role=f"steering_vector:{name}")
            destination = target / name
            if destination.exists():
                existing = read_stable_bytes(destination, max_bytes=16 * 1024 * 1024)
                if existing != payload:
                    raise ModelBoundSteeringError("steering_materialization_collision")
            else:
                gateway.write_bytes(
                    destination,
                    payload,
                    source="caa.materialize_qualified_generation",
                )
        metadata_target = target / "caa_steering_meta.json"
        if metadata_target.exists():
            existing_metadata = read_stable_bytes(metadata_target, max_bytes=16 * 1024 * 1024)
            if existing_metadata != metadata_payload:
                raise ModelBoundSteeringError("steering_metadata_collision")
        else:
            gateway.write_bytes(
                metadata_target,
                metadata_payload,
                source="caa.materialize_qualified_generation",
            )
    return target


def resolve_active_generation(
    *,
    descriptor_sha256: str,
    model_cache_root: Path,
) -> SteeringGenerationResolution:
    """Resolve signed steering authority without collapsing deferral into absence."""

    from core.brain.llm.model_registry import get_active_cortex_spec

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        return SteeringGenerationResolution("unmanaged", reason="active_cortex_absent")
    if spec.descriptor_sha256 != descriptor_sha256:
        # The active cortex is a different checkpoint from the one this tissue
        # was built against. During a model migration that is the expected
        # state, not a broken authority, and calling it "invalid" put an
        # expected condition and a forged one behind the same word -- so the
        # runtime logged an error every attach and nobody could tell a
        # migration from corruption.
        return SteeringGenerationResolution(
            CHECKPOINT_INCOMPATIBLE,
            reason="active_cortex_descriptor_mismatch",
        )
    migration = spec.migration_contract()
    if not isinstance(migration, Mapping):
        return SteeringGenerationResolution(
            "invalid",
            reason="active_cortex_migration_contract_missing",
        )
    components = migration.get("components")
    authority = components.get("steering") if isinstance(components, Mapping) else None
    if not isinstance(authority, Mapping):
        return SteeringGenerationResolution(
            "invalid",
            reason="active_cortex_steering_authority_missing",
        )
    status = str(authority.get("status") or "")
    if status in {"deferred", "retired"}:
        validate_component_authority(
            authority,
            component="steering",
            descriptor_sha256=descriptor_sha256,
        )
        return SteeringGenerationResolution(
            status,
            reason=f"steering_generation_{status}",
        )
    if status != "qualified":
        return SteeringGenerationResolution(
            "invalid",
            reason="active_cortex_steering_status_invalid",
        )
    return SteeringGenerationResolution(
        "qualified",
        cache_dir=materialize_qualified_generation(
            authority,
            descriptor_sha256=descriptor_sha256,
            model_cache_root=model_cache_root,
        ),
    )


def resolve_active_qualified_generation(
    *,
    descriptor_sha256: str,
    model_cache_root: Path,
) -> Path | None:
    """Compatibility view for callers that only consume qualified bytes."""

    resolution = resolve_active_generation(
        descriptor_sha256=descriptor_sha256,
        model_cache_root=model_cache_root,
    )
    return resolution.cache_dir if resolution.status == "qualified" else None


__all__ = [
    "ModelBoundSteeringError",
    "SteeringGenerationResolution",
    "materialize_qualified_generation",
    "resolve_active_generation",
    "resolve_active_qualified_generation",
]
