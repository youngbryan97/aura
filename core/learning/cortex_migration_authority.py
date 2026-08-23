"""Typed authority for moving model-facing tissue to a new cortex basis."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.brain.llm.unified_recurrent_qualified_activation import (
    qualified_activation_errors,
)
from core.learning.model_tissue_migration_inventory import (
    TissueInventoryError,
    validate_tissue_migration_inventory,
)
from core.runtime.file_read_gateway import StableFileReadError, read_stable_bytes
from core.runtime.state_ownership import state_root

COMPONENT_AUTHORITY_SCHEMA: Final = "aura.cortex_upgrade.component_authority.v1"
CAA_EVALUATION_SCHEMA: Final = "aura.caa.causal_evaluation.v1"
RECURRENT_MODEL_BINDING_SCHEMA: Final = "aura.recurrent.model_binding.v1"
MAX_EVIDENCE_BYTES: Final = 16 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_COMPONENT_SPECS: Final = {
    "persona_crsm": ("fused_persona_crsm", "qualified"),
    "steering": ("caa_model_bound", "qualified"),
    "expert_adapters": ("retirement_inventory", "retired"),
    "recurrence_native": ("qualified_recurrent_activation", "qualified"),
}
_CLAIM_FIELDS: Final = {
    "persona_crsm": {"fusion_plan_sha256", "fusion_receipt_sha256"},
    "steering": {
        "generation_sha256",
        "extraction_contract_sha256",
        "causal_evaluation_sha256",
    },
    "expert_adapters": {"inventory_sha256"},
    "recurrence_native": {
        "activation_sha256",
        "package_id",
        "manifest_sha256",
        "model_binding_sha256",
    },
}


class CortexMigrationAuthorityError(ValueError):
    """Migration evidence is absent, forged, stale, or semantically invalid."""


def _fail(code: str) -> None:
    raise CortexMigrationAuthorityError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CortexMigrationAuthorityError("migration_authority_not_canonical") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def default_authority_key_path() -> Path:
    return state_root() / "private/cortex-upgrade/migration-authority.key"


def _authority_key(path: Path | None) -> bytes:
    target = (path or default_authority_key_path()).expanduser().absolute()
    try:
        metadata = target.lstat()
        key = read_stable_bytes(target, max_bytes=32)
    except (OSError, StableFileReadError, ValueError) as exc:
        raise CortexMigrationAuthorityError("migration_authority_key_unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or len(key) != 32
    ):
        _fail("migration_authority_key_custody_invalid")
    return key


def _strict_json(payload: bytes, *, role: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                _fail(f"{role}_duplicate_key")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(f"{role}_non_finite"),
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CortexMigrationAuthorityError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _read_binding(raw: Any, *, role: str, custody_root: Path) -> tuple[dict[str, Any], bytes]:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
        _fail(f"{role}_binding_invalid")
    path_value = raw.get("path")
    size = raw.get("size_bytes")
    digest = raw.get("sha256")
    if (
        not isinstance(path_value, str)
        or type(size) is not int
        or not 0 < size <= MAX_EVIDENCE_BYTES
        or not _is_sha(digest)
    ):
        _fail(f"{role}_binding_invalid")
    path = Path(path_value).expanduser().absolute()
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(custody_root)
        payload = read_stable_bytes(resolved, max_bytes=MAX_EVIDENCE_BYTES)
    except (OSError, StableFileReadError, ValueError) as exc:
        raise CortexMigrationAuthorityError(f"{role}_custody_invalid") from exc
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        _fail(f"{role}_binding_drift")
    normalized = {"path": str(resolved), "size_bytes": len(payload), "sha256": digest}
    if dict(raw) != normalized:
        _fail(f"{role}_binding_identity_invalid")
    return normalized, payload


def _validate_persona(
    evidence: Mapping[str, bytes], claims: Mapping[str, Any], descriptor_sha256: str
) -> None:
    if set(evidence) != {"fusion_plan", "fusion_receipt"}:
        _fail("persona_crsm_evidence_incomplete")
    plan = _strict_json(evidence["fusion_plan"], role="fusion_plan")
    receipt = _strict_json(evidence["fusion_receipt"], role="fusion_receipt")
    plan_material = dict(plan)
    plan_digest = plan_material.pop("fusion_plan_sha256", None)
    receipt_material = dict(receipt)
    receipt_digest = receipt_material.pop("receipt_sha256", None)
    if (
        plan.get("schema") != "aura.candidate_cortex_fusion.plan.v1"
        or receipt.get("schema") != "aura.candidate_cortex_fusion.receipt.v1"
        or plan_digest != claims["fusion_plan_sha256"]
        or plan_digest != _sha(plan_material)
        or receipt_digest != claims["fusion_receipt_sha256"]
        or receipt_digest != _sha(receipt_material)
        or receipt.get("fusion_plan_sha256") != plan.get("fusion_plan_sha256")
        or receipt.get("descriptor_sha256") != descriptor_sha256
    ):
        _fail("persona_crsm_fusion_authority_invalid")


def _validate_steering(
    evidence: Mapping[str, bytes], claims: Mapping[str, Any], descriptor_sha256: str
) -> None:
    if "metadata" not in evidence or "causal_evaluation" not in evidence:
        _fail("steering_evidence_incomplete")
    metadata = _strict_json(evidence["metadata"], role="steering_metadata")
    evaluation = _strict_json(
        evidence["causal_evaluation"], role="steering_causal_evaluation"
    )
    model_identity = metadata.get("model_identity")
    extraction = metadata.get("extraction_contract")
    vectors = metadata.get("vector_files")
    generation_sha256 = _sha(
        {
            "extraction_contract_sha256": (
                extraction.get("extraction_contract_sha256")
                if isinstance(extraction, Mapping)
                else None
            ),
            "vector_files": vectors,
        }
    )
    if (
        not isinstance(model_identity, Mapping)
        or model_identity.get("model_descriptor_sha256") != descriptor_sha256
        or not isinstance(extraction, Mapping)
        or extraction.get("extraction_contract_sha256")
        != claims["extraction_contract_sha256"]
        or metadata.get("generation_sha256") != claims["generation_sha256"]
        or metadata.get("generation_sha256") != generation_sha256
        or not isinstance(vectors, list)
        or not vectors
    ):
        _fail("steering_generation_invalid")
    expected_roles = {"metadata", "causal_evaluation"}
    for vector in vectors:
        if (
            not isinstance(vector, Mapping)
            or set(vector) != {"name", "size_bytes", "sha256"}
            or not isinstance(vector.get("name"), str)
            or not vector["name"]
            or Path(vector["name"]).name != vector["name"]
        ):
            _fail("steering_vector_manifest_invalid")
        role = f"vector:{vector['name']}"
        expected_roles.add(role)
        payload = evidence.get(role)
        if (
            payload is None
            or len(payload) != vector.get("size_bytes")
            or hashlib.sha256(payload).hexdigest() != vector.get("sha256")
        ):
            _fail("steering_vector_binding_invalid")
    required_evaluation = {
        "schema",
        "model_descriptor_sha256",
        "generation_sha256",
        "verdict",
        "qualified",
        "sample_count",
        "treatment_successes",
        "matched_control_successes",
        "lesion_successes",
        "no_regression",
        "causal_effect_positive",
        "independent_verifier",
        "evaluation_sha256",
    }
    material = dict(evaluation)
    claimed_evaluation = material.pop("evaluation_sha256", None)
    sample_count = evaluation.get("sample_count")
    treatment_successes = evaluation.get("treatment_successes")
    matched_control_successes = evaluation.get("matched_control_successes")
    lesion_successes = evaluation.get("lesion_successes")
    verifier = evaluation.get("independent_verifier")
    if (
        set(evidence) != expected_roles
        or set(evaluation) != required_evaluation
        or evaluation.get("schema") != CAA_EVALUATION_SCHEMA
        or evaluation.get("model_descriptor_sha256") != descriptor_sha256
        or evaluation.get("generation_sha256") != claims["generation_sha256"]
        or evaluation.get("verdict") != "PASS"
        or evaluation.get("qualified") is not True
        or type(sample_count) is not int
        or sample_count < 24
        or type(treatment_successes) is not int
        or not 0 <= treatment_successes <= sample_count
        or type(matched_control_successes) is not int
        or not 0 <= matched_control_successes <= sample_count
        or treatment_successes <= matched_control_successes
        or not isinstance(lesion_successes, Mapping)
        or not lesion_successes
        or any(
            type(successes) is not int
            or not 0 <= successes < treatment_successes
            for successes in lesion_successes.values()
        )
        or evaluation.get("no_regression") is not True
        or evaluation.get("causal_effect_positive") is not True
        or not isinstance(verifier, Mapping)
        or set(verifier) != {"name", "version", "evidence_sha256"}
        or not isinstance(verifier.get("name"), str)
        or not verifier["name"]
        or not isinstance(verifier.get("version"), str)
        or not verifier["version"]
        or not _is_sha(verifier.get("evidence_sha256"))
        or claimed_evaluation != claims["causal_evaluation_sha256"]
        or claimed_evaluation != _sha(material)
    ):
        _fail("steering_causal_evaluation_invalid")


def _validate_expert_retirement(
    evidence: Mapping[str, bytes], claims: Mapping[str, Any], descriptor_sha256: str
) -> None:
    if set(evidence) != {"migration_inventory"}:
        _fail("expert_retirement_evidence_incomplete")
    inventory = _strict_json(evidence["migration_inventory"], role="migration_inventory")
    try:
        inventory = validate_tissue_migration_inventory(inventory)
    except TissueInventoryError as exc:
        raise CortexMigrationAuthorityError("expert_retirement_inventory_invalid") from exc
    families = inventory.get("families")
    expert = next(
        (
            item
            for item in families
            if isinstance(item, Mapping) and item.get("family") == "expert_adapters"
        ),
        None,
    )
    candidate = inventory.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("descriptor_sha256") != descriptor_sha256
        or not isinstance(expert, Mapping)
        or expert.get("outcome") != "retire"
        or inventory.get("inventory_sha256") != claims["inventory_sha256"]
    ):
        _fail("expert_retirement_inventory_not_authoritative")


def _validate_recurrence(
    evidence: Mapping[str, bytes], claims: Mapping[str, Any], descriptor_sha256: str
) -> None:
    if set(evidence) != {"activation", "model_binding"}:
        _fail("recurrence_evidence_incomplete")
    activation = _strict_json(evidence["activation"], role="recurrent_activation")
    binding = _strict_json(evidence["model_binding"], role="recurrent_model_binding")
    material = dict(binding)
    binding_sha256 = material.pop("binding_sha256", None)
    if (
        qualified_activation_errors(activation)
        or activation.get("serving_authority") is not True
        or activation.get("activation_sha256") != claims["activation_sha256"]
        or activation.get("package_id") != claims["package_id"]
        or activation.get("manifest_sha256") != claims["manifest_sha256"]
        or binding.get("schema") != RECURRENT_MODEL_BINDING_SCHEMA
        or binding.get("model_descriptor_sha256") != descriptor_sha256
        or binding.get("package_id") != activation.get("package_id")
        or binding.get("manifest_sha256") != activation.get("manifest_sha256")
        or binding.get("activation_sha256") != activation.get("activation_sha256")
        or binding_sha256 != claims["model_binding_sha256"]
        or binding_sha256 != _sha(material)
    ):
        _fail("recurrence_model_authority_invalid")


_SEMANTIC_VALIDATORS: Final = {
    "persona_crsm": _validate_persona,
    "steering": _validate_steering,
    "expert_adapters": _validate_expert_retirement,
    "recurrence_native": _validate_recurrence,
}


def validate_component_authority(
    value: Mapping[str, Any],
    *,
    component: str,
    descriptor_sha256: str,
    authority_key_path: Path | None = None,
) -> dict[str, Any]:
    """Reopen and semantically validate one signed component authority."""

    required = {
        "schema",
        "component",
        "authority_kind",
        "status",
        "model_descriptor_sha256",
        "custody_root",
        "evidence",
        "claims",
        "issued_at",
        "authority_sha256",
        "authority_hmac_sha256",
    }
    if component not in _COMPONENT_SPECS or not isinstance(value, Mapping):
        _fail(f"migration_component_authority_invalid:{component}")
    if set(value) != required or value.get("schema") != COMPONENT_AUTHORITY_SCHEMA:
        _fail(f"migration_component_authority_schema_invalid:{component}")
    kind, status = _COMPONENT_SPECS[component]
    claims = value.get("claims")
    evidence_raw = value.get("evidence")
    if (
        value.get("component") != component
        or value.get("authority_kind") != kind
        or value.get("status") != status
        or value.get("model_descriptor_sha256") != descriptor_sha256
        or not isinstance(claims, Mapping)
        or set(claims) != _CLAIM_FIELDS[component]
        or any(not _is_sha(item) for key, item in claims.items() if key != "package_id")
        or (
            component == "recurrence_native"
            and (not isinstance(claims.get("package_id"), str) or not claims["package_id"])
        )
        or not isinstance(evidence_raw, Mapping)
        or not evidence_raw
        or not isinstance(value.get("issued_at"), (int, float))
        or isinstance(value.get("issued_at"), bool)
    ):
        _fail(f"migration_component_authority_invalid:{component}")
    material = dict(value)
    claimed_sha256 = material.pop("authority_sha256", None)
    signature = material.pop("authority_hmac_sha256", None)
    observed_sha256 = _sha(material)
    key = _authority_key(authority_key_path)
    expected_signature = hmac.new(
        key, bytes.fromhex(observed_sha256), hashlib.sha256
    ).hexdigest()
    if (
        not _is_sha(claimed_sha256)
        or not _is_sha(signature)
        or not hmac.compare_digest(claimed_sha256, observed_sha256)
        or not hmac.compare_digest(signature, expected_signature)
    ):
        _fail(f"migration_component_authority_signature_invalid:{component}")
    custody_value = value.get("custody_root")
    if not isinstance(custody_value, str) or not custody_value:
        _fail(f"migration_component_custody_invalid:{component}")
    try:
        custody_root = Path(custody_value).expanduser().resolve(strict=True)
        custody_stat = custody_root.lstat()
    except OSError as exc:
        raise CortexMigrationAuthorityError(
            f"migration_component_custody_invalid:{component}"
        ) from exc
    if (
        custody_value != str(custody_root)
        or not stat.S_ISDIR(custody_stat.st_mode)
        or custody_stat.st_uid != os.geteuid()
        or stat.S_IMODE(custody_stat.st_mode) & 0o077
    ):
        _fail(f"migration_component_custody_invalid:{component}")
    evidence: dict[str, bytes] = {}
    normalized_bindings: dict[str, dict[str, Any]] = {}
    for role, binding in sorted(evidence_raw.items()):
        if not isinstance(role, str) or not role:
            _fail(f"migration_component_evidence_role_invalid:{component}")
        normalized, payload = _read_binding(
            binding,
            role=f"migration_component:{component}:{role}",
            custody_root=custody_root,
        )
        normalized_bindings[role] = normalized
        evidence[role] = payload
    if dict(evidence_raw) != normalized_bindings:
        _fail(f"migration_component_evidence_identity_invalid:{component}")
    _SEMANTIC_VALIDATORS[component](evidence, claims, descriptor_sha256)
    return dict(value)


__all__ = [
    "CAA_EVALUATION_SCHEMA",
    "COMPONENT_AUTHORITY_SCHEMA",
    "CortexMigrationAuthorityError",
    "RECURRENT_MODEL_BINDING_SCHEMA",
    "default_authority_key_path",
    "validate_component_authority",
]
