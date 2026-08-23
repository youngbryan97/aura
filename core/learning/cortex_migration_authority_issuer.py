"""Component-owned issuance for cortex migration authority."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.governance_context import local_internal_governed_scope
from core.learning.candidate_cortex_fusion import (
    load_and_validate_fusion_plan,
    validate_fusion_receipt,
)
from core.learning.cortex_migration_authority import (
    COMPONENT_AUTHORITY_SCHEMA,
    RECURRENT_MODEL_BINDING_SCHEMA,
    default_authority_key_path,
    validate_component_authority,
    validate_component_evidence,
)
from core.learning.model_tissue_migration_inventory import (
    TissueInventoryError,
    validate_tissue_migration_inventory,
)
from core.runtime.file_read_gateway import StableFileReadError, read_stable_bytes
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.runtime.state_ownership import state_root

MAX_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
_COMPONENT_SPECS: Final = {
    "persona_crsm": ("fused_persona_crsm", "qualified"),
    "steering": ("caa_model_bound", "qualified"),
    "expert_adapters": ("retirement_inventory", "retired"),
    "recurrence_native": ("qualified_recurrent_activation", "qualified"),
}


class CortexMigrationAuthorityIssuanceError(ValueError):
    """A producer could not prove or retain its migration authority."""


def _fail(code: str) -> None:
    raise CortexMigrationAuthorityIssuanceError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CortexMigrationAuthorityIssuanceError(
            "migration_authority_not_canonical"
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                _fail(f"{role}_duplicate_key")
            value[key] = item
        return value

    try:
        payload = read_stable_bytes(path.expanduser().absolute(), max_bytes=MAX_DOCUMENT_BYTES)
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(f"{role}_non_finite"),
        )
    except (OSError, StableFileReadError, UnicodeDecodeError, ValueError) as exc:
        raise CortexMigrationAuthorityIssuanceError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value, payload


def _provision_key() -> bytes:
    with local_internal_governed_scope(
        "cortex_migration_authority.issue_key", domain="file_write"
    ):
        return get_file_write_gateway().provision_private_bytes(
            default_authority_key_path(),
            secrets.token_bytes(32),
            expected_size=32,
            mode=0o600,
            source="cortex_migration_authority.issue_key",
        )


def _retain_and_sign(
    *,
    component: str,
    descriptor_sha256: str,
    claims: Mapping[str, Any],
    evidence: Mapping[str, bytes],
    issued_at: float | None,
    custody_base: Path | None,
) -> dict[str, Any]:
    validate_component_evidence(
        component=component,
        evidence=evidence,
        claims=claims,
        descriptor_sha256=descriptor_sha256,
    )
    fingerprint = _sha(
        {
            "component": component,
            "descriptor_sha256": descriptor_sha256,
            "claims": dict(claims),
            "evidence": {
                role: hashlib.sha256(payload).hexdigest()
                for role, payload in sorted(evidence.items())
            },
        }
    )
    base = custody_base or (
        state_root() / "private/cortex-upgrade/migration-evidence"
    )
    relative_root = Path(descriptor_sha256[:16]) / component / fingerprint
    try:
        custody = DirectoryCustody.acquire(base, create=True, private=True)
    except SecurePathCustodyError as exc:
        raise CortexMigrationAuthorityIssuanceError(
            "migration_component_custody_unavailable"
        ) from exc
    with custody:
        try:
            custody_root = custody.ensure_directory(relative_root).resolve(strict=True)
        except (OSError, SecurePathCustodyError) as exc:
            raise CortexMigrationAuthorityIssuanceError(
                "migration_component_custody_unavailable"
            ) from exc
        bindings: dict[str, dict[str, Any]] = {}
        for index, (role, payload) in enumerate(sorted(evidence.items())):
            role_id = hashlib.sha256(role.encode("utf-8")).hexdigest()[:12]
            relative_target = relative_root / f"{index:02d}-{role_id}.evidence"
            target = custody.path / relative_target
            try:
                custody.write_bytes_once(relative_target, payload, mode=0o600)
            except SecurePathCustodyError as exc:
                raise CortexMigrationAuthorityIssuanceError(
                    f"migration_component_evidence_publish_failed:{component}:{role}"
                ) from exc
            retained = read_stable_bytes(target, max_bytes=MAX_DOCUMENT_BYTES)
            if retained != payload:
                _fail(f"migration_component_evidence_collision:{component}:{role}")
            bindings[role] = {
                "path": str(target.resolve(strict=True)),
                "size_bytes": len(retained),
                "sha256": hashlib.sha256(retained).hexdigest(),
            }

        kind, status = _COMPONENT_SPECS[component]
        body = {
            "schema": COMPONENT_AUTHORITY_SCHEMA,
            "component": component,
            "authority_kind": kind,
            "status": status,
            "model_descriptor_sha256": descriptor_sha256,
            "custody_root": str(custody_root),
            "evidence": bindings,
            "claims": dict(claims),
            "issued_at": float(time.time() if issued_at is None else issued_at),
        }
        digest = _sha(body)
        key = _provision_key()
        authority = {
            **body,
            "authority_sha256": digest,
            "authority_hmac_sha256": hmac.new(
                key, bytes.fromhex(digest), hashlib.sha256
            ).hexdigest(),
        }
        validated = validate_component_authority(
            authority,
            component=component,
            descriptor_sha256=descriptor_sha256,
        )
        authority_payload = _canonical(validated)
        relative_authority = relative_root / f"authority-{digest}.json"
        authority_path = custody.path / relative_authority
        try:
            custody.write_bytes_once(relative_authority, authority_payload, mode=0o600)
        except SecurePathCustodyError as exc:
            raise CortexMigrationAuthorityIssuanceError(
                f"migration_component_authority_publish_failed:{component}"
            ) from exc
        if (
            read_stable_bytes(authority_path, max_bytes=MAX_DOCUMENT_BYTES)
            != authority_payload
        ):
            _fail(f"migration_component_authority_collision:{component}")
        return validated


def issue_persona_crsm_authority(
    *,
    fusion_plan_path: Path,
    fusion_receipt_path: Path,
    journal_key_path: Path,
    descriptor_sha256: str,
    custody_base: Path | None = None,
    issued_at: float | None = None,
) -> dict[str, Any]:
    """Issue authority only for a fully validated fused persona checkpoint."""

    plan = load_and_validate_fusion_plan(
        fusion_plan_path,
        journal_key_path=journal_key_path,
        verify_full_model=True,
    )
    receipt, receipt_payload = _strict_json(
        fusion_receipt_path, role="fusion_receipt"
    )
    validate_fusion_receipt(plan, receipt, verify_full_model=True)
    if receipt.get("descriptor_sha256") != descriptor_sha256:
        _fail("persona_crsm_model_identity_mismatch")
    plan_payload = read_stable_bytes(
        fusion_plan_path.expanduser().absolute(), max_bytes=MAX_DOCUMENT_BYTES
    )
    return _retain_and_sign(
        component="persona_crsm",
        descriptor_sha256=descriptor_sha256,
        claims={
            "fusion_plan_sha256": plan["fusion_plan_sha256"],
            "fusion_receipt_sha256": receipt["receipt_sha256"],
        },
        evidence={"fusion_plan": plan_payload, "fusion_receipt": receipt_payload},
        issued_at=issued_at,
        custody_base=custody_base,
    )


def issue_steering_authority(
    *,
    metadata_path: Path,
    causal_evaluation_path: Path,
    independent_evidence_path: Path,
    descriptor_sha256: str,
    custody_base: Path | None = None,
    issued_at: float | None = None,
) -> dict[str, Any]:
    """Issue authority for a model-bound, causally qualified CAA generation."""

    metadata, metadata_payload = _strict_json(metadata_path, role="steering_metadata")
    evaluation, evaluation_payload = _strict_json(
        causal_evaluation_path, role="steering_causal_evaluation"
    )
    extraction = metadata.get("extraction_contract")
    vectors = metadata.get("vector_files")
    if not isinstance(extraction, Mapping) or not isinstance(vectors, list):
        _fail("steering_generation_invalid")
    evidence: dict[str, bytes] = {
        "metadata": metadata_payload,
        "causal_evaluation": evaluation_payload,
        "independent_verifier": read_stable_bytes(
            independent_evidence_path, max_bytes=MAX_DOCUMENT_BYTES
        ),
    }
    for vector in vectors:
        if not isinstance(vector, Mapping) or not isinstance(vector.get("name"), str):
            _fail("steering_vector_manifest_invalid")
        vector_path = metadata_path.parent / vector["name"]
        evidence[f"vector:{vector['name']}"] = read_stable_bytes(
            vector_path, max_bytes=MAX_DOCUMENT_BYTES
        )
    claims = {
        "generation_sha256": metadata.get("generation_sha256"),
        "extraction_contract_sha256": extraction.get("extraction_contract_sha256"),
        "causal_evaluation_sha256": evaluation.get("evaluation_sha256"),
    }
    return _retain_and_sign(
        component="steering",
        descriptor_sha256=descriptor_sha256,
        claims=claims,
        evidence=evidence,
        issued_at=issued_at,
        custody_base=custody_base,
    )


def issue_expert_retirement_authority(
    *,
    inventory_path: Path,
    descriptor_sha256: str,
    custody_base: Path | None = None,
    issued_at: float | None = None,
) -> dict[str, Any]:
    """Issue retirement authority from the validated migration inventory."""

    inventory, payload = _strict_json(inventory_path, role="migration_inventory")
    try:
        inventory = validate_tissue_migration_inventory(inventory)
    except TissueInventoryError as exc:
        raise CortexMigrationAuthorityIssuanceError(
            "expert_retirement_inventory_invalid"
        ) from exc
    return _retain_and_sign(
        component="expert_adapters",
        descriptor_sha256=descriptor_sha256,
        claims={"inventory_sha256": inventory["inventory_sha256"]},
        evidence={"migration_inventory": payload},
        issued_at=issued_at,
        custody_base=custody_base,
    )


def issue_recurrence_authority(
    *,
    activation_path: Path,
    descriptor_sha256: str,
    custody_base: Path | None = None,
    issued_at: float | None = None,
) -> dict[str, Any]:
    """Issue authority for qualified recurrent tissue on one model identity."""

    activation, activation_payload = _strict_json(
        activation_path, role="recurrent_activation"
    )
    binding_body = {
        "schema": RECURRENT_MODEL_BINDING_SCHEMA,
        "model_descriptor_sha256": descriptor_sha256,
        "package_id": activation.get("package_id"),
        "manifest_sha256": activation.get("manifest_sha256"),
        "activation_sha256": activation.get("activation_sha256"),
    }
    binding = {**binding_body, "binding_sha256": _sha(binding_body)}
    return _retain_and_sign(
        component="recurrence_native",
        descriptor_sha256=descriptor_sha256,
        claims={
            "activation_sha256": activation.get("activation_sha256"),
            "package_id": activation.get("package_id"),
            "manifest_sha256": activation.get("manifest_sha256"),
            "model_binding_sha256": binding["binding_sha256"],
        },
        evidence={
            "activation": activation_payload,
            "model_binding": _canonical(binding),
        },
        issued_at=issued_at,
        custody_base=custody_base,
    )


__all__ = [
    "CortexMigrationAuthorityIssuanceError",
    "issue_expert_retirement_authority",
    "issue_persona_crsm_authority",
    "issue_recurrence_authority",
    "issue_steering_authority",
]
