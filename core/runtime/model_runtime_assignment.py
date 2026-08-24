"""Immutable semantic and resource assignment for one model runtime.

Model paths identify bytes. They do not identify authority. This contract is
issued once by the registry (or by a trusted workload declaration), carried by
the client that loads the artifact, and persisted with every durable lane
claim. Replaying a claim therefore restores the original role and QoS instead
of re-interpreting a path under newer configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_RUNTIME_ASSIGNMENT_SCHEMA = "aura.model_runtime_assignment.v1"

_SERVING_ROLE_POLICY: dict[str, tuple[str, str]] = {
    "cortex": ("cortex", "guaranteed"),
    "solver": ("solver", "burstable"),
    "brainstem": ("brainstem", "burstable"),
    "reflex": ("reflex", "burstable"),
    "auxiliary": ("auxiliary", "best_effort"),
}
_WORKLOAD_PURPOSES = frozenset({"train", "compound", "fuse", "benchmark"})
_IDENTITY_KINDS = frozenset(
    {
        "model_descriptor_sha256",
        "artifact_profile_fingerprint",
        "canonical_locator_sha256",
    }
)


def canonical_model_locator(model_path: str | os.PathLike[str]) -> str:
    """Normalize a local path or repository locator without requiring I/O."""

    raw = str(model_path or "").strip()
    if not raw:
        raise ValueError("model_runtime_assignment_model_path_missing")
    expanded = os.path.expanduser(raw)
    if os.path.isabs(expanded):
        return os.path.realpath(expanded)
    candidate = Path(expanded)
    if candidate.exists():
        return os.path.realpath(str(candidate))
    return raw.rstrip("/")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("model_runtime_assignment_not_canonical_json") from exc


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def normalize_model_runtime_purpose(purpose: object) -> str:
    """Collapse caller vocabulary into one stable serving/workload purpose.

    Model tools historically used ``train``/``training`` and
    ``benchmark``/``evaluation``/``proof`` interchangeably. Exact string
    matching therefore let some offline jobs acquire serving semantics. The
    normalized value is part of the signed assignment identity, so a replay
    cannot change class merely because a caller or configuration later uses a
    different alias.
    """

    words = frozenset(
        word
        for word in re.sub(r"[^a-z0-9]+", " ", str(purpose or "serve").lower()).split()
        if word
    )
    if words & {"compound", "compounding"}:
        return "compound"
    if words & {"fuse", "fusion", "merge", "merging"}:
        return "fuse"
    if words & {
        "train",
        "training",
        "finetune",
        "finetuning",
        "lora",
        "distill",
        "distillation",
    }:
        return "train"
    if words & {
        "ablation",
        "benchmark",
        "canary",
        "eval",
        "evaluate",
        "evaluation",
        "measure",
        "measurement",
        "proof",
        "replay",
        "validate",
        "validation",
        "verification",
        "verify",
    }:
        return "benchmark"
    return "serve"


@dataclass(frozen=True, slots=True)
class ModelRuntimeAssignment:
    """One immutable role/QoS decision bound to a concrete artifact identity."""

    model_path: str
    artifact_identity: str
    artifact_identity_kind: str
    artifact_identity_exact: bool
    role: str
    lane: str
    qos: str
    purpose: str
    authority_source: str
    evidence_receipt_id: str
    assignment_sha256: str
    schema: str = MODEL_RUNTIME_ASSIGNMENT_SCHEMA

    @classmethod
    def issue(
        cls,
        *,
        model_path: str | os.PathLike[str],
        artifact_identity: str,
        artifact_identity_kind: str,
        artifact_identity_exact: bool,
        role: str,
        purpose: str,
        authority_source: str,
        evidence_receipt_id: str = "",
    ) -> ModelRuntimeAssignment:
        normalized_purpose = normalize_model_runtime_purpose(purpose)
        normalized_role = str(role or "auxiliary").strip().lower()
        if normalized_purpose in _WORKLOAD_PURPOSES:
            normalized_role = "trainer"
            lane, qos = "trainer", "best_effort"
        else:
            normalized_purpose = "serve"
            try:
                lane, qos = _SERVING_ROLE_POLICY[normalized_role]
            except KeyError as exc:
                raise ValueError("model_runtime_assignment_role_invalid") from exc
        material = {
            "schema": MODEL_RUNTIME_ASSIGNMENT_SCHEMA,
            "model_path": canonical_model_locator(model_path),
            "artifact_identity": str(artifact_identity or "").strip().lower(),
            "artifact_identity_kind": str(artifact_identity_kind or "").strip(),
            "artifact_identity_exact": bool(artifact_identity_exact),
            "role": normalized_role,
            "lane": lane,
            "qos": qos,
            "purpose": normalized_purpose,
            "authority_source": str(authority_source or "").strip(),
            "evidence_receipt_id": str(evidence_receipt_id or "").strip(),
        }
        cls._validate_material(material)
        return cls(**material, assignment_sha256=_digest(material))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelRuntimeAssignment:
        if not isinstance(value, Mapping):
            raise ValueError("model_runtime_assignment_not_mapping")
        required = {
            "schema",
            "model_path",
            "artifact_identity",
            "artifact_identity_kind",
            "artifact_identity_exact",
            "role",
            "lane",
            "qos",
            "purpose",
            "authority_source",
            "evidence_receipt_id",
            "assignment_sha256",
        }
        if set(value) != required:
            raise ValueError("model_runtime_assignment_schema_invalid")
        material = {key: value[key] for key in required if key != "assignment_sha256"}
        cls._validate_material(material)
        claimed = str(value.get("assignment_sha256") or "")
        if not _is_sha256(claimed) or claimed != _digest(material):
            raise ValueError("model_runtime_assignment_digest_invalid")
        return cls(
            model_path=str(material["model_path"]),
            artifact_identity=str(material["artifact_identity"]),
            artifact_identity_kind=str(material["artifact_identity_kind"]),
            artifact_identity_exact=bool(material["artifact_identity_exact"]),
            role=str(material["role"]),
            lane=str(material["lane"]),
            qos=str(material["qos"]),
            purpose=str(material["purpose"]),
            authority_source=str(material["authority_source"]),
            evidence_receipt_id=str(material["evidence_receipt_id"]),
            assignment_sha256=claimed,
            schema=str(material["schema"]),
        )

    @staticmethod
    def _validate_material(material: Mapping[str, Any]) -> None:
        if material.get("schema") != MODEL_RUNTIME_ASSIGNMENT_SCHEMA:
            raise ValueError("model_runtime_assignment_schema_invalid")
        if canonical_model_locator(str(material.get("model_path") or "")) != material.get(
            "model_path"
        ):
            raise ValueError("model_runtime_assignment_path_not_canonical")
        identity = str(material.get("artifact_identity") or "")
        if not _is_sha256(identity):
            raise ValueError("model_runtime_assignment_artifact_identity_invalid")
        identity_kind = str(material.get("artifact_identity_kind") or "")
        if identity_kind not in _IDENTITY_KINDS:
            raise ValueError("model_runtime_assignment_identity_kind_invalid")
        exact = bool(material.get("artifact_identity_exact"))
        if exact is not (identity_kind == "model_descriptor_sha256"):
            raise ValueError("model_runtime_assignment_exactness_invalid")
        purpose = str(material.get("purpose") or "")
        role = str(material.get("role") or "")
        lane = str(material.get("lane") or "")
        qos = str(material.get("qos") or "")
        if purpose in _WORKLOAD_PURPOSES:
            if (role, lane, qos) != ("trainer", "trainer", "best_effort"):
                raise ValueError("model_runtime_assignment_workload_policy_invalid")
        elif purpose == "serve":
            if role not in _SERVING_ROLE_POLICY or _SERVING_ROLE_POLICY[role] != (
                lane,
                qos,
            ):
                raise ValueError("model_runtime_assignment_serving_policy_invalid")
        else:
            raise ValueError("model_runtime_assignment_purpose_invalid")
        if not str(material.get("authority_source") or "").strip():
            raise ValueError("model_runtime_assignment_authority_missing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_path": self.model_path,
            "artifact_identity": self.artifact_identity,
            "artifact_identity_kind": self.artifact_identity_kind,
            "artifact_identity_exact": self.artifact_identity_exact,
            "role": self.role,
            "lane": self.lane,
            "qos": self.qos,
            "purpose": self.purpose,
            "authority_source": self.authority_source,
            "evidence_receipt_id": self.evidence_receipt_id,
            "assignment_sha256": self.assignment_sha256,
        }

    def assert_bound_to(
        self,
        *,
        model_path: str | os.PathLike[str],
        purpose: str | None = None,
    ) -> None:
        if canonical_model_locator(model_path) != self.model_path:
            raise ValueError("model_runtime_assignment_model_path_mismatch")
        if purpose is not None:
            normalized = normalize_model_runtime_purpose(purpose)
            if normalized != self.purpose:
                raise ValueError("model_runtime_assignment_purpose_mismatch")


def locator_identity(model_path: str | os.PathLike[str]) -> str:
    """Fail-honest identity when no measured artifact descriptor is available."""

    return hashlib.sha256(canonical_model_locator(model_path).encode("utf-8")).hexdigest()


def issue_unqualified_model_runtime_assignment(
    *,
    model_path: str | os.PathLike[str],
    purpose: object,
    authority_source: str,
) -> ModelRuntimeAssignment:
    """Issue conservative ownership for an artifact with no registry authority.

    An explicit workload remains a best-effort trainer. An unregistered
    serving artifact remains auxiliary. Neither a suggestive filename nor a
    large parameter count can promote it to Cortex or Solver.
    """

    locator = canonical_model_locator(model_path)
    return ModelRuntimeAssignment.issue(
        model_path=locator,
        artifact_identity=locator_identity(locator),
        artifact_identity_kind="canonical_locator_sha256",
        artifact_identity_exact=False,
        role="auxiliary",
        purpose=normalize_model_runtime_purpose(purpose),
        authority_source=authority_source,
    )


__all__ = [
    "MODEL_RUNTIME_ASSIGNMENT_SCHEMA",
    "ModelRuntimeAssignment",
    "canonical_model_locator",
    "issue_unqualified_model_runtime_assignment",
    "locator_identity",
    "normalize_model_runtime_purpose",
]
