"""Cross-session identity proof for frozen semantic transducers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from core.brain.llm.latent_cortex.runtime_identity import (
    worker_representation_basis,
)
from core.learning.semantic_program_transducer import (
    SemanticProgramTransducer,
    SemanticTransducerTrainingExample,
)

SEMANTIC_REPRESENTATION_COMPATIBILITY_SCHEMA: Final = (
    "aura.semantic_representation_compatibility.v1"
)
_SESSION_ONLY_FIELDS: Final = (
    "worker_action_capture_identity",
    "worker_boot_id",
    "worker_pid",
)


class SemanticRepresentationCompatibilityError(RuntimeError):
    """Two feature bundles do not prove the same neural function."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_bases(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("model_bases")
    if not isinstance(entries, list) or not entries:
        raise SemanticRepresentationCompatibilityError(
            "semantic feature manifest has no model bases"
        )
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"sha256", "receipt"}:
            raise SemanticRepresentationCompatibilityError(
                "semantic feature manifest model basis is malformed"
            )
        basis_hash = entry["sha256"]
        receipt = entry["receipt"]
        if (
            not _is_sha256(basis_hash)
            or not isinstance(receipt, Mapping)
            or _sha(receipt) != basis_hash
            or receipt.get("worker_stack_identity_gaps") != []
        ):
            raise SemanticRepresentationCompatibilityError(
                "semantic feature manifest model basis is incomplete"
            )
        result[str(basis_hash)] = json.loads(_canonical_bytes(receipt))
    if len(result) != len(entries):
        raise SemanticRepresentationCompatibilityError(
            "semantic feature manifest repeats a model basis"
        )
    return result


def establish_semantic_representation_compatibility(
    *,
    model: SemanticProgramTransducer,
    training_manifest: Mapping[str, Any],
    replication_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that fresh hidden states came from the trained neural function."""

    training_bases = _manifest_bases(training_manifest)
    replication_bases = _manifest_bases(replication_manifest)
    if set(training_bases) != {model.model_basis_sha256}:
        raise SemanticRepresentationCompatibilityError(
            "frozen transducer does not match its training feature basis"
        )
    training_representation = worker_representation_basis(
        training_bases[model.model_basis_sha256]
    )
    representation_sha256 = _sha(training_representation)
    replication_representation_sha256s = {
        _sha(worker_representation_basis(receipt))
        for receipt in replication_bases.values()
    }
    if replication_representation_sha256s != {representation_sha256}:
        raise SemanticRepresentationCompatibilityError(
            "fresh cohort neural function differs from the trained representation"
        )
    training_manifest_sha256 = training_manifest.get("manifest_sha256")
    replication_manifest_sha256 = replication_manifest.get("manifest_sha256")
    if (
        not _is_sha256(training_manifest_sha256)
        or not _is_sha256(replication_manifest_sha256)
        or training_manifest_sha256 == replication_manifest_sha256
    ):
        raise SemanticRepresentationCompatibilityError(
            "semantic replication manifest identities are invalid"
        )
    body = {
        "schema": SEMANTIC_REPRESENTATION_COMPATIBILITY_SCHEMA,
        "transducer_receipt_sha256": model.receipt_sha256,
        "coefficient_sha256": model.training_receipt["coefficient_sha256"],
        "training_feature_manifest_sha256": training_manifest_sha256,
        "replication_feature_manifest_sha256": replication_manifest_sha256,
        "training_session_basis_sha256": model.model_basis_sha256,
        "replication_session_basis_sha256s": sorted(replication_bases),
        "representation_basis_sha256": representation_sha256,
        "session_only_fields": list(_SESSION_ONLY_FIELDS),
        "coefficients_changed": False,
        "hidden_states_changed": False,
        "serving_authority": False,
    }
    return {**body, "receipt_sha256": _sha(body)}


def bind_examples_to_compatible_training_session(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    compatibility: Mapping[str, Any],
) -> tuple[SemanticTransducerTrainingExample, ...]:
    """Apply a proven session substitution without changing measured features."""

    body = {key: value for key, value in compatibility.items() if key != "receipt_sha256"}
    if (
        compatibility.get("schema")
        != SEMANTIC_REPRESENTATION_COMPATIBILITY_SCHEMA
        or compatibility.get("receipt_sha256") != _sha(body)
        or compatibility.get("coefficients_changed") is not False
        or compatibility.get("hidden_states_changed") is not False
        or compatibility.get("serving_authority") is not False
    ):
        raise SemanticRepresentationCompatibilityError(
            "semantic representation compatibility receipt is invalid"
        )
    source_bases = set(compatibility.get("replication_session_basis_sha256s", ()))
    target_basis = compatibility.get("training_session_basis_sha256")
    if (
        not source_bases
        or not _is_sha256(target_basis)
        or any(
            item.ir.model_basis_receipt_sha256 not in source_bases
            for item in examples
        )
    ):
        raise SemanticRepresentationCompatibilityError(
            "semantic examples are outside the compatible fresh sessions"
        )
    return tuple(
        replace(
            item,
            ir=replace(item.ir, model_basis_receipt_sha256=target_basis),
        )
        for item in examples
    )


__all__ = [
    "SEMANTIC_REPRESENTATION_COMPATIBILITY_SCHEMA",
    "SemanticRepresentationCompatibilityError",
    "bind_examples_to_compatible_training_session",
    "establish_semantic_representation_compatibility",
]
