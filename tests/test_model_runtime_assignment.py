from __future__ import annotations

import hashlib

import pytest

from core.runtime.model_runtime_assignment import (
    ModelRuntimeAssignment,
    issue_unqualified_model_runtime_assignment,
    normalize_model_runtime_purpose,
)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_assignment_round_trip_is_self_authenticating(tmp_path) -> None:
    assignment = ModelRuntimeAssignment.issue(
        model_path=tmp_path,
        artifact_identity=_identity("resident"),
        artifact_identity_kind="model_descriptor_sha256",
        artifact_identity_exact=True,
        role="cortex",
        purpose="serve",
        authority_source="active_cortex_pointer",
        evidence_receipt_id="pointer-123",
    )

    restored = ModelRuntimeAssignment.from_dict(assignment.to_dict())

    assert restored == assignment
    assert restored.lane == "cortex"
    assert restored.qos == "guaranteed"


def test_assignment_rejects_role_or_qos_tampering(tmp_path) -> None:
    assignment = ModelRuntimeAssignment.issue(
        model_path=tmp_path,
        artifact_identity=_identity("specialist"),
        artifact_identity_kind="artifact_profile_fingerprint",
        artifact_identity_exact=False,
        role="solver",
        purpose="serve",
        authority_source="model_registry",
    )
    tampered = assignment.to_dict()
    tampered["qos"] = "guaranteed"

    with pytest.raises(ValueError, match="serving_policy|digest"):
        ModelRuntimeAssignment.from_dict(tampered)


def test_training_purpose_cannot_inherit_cortex_authority(tmp_path) -> None:
    assignment = ModelRuntimeAssignment.issue(
        model_path=tmp_path,
        artifact_identity=_identity("candidate"),
        artifact_identity_kind="canonical_locator_sha256",
        artifact_identity_exact=False,
        role="cortex",
        purpose="train",
        authority_source="declared_model_process",
    )

    assert assignment.role == "trainer"
    assert assignment.lane == "trainer"
    assert assignment.qos == "best_effort"


def test_assignment_refuses_different_path_or_purpose(tmp_path) -> None:
    assignment = ModelRuntimeAssignment.issue(
        model_path=tmp_path / "one",
        artifact_identity=_identity("one"),
        artifact_identity_kind="canonical_locator_sha256",
        artifact_identity_exact=False,
        role="auxiliary",
        purpose="serve",
        authority_source="unassigned_artifact",
    )

    with pytest.raises(ValueError, match="model_path_mismatch"):
        assignment.assert_bound_to(model_path=tmp_path / "two")
    with pytest.raises(ValueError, match="purpose_mismatch"):
        assignment.assert_bound_to(model_path=tmp_path / "one", purpose="benchmark")


def test_exact_identity_requires_full_descriptor_kind(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactness"):
        ModelRuntimeAssignment.issue(
            model_path=tmp_path,
            artifact_identity=_identity("profile"),
            artifact_identity_kind="artifact_profile_fingerprint",
            artifact_identity_exact=True,
            role="auxiliary",
            purpose="serve",
            authority_source="model_registry",
        )


@pytest.mark.parametrize(
    ("purpose", "expected"),
    [
        ("training", "train"),
        ("train_frozen_controller", "train"),
        ("evaluation", "benchmark"),
        ("benchmark_evaluation", "benchmark"),
        ("proof", "benchmark"),
        ("measurement", "benchmark"),
        ("fused-model merge", "fuse"),
        ("compound", "compound"),
        ("user_reply", "serve"),
    ],
)
def test_purpose_aliases_normalize_to_stable_assignment_class(
    purpose: str,
    expected: str,
) -> None:
    assert normalize_model_runtime_purpose(purpose) == expected


def test_unqualified_assignment_never_infers_authority_from_name(tmp_path) -> None:
    assignment = issue_unqualified_model_runtime_assignment(
        model_path=tmp_path / "Massive-999B-Cortex-Solver",
        purpose="serve",
        authority_source="external_process_discovery",
    )

    assert assignment.role == "auxiliary"
    assert assignment.lane == "auxiliary"
    assert assignment.qos == "best_effort"


def test_unqualified_evaluation_alias_is_a_trainer(tmp_path) -> None:
    assignment = issue_unqualified_model_runtime_assignment(
        model_path=tmp_path / "candidate",
        purpose="evaluation",
        authority_source="standalone_model_lane",
    )

    assert assignment.purpose == "benchmark"
    assert assignment.role == "trainer"
    assert assignment.lane == "trainer"
