from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import core.learning.semantic_operation_transfer_verification as verification_module
from core.learning.semantic_operation_transfer import SEMANTIC_OPERATION_TRANSFER_SCHEMA
from core.learning.semantic_operation_transfer_verification import (
    SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES,
    verify_semantic_operation_transfer,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _report() -> dict[str, object]:
    arms = {
        name: {"operation_exact": 1, "program_exact": 1}
        for name in (
            "treatment",
            "coefficient_lesion",
            "label_permutation",
            "geometry_only",
            "token_lookup",
        )
    }
    paired = {
        name: {
            "treatment_only": 0,
            "control_only": 0,
            "discordant": 0,
            "one_sided_exact_p": 1.0,
        }
        for name in arms
        if name != "treatment"
    }
    split = {
        "operation_count": 1,
        "program_count": 1,
        "surface_overlap_count": 0,
        "surface_unseen_count": 1,
        "arms": arms,
        "paired_program_tests": paired,
    }
    directions = {
        name: {
            "source_training_operation_count": 4,
            "source_training_surface_count": 4,
            "splits": {"validation": split, "test": split},
        }
        for name in (
            "arithmetic_to_fork_join",
            "arithmetic_to_sequence_binary",
            "fork_join_to_arithmetic",
            "fork_join_to_sequence_binary",
            "sequence_binary_to_arithmetic",
            "sequence_binary_to_fork_join",
        )
    }
    body = {
        "schema": SEMANTIC_OPERATION_TRANSFER_SCHEMA,
        "feature_manifest_sha256s": {
            "arithmetic": "a" * 64,
            "fork_join": "b" * 64,
            "sequence_binary": "c" * 64,
        },
        "representation_compatibility": {"compatible": True},
        "representation_views": {
            "absolute_span_mean": {
                "counterfactual_target_batch_required": False,
                "directions": directions,
            },
            "counterfactual_centered_span_mean": {
                "counterfactual_target_batch_required": True,
                "directions": directions,
            },
        },
        "gold_operation_spans_available": True,
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": False,
        "family_identity_available_to_classifier": False,
        "geometry_available_to_treatment_classifier": False,
        "serving_authority": False,
        "claim_boundary": "diagnostic",
    }
    return {**body, "report_sha256": _sha(body)}


def _bundles() -> dict[str, SimpleNamespace]:
    return {
        "arithmetic": SimpleNamespace(
            manifest={"manifest_sha256": "a" * 64}, examples=(1,)
        ),
        "fork_join": SimpleNamespace(
            manifest={"manifest_sha256": "b" * 64}, examples=(1,)
        ),
        "sequence_binary": SimpleNamespace(
            manifest={"manifest_sha256": "c" * 64}, examples=(1,)
        ),
    }


def _sources() -> dict[str, str]:
    return {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES
    }


def test_operation_transfer_verifier_requires_exact_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    monkeypatch.setattr(
        verification_module,
        "run_semantic_operation_transfer",
        lambda bundles: report,
    )

    verification = verify_semantic_operation_transfer(
        _bundles(), stored_report=report, source_sha256s=_sources()
    )

    assert verification["verified"] is True
    assert verification["raw_records_reloaded"] == 3
    assert verification["directed_transfer_count"] == 12
    assert verification["frozen_replay_exact"] is True


def test_operation_transfer_verifier_rejects_control_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    del report["representation_views"]["absolute_span_mean"]["directions"][
        "arithmetic_to_fork_join"
    ]["splits"]["test"]["arms"]["geometry_only"]
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = _sha(body)
    monkeypatch.setattr(
        verification_module,
        "run_semantic_operation_transfer",
        lambda bundles: report,
    )

    with pytest.raises(ValueError, match="controls differ"):
        verify_semantic_operation_transfer(
            _bundles(), stored_report=report, source_sha256s=_sources()
        )
