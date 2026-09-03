from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.learning.semantic_program_compositional_verification as verification
from core.learning.semantic_program_compositional_verification import (
    COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES,
    CompositionalReplicationCohort,
    SemanticCohortInventory,
    verify_compositional_semantic_replications,
)


def _inventory(prefix: str) -> SemanticCohortInventory:
    return SemanticCohortInventory(
        manifest_sha256=(prefix * 64)[:64],
        exact_model_path="/models/aura-27b",
        tokenizer_identity_sha256="a" * 64,
        example_count=8,
        example_ids=tuple(f"{prefix}-example-{index}" for index in range(8)),
        held_source_text_sha256s=tuple(
            f"{index + 1:064x}" for index in range(5)
        ),
        worker_stack_identity_gaps=(),
    )


def _row(identity: str, *, correct: bool) -> dict[str, object]:
    return {
        "source_text_sha256": identity,
        "accepted": correct,
        "program_exact": correct,
        "operation_exact": correct,
        "argument_exact": correct,
        "arity_exact": correct,
        "step_count_exact": correct,
        "geometry_exact": correct,
        "input_span_exact": correct,
        "answer_emitted": correct,
        "answer_exact": correct,
    }


def _arm(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        field: sum(row[field] is True for row in rows)
        for field in verification._COUNT_FIELDS
    }
    return {"total": len(rows), **counts, "rows": rows}


def _report(inventory: SemanticCohortInventory, lesion: str) -> dict[str, object]:
    ids = list(inventory.held_source_text_sha256s)
    treatment_rows = [_row(identity, correct=True) for identity in ids]
    lesion_rows = [_row(identity, correct=False) for identity in ids]
    body = {
        "schema": "aura.semantic_program_compositional_lesions.v1",
        "transducer_receipt_sha256": "b" * 64,
        "example_ids_sha256": "c" * 64,
        "evaluated_arms": ["treatment", lesion],
        "arms": {
            "treatment": {
                "validation": _arm(treatment_rows[:3]),
                "test": _arm(treatment_rows[3:]),
            },
            lesion: {
                "validation": _arm(lesion_rows[:3]),
                "test": _arm(lesion_rows[3:]),
            },
        },
        "fit_or_refit_calls": 0,
        "expected_answers_available_to_decode": False,
        "serving_authority": False,
    }
    return {**body, "report_sha256": verification._sha(body)}


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    model_payload = {"schema": "test-model"}
    model = SimpleNamespace(
        receipt_sha256="b" * 64,
        model_basis_sha256="d" * 64,
        training_receipt={
            "correctness_authority": False,
            "expected_answers_available": False,
            "verifier_traces_available": False,
            "generated_compiler_text_available": False,
            "family_router_present": False,
        },
        to_dict=lambda: model_payload,
    )
    monkeypatch.setattr(
        verification,
        "compositional_semantic_program_transducer_from_dict",
        lambda payload: model,
    )
    source = {"arithmetic": _inventory("1"), "fork_join": _inventory("2")}
    fresh = {"arithmetic": _inventory("3"), "fork_join": _inventory("4")}
    cohorts = [
        CompositionalReplicationCohort(
            family="arithmetic",
            source=source["arithmetic"],
            fresh=fresh["arithmetic"],
            lesion_arm="relation_tissue_lesion",
            report=_report(fresh["arithmetic"], "relation_tissue_lesion"),
            transfer_kind="family_withheld",
            evaluation_source_commit="e" * 40,
        ),
        CompositionalReplicationCohort(
            family="fork_join",
            source=source["fork_join"],
            fresh=fresh["fork_join"],
            lesion_arm="argument_proposal_lesion",
            report=_report(fresh["fork_join"], "argument_proposal_lesion"),
            transfer_kind="disjoint_seed",
            evaluation_source_commit="f" * 40,
        ),
    ]
    source_body = {
        "schema": "aura.semantic_program_compositional_leave_family_out.v1",
        "feature_manifest_sha256s": {
            family: inventory.manifest_sha256 for family, inventory in source.items()
        },
        "transducer_receipt_sha256": model.receipt_sha256,
        "model_basis_sha256": model.model_basis_sha256,
        "expected_answers_available_to_training": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
    }
    return {
        "source_manifest_sha256s": {
            family: inventory.manifest_sha256 for family, inventory in source.items()
        },
        "trained_model_payload": model_payload,
        "source_report": {
            **source_body,
            "report_sha256": verification._sha(source_body),
        },
        "cohorts": cohorts,
        "source_sha256s": {
            path: "9" * 64
            for path in COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES
        },
        "stored_file_sha256s": {"model": "8" * 64},
    }


def test_verifier_accepts_two_disjoint_significant_lesions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = verify_compositional_semantic_replications(**_inputs(monkeypatch))

    assert result["verified"] is True
    assert result["frozen_model_unchanged"] is True
    assert result["cohorts"][0]["paired_exact_tests"]["program_exact"] == {
        "metric": "program_exact",
        "treatment_only": 5,
        "control_only": 0,
        "discordant": 5,
        "one_sided_exact_p_numerator": 1,
        "one_sided_exact_p_denominator": 32,
        "one_sided_exact_p": 0.03125,
    }


def test_verifier_rejects_source_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(monkeypatch)
    cohort = inputs["cohorts"][0]
    inputs["cohorts"][0] = CompositionalReplicationCohort(
        family=cohort.family,
        source=cohort.source,
        fresh=SemanticCohortInventory(
            manifest_sha256=cohort.fresh.manifest_sha256,
            exact_model_path=cohort.fresh.exact_model_path,
            tokenizer_identity_sha256=cohort.fresh.tokenizer_identity_sha256,
            example_count=cohort.fresh.example_count,
            example_ids=(cohort.source.example_ids[0], *cohort.fresh.example_ids[1:]),
            held_source_text_sha256s=cohort.fresh.held_source_text_sha256s,
            worker_stack_identity_gaps=(),
        ),
        lesion_arm=cohort.lesion_arm,
        report=cohort.report,
        transfer_kind=cohort.transfer_kind,
        evaluation_source_commit=cohort.evaluation_source_commit,
    )

    with pytest.raises(ValueError, match="not disjoint"):
        verify_compositional_semantic_replications(**inputs)


def test_verifier_rejects_aggregate_not_supported_by_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    report = inputs["cohorts"][0].report
    report["arms"]["treatment"]["test"]["answer_exact"] -= 1
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = verification._sha(body)

    with pytest.raises(ValueError, match="aggregate differs"):
        verify_compositional_semantic_replications(**inputs)
