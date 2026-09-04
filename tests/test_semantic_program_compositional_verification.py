from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.learning.semantic_program_compositional_verification as verification
from core.learning.semantic_program_compositional_verification import (
    COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES,
    CompositionalReplicationCohort,
    SemanticCohortInventory,
    verify_compositional_family_withheld_replication,
    verify_compositional_semantic_replications,
)


def _inventory(prefix: str) -> SemanticCohortInventory:
    offset = int(prefix, 16) * 100
    hashes = tuple(f"{offset + index + 1:064x}" for index in range(8))
    return SemanticCohortInventory(
        manifest_sha256=(prefix * 64)[:64],
        exact_model_path="/models/aura-27b",
        tokenizer_identity_sha256="a" * 64,
        example_count=8,
        example_ids=tuple(f"{prefix}-example-{index}" for index in range(8)),
        source_text_sha256s=hashes,
        split_source_text_sha256s=(
            ("train", hashes[:3]),
            ("validation", hashes[3:6]),
            ("test", hashes[6:]),
        ),
        held_source_text_sha256s=hashes[3:],
        session_basis_sha256s=((prefix * 64)[:64],),
        representation_basis_sha256="b" * 64,
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
            "coefficient_sha256": "c" * 64,
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
            source_text_sha256s=cohort.fresh.source_text_sha256s,
            split_source_text_sha256s=cohort.fresh.split_source_text_sha256s,
            held_source_text_sha256s=cohort.fresh.held_source_text_sha256s,
            session_basis_sha256s=cohort.fresh.session_basis_sha256s,
            representation_basis_sha256=cohort.fresh.representation_basis_sha256,
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


def _receipt(schema: str, **values: object) -> dict[str, object]:
    body = {"schema": schema, **values}
    return {**body, "receipt_sha256": verification._sha(body)}


def _source_family_report(inventory: SemanticCohortInventory) -> dict[str, object]:
    split_rows = {
        split: [_row(identity, correct=True) for identity in identities]
        for split, identities in inventory.split_source_text_sha256s
    }
    split_arms = {split: _arm(rows) for split, rows in split_rows.items()}
    return {
        "example_count": inventory.example_count,
        "splits": split_arms,
        "held_out_program_exact": len(inventory.held_source_text_sha256s),
        "held_out_argument_exact": len(inventory.held_source_text_sha256s),
        "held_out_answer_exact": len(inventory.held_source_text_sha256s),
        "held_out_total": len(inventory.held_source_text_sha256s),
    }


def _family_withheld_inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    model_payload = {"schema": "test-family-withheld-model"}
    source = {
        "arithmetic": _inventory("1"),
        "fork_join": _inventory("2"),
        "sequence": _inventory("3"),
    }
    fresh = _inventory("4")
    model = SimpleNamespace(
        receipt_sha256="b" * 64,
        model_basis_sha256="1" * 64,
        training_receipt={
            "coefficient_sha256": "c" * 64,
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
    fit_compatibility = _receipt(
        "aura.semantic_training_representation_compatibility.v1",
        source_feature_manifest_sha256s={
            family: source[family].manifest_sha256
            for family in ("arithmetic", "fork_join")
        },
        source_session_basis_sha256s={
            family: list(source[family].session_basis_sha256s)
            for family in ("arithmetic", "fork_join")
        },
        representation_basis_sha256="b" * 64,
        target_training_session_basis_sha256=model.model_basis_sha256,
        hidden_states_changed=False,
        serving_authority=False,
    )
    source_compatibility = _receipt(
        "aura.semantic_representation_compatibility.v1",
        transducer_receipt_sha256=model.receipt_sha256,
        coefficient_sha256=model.training_receipt["coefficient_sha256"],
        training_feature_manifest_sha256=source["arithmetic"].manifest_sha256,
        replication_feature_manifest_sha256=source["sequence"].manifest_sha256,
        training_session_basis_sha256=model.model_basis_sha256,
        replication_session_basis_sha256s=list(source["sequence"].session_basis_sha256s),
        representation_basis_sha256="b" * 64,
        coefficients_changed=False,
        hidden_states_changed=False,
        serving_authority=False,
    )
    source_body = {
        "schema": "aura.semantic_program_compositional_leave_family_out.v1",
        "held_out_family": "sequence",
        "fit_families": ["arithmetic", "fork_join"],
        "evaluated_families": ["sequence"],
        "feature_manifest_sha256s": {
            family: inventory.manifest_sha256
            for family, inventory in sorted(source.items())
        },
        "representation_compatibility": fit_compatibility,
        "held_out_representation_compatibility": source_compatibility,
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "fit_example_count": source["arithmetic"].example_count
        + source["fork_join"].example_count,
        "families": {"sequence": _source_family_report(source["sequence"])},
        "held_out_family_was_available_to_fit": False,
        "expected_answers_available_to_training": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
    }
    source_report = {
        **source_body,
        "report_sha256": verification._sha(source_body),
    }
    fresh_compatibility = _receipt(
        "aura.semantic_representation_compatibility.v1",
        transducer_receipt_sha256=model.receipt_sha256,
        coefficient_sha256=model.training_receipt["coefficient_sha256"],
        training_feature_manifest_sha256=source["arithmetic"].manifest_sha256,
        replication_feature_manifest_sha256=fresh.manifest_sha256,
        training_session_basis_sha256=model.model_basis_sha256,
        replication_session_basis_sha256s=list(fresh.session_basis_sha256s),
        representation_basis_sha256="b" * 64,
        coefficients_changed=False,
        hidden_states_changed=False,
        serving_authority=False,
    )
    fresh_base = _report(fresh, "coefficient_lesion")
    fresh_body = {
        key: value for key, value in fresh_base.items() if key != "report_sha256"
    }
    fresh_body["representation_compatibility"] = fresh_compatibility
    fresh_report = {
        **fresh_body,
        "report_sha256": verification._sha(fresh_body),
    }
    return {
        "source_inventories": source,
        "fresh_inventory": fresh,
        "trained_model_payload": model_payload,
        "source_report": source_report,
        "fresh_report": fresh_report,
        "lesion_arm": "coefficient_lesion",
        "evaluation_source_commit": "e" * 40,
        "source_sha256s": {
            path: "9" * 64
            for path in COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES
        },
        "stored_file_sha256s": {"model": "8" * 64},
    }


def test_family_withheld_verifier_recounts_source_and_fresh_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = verify_compositional_family_withheld_replication(
        **_family_withheld_inputs(monkeypatch)
    )

    assert result["verified"] is True
    assert result["held_out_family"] == "sequence"
    assert result["held_out_family_was_available_to_fit"] is False
    assert result["source_recount"]["held_out"]["answer_exact"] == 5
    assert result["fresh_replication"]["paired_exact_tests"]["answer_exact"][
        "one_sided_exact_p"
    ] == 0.03125


def test_family_withheld_verifier_rejects_target_family_in_fit_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _family_withheld_inputs(monkeypatch)
    report = inputs["source_report"]
    compatibility = report["representation_compatibility"]
    compatibility["source_feature_manifest_sha256s"]["sequence"] = (
        inputs["source_inventories"]["sequence"].manifest_sha256
    )
    compatibility_body = {
        key: value
        for key, value in compatibility.items()
        if key != "receipt_sha256"
    }
    compatibility["receipt_sha256"] = verification._sha(compatibility_body)
    report_body = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    report["report_sha256"] = verification._sha(report_body)

    with pytest.raises(ValueError, match="fit basis includes"):
        verify_compositional_family_withheld_replication(**inputs)
