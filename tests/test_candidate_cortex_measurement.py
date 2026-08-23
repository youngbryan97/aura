from __future__ import annotations

from copy import deepcopy

import pytest

from core.learning import candidate_cortex_measurement as measurement


def _plan() -> dict:
    return {
        "plan_sha256": "1" * 64,
        "model": {"descriptor_sha256": "2" * 64},
        "dataset": {"receipt_sha256": "3" * 64},
    }


def _loss(sample_id: str, domain: str, baseline: float, candidate: float, tokens: int) -> dict:
    return {
        "schema": measurement.LOSS_ROW_SCHEMA,
        "sample_id": sample_id,
        "domain": domain,
        "baseline_nll_sum": baseline,
        "candidate_nll_sum": candidate,
        "tokens": tokens,
    }


def _behavior(probe_id: str, *, baseline: bool = True, candidate: bool = True) -> dict:
    return {
        "probe_id": probe_id,
        "family": "grounding",
        "baseline_passed": baseline,
        "candidate_passed": candidate,
        "evaluator_sha256": "5" * 64,
    }


def test_measurement_compiler_uses_token_weighted_losses_and_canonical_order() -> None:
    evidence = measurement.compile_checkpoint_evidence(
        plan=_plan(),
        stage_index=0,
        checkpoint_sha256="4" * 64,
        persona_rows=[
            _loss("b", "voice", 8.0, 4.0, 4),
            _loss("a", "voice", 2.0, 2.0, 2),
            _loss("c", "identity", 6.0, 9.0, 3),
        ],
        retention_rows=[_loss("r", "retention", 2.0, 1.0, 2)],
        behavior_rows=[_behavior("z"), _behavior("a")],
    )
    assert evidence["persona"]["baseline_loss"] == pytest.approx(16.0 / 9.0)
    assert evidence["persona"]["candidate_loss"] == pytest.approx(15.0 / 9.0)
    assert evidence["persona"]["domain_losses"]["voice"] == {
        "baseline_loss": pytest.approx(10.0 / 6.0),
        "candidate_loss": pytest.approx(6.0 / 6.0),
        "samples": 2,
        "tokens": 6,
    }
    assert [row["probe_id"] for row in evidence["behavior"]] == ["a", "z"]
    assert len(evidence["measurement_sha256"]) == 64


def test_measurement_compiler_binds_current_baseline_generation() -> None:
    evidence = measurement.compile_checkpoint_evidence(
        plan=_plan(),
        stage_index=0,
        checkpoint_sha256="4" * 64,
        persona_rows=[_loss("p", "voice", 2.0, 1.0, 2)],
        retention_rows=[_loss("r", "retention", 2.0, 1.0, 2)],
        behavior_rows=[_behavior("b")],
        measurement_contract_sha256="6" * 64,
        baseline_sha256="7" * 64,
    )

    assert evidence["schema"].endswith(".v2")
    assert evidence["measurement_contract_sha256"] == "6" * 64
    assert evidence["baseline_sha256"] == "7" * 64


def test_measurement_compiler_rejects_partial_baseline_binding() -> None:
    with pytest.raises(
        measurement.CandidateCortexMeasurementError,
        match="baseline_binding_incomplete",
    ):
        measurement.compile_checkpoint_evidence(
            plan=_plan(),
            stage_index=0,
            checkpoint_sha256="4" * 64,
            persona_rows=[_loss("p", "voice", 2.0, 1.0, 2)],
            retention_rows=[_loss("r", "retention", 2.0, 1.0, 2)],
            behavior_rows=[_behavior("b")],
            measurement_contract_sha256="6" * 64,
        )


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("sample_id", "row_invalid"),
        ("domain", "row_invalid"),
        ("tokens", "row_invalid"),
        ("baseline_nll_sum", "loss_invalid"),
        ("candidate_nll_sum", "loss_invalid"),
    ],
)
def test_measurement_compiler_rejects_malformed_loss_rows(field: str, reason: str) -> None:
    row = _loss("a", "voice", 1.0, 1.0, 1)
    row[field] = {"sample_id": "", "domain": "", "tokens": 0}.get(field, float("nan"))
    with pytest.raises(measurement.CandidateCortexMeasurementError, match=reason):
        measurement.compile_checkpoint_evidence(
            plan=_plan(),
            stage_index=0,
            checkpoint_sha256="4" * 64,
            persona_rows=[row],
            retention_rows=[_loss("r", "retention", 1.0, 1.0, 1)],
            behavior_rows=[_behavior("p")],
        )


def test_measurement_compiler_rejects_duplicate_evidence() -> None:
    duplicate = _loss("same", "voice", 1.0, 1.0, 1)
    with pytest.raises(measurement.CandidateCortexMeasurementError, match="row_invalid"):
        measurement.compile_checkpoint_evidence(
            plan=_plan(),
            stage_index=0,
            checkpoint_sha256="4" * 64,
            persona_rows=[duplicate, deepcopy(duplicate)],
            retention_rows=[_loss("r", "retention", 1.0, 1.0, 1)],
            behavior_rows=[_behavior("p")],
        )
    with pytest.raises(measurement.CandidateCortexMeasurementError, match="behavior_row_invalid"):
        measurement.compile_checkpoint_evidence(
            plan=_plan(),
            stage_index=0,
            checkpoint_sha256="4" * 64,
            persona_rows=[_loss("p", "voice", 1.0, 1.0, 1)],
            retention_rows=[_loss("r", "retention", 1.0, 1.0, 1)],
            behavior_rows=[_behavior("same"), _behavior("same")],
        )
