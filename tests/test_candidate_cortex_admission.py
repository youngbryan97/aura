from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from core.learning import candidate_cortex_admission as admission
from core.learning.candidate_cortex_training import document_sha256


def _plan() -> dict[str, Any]:
    return {
        "plan_sha256": "1" * 64,
        "model": {"descriptor_sha256": "2" * 64},
        "dataset": {"receipt_sha256": "3" * 64},
        "stages": {
            "initial_iterations": 100,
            "growth_factor": 2,
            "max_stages": 5,
            "min_stages": 2,
            "patience": 2,
            "min_loss_improvement": 0.002,
            "max_loss_regression_fraction": 0.02,
            "persona_floor": 0.9,
            "retention_floor": 0.98,
            "no_regression_floor": 1.0,
            "min_eval_samples": 32,
        },
    }


def _loss(baseline: float, candidate: float, *, domain: str) -> dict[str, Any]:
    return {
        "baseline_loss": baseline,
        "candidate_loss": candidate,
        "samples": 16,
        "tokens": 160,
        "domain_losses": {
            domain: {
                "baseline_loss": baseline,
                "candidate_loss": candidate,
                "samples": 16,
                "tokens": 160,
            }
        },
    }


def _evidence() -> dict[str, Any]:
    body = {
        "schema": admission.EVIDENCE_SCHEMA,
        "stage_index": 0,
        "plan_sha256": "1" * 64,
        "model_descriptor_sha256": "2" * 64,
        "dataset_receipt_sha256": "3" * 64,
        "checkpoint_sha256": "4" * 64,
        "persona": _loss(2.0, 1.8, domain="persona"),
        "retention": _loss(1.0, 1.01, domain="general"),
        "behavior": [
            {
                "probe_id": f"probe-{index}",
                "family": "exact",
                "baseline_passed": index != 3,
                "candidate_passed": True,
                "evaluator_sha256": "5" * 64,
            }
            for index in range(8)
        ],
    }
    return {**body, "measurement_sha256": document_sha256(body)}


def test_mechanical_admission_preserves_base_successes_and_loss_surfaces() -> None:
    result = admission.adjudicate_checkpoint_evidence(_evidence(), plan=_plan(), stage_index=0)
    assert result["persona_score"] == 1.0
    assert result["retention_score"] == pytest.approx(1.0 / 1.01)
    assert result["no_regression_score"] == 1.0
    assert result["regressions"] == 0
    assert result["checks"] == 40
    assert result["model_free"] is True


def test_mechanical_admission_accepts_exact_bound_v2_evidence() -> None:
    evidence = _evidence()
    evidence["schema"] = admission.EVIDENCE_SCHEMA_V2
    evidence["measurement_contract_sha256"] = "6" * 64
    evidence["baseline_sha256"] = "7" * 64
    body = dict(evidence)
    body.pop("measurement_sha256")
    evidence["measurement_sha256"] = document_sha256(body)

    result = admission.adjudicate_checkpoint_evidence(evidence, plan=_plan(), stage_index=0)

    assert result["evidence_sha256"] == evidence["measurement_sha256"]


def test_one_lost_baseline_success_is_an_explicit_regression() -> None:
    evidence = _evidence()
    evidence["behavior"][0]["candidate_passed"] = False
    body = dict(evidence)
    body.pop("measurement_sha256")
    evidence["measurement_sha256"] = document_sha256(body)
    result = admission.adjudicate_checkpoint_evidence(evidence, plan=_plan(), stage_index=0)
    assert result["regressions"] == 1
    assert result["no_regression_score"] == pytest.approx(6.0 / 7.0)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda value: value.update(plan_sha256="9" * 64), "binding_mismatch"),
        (
            lambda value: value["persona"].update(samples=15),
            "domain_totals_mismatch",
        ),
        (
            lambda value: value["behavior"].append(deepcopy(value["behavior"][0])),
            "behavior_evidence_invalid",
        ),
        (lambda value: value.update(measurement_sha256="0" * 64), "digest_invalid"),
    ],
)
def test_forged_or_incoherent_evidence_is_rejected(mutation: Any, reason: str) -> None:
    evidence = _evidence()
    mutation(evidence)
    if reason != "digest_invalid":
        body = dict(evidence)
        body.pop("measurement_sha256")
        evidence["measurement_sha256"] = document_sha256(body)
    with pytest.raises(admission.CandidateCortexAdmissionError, match=reason):
        admission.adjudicate_checkpoint_evidence(evidence, plan=_plan(), stage_index=0)
