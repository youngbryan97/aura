from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.learning.candidate_cortex_admission import adjudicate_checkpoint_evidence
from core.learning.candidate_cortex_measurement import (
    LOSS_ROW_SCHEMA,
    compile_checkpoint_evidence,
)
from core.learning.candidate_cortex_reconciliation import (
    DETAIL_SCHEMA,
    reconcile_preserved_measurement,
)
from core.learning.candidate_cortex_training import (
    STAGE_RECONCILIATION_SCHEMA,
    document_sha256,
    effective_stage_evidence,
)
from core.learning.recurrent_sft_behavior_canaries import (
    build_generated_behavior_canaries,
    grade_generated_behavior_text,
)


def _plan() -> dict:
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


def _loss(sample_id: str, domain: str) -> dict:
    return {
        "schema": LOSS_ROW_SCHEMA,
        "sample_id": sample_id,
        "domain": domain,
        "baseline_nll_sum": 2.0,
        "candidate_nll_sum": 1.0,
        "tokens": 1,
    }


def _behavior_row(case: dict, text: str, *, force_old_failure: bool = False) -> dict:
    grade = grade_generated_behavior_text(case, text)
    if force_old_failure:
        grade = deepcopy(grade)
        grade["required_groups"][-1]["matched"] = []
        grade["required_groups"][-1]["passed"] = False
        grade["passed"] = False
        body = dict(grade)
        body.pop("grade_sha256")
        grade["grade_sha256"] = document_sha256(body)
    return {
        "probe_id": case["case_id"],
        "family": case["family"],
        "passed": grade["passed"],
        "evaluator_sha256": "5" * 64,
        "text": text,
        "text_sha256": grade["text_sha256"],
        "finish_reason": "stop",
        "grade": grade,
    }


def _fixture() -> tuple[dict, dict, dict, dict]:
    plan = _plan()
    baseline = []
    candidate = []
    for case in build_generated_behavior_canaries():
        text = " ".join(group[0] for group in case["required_groups"])
        baseline.append(_behavior_row(case, text))
        if case["name"] == "untrusted_web_secret_request":
            candidate.append(
                _behavior_row(
                    case,
                    "No. I don't hand over credentials to an untrusted webpage.",
                    force_old_failure=True,
                )
            )
        else:
            candidate.append(_behavior_row(case, text))
    old_behavior = [
        {
            "probe_id": left["probe_id"],
            "family": left["family"],
            "baseline_passed": left["passed"],
            "candidate_passed": right["passed"],
            "evaluator_sha256": left["evaluator_sha256"],
        }
        for left, right in zip(baseline, candidate, strict=True)
    ]
    persona = [_loss(f"persona-{index}", "persona") for index in range(10)]
    retention = [_loss(f"retention-{index}", "retention") for index in range(10)]
    evidence = compile_checkpoint_evidence(
        plan=plan,
        stage_index=0,
        checkpoint_sha256="4" * 64,
        persona_rows=persona,
        retention_rows=retention,
        behavior_rows=old_behavior,
    )
    detail_body = {
        "schema": DETAIL_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "stage_index": 0,
        "checkpoint": {"sha256": "4" * 64},
        "persona_rows": persona,
        "retention_rows": retention,
        "baseline_behavior": baseline,
        "candidate_behavior": candidate,
        "baseline_path": "/immutable/baseline.json",
        "baseline_sha256": "6" * 64,
        "baseline_reused": True,
        "evidence_sha256": evidence["measurement_sha256"],
    }
    detail = {**detail_body, "detail_sha256": document_sha256(detail_body)}
    prior = adjudicate_checkpoint_evidence(evidence, plan=plan, stage_index=0)
    assert prior["regressions"] == 1
    return plan, detail, evidence, prior


def test_preserved_measurement_reconciliation_corrects_only_the_grader(tmp_path: Path) -> None:
    plan, detail, evidence, prior = _fixture()
    source = tmp_path / "evaluator.py"
    source.write_text("corrected evaluator\n", encoding="utf-8")
    result = reconcile_preserved_measurement(
        plan=plan,
        stage_index=0,
        detail=detail,
        original_evidence=evidence,
        prior_admission=prior,
        evaluator_source_path=source,
    )
    assert result["schema"] == STAGE_RECONCILIATION_SCHEMA
    assert result["admission"]["regressions"] == 0
    assert result["admission"]["no_regression_score"] == 1.0
    assert result["prior_evidence_sha256"] == evidence["measurement_sha256"]
    corrected = result["corrected_evidence"]
    assert corrected["persona"] == evidence["persona"]
    assert corrected["retention"] == evidence["retention"]
    changed = [
        row
        for row in corrected["behavior"]
        if row["candidate_passed"]
        and not next(
            old["candidate_passed"]
            for old in evidence["behavior"]
            if old["probe_id"] == row["probe_id"]
        )
    ]
    assert len(changed) == 1


def test_effective_stage_evidence_applies_bound_reconciliation(tmp_path: Path) -> None:
    plan, detail, evidence, prior = _fixture()
    source = tmp_path / "evaluator.py"
    source.write_text("corrected evaluator\n", encoding="utf-8")
    result = reconcile_preserved_measurement(
        plan=plan,
        stage_index=0,
        detail=detail,
        original_evidence=evidence,
        prior_admission=prior,
        evaluator_source_path=source,
    )
    event_keys = {
        "schema",
        "plan_sha256",
        "stage_index",
        "prior_admission_sha256",
        "prior_evidence_sha256",
        "detail_sha256",
        "evaluator_source_sha256",
        "reconciled_evidence_sha256",
        "admission",
        "reconciliation_sha256",
    }
    reconciliation = {key: result[key] for key in event_keys}
    observation = {"stage_index": 0, "validation_loss": 1.0}
    observations, admissions = effective_stage_evidence(
        [
            {"event_type": "stage_observed", "payload": observation},
            {"event_type": "stage_admitted", "payload": prior},
            {"event_type": "stage_reconciled", "payload": reconciliation},
        ]
    )
    assert observations == [observation]
    assert admissions == [result["admission"]]
