"""Append-only re-adjudication of preserved candidate-cortex measurements."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from core.learning.candidate_cortex_admission import adjudicate_checkpoint_evidence
from core.learning.candidate_cortex_measurement import compile_checkpoint_evidence
from core.learning.candidate_cortex_training import (
    STAGE_RECONCILIATION_SCHEMA,
    document_sha256,
    file_sha256,
)
from core.learning.recurrent_sft_behavior_canaries import (
    build_generated_behavior_canaries,
    grade_generated_behavior_text,
)

DETAIL_SCHEMA: Final = "aura.candidate_cortex_training.checkpoint_measurement_detail.v1"
DETAIL_SCHEMA_V2: Final = "aura.candidate_cortex_training.checkpoint_measurement_detail.v2"


class CandidateCortexReconciliationError(ValueError):
    """Preserved measurement evidence cannot support a correction."""


def _fail(code: str) -> None:
    raise CandidateCortexReconciliationError(code)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_detail(
    detail: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    stage_index: int,
) -> dict[str, Any]:
    required = {
        "schema",
        "plan_sha256",
        "stage_index",
        "checkpoint",
        "persona_rows",
        "retention_rows",
        "baseline_behavior",
        "candidate_behavior",
        "baseline_path",
        "baseline_sha256",
        "baseline_reused",
        "evidence_sha256",
        "detail_sha256",
    }
    schema = detail.get("schema")
    if schema == DETAIL_SCHEMA_V2:
        required.add("measurement_contract_sha256")
    material = dict(detail)
    claimed = material.pop("detail_sha256", None)
    if (
        set(detail) != required
        or schema not in {DETAIL_SCHEMA, DETAIL_SCHEMA_V2}
        or detail.get("plan_sha256") != plan.get("plan_sha256")
        or detail.get("stage_index") != stage_index
        or claimed != document_sha256(material)
    ):
        _fail("reconciliation_detail_invalid")
    for role in ("persona_rows", "retention_rows", "baseline_behavior", "candidate_behavior"):
        rows = detail.get(role)
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            _fail("reconciliation_detail_rows_invalid")
    return dict(detail)


def _validated_behavior_rows(
    raw: Sequence[Mapping[str, Any]],
    *,
    evaluator_source_sha256: str,
) -> list[dict[str, Any]]:
    cases = {case["case_id"]: case for case in build_generated_behavior_canaries()}
    if len(raw) != len(cases):
        _fail("reconciliation_behavior_count_invalid")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        required = {
            "probe_id",
            "family",
            "passed",
            "evaluator_sha256",
            "text",
            "text_sha256",
            "finish_reason",
            "grade",
        }
        probe_id = value.get("probe_id")
        case = cases.get(probe_id)
        text = value.get("text")
        old_grade = value.get("grade")
        if (
            set(value) != required
            or not isinstance(probe_id, str)
            or probe_id in seen
            or case is None
            or value.get("family") != case["family"]
            or not isinstance(text, str)
            or value.get("text_sha256") != _text_sha256(text)
            or not isinstance(old_grade, Mapping)
            or old_grade.get("case_id") != probe_id
            or old_grade.get("text_sha256") != value.get("text_sha256")
            or value.get("passed") is not old_grade.get("passed")
            or not isinstance(value.get("finish_reason"), str)
        ):
            _fail("reconciliation_behavior_row_invalid")
        seen.add(probe_id)
        grade = grade_generated_behavior_text(case, text)
        evaluator_sha = document_sha256(
            {
                "case": case,
                "grader_source_sha256": evaluator_source_sha256,
                "max_tokens": 160,
                "temperature": 0.0,
                "thinking": False,
            }
        )
        rows.append(
            {
                **dict(value),
                "passed": bool(grade["passed"]),
                "evaluator_sha256": evaluator_sha,
                "grade": grade,
            }
        )
    if seen != set(cases):
        _fail("reconciliation_behavior_identity_invalid")
    return rows


def _paired_behavior(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left = {str(row["probe_id"]): row for row in baseline}
    right = {str(row["probe_id"]): row for row in candidate}
    if set(left) != set(right) or len(left) != len(baseline) or len(right) != len(candidate):
        _fail("reconciliation_behavior_pair_invalid")
    result: list[dict[str, Any]] = []
    for probe_id in sorted(left):
        base = left[probe_id]
        adapted = right[probe_id]
        if (
            base["family"] != adapted["family"]
            or base["evaluator_sha256"] != adapted["evaluator_sha256"]
        ):
            _fail("reconciliation_behavior_pair_invalid")
        result.append(
            {
                "probe_id": probe_id,
                "family": base["family"],
                "baseline_passed": bool(base["passed"]),
                "candidate_passed": bool(adapted["passed"]),
                "evaluator_sha256": base["evaluator_sha256"],
            }
        )
    return result


def reconcile_preserved_measurement(
    *,
    plan: Mapping[str, Any],
    stage_index: int,
    detail: Mapping[str, Any],
    original_evidence: Mapping[str, Any],
    prior_admission: Mapping[str, Any],
    evaluator_source_path: Path,
) -> dict[str, Any]:
    """Regrade exact preserved texts and bind the correction to old evidence."""

    validated = _validate_detail(detail, plan=plan, stage_index=stage_index)
    if (
        original_evidence.get("measurement_sha256") != validated["evidence_sha256"]
        or prior_admission.get("evidence_sha256") != validated["evidence_sha256"]
        or prior_admission.get("stage_index") != stage_index
    ):
        _fail("reconciliation_evidence_binding_invalid")

    # Rebuild the original aggregate from the preserved raw rows before any
    # corrected grade is trusted. This proves the detail is the source of the
    # exact admission being superseded.
    old_behavior = _paired_behavior(
        list(validated["baseline_behavior"]),
        list(validated["candidate_behavior"]),
    )
    baseline_binding = (
        {
            "measurement_contract_sha256": str(validated["measurement_contract_sha256"]),
            "baseline_sha256": str(validated["baseline_sha256"]),
        }
        if validated["schema"] == DETAIL_SCHEMA_V2
        else {}
    )
    rebuilt_original = compile_checkpoint_evidence(
        plan=plan,
        stage_index=stage_index,
        checkpoint_sha256=str(validated["checkpoint"]["sha256"]),
        persona_rows=list(validated["persona_rows"]),
        retention_rows=list(validated["retention_rows"]),
        behavior_rows=old_behavior,
        **baseline_binding,
    )
    if rebuilt_original != dict(original_evidence):
        _fail("reconciliation_original_replay_mismatch")

    evaluator_source_sha256 = file_sha256(evaluator_source_path.resolve(strict=True))
    baseline = _validated_behavior_rows(
        list(validated["baseline_behavior"]),
        evaluator_source_sha256=evaluator_source_sha256,
    )
    candidate = _validated_behavior_rows(
        list(validated["candidate_behavior"]),
        evaluator_source_sha256=evaluator_source_sha256,
    )
    corrected_evidence = compile_checkpoint_evidence(
        plan=plan,
        stage_index=stage_index,
        checkpoint_sha256=str(validated["checkpoint"]["sha256"]),
        persona_rows=list(validated["persona_rows"]),
        retention_rows=list(validated["retention_rows"]),
        behavior_rows=_paired_behavior(baseline, candidate),
        **baseline_binding,
    )
    admission = adjudicate_checkpoint_evidence(
        corrected_evidence,
        plan=plan,
        stage_index=stage_index,
    )
    body = {
        "schema": STAGE_RECONCILIATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "stage_index": stage_index,
        "prior_admission_sha256": document_sha256(prior_admission),
        "prior_evidence_sha256": validated["evidence_sha256"],
        "detail_sha256": validated["detail_sha256"],
        "evaluator_source_sha256": evaluator_source_sha256,
        "reconciled_evidence_sha256": corrected_evidence["measurement_sha256"],
        "admission": admission,
    }
    return {
        **body,
        "reconciliation_sha256": document_sha256(body),
        "corrected_evidence": corrected_evidence,
        "baseline_behavior": baseline,
        "candidate_behavior": candidate,
    }


__all__ = [
    "CandidateCortexReconciliationError",
    "reconcile_preserved_measurement",
]
