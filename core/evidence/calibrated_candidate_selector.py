"""Select between viable candidates using independently calibrated evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.evidence.calibrated_binary import (
    CalibratedBinaryScorer,
    calibrated_binary_scorer_from_dict,
)
from core.evidence.necessary_condition_selector import (
    CANDIDATE_SELECTION_DECISION_SCHEMA,
    CandidateSelectionDecision,
    NecessaryConditionSelector,
    PairwiseSelectionEvidence,
    necessary_condition_selector_from_dict,
)
from core.evidence.packet import derive

CALIBRATED_CANDIDATE_SELECTOR_SCHEMA: Final = (
    "aura.evidence.calibrated_candidate_selector.v1"
)
CALIBRATED_CANDIDATE_SELECTOR_RECEIPT_SCHEMA: Final = (
    "aura.evidence.calibrated_candidate_selector_receipt.v1"
)
MIN_PAIRWISE_CALIBRATION_OBSERVATIONS: Final = 12
MIN_PAIRWISE_ADMISSION_OBSERVATIONS: Final = 12


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedPairwiseObservation:
    """One candidate pair graded by an independent verifier."""

    incumbent: tuple[tuple[str, float], ...]
    challenger: tuple[tuple[str, float], ...]
    incumbent_correct: bool
    challenger_correct: bool
    source_ref: str

    def __post_init__(self) -> None:
        incumbent_names = tuple(name for name, _value in self.incumbent)
        challenger_names = tuple(name for name, _value in self.challenger)
        if (
            not self.incumbent
            or incumbent_names != tuple(sorted(incumbent_names))
            or challenger_names != incumbent_names
            or len(set(incumbent_names)) != len(incumbent_names)
            or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for name, value in (*self.incumbent, *self.challenger)
            )
            or type(self.incumbent_correct) is not bool
            or type(self.challenger_correct) is not bool
            or not isinstance(self.source_ref, str)
            or not self.source_ref
        ):
            raise ValueError("verified pairwise observation is invalid")

    @classmethod
    def from_mappings(
        cls,
        *,
        incumbent: Mapping[str, float],
        challenger: Mapping[str, float],
        incumbent_correct: bool,
        challenger_correct: bool,
        source_ref: str,
    ) -> VerifiedPairwiseObservation:
        def normalized(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
            if not isinstance(values, Mapping):
                raise ValueError("pairwise observation features must be mappings")
            return tuple(sorted((str(name), float(value)) for name, value in values.items()))

        return cls(
            incumbent=normalized(incumbent),
            challenger=normalized(challenger),
            incumbent_correct=incumbent_correct,
            challenger_correct=challenger_correct,
            source_ref=source_ref,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incumbent": [list(row) for row in self.incumbent],
            "challenger": [list(row) for row in self.challenger],
            "incumbent_correct": self.incumbent_correct,
            "challenger_correct": self.challenger_correct,
            "source_ref": self.source_ref,
        }


def _missing(
    selector: NecessaryConditionSelector,
    values: Mapping[str, float],
) -> tuple[str, ...]:
    return tuple(
        condition.name
        for condition in selector.conditions
        if values[condition.name] < condition.minimum
    )


@dataclass(frozen=True, slots=True)
class CalibratedCandidateSelector:
    """Necessary-condition repair plus calibrated comparison of viable paths."""

    necessary: NecessaryConditionSelector
    scorer: CalibratedBinaryScorer
    switch_margin: float
    construction_receipt: dict[str, Any]
    schema: str = CALIBRATED_CANDIDATE_SELECTOR_SCHEMA

    def __post_init__(self) -> None:
        body = {
            key: value
            for key, value in self.construction_receipt.items()
            if key != "receipt_sha256"
        }
        if (
            self.schema != CALIBRATED_CANDIDATE_SELECTOR_SCHEMA
            or not math.isfinite(self.switch_margin)
            or self.switch_margin < 0.0
            or self.construction_receipt.get("schema")
            != CALIBRATED_CANDIDATE_SELECTOR_RECEIPT_SCHEMA
            or self.construction_receipt.get("necessary_selector_receipt_sha256")
            != self.necessary.receipt_sha256
            or self.construction_receipt.get("scorer_receipt_sha256")
            != self.scorer.receipt_sha256
            or self.construction_receipt.get("switch_margin") != self.switch_margin
            or self.construction_receipt.get("labels_available_to_calibration") is not True
            or self.construction_receipt.get("labels_available_to_runtime") is not False
            or self.construction_receipt.get("text_available_to_selector") is not False
            or self.construction_receipt.get("domain_identity_available_to_selector")
            is not False
            or self.construction_receipt.get("admitted") is not True
            or self.construction_receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("calibrated candidate selector envelope is invalid")
        object.__setattr__(
            self,
            "construction_receipt",
            json.loads(json.dumps(self.construction_receipt)),
        )

    @property
    def receipt_sha256(self) -> str:
        return str(self.construction_receipt["receipt_sha256"])

    def select(
        self,
        *,
        incumbent: str,
        challenger: str,
        evidence: PairwiseSelectionEvidence,
    ) -> CandidateSelectionDecision:
        if not incumbent or not challenger or incumbent == challenger:
            raise ValueError("selection candidates are invalid")
        necessary_names = {condition.name for condition in self.necessary.conditions}
        if not necessary_names.issubset(evidence.names):
            raise ValueError("selection evidence omits a necessary condition")
        if set(evidence.names) != set(self.scorer.feature_names):
            raise ValueError("selection evidence differs from calibrated feature schema")
        incumbent_values = evidence.values("incumbent")
        challenger_values = evidence.values("challenger")
        incumbent_missing = _missing(self.necessary, incumbent_values)
        challenger_missing = _missing(self.necessary, challenger_values)
        incumbent_score = self.scorer.predict(incumbent_values)
        challenger_score = self.scorer.predict(challenger_values)
        if incumbent_missing and not challenger_missing:
            selected = challenger
            reason = "challenger_repairs_necessary_condition_failure"
        elif not incumbent_missing and challenger_missing:
            selected = incumbent
            reason = "challenger_fails_necessary_condition"
        elif (
            not incumbent_missing
            and not challenger_missing
            and challenger_score - incumbent_score > self.switch_margin
        ):
            selected = challenger
            reason = "challenger_exceeds_calibrated_quality_margin"
        else:
            selected = incumbent
            reason = "incumbent_retained_without_admitted_advantage"
        result_evidence = derive(
            1.0,
            (evidence.packet,),
            subject=evidence.packet.subject,
            produced_by="core.evidence.calibrated_candidate_selector",
        )
        body = {
            "schema": CANDIDATE_SELECTION_DECISION_SCHEMA,
            "selector_receipt_sha256": self.receipt_sha256,
            "evidence_sha256": evidence.evidence_sha256,
            "evidence_sources": sorted(evidence.packet.sources),
            "incumbent": incumbent,
            "challenger": challenger,
            "incumbent_missing": list(incumbent_missing),
            "challenger_missing": list(challenger_missing),
            "incumbent_quality": round(incumbent_score, 10),
            "challenger_quality": round(challenger_score, 10),
            "switch_margin": self.switch_margin,
            "selected": selected,
            "reason": reason,
        }
        return CandidateSelectionDecision(
            selected=selected,
            evidence=result_evidence,
            receipt={**body, "receipt_sha256": _sha(body)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "necessary": self.necessary.to_dict(),
            "scorer": self.scorer.to_dict(),
            "switch_margin": self.switch_margin,
            "construction_receipt": self.construction_receipt,
        }


def build_calibrated_candidate_selector(
    *,
    necessary: NecessaryConditionSelector,
    scorer: CalibratedBinaryScorer,
    calibration_rows: Sequence[VerifiedPairwiseObservation],
    admission_rows: Sequence[VerifiedPairwiseObservation],
    maximum_regressions: int = 0,
) -> tuple[CalibratedCandidateSelector | None, dict[str, Any]]:
    """Choose a margin, then admit it on pairwise outcomes it did not tune on."""
    if len(calibration_rows) < MIN_PAIRWISE_CALIBRATION_OBSERVATIONS:
        return None, {
            "admitted": False,
            "reason": "insufficient_pairwise_calibration_observations",
            "observations": len(calibration_rows),
            "required": MIN_PAIRWISE_CALIBRATION_OBSERVATIONS,
        }
    if len(admission_rows) < MIN_PAIRWISE_ADMISSION_OBSERVATIONS:
        return None, {
            "admitted": False,
            "reason": "insufficient_pairwise_admission_observations",
            "observations": len(admission_rows),
            "required": MIN_PAIRWISE_ADMISSION_OBSERVATIONS,
        }
    if type(maximum_regressions) is not int or maximum_regressions < 0:
        raise ValueError("maximum_regressions must be a non-negative integer")
    feature_names = set(scorer.feature_names)
    necessary_names = {condition.name for condition in necessary.conditions}
    if (
        not necessary_names.issubset(feature_names)
        or any(
            {name for name, _value in row.incumbent} != feature_names
            or {name for name, _value in row.challenger} != feature_names
            for row in (*calibration_rows, *admission_rows)
        )
        or len({row.source_ref for row in calibration_rows}) != len(calibration_rows)
        or len({row.source_ref for row in admission_rows}) != len(admission_rows)
        or {row.source_ref for row in calibration_rows}
        & {row.source_ref for row in admission_rows}
    ):
        raise ValueError("pairwise calibration schema or identity differs")

    def result(row: VerifiedPairwiseObservation, margin: float) -> str:
        incumbent_values = dict(row.incumbent)
        challenger_values = dict(row.challenger)
        incumbent_missing = _missing(necessary, incumbent_values)
        challenger_missing = _missing(necessary, challenger_values)
        if incumbent_missing and not challenger_missing:
            return "challenger"
        if not incumbent_missing and challenger_missing:
            return "incumbent"
        if incumbent_missing or challenger_missing:
            return "incumbent"
        return (
            "challenger"
            if scorer.predict(challenger_values) - scorer.predict(incumbent_values) > margin
            else "incumbent"
        )

    deltas = sorted(
        {
            max(
                0.0,
                scorer.predict(dict(row.challenger))
                - scorer.predict(dict(row.incumbent)),
            )
            for row in calibration_rows
        }
    )
    thresholds = sorted({0.0, *(math.nextafter(delta, -math.inf) for delta in deltas)})
    candidates = []
    for margin in thresholds:
        selections = [result(row, margin) for row in calibration_rows]
        improvements = sum(
            selected == "challenger"
            and row.challenger_correct
            and not row.incumbent_correct
            for selected, row in zip(selections, calibration_rows, strict=True)
        )
        regressions = sum(
            selected == "challenger"
            and row.incumbent_correct
            and not row.challenger_correct
            for selected, row in zip(selections, calibration_rows, strict=True)
        )
        selected_correct = sum(
            row.challenger_correct if selected == "challenger" else row.incumbent_correct
            for selected, row in zip(selections, calibration_rows, strict=True)
        )
        switches = sum(selected == "challenger" for selected in selections)
        if regressions <= maximum_regressions:
            candidates.append(
                (selected_correct, improvements, -switches, margin, regressions, switches)
            )
    if not candidates:
        return None, {"admitted": False, "reason": "no_margin_meets_regression_ceiling"}
    best = max(candidates, key=lambda item: item[:3])
    (
        calibration_selected_correct,
        calibration_improvements,
        _negative_switches,
        margin,
        calibration_regressions,
        calibration_switches,
    ) = best
    admission_selections = [result(row, margin) for row in admission_rows]
    admission_improvements = sum(
        selected == "challenger"
        and row.challenger_correct
        and not row.incumbent_correct
        for selected, row in zip(admission_selections, admission_rows, strict=True)
    )
    admission_regressions = sum(
        selected == "challenger"
        and row.incumbent_correct
        and not row.challenger_correct
        for selected, row in zip(admission_selections, admission_rows, strict=True)
    )
    admission_selected_correct = sum(
        row.challenger_correct if selected == "challenger" else row.incumbent_correct
        for selected, row in zip(admission_selections, admission_rows, strict=True)
    )
    admission_switches = sum(selected == "challenger" for selected in admission_selections)
    admission_incumbent_correct = sum(row.incumbent_correct for row in admission_rows)
    admitted = bool(
        admission_improvements > 0
        and admission_regressions <= maximum_regressions
        and admission_selected_correct > admission_incumbent_correct
    )
    report = {
        "admitted": admitted,
        "reason": "independent_pairwise_admission_passed" if admitted else "no_admission_gain",
        "calibration_observations": len(calibration_rows),
        "calibration_selected_correct": calibration_selected_correct,
        "calibration_improvements": calibration_improvements,
        "calibration_regressions": calibration_regressions,
        "calibration_switches": calibration_switches,
        "admission_observations": len(admission_rows),
        "admission_incumbent_correct": admission_incumbent_correct,
        "admission_selected_correct": admission_selected_correct,
        "admission_improvements": admission_improvements,
        "admission_regressions": admission_regressions,
        "admission_switches": admission_switches,
        "maximum_regressions": maximum_regressions,
        "switch_margin": margin,
    }
    if not admitted:
        return None, report
    receipt_body = {
        "schema": CALIBRATED_CANDIDATE_SELECTOR_RECEIPT_SCHEMA,
        "algorithm": "necessary_then_calibrated_pairwise_v1",
        "necessary_selector_receipt_sha256": necessary.receipt_sha256,
        "scorer_receipt_sha256": scorer.receipt_sha256,
        "pairwise_calibration_observations": len(calibration_rows),
        "pairwise_calibration_sha256": _sha([row.to_dict() for row in calibration_rows]),
        "pairwise_admission_observations": len(admission_rows),
        "pairwise_admission_sha256": _sha([row.to_dict() for row in admission_rows]),
        "pairwise_report": report,
        "switch_margin": margin,
        "labels_available_to_calibration": True,
        "labels_available_to_runtime": False,
        "text_available_to_selector": False,
        "domain_identity_available_to_selector": False,
        "admitted": True,
    }
    selector = CalibratedCandidateSelector(
        necessary=necessary,
        scorer=scorer,
        switch_margin=margin,
        construction_receipt={**receipt_body, "receipt_sha256": _sha(receipt_body)},
    )
    return selector, report


def calibrated_candidate_selector_from_dict(value: Any) -> CalibratedCandidateSelector:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "necessary",
        "scorer",
        "switch_margin",
        "construction_receipt",
    }:
        raise ValueError("calibrated candidate selector payload is invalid")
    return CalibratedCandidateSelector(
        schema=str(value["schema"]),
        necessary=necessary_condition_selector_from_dict(value["necessary"]),
        scorer=calibrated_binary_scorer_from_dict(value["scorer"]),
        switch_margin=float(value["switch_margin"]),
        construction_receipt=dict(value["construction_receipt"]),
    )


__all__ = [
    "CALIBRATED_CANDIDATE_SELECTOR_RECEIPT_SCHEMA",
    "CALIBRATED_CANDIDATE_SELECTOR_SCHEMA",
    "MIN_PAIRWISE_CALIBRATION_OBSERVATIONS",
    "MIN_PAIRWISE_ADMISSION_OBSERVATIONS",
    "CalibratedCandidateSelector",
    "VerifiedPairwiseObservation",
    "build_calibrated_candidate_selector",
    "calibrated_candidate_selector_from_dict",
]
