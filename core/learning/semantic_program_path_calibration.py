"""Calibrate semantic path arbitration from independently verified source tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from core.evidence.calibrated_binary import (
    VerifiedBinaryObservation,
    fit_calibrated_binary_scorer,
)
from core.evidence.calibrated_candidate_selector import (
    CalibratedCandidateSelector,
    VerifiedPairwiseObservation,
    build_calibrated_candidate_selector,
)
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    build_necessary_condition_selector,
)
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_path_ensemble import (
    EXECUTABLE_PROGRAM_CONDITION,
    EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
    SEMANTIC_PATH_QUALITY_FEATURES,
    semantic_path_selection_values,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

SEMANTIC_PATH_CALIBRATION_SCHEMA: Final = "aura.semantic_program_path_calibration.v1"
SEMANTIC_PATH_EVIDENCE_CALIBRATION_SCHEMA: Final = (
    "aura.semantic_program_path_evidence_calibration.v1"
)
_CALIBRATION_SPLITS: Final = frozenset({"validation", "tuning", "admission"})


class _SemanticPath(Protocol):
    receipt_sha256: str
    model_basis_sha256: str

    def decode(
        self,
        *,
        source_token_ids: Sequence[int],
        hidden_states: Any,
        public_inputs: Sequence[Any],
        source_text_sha256: str,
        model_basis_sha256: str,
    ) -> SemanticTransductionOutcome: ...


@dataclass(frozen=True, slots=True)
class VerifiedSemanticPathObservation:
    """One text-blind path comparison whose outcomes were verified externally."""

    pair: VerifiedPairwiseObservation
    calibration_split: str
    construction_id: str
    topology_id: str

    def __post_init__(self) -> None:
        if (
            self.calibration_split not in _CALIBRATION_SPLITS
            or not isinstance(self.construction_id, str)
            or not self.construction_id
            or not isinstance(self.topology_id, str)
            or not self.topology_id
            or {name for name, _value in self.pair.incumbent}
            != set(SEMANTIC_PATH_QUALITY_FEATURES)
        ):
            raise ValueError("verified semantic path observation is invalid")

    @classmethod
    def from_mappings(
        cls,
        *,
        incumbent: Mapping[str, float],
        challenger: Mapping[str, float],
        incumbent_correct: bool,
        challenger_correct: bool,
        source_ref: str,
        calibration_split: str,
        construction_id: str,
        topology_id: str,
    ) -> VerifiedSemanticPathObservation:
        return cls(
            pair=VerifiedPairwiseObservation.from_mappings(
                incumbent=incumbent,
                challenger=challenger,
                incumbent_correct=incumbent_correct,
                challenger_correct=challenger_correct,
                source_ref=source_ref,
            ),
            calibration_split=calibration_split,
            construction_id=construction_id,
            topology_id=topology_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ref": self.pair.source_ref,
            "calibration_split": self.calibration_split,
            "construction_id": self.construction_id,
            "topology_id": self.topology_id,
            "incumbent_correct": self.pair.incumbent_correct,
            "challenger_correct": self.pair.challenger_correct,
            "incumbent_selection_values": dict(self.pair.incumbent),
            "challenger_selection_values": dict(self.pair.challenger),
        }


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


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_correct(
    item: SemanticTransducerTrainingExample,
    outcome: SemanticTransductionOutcome,
) -> bool:
    if outcome.ir is None:
        return False
    try:
        actual = execute_semantic_program(outcome.ir, item.public_inputs).result
    except (RuntimeError, TypeError, ValueError):
        return False
    expected = item.ir.to_program().run(item.public_inputs)
    return actual == expected


def _decode(
    path: _SemanticPath,
    item: SemanticTransducerTrainingExample,
) -> SemanticTransductionOutcome:
    return path.decode(
        source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states,
        public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
    )


def calibrate_semantic_program_path_evidence(
    *,
    model_basis_sha256: str,
    incumbent_receipt_sha256: str,
    challenger_receipt_sha256: str,
    observations: Sequence[VerifiedSemanticPathObservation],
    evidence_source_receipts: Sequence[str],
) -> tuple[CalibratedCandidateSelector | None, dict[str, Any]]:
    """Fit and independently admit arbitration from reusable verified evidence."""
    rows = tuple(observations)
    source_receipts = tuple(sorted(evidence_source_receipts))
    if (
        not _is_sha256(model_basis_sha256)
        or not _is_sha256(incumbent_receipt_sha256)
        or not _is_sha256(challenger_receipt_sha256)
        or incumbent_receipt_sha256 == challenger_receipt_sha256
        or not rows
        or not source_receipts
        or len(set(source_receipts)) != len(source_receipts)
        or any(not _is_sha256(receipt) for receipt in source_receipts)
        or any(not isinstance(row, VerifiedSemanticPathObservation) for row in rows)
        or len({row.pair.source_ref for row in rows}) != len(rows)
        or {row.calibration_split for row in rows} != _CALIBRATION_SPLITS
    ):
        raise ValueError("semantic path evidence calibration contract is invalid")

    fit_rows: list[VerifiedBinaryObservation] = []
    scorer_calibration_rows: list[VerifiedBinaryObservation] = []
    pairwise_tuning: list[VerifiedPairwiseObservation] = []
    pairwise_admission: list[VerifiedPairwiseObservation] = []
    counts = {
        split: {"examples": 0, "incumbent": 0, "challenger": 0}
        for split in sorted(_CALIBRATION_SPLITS)
    }
    for row in rows:
        pair = row.pair
        split_counts = counts[row.calibration_split]
        split_counts["examples"] += 1
        split_counts["incumbent"] += int(pair.incumbent_correct)
        split_counts["challenger"] += int(pair.challenger_correct)
        if row.calibration_split in {"validation", "tuning"}:
            destination = (
                fit_rows
                if row.calibration_split == "validation"
                else scorer_calibration_rows
            )
            destination.extend(
                (
                    VerifiedBinaryObservation(
                        values=pair.incumbent,
                        verified_correct=pair.incumbent_correct,
                        source_ref=f"{pair.source_ref}:incumbent",
                    ),
                    VerifiedBinaryObservation(
                        values=pair.challenger,
                        verified_correct=pair.challenger_correct,
                        source_ref=f"{pair.source_ref}:challenger",
                    ),
                )
            )
        if row.calibration_split == "tuning":
            pairwise_tuning.append(pair)
        elif row.calibration_split == "admission":
            pairwise_admission.append(pair)

    scorer, scorer_report = fit_calibrated_binary_scorer(
        fit_rows,
        scorer_calibration_rows,
    )
    selector: CalibratedCandidateSelector | None = None
    selector_report: dict[str, Any] = {
        "admitted": False,
        "reason": "candidate_quality_scorer_not_admitted",
    }
    if scorer is not None:
        necessary = build_necessary_condition_selector(
            (
                NecessaryEvidenceCondition(
                    name=EXECUTABLE_PROGRAM_CONDITION,
                    minimum=1.0,
                    necessity_contract=EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
                ),
            )
        )
        selector, selector_report = build_calibrated_candidate_selector(
            necessary=necessary,
            scorer=scorer,
            calibration_rows=pairwise_tuning,
            admission_rows=pairwise_admission,
            maximum_regressions=0,
        )

    serialized_rows = [row.to_dict() for row in rows]
    body = {
        "schema": SEMANTIC_PATH_EVIDENCE_CALIBRATION_SCHEMA,
        "admitted": selector is not None,
        "reason": (
            "independent_mixed_evidence_calibration_passed"
            if selector is not None
            else (
                "candidate_quality_scorer_not_admitted"
                if scorer is None
                else "pairwise_selector_not_admitted"
            )
        ),
        "model_basis_sha256": model_basis_sha256,
        "incumbent_receipt_sha256": incumbent_receipt_sha256,
        "challenger_receipt_sha256": challenger_receipt_sha256,
        "evidence_source_receipts": list(source_receipts),
        "evidence_source_receipts_sha256": _sha(source_receipts),
        "evidence_rows_sha256": _sha(serialized_rows),
        "counts": counts,
        "evidence_rows": serialized_rows,
        "scorer_report": scorer_report,
        "selector_report": selector_report,
        "selector_receipt_sha256": (
            selector.receipt_sha256 if selector is not None else None
        ),
        "verified_outcomes_available_to_calibration": True,
        "verified_outcomes_available_to_runtime": False,
        "fresh_target_examples_available_to_build": False,
        "text_available_to_selector": False,
        "domain_identity_available_to_selector": False,
        "serving_authority": False,
    }
    return selector, {**body, "report_sha256": _sha(body)}


def calibrate_semantic_program_paths(
    *,
    incumbent: _SemanticPath,
    challenger: _SemanticPath,
    source_examples: Sequence[SemanticTransducerTrainingExample],
) -> tuple[CalibratedCandidateSelector | None, dict[str, Any]]:
    """Fit on source validation, tune on source test, admit on disjoint source test."""
    examples = tuple(source_examples)
    if (
        not examples
        or incumbent.receipt_sha256 == challenger.receipt_sha256
        or incumbent.model_basis_sha256 != challenger.model_basis_sha256
        or any(item.split not in {"validation", "test"} for item in examples)
        or any(
            item.ir.model_basis_receipt_sha256 != incumbent.model_basis_sha256
            for item in examples
        )
        or len({item.ir.source_text_sha256 for item in examples}) != len(examples)
    ):
        raise ValueError("semantic path calibration source contract is invalid")
    validation = tuple(item for item in examples if item.split == "validation")
    test = tuple(sorted(
        (item for item in examples if item.split == "test"),
        key=lambda item: item.ir.source_text_sha256,
    ))
    tuning = test[::2]
    admission = test[1::2]
    if not validation or not tuning or not admission:
        raise ValueError("semantic path calibration requires three non-empty source splits")

    fit_rows: list[VerifiedBinaryObservation] = []
    tuning_rows: list[VerifiedBinaryObservation] = []
    pairwise_tuning: list[VerifiedPairwiseObservation] = []
    pairwise_admission: list[VerifiedPairwiseObservation] = []
    counts = {
        "validation": {"incumbent": 0, "challenger": 0},
        "tuning": {"incumbent": 0, "challenger": 0},
        "admission": {"incumbent": 0, "challenger": 0},
    }
    evidence_rows: list[dict[str, Any]] = []
    for split_name, split_examples in (
        ("validation", validation),
        ("tuning", tuning),
        ("admission", admission),
    ):
        for item in split_examples:
            incumbent_outcome = _decode(incumbent, item)
            challenger_outcome = _decode(challenger, item)
            incumbent_correct = _is_correct(item, incumbent_outcome)
            challenger_correct = _is_correct(item, challenger_outcome)
            counts[split_name]["incumbent"] += int(incumbent_correct)
            counts[split_name]["challenger"] += int(challenger_correct)
            incumbent_values = semantic_path_selection_values(incumbent_outcome)
            challenger_values = semantic_path_selection_values(challenger_outcome)
            evidence_rows.append(
                {
                    "source_text_sha256": item.ir.source_text_sha256,
                    "source_split": item.split,
                    "calibration_split": split_name,
                    "construction_id": item.construction_id,
                    "topology_id": item.topology_id,
                    "incumbent_correct": incumbent_correct,
                    "challenger_correct": challenger_correct,
                    "incumbent_selection_values": incumbent_values,
                    "challenger_selection_values": challenger_values,
                }
            )
            if split_name == "validation":
                destination = fit_rows
            elif split_name == "tuning":
                destination = tuning_rows
            else:
                destination = None
            if destination is not None:
                destination.extend(
                    (
                        VerifiedBinaryObservation.from_mapping(
                            incumbent_values,
                            verified_correct=incumbent_correct,
                            source_ref=(
                                f"semantic_source:{split_name}:incumbent:"
                                f"{item.ir.source_text_sha256}"
                            ),
                        ),
                        VerifiedBinaryObservation.from_mapping(
                            challenger_values,
                            verified_correct=challenger_correct,
                            source_ref=(
                                f"semantic_source:{split_name}:challenger:"
                                f"{item.ir.source_text_sha256}"
                            ),
                        ),
                    )
                )
            pair = VerifiedPairwiseObservation.from_mappings(
                incumbent=incumbent_values,
                challenger=challenger_values,
                incumbent_correct=incumbent_correct,
                challenger_correct=challenger_correct,
                source_ref=f"semantic_source:{split_name}:{item.ir.source_text_sha256}",
            )
            if split_name == "tuning":
                pairwise_tuning.append(pair)
            elif split_name == "admission":
                pairwise_admission.append(pair)

    scorer, scorer_report = fit_calibrated_binary_scorer(fit_rows, tuning_rows)
    if scorer is None:
        return None, {
            "schema": SEMANTIC_PATH_CALIBRATION_SCHEMA,
            "admitted": False,
            "reason": "candidate_quality_scorer_not_admitted",
            "counts": counts,
            "scorer_report": scorer_report,
        }
    necessary = build_necessary_condition_selector(
        (
            NecessaryEvidenceCondition(
                name=EXECUTABLE_PROGRAM_CONDITION,
                minimum=1.0,
                necessity_contract=EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
            ),
        )
    )
    selector, selector_report = build_calibrated_candidate_selector(
        necessary=necessary,
        scorer=scorer,
        calibration_rows=pairwise_tuning,
        admission_rows=pairwise_admission,
        maximum_regressions=0,
    )
    body = {
        "schema": SEMANTIC_PATH_CALIBRATION_SCHEMA,
        "admitted": selector is not None,
        "reason": (
            "independent_source_calibration_passed"
            if selector is not None
            else "pairwise_selector_not_admitted"
        ),
        "model_basis_sha256": incumbent.model_basis_sha256,
        "incumbent_receipt_sha256": incumbent.receipt_sha256,
        "challenger_receipt_sha256": challenger.receipt_sha256,
        "validation_examples": len(validation),
        "tuning_examples": len(tuning),
        "admission_examples": len(admission),
        "source_examples_sha256": _sha(
            [
                {
                    "source_text_sha256": item.ir.source_text_sha256,
                    "split": item.split,
                    "construction_id": item.construction_id,
                    "topology_id": item.topology_id,
                }
                for item in examples
            ]
        ),
        "counts": counts,
        "evidence_rows": evidence_rows,
        "scorer_report": scorer_report,
        "selector_report": selector_report,
        "selector_receipt_sha256": (
            selector.receipt_sha256 if selector is not None else None
        ),
        "expected_answers_available_to_source_verifier": True,
        "expected_answers_available_to_paths": False,
        "expected_answers_available_to_runtime": False,
        "target_examples_available_to_build": False,
        "text_available_to_selector": False,
        "domain_identity_available_to_selector": False,
    }
    return selector, {**body, "report_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PATH_CALIBRATION_SCHEMA",
    "SEMANTIC_PATH_EVIDENCE_CALIBRATION_SCHEMA",
    "VerifiedSemanticPathObservation",
    "calibrate_semantic_program_path_evidence",
    "calibrate_semantic_program_paths",
]
