"""Calibrate semantic path arbitration from independently verified source tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
    semantic_path_selection_values,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

SEMANTIC_PATH_CALIBRATION_SCHEMA: Final = "aura.semantic_program_path_calibration.v1"


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
    "calibrate_semantic_program_paths",
]
