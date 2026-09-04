from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)
from core.learning.semantic_program_path_calibration import (
    SEMANTIC_PATH_EVIDENCE_CALIBRATION_SCHEMA,
    VerifiedSemanticPathObservation,
    calibrate_semantic_program_path_evidence,
    calibrate_semantic_program_paths,
)
from core.learning.semantic_program_path_ensemble import SEMANTIC_PATH_QUALITY_FEATURES
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

MODEL_BASIS = "a" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _ir(index: int, *, op: str = "add") -> SemanticProgramIR:
    return SemanticProgramIR(
        source_token_ids=(10 + index, 20, 30, 40),
        source_text_sha256=_sha(f"source:{index}"),
        input_spans=(TokenSpan(0, 1), TokenSpan(1, 2)),
        instructions=(
            SemanticIRInstruction(
                op=op,
                args=(0, 1),
                operation_span=TokenSpan(2, 3),
                argument_spans=(TokenSpan(0, 1), TokenSpan(1, 2)),
                depends_on=(),
            ),
        ),
        report_value=2,
        model_basis_receipt_sha256=MODEL_BASIS,
        transducer_receipt_sha256="b" * 64,
    )


def _examples() -> tuple[SemanticTransducerTrainingExample, ...]:
    hidden = np.asarray(
        ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=np.float32,
    )
    return tuple(
        SemanticTransducerTrainingExample(
            ir=_ir(index),
            hidden_states=hidden,
            split="validation" if index < 24 else "test",
            construction_id=f"construction:{index}",
            topology_id="binary",
            public_inputs=(index + 2, 3),
        )
        for index in range(48)
    )


@dataclass
class _Path:
    receipt_sha256: str
    correct: bool
    quality: float
    model_basis_sha256: str = MODEL_BASIS

    def decode(self, **kwargs):
        source_text_sha256 = kwargs["source_text_sha256"]
        index = next(
            index for index in range(48) if _sha(f"source:{index}") == source_text_sha256
        )
        ir = _ir(index, op="add" if self.correct else "mul")
        return SemanticTransductionOutcome(
            ir=ir,
            refusal=None,
            pointer_scores={
                "input:0": self.quality,
                "input:1": self.quality,
                "operation:0": self.quality,
                "argument_graph_mean": self.quality,
            },
            classification_confidences={
                "operation:0": self.quality,
                "argument_graph_margin": self.quality,
                "argument_graph_runner_up_available": 1.0,
            },
        )


def test_semantic_paths_are_calibrated_on_three_disjoint_source_splits() -> None:
    selector, report = calibrate_semantic_program_paths(
        incumbent=_Path("c" * 64, correct=False, quality=0.1),
        challenger=_Path("d" * 64, correct=True, quality=0.9),
        source_examples=_examples(),
    )

    assert selector is not None
    assert report["admitted"] is True
    assert report["validation_examples"] == 24
    assert report["tuning_examples"] == 12
    assert report["admission_examples"] == 12
    assert report["selector_report"]["admission_improvements"] == 12
    assert report["selector_report"]["admission_regressions"] == 0
    assert len(report["evidence_rows"]) == 48
    assert all("source_text" not in row for row in report["evidence_rows"])
    assert report["expected_answers_available_to_paths"] is False
    assert report["expected_answers_available_to_runtime"] is False
    assert report["target_examples_available_to_build"] is False


def test_semantic_calibration_refuses_same_path_and_duplicate_source() -> None:
    path = _Path("c" * 64, correct=True, quality=0.9)
    examples = _examples()

    try:
        calibrate_semantic_program_paths(
            incumbent=path,
            challenger=path,
            source_examples=examples,
        )
    except ValueError as exc:
        assert "source contract" in str(exc)
    else:
        raise AssertionError("same path must not calibrate against itself")

    try:
        calibrate_semantic_program_paths(
            incumbent=path,
            challenger=_Path("d" * 64, correct=False, quality=0.1),
            source_examples=(*examples, examples[0]),
        )
    except ValueError as exc:
        assert "source contract" in str(exc)
    else:
        raise AssertionError("duplicate source evidence must be refused")


def _selection_values(quality: float) -> dict[str, float]:
    values = {name: quality for name in SEMANTIC_PATH_QUALITY_FEATURES}
    values["executable_program"] = 1.0
    values["input_count"] = 5.0
    values["instruction_count"] = 4.0
    return values


def test_verified_evidence_recalibrates_across_multiple_geometries() -> None:
    observations = tuple(
        VerifiedSemanticPathObservation.from_mappings(
            incumbent=_selection_values(0.1),
            challenger=_selection_values(0.9),
            incumbent_correct=False,
            challenger_correct=True,
            source_ref=f"geometry:{index // 24}:{index}",
            calibration_split=(
                "validation" if index < 24 else "tuning" if index < 36 else "admission"
            ),
            construction_id=f"construction:{index}",
            topology_id="four_by_three" if index < 24 else "five_by_four",
        )
        for index in range(48)
    )

    selector, report = calibrate_semantic_program_path_evidence(
        model_basis_sha256="a" * 64,
        incumbent_receipt_sha256="b" * 64,
        challenger_receipt_sha256="c" * 64,
        observations=observations,
        evidence_source_receipts=("d" * 64, "e" * 64),
    )

    assert selector is not None
    assert report["schema"] == SEMANTIC_PATH_EVIDENCE_CALIBRATION_SCHEMA
    assert report["counts"]["admission"] == {
        "examples": 12,
        "incumbent": 0,
        "challenger": 12,
    }
    assert report["selector_report"]["admission_improvements"] == 12
    assert report["selector_report"]["admission_regressions"] == 0
    assert report["fresh_target_examples_available_to_build"] is False
    assert report["text_available_to_selector"] is False


def test_verified_evidence_refuses_duplicate_sources_and_missing_splits() -> None:
    row = VerifiedSemanticPathObservation.from_mappings(
        incumbent=_selection_values(0.1),
        challenger=_selection_values(0.9),
        incumbent_correct=False,
        challenger_correct=True,
        source_ref="duplicate",
        calibration_split="validation",
        construction_id="construction",
        topology_id="topology",
    )

    for observations in ((row,), (row, row)):
        try:
            calibrate_semantic_program_path_evidence(
                model_basis_sha256="a" * 64,
                incumbent_receipt_sha256="b" * 64,
                challenger_receipt_sha256="c" * 64,
                observations=observations,
                evidence_source_receipts=("d" * 64,),
            )
        except ValueError as exc:
            assert "contract" in str(exc)
        else:
            raise AssertionError("invalid reusable evidence must be refused")
