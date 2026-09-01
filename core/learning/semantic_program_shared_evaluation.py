"""Geometry-stratified evaluation for the shared semantic transducer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_shared_transducer import (
    SharedSemanticProgramTransducer,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
)


@dataclass(frozen=True, slots=True)
class SharedSemanticProgramEvaluation:
    arm: str
    split: str
    total: int
    accepted: int
    geometry_exact: int
    step_count_exact: int
    arity_exact: int
    program_exact: int
    operation_exact: int
    argument_exact: int
    input_span_exact: int
    answer_emitted: int
    answer_exact: int
    by_geometry: dict[str, dict[str, int]]
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "split": self.split,
            "total": self.total,
            "accepted": self.accepted,
            "geometry_exact": self.geometry_exact,
            "step_count_exact": self.step_count_exact,
            "arity_exact": self.arity_exact,
            "program_exact": self.program_exact,
            "operation_exact": self.operation_exact,
            "argument_exact": self.argument_exact,
            "input_span_exact": self.input_span_exact,
            "answer_emitted": self.answer_emitted,
            "answer_exact": self.answer_exact,
            "by_geometry": self.by_geometry,
            "geometry_macro_program_accuracy": (
                sum(
                    values["program_exact"] / values["total"]
                    for values in self.by_geometry.values()
                )
                / len(self.by_geometry)
            ),
            "geometry_macro_answer_accuracy": (
                sum(
                    values["answer_exact"] / values["total"] for values in self.by_geometry.values()
                )
                / len(self.by_geometry)
            ),
            "rows": list(self.rows),
        }


def evaluate_shared_semantic_program_transducer(
    model: SharedSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    split: str,
    arm: str = "treatment",
    hidden_transform: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> SharedSemanticProgramEvaluation:
    """Measure shape and semantics without supplying gold geometry to decode."""

    selected = tuple(item for item in examples if item.split == split)
    if not selected:
        raise ValueError(f"shared semantic evaluation split is empty: {split}")
    keys = (
        "accepted",
        "geometry_exact",
        "step_count_exact",
        "arity_exact",
        "program_exact",
        "operation_exact",
        "argument_exact",
        "input_span_exact",
        "answer_emitted",
        "answer_exact",
    )
    counts = {key: 0 for key in keys}
    by_geometry: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, **{key: 0 for key in keys}}
    )
    rows: list[dict[str, Any]] = []
    for item in selected:
        hidden = item.hidden_states
        if hidden_transform is not None:
            hidden = hidden_transform(np.array(hidden, copy=True), item.ir.source_text_sha256)
        outcome = model.decode(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=hidden,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        predicted = outcome.ir
        predicted_answer: int | None = None
        if predicted is not None:
            try:
                predicted_answer = execute_semantic_program(predicted, item.public_inputs).result
            except (RuntimeError, TypeError, ValueError):
                predicted_answer = None
        expected_answer = item.ir.to_program().run(item.public_inputs)
        expected_geometry = (
            item.ir.n_inputs,
            tuple(len(instruction.args) for instruction in item.ir.instructions),
        )
        predicted_geometry = (
            (
                predicted.n_inputs,
                tuple(len(instruction.args) for instruction in predicted.instructions),
            )
            if predicted is not None
            else None
        )
        values = {
            "accepted": predicted is not None,
            "geometry_exact": predicted_geometry == expected_geometry,
            "step_count_exact": bool(
                predicted is not None and len(predicted.instructions) == len(item.ir.instructions)
            ),
            "arity_exact": bool(
                predicted is not None
                and tuple(len(step.args) for step in predicted.instructions)
                == tuple(len(step.args) for step in item.ir.instructions)
            ),
            "program_exact": bool(
                predicted is not None and predicted.to_program() == item.ir.to_program()
            ),
            "operation_exact": bool(
                predicted is not None
                and tuple(step.op for step in predicted.instructions)
                == tuple(step.op for step in item.ir.instructions)
            ),
            "argument_exact": bool(
                predicted is not None
                and tuple(step.args for step in predicted.instructions)
                == tuple(step.args for step in item.ir.instructions)
            ),
            "input_span_exact": bool(
                predicted is not None and predicted.input_spans == item.ir.input_spans
            ),
            "answer_emitted": predicted_answer is not None,
            "answer_exact": bool(
                predicted_answer is not None and predicted_answer == expected_answer
            ),
        }
        geometry_name = f"inputs:{expected_geometry[0]}|arities:" + ",".join(
            str(value) for value in expected_geometry[1]
        )
        by_geometry[geometry_name]["total"] += 1
        for key, value in values.items():
            counts[key] += int(value)
            by_geometry[geometry_name][key] += int(value)
        rows.append(
            {
                "source_text_sha256": item.ir.source_text_sha256,
                "construction_id": item.construction_id,
                "topology_id": item.topology_id,
                "geometry": geometry_name,
                "refusal": outcome.refusal,
                **values,
            }
        )
    return SharedSemanticProgramEvaluation(
        arm=arm,
        split=split,
        total=len(selected),
        by_geometry={key: dict(value) for key, value in sorted(by_geometry.items())},
        rows=tuple(rows),
        **counts,
    )


__all__ = [
    "SharedSemanticProgramEvaluation",
    "evaluate_shared_semantic_program_transducer",
]
