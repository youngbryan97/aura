"""Matched evaluation and lesions for the learned semantic transducer."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_ir import SemanticIRInstruction, SemanticProgramIR
from core.learning.semantic_program_transducer import (
    LinearClassifierHead,
    LinearPointerHead,
    MultiViewClassifierHead,
    SemanticProgramTransducer,
    SemanticTransducerTrainingExample,
)


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
class SemanticProgramEvaluation:
    arm: str
    split: str
    total: int
    accepted: int
    program_exact: int
    operation_exact: int
    argument_exact: int
    input_span_exact: int
    attribution_exact: int
    full_ir_exact: int
    answer_emitted: int
    answer_exact: int
    rows: tuple[dict[str, Any], ...]

    @property
    def program_accuracy(self) -> float:
        return self.program_exact / self.total if self.total else 0.0

    @property
    def full_ir_accuracy(self) -> float:
        return self.full_ir_exact / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "split": self.split,
            "total": self.total,
            "accepted": self.accepted,
            "program_exact": self.program_exact,
            "program_accuracy": self.program_accuracy,
            "operation_exact": self.operation_exact,
            "argument_exact": self.argument_exact,
            "input_span_exact": self.input_span_exact,
            "attribution_exact": self.attribution_exact,
            "full_ir_exact": self.full_ir_exact,
            "full_ir_accuracy": self.full_ir_accuracy,
            "answer_emitted": self.answer_emitted,
            "answer_exact": self.answer_exact,
            "rows": list(self.rows),
        }


def evaluate_semantic_program_transducer(
    model: SemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    split: str,
    arm: str = "treatment",
    hidden_transform: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> SemanticProgramEvaluation:
    """Measure exact program and grounding without consulting task outputs."""

    selected = tuple(item for item in examples if item.split == split)
    if not selected:
        raise ValueError(f"semantic evaluation split is empty: {split}")
    counts = {
        "accepted": 0,
        "program_exact": 0,
        "operation_exact": 0,
        "argument_exact": 0,
        "input_span_exact": 0,
        "attribution_exact": 0,
        "full_ir_exact": 0,
        "answer_emitted": 0,
        "answer_exact": 0,
    }
    rows: list[dict[str, Any]] = []
    for item in selected:
        hidden = item.hidden_states
        if hidden_transform is not None:
            hidden = hidden_transform(np.array(hidden, copy=True), item.ir.source_text_sha256)
        outcome = model.decode(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=hidden,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        predicted = outcome.ir
        predicted_answer: int | None = None
        if predicted is not None:
            try:
                predicted_answer = execute_semantic_program(
                    predicted,
                    item.public_inputs,
                ).result
            except (RuntimeError, TypeError, ValueError):
                predicted_answer = None
        expected_answer = item.ir.to_program().run(item.public_inputs)
        accepted = predicted is not None
        program_exact = bool(
            predicted is not None and predicted.to_program() == item.ir.to_program()
        )
        operation_exact = bool(
            predicted is not None
            and tuple(step.op for step in predicted.instructions)
            == tuple(step.op for step in item.ir.instructions)
        )
        argument_exact = bool(
            predicted is not None
            and tuple(step.args for step in predicted.instructions)
            == tuple(step.args for step in item.ir.instructions)
        )
        input_span_exact = bool(
            predicted is not None and predicted.input_spans == item.ir.input_spans
        )
        attribution_exact = bool(
            predicted is not None
            and tuple((step.operation_span, step.argument_spans) for step in predicted.instructions)
            == tuple((step.operation_span, step.argument_spans) for step in item.ir.instructions)
        )
        full_ir_exact = bool(
            program_exact
            and input_span_exact
            and attribution_exact
            and predicted is not None
            and predicted.source_token_ids == item.ir.source_token_ids
        )
        answer_emitted = predicted_answer is not None
        answer_exact = bool(predicted_answer is not None and predicted_answer == expected_answer)
        values = {
            "accepted": accepted,
            "program_exact": program_exact,
            "operation_exact": operation_exact,
            "argument_exact": argument_exact,
            "input_span_exact": input_span_exact,
            "attribution_exact": attribution_exact,
            "full_ir_exact": full_ir_exact,
            "answer_emitted": answer_emitted,
            "answer_exact": answer_exact,
        }
        for key, value in values.items():
            counts[key] += int(value)
        rows.append(
            {
                "source_text_sha256": item.ir.source_text_sha256,
                "construction_id": item.construction_id,
                "topology_id": item.topology_id,
                "refusal": outcome.refusal,
                **values,
            }
        )
    return SemanticProgramEvaluation(
        arm=arm,
        split=split,
        total=len(selected),
        rows=tuple(rows),
        **counts,
    )


def shuffle_hidden_tokens(hidden: np.ndarray, identity: str) -> np.ndarray:
    """Matched grounding lesion: retain vectors and compute, move token binding."""

    seed = int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:16], 16)
    order = list(range(hidden.shape[0]))
    random.Random(seed).shuffle(order)
    if order == list(range(hidden.shape[0])) and len(order) > 1:
        order = order[1:] + order[:1]
    return np.ascontiguousarray(hidden[order])


def coefficient_lesion(
    model: SemanticProgramTransducer,
) -> SemanticProgramTransducer:
    """Remove feature dependence while preserving topology and learned biases."""

    pointers = {
        role: LinearPointerHead(
            np.zeros_like(head.start_weight),
            head.start_bias,
            np.zeros_like(head.end_weight),
            head.end_bias,
        )
        for role, head in model.pointer_heads.items()
    }

    def lesion_classifier(
        head: LinearClassifierHead | MultiViewClassifierHead,
    ) -> LinearClassifierHead | MultiViewClassifierHead:
        if isinstance(head, MultiViewClassifierHead):
            return MultiViewClassifierHead(
                head.modes,
                tuple(
                    lesion_classifier(component)
                    for component in head.heads
                    if isinstance(component, LinearClassifierHead)
                ),
            )
        return LinearClassifierHead(
            head.labels,
            np.zeros_like(head.weight),
            np.array(head.bias, copy=True),
        )

    operations = tuple(lesion_classifier(head) for head in model.operation_heads)
    arguments = tuple(
        tuple(lesion_classifier(head) for head in heads) for heads in model.argument_heads
    )
    coefficient_body = {
        "pointer_heads": {role: pointers[role].to_dict() for role in sorted(pointers)},
        "operation_heads": [head.to_dict() for head in operations],
        "argument_heads": [[head.to_dict() for head in heads] for heads in arguments],
    }
    receipt_body = {
        key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"
    }
    receipt_body["coefficient_sha256"] = _sha(coefficient_body)
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    return SemanticProgramTransducer(
        schema=model.schema,
        hidden_size=model.hidden_size,
        model_basis_sha256=model.model_basis_sha256,
        input_count=model.input_count,
        step_count=model.step_count,
        argument_arities=model.argument_arities,
        pointer_heads=pointers,
        operation_heads=operations,
        argument_heads=arguments,
        training_receipt=receipt,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )


def label_permuted_training_examples(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    seed: int = 982451653,
) -> tuple[SemanticTransducerTrainingExample, ...]:
    """Pair each training hidden sequence with a different valid program label."""

    training = [item for item in examples if item.split == "train"]
    if len(training) < 2:
        raise ValueError("semantic label null needs at least two training examples")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(training):
        groups.setdefault(item.ir.to_program().sha(), []).append(index)
    largest_group = max(len(indices) for indices in groups.values())
    if largest_group * 2 > len(training):
        raise ValueError("semantic label null support is too concentrated to derange")
    group_order = sorted(groups)
    rng.shuffle(group_order)
    ordered_indices: list[int] = []
    for key in group_order:
        indices = list(groups[key])
        rng.shuffle(indices)
        ordered_indices.extend(indices)
    rotated = ordered_indices[largest_group:] + ordered_indices[:largest_group]
    donor_indices = [0] * len(training)
    for recipient, donor in zip(ordered_indices, rotated, strict=True):
        donor_indices[recipient] = donor
    if not all(
        training[index].ir.to_program() != training[donor].ir.to_program()
        for index, donor in enumerate(donor_indices)
    ):
        raise AssertionError("validated semantic label derangement construction failed")

    replacements: dict[int, SemanticTransducerTrainingExample] = {}
    for index, donor_index in enumerate(donor_indices):
        current = training[index]
        donor = training[donor_index]
        instructions = tuple(
            SemanticIRInstruction(
                op=donor.ir.instructions[step].op,
                args=donor.ir.instructions[step].args,
                operation_span=current.ir.instructions[step].operation_span,
                argument_spans=current.ir.instructions[step].argument_spans,
                depends_on=donor.ir.instructions[step].depends_on,
            )
            for step in range(len(current.ir.instructions))
        )
        ir = SemanticProgramIR(
            source_token_ids=current.ir.source_token_ids,
            source_text_sha256=current.ir.source_text_sha256,
            input_spans=current.ir.input_spans,
            instructions=instructions,
            report_value=current.ir.report_value,
            model_basis_receipt_sha256=current.ir.model_basis_receipt_sha256,
            transducer_receipt_sha256=current.ir.transducer_receipt_sha256,
        )
        replacements[id(current)] = replace(current, ir=ir)
    return tuple(replacements.get(id(item), item) for item in examples)


__all__ = [
    "SemanticProgramEvaluation",
    "coefficient_lesion",
    "evaluate_semantic_program_transducer",
    "label_permuted_training_examples",
    "shuffle_hidden_tokens",
]
