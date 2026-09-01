"""Typed semantic program proposed by a learned token-level transducer.

This is the boundary between language understanding and execution. A model may
propose operations and pointers into public source tokens; this module decides
whether that proposal is a complete, well-typed, causally connected program.
It never reads an expected answer and never repairs a malformed proposal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.induced_neural_procedure import (
    LoweredInducedProcedure,
    lower_induced_program,
)
from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Instruction, Program

SEMANTIC_PROGRAM_IR_SCHEMA: Final = "aura.semantic_program_ir.v1"
SEMANTIC_PROGRAM_IR_RECEIPT_SCHEMA: Final = "aura.semantic_program_ir_receipt.v1"
MAX_SEMANTIC_PROGRAM_TOKENS: Final = 512
MAX_SEMANTIC_PROGRAM_INPUTS: Final = 8
MAX_SEMANTIC_PROGRAM_STEPS: Final = 16
MAX_SEMANTIC_SEQUENCE_ITEMS: Final = 64

type SemanticValue = int | tuple[int, ...]


def normalize_semantic_value(value: Any) -> SemanticValue:
    """Convert untrusted JSON-shaped data into the closed exact value algebra."""

    if type(value) is int:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= MAX_SEMANTIC_SEQUENCE_ITEMS:
        normalized = tuple(value)
        if all(type(item) is int for item in normalized):
            return normalized
    raise ValueError("semantic value is outside the exact integer/sequence algebra")


def semantic_value_to_json(value: SemanticValue) -> int | list[int]:
    """Return the canonical JSON representation of one validated value."""

    normalized = normalize_semantic_value(value)
    return list(normalized) if isinstance(normalized, tuple) else normalized


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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


@dataclass(frozen=True, slots=True, order=True)
class TokenSpan:
    """One non-empty half-open span in the measured source token sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("semantic token span is invalid")

    def validate_bound(self, token_count: int) -> None:
        if self.end > token_count:
            raise ValueError("semantic token span exceeds the measured source")

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class SemanticIRInstruction:
    """One SSA instruction with learned source and pointer attribution."""

    op: str
    args: tuple[int, ...]
    operation_span: TokenSpan
    argument_spans: tuple[TokenSpan, ...]
    depends_on: tuple[int, ...]

    def __post_init__(self) -> None:
        primitive = PRIMITIVES_BY_NAME.get(self.op)
        if primitive is None:
            raise ValueError("semantic IR operation is outside the frozen vocabulary")
        if (
            not isinstance(self.args, tuple)
            or len(self.args) != primitive.arity
            or any(type(value) is not int or value < 0 for value in self.args)
            or not isinstance(self.argument_spans, tuple)
            or len(self.argument_spans) != primitive.arity
            or not isinstance(self.depends_on, tuple)
            or tuple(sorted(set(self.depends_on))) != self.depends_on
            or any(type(value) is not int or value < 0 for value in self.depends_on)
        ):
            raise ValueError("semantic IR instruction fields are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": list(self.args),
            "operation_span": self.operation_span.to_dict(),
            "argument_spans": [span.to_dict() for span in self.argument_spans],
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True, slots=True)
class SemanticProgramIR:
    """A complete answer-blind program grounded in measured source tokens."""

    source_token_ids: tuple[int, ...]
    source_text_sha256: str
    input_spans: tuple[TokenSpan, ...]
    instructions: tuple[SemanticIRInstruction, ...]
    report_value: int
    model_basis_receipt_sha256: str
    transducer_receipt_sha256: str
    schema: str = SEMANTIC_PROGRAM_IR_SCHEMA

    def __post_init__(self) -> None:
        token_count = len(self.source_token_ids)
        if (
            self.schema != SEMANTIC_PROGRAM_IR_SCHEMA
            or not 1 <= token_count <= MAX_SEMANTIC_PROGRAM_TOKENS
            or any(type(token) is not int or token < 0 for token in self.source_token_ids)
            or not _is_sha256(self.source_text_sha256)
            or not 1 <= len(self.input_spans) <= MAX_SEMANTIC_PROGRAM_INPUTS
            or len(set(self.input_spans)) != len(self.input_spans)
            or not 1 <= len(self.instructions) <= MAX_SEMANTIC_PROGRAM_STEPS
            or not _is_sha256(self.model_basis_receipt_sha256)
            or not _is_sha256(self.transducer_receipt_sha256)
        ):
            raise ValueError("semantic program IR envelope is invalid")
        for span in self.input_spans:
            span.validate_bound(token_count)
        if any(
            left.start < right.end and right.start < left.end
            for index, left in enumerate(self.input_spans)
            for right in self.input_spans[index + 1 :]
        ):
            raise ValueError("semantic program input spans overlap")

        n_inputs = len(self.input_spans)
        for ordinal, instruction in enumerate(self.instructions):
            output = n_inputs + ordinal
            if any(argument >= output for argument in instruction.args):
                raise ValueError("semantic program is not forward SSA")
            instruction.operation_span.validate_bound(token_count)
            for span in instruction.argument_spans:
                span.validate_bound(token_count)
            expected_dependencies = tuple(
                sorted(
                    {argument - n_inputs for argument in instruction.args if argument >= n_inputs}
                )
            )
            if instruction.depends_on != expected_dependencies:
                raise ValueError("semantic program dependency receipt differs from its pointers")

        terminal = n_inputs + len(self.instructions) - 1
        if type(self.report_value) is not int or self.report_value != terminal:
            raise ValueError("semantic program must report its terminal SSA value")
        required = {self.report_value}
        for ordinal in range(len(self.instructions) - 1, -1, -1):
            output = n_inputs + ordinal
            if output in required:
                required.update(self.instructions[ordinal].args)
        expected_outputs = set(range(n_inputs, terminal + 1))
        if not expected_outputs.issubset(required):
            raise ValueError("semantic program contains non-causal decorative steps")

    @property
    def n_inputs(self) -> int:
        return len(self.input_spans)

    def to_program(self) -> Program:
        return Program(
            n_inputs=self.n_inputs,
            instructions=tuple(
                Instruction(instruction.op, instruction.args) for instruction in self.instructions
            ),
        )

    @property
    def alpha_normalized_sha256(self) -> str:
        """Identity of the computation independent of wording and span position."""

        return _sha(
            {
                "n_inputs": self.n_inputs,
                "instructions": [
                    [instruction.op, list(instruction.args)] for instruction in self.instructions
                ],
                "report_value": self.report_value,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_token_ids": list(self.source_token_ids),
            "source_text_sha256": self.source_text_sha256,
            "input_spans": [span.to_dict() for span in self.input_spans],
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "report_value": self.report_value,
            "model_basis_receipt_sha256": self.model_basis_receipt_sha256,
            "transducer_receipt_sha256": self.transducer_receipt_sha256,
        }

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": SEMANTIC_PROGRAM_IR_RECEIPT_SCHEMA,
            "ir_sha256": _sha(self.to_dict()),
            "alpha_normalized_sha256": self.alpha_normalized_sha256,
            "source_token_ids_sha256": _sha(self.source_token_ids),
            "source_text_sha256": self.source_text_sha256,
            "model_basis_receipt_sha256": self.model_basis_receipt_sha256,
            "transducer_receipt_sha256": self.transducer_receipt_sha256,
            "input_count": self.n_inputs,
            "step_count": len(self.instructions),
            "all_steps_causally_load_bearing": True,
            "expected_answer_available": False,
            "verifier_trace_available": False,
            "generated_compiler_text_available": False,
            "correctness_authority": False,
        }
        return {**body, "receipt_sha256": _sha(body)}

    def lower(self, inputs: tuple[Any, ...]) -> LoweredInducedProcedure:
        """Lower validated learned semantics into the proven neural executor."""

        if len(inputs) != self.n_inputs:
            raise ValueError("semantic program public inputs do not match its pointers")
        return lower_induced_program(self.to_program(), inputs)


def semantic_program_ir_from_dict(payload: dict[str, Any]) -> SemanticProgramIR:
    """Reconstruct and revalidate an untrusted serialized proposal."""

    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "source_token_ids",
        "source_text_sha256",
        "input_spans",
        "instructions",
        "report_value",
        "model_basis_receipt_sha256",
        "transducer_receipt_sha256",
    }:
        raise ValueError("semantic program IR fields are invalid")

    def span(value: Any) -> TokenSpan:
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            raise ValueError("semantic token span fields are invalid")
        return TokenSpan(start=value["start"], end=value["end"])

    raw_instructions = payload["instructions"]
    if not isinstance(raw_instructions, list):
        raise ValueError("semantic program instruction list is invalid")
    instructions = []
    for value in raw_instructions:
        if not isinstance(value, dict) or set(value) != {
            "op",
            "args",
            "operation_span",
            "argument_spans",
            "depends_on",
        }:
            raise ValueError("semantic IR instruction fields are invalid")
        instructions.append(
            SemanticIRInstruction(
                op=value["op"],
                args=tuple(value["args"]),
                operation_span=span(value["operation_span"]),
                argument_spans=tuple(span(item) for item in value["argument_spans"]),
                depends_on=tuple(value["depends_on"]),
            )
        )
    input_spans = payload["input_spans"]
    if not isinstance(input_spans, list):
        raise ValueError("semantic program input spans are invalid")
    return SemanticProgramIR(
        schema=payload["schema"],
        source_token_ids=tuple(payload["source_token_ids"]),
        source_text_sha256=payload["source_text_sha256"],
        input_spans=tuple(span(item) for item in input_spans),
        instructions=tuple(instructions),
        report_value=payload["report_value"],
        model_basis_receipt_sha256=payload["model_basis_receipt_sha256"],
        transducer_receipt_sha256=payload["transducer_receipt_sha256"],
    )


__all__ = [
    "MAX_SEMANTIC_SEQUENCE_ITEMS",
    "MAX_SEMANTIC_PROGRAM_INPUTS",
    "MAX_SEMANTIC_PROGRAM_STEPS",
    "MAX_SEMANTIC_PROGRAM_TOKENS",
    "SEMANTIC_PROGRAM_IR_SCHEMA",
    "SemanticIRInstruction",
    "SemanticProgramIR",
    "SemanticValue",
    "TokenSpan",
    "normalize_semantic_value",
    "semantic_value_to_json",
    "semantic_program_ir_from_dict",
]
