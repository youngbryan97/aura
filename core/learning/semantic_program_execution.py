"""Exact execution and answer emission for learned semantic programs.

Language interpretation is neural; objective arithmetic is not. Once a
model-bound transducer has proposed a validated SSA program, this module runs
the repository's closed primitive implementation and emits the terminal answer
without asking a language model to reproduce or re-compute it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.procedure_induction import PRIMITIVE_SET_SHA
from core.learning.semantic_program_ir import (
    SemanticProgramIR,
    SemanticValue,
    normalize_semantic_value,
    semantic_value_to_json,
)

SEMANTIC_PROGRAM_EXECUTION_SCHEMA: Final = "aura.semantic_program_execution.v1"
SEMANTIC_PROGRAM_ANSWER_SCHEMA: Final = "aura.semantic_program_answer.v1"
SEMANTIC_PROGRAM_SEQUENCE_EXECUTION_SCHEMA: Final = "aura.semantic_program_execution.v2"
SEMANTIC_PROGRAM_SEQUENCE_ANSWER_SCHEMA: Final = "aura.semantic_program_answer.v2"


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
class SemanticProgramExecution:
    """Authenticated result of one validated learned program."""

    result: SemanticValue
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            result = normalize_semantic_value(self.result)
        except ValueError as exc:
            raise ValueError("semantic program execution result is invalid") from exc
        if result != self.result or not isinstance(self.receipt, dict):
            raise ValueError("semantic program execution result is invalid")
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (
            body.get("schema")
            not in {
                SEMANTIC_PROGRAM_EXECUTION_SCHEMA,
                SEMANTIC_PROGRAM_SEQUENCE_EXECUTION_SCHEMA,
            }
            or self.receipt.get("receipt_sha256") != _sha(body)
            or body.get("result_sha256") != _sha(semantic_value_to_json(result))
        ):
            raise ValueError("semantic program execution receipt is invalid")


def execute_semantic_program(
    ir: SemanticProgramIR,
    public_inputs: tuple[Any, ...],
) -> SemanticProgramExecution:
    """Execute validated semantics without any expected answer or model decode."""

    if not isinstance(ir, SemanticProgramIR):
        raise TypeError("semantic execution requires validated program IR")
    if not isinstance(public_inputs, tuple) or len(public_inputs) != ir.n_inputs:
        raise ValueError("semantic execution public inputs are invalid")
    try:
        inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    except ValueError as exc:
        raise ValueError("semantic execution public inputs are invalid") from exc
    try:
        result = normalize_semantic_value(ir.to_program().run(inputs))
    except ValueError as exc:
        raise RuntimeError("semantic program left the exact value algebra") from exc
    sequence_result = isinstance(result, tuple)
    body = {
        "schema": (
            SEMANTIC_PROGRAM_SEQUENCE_EXECUTION_SCHEMA
            if sequence_result
            else SEMANTIC_PROGRAM_EXECUTION_SCHEMA
        ),
        "ir_receipt_sha256": ir.receipt()["receipt_sha256"],
        "alpha_normalized_sha256": ir.alpha_normalized_sha256,
        "primitive_set_sha256": PRIMITIVE_SET_SHA,
        "public_inputs_sha256": _sha([semantic_value_to_json(value) for value in inputs]),
        "result_sha256": _sha(semantic_value_to_json(result)),
        "execution_engine": "closed_exact_objective_program",
        "expected_answer_available": False,
        "verifier_trace_available": False,
        "generated_text_available": False,
        "correctness_authority": False,
    }
    return SemanticProgramExecution(
        result=result,
        receipt={**body, "receipt_sha256": _sha(body)},
    )


def render_semantic_program_answer(execution: SemanticProgramExecution) -> str:
    """Emit one deterministic typed answer from authenticated execution state."""

    if not isinstance(execution, SemanticProgramExecution):
        raise TypeError("semantic answer requires authenticated execution")
    payload = {
        "schema": (
            SEMANTIC_PROGRAM_SEQUENCE_ANSWER_SCHEMA
            if isinstance(execution.result, tuple)
            else SEMANTIC_PROGRAM_ANSWER_SCHEMA
        ),
        "result": semantic_value_to_json(execution.result),
        "execution_receipt_sha256": execution.receipt["receipt_sha256"],
    }
    return "FINAL_ANSWER:" + json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = [
    "SEMANTIC_PROGRAM_ANSWER_SCHEMA",
    "SEMANTIC_PROGRAM_EXECUTION_SCHEMA",
    "SEMANTIC_PROGRAM_SEQUENCE_ANSWER_SCHEMA",
    "SEMANTIC_PROGRAM_SEQUENCE_EXECUTION_SCHEMA",
    "SemanticProgramExecution",
    "execute_semantic_program",
    "render_semantic_program_answer",
]
