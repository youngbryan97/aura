"""Typed bridge from learned operation composition to the language surface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.semantic_neural_composition import (
    SemanticNeuralCompositionState,
    execute_public_typed_workflow,
    public_typed_workflow_document,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine

COMPOSITION_DECODE_SCHEMA: Final = "aura.semantic_neural_composition_decode.v1"
COMPOSITION_STATE_CHANNEL_SCHEMA: Final = "aura.computed_typed_state.v1"
FINAL_MARKER: Final = "FINAL_ANSWER:"


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
class SemanticNeuralCompositionDecodeState:
    objective_sha256: str
    report: tuple[str, ...]
    semantic_result: dict[str, int]
    state_trajectory_sha256: str
    program_receipt_sha256: str
    transition_receipt_sha256s: tuple[str, ...]
    tissue_sha256: str
    schema: str = COMPOSITION_DECODE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != COMPOSITION_DECODE_SCHEMA
            or len(self.objective_sha256) != 64
            or not self.report
            or len(self.report) != len(set(self.report))
            or tuple(self.semantic_result) != self.report
            or any(type(value) is not int for value in self.semantic_result.values())
            or len(self.state_trajectory_sha256) != 64
            or len(self.program_receipt_sha256) != 64
            or not self.transition_receipt_sha256s
            or any(len(value) != 64 for value in self.transition_receipt_sha256s)
            or len(self.tissue_sha256) != 64
        ):
            raise ValueError("semantic composition decode state is invalid")

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "objective_sha256": self.objective_sha256,
            "report": list(self.report),
            "semantic_result_sha256": _sha(self.semantic_result),
            "state_trajectory_sha256": self.state_trajectory_sha256,
            "program_receipt_sha256": self.program_receipt_sha256,
            "transition_receipt_sha256s": list(self.transition_receipt_sha256s),
            "tissue_sha256": self.tissue_sha256,
            "student_rollin": True,
            "teacher_available": False,
            "private_trace_available": False,
            "verifier_available": False,
            "answer_key_available": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def execute_composition_decode_state(
    public_workflow: str,
    *,
    machine: SemanticNeuralMachine | None = None,
) -> SemanticNeuralCompositionDecodeState:
    """Execute one public workflow and preserve only authenticated result state."""

    document = public_typed_workflow_document(public_workflow)
    state: SemanticNeuralCompositionState = execute_public_typed_workflow(
        public_workflow,
        machine=machine,
    )
    program_receipt = state.program_receipt
    transition_receipts = state.transition_receipts
    if (
        program_receipt.get("verifier_answer_available") is not False
        or program_receipt.get("private_state_trace_available") is not False
        or any(
            receipt.get("teacher_available") is not False
            or receipt.get("private_trace_available") is not False
            for receipt in transition_receipts
        )
    ):
        raise RuntimeError("composition state crossed a private supervision boundary")
    return SemanticNeuralCompositionDecodeState(
        objective_sha256=state.objective_sha256,
        report=tuple(document["report"]),
        semantic_result=dict(state.semantic_result),
        state_trajectory_sha256=_sha(state.states),
        program_receipt_sha256=program_receipt["receipt_sha256"],
        transition_receipt_sha256s=tuple(
            receipt["receipt_sha256"] for receipt in transition_receipts
        ),
        tissue_sha256=state.tissue_sha256,
    )


def render_composition_decode_objective(public_workflow: str) -> str:
    """Expose a fixed public protocol without a result or private trace."""

    document = public_typed_workflow_document(public_workflow)
    fields = ",".join(document["report"])
    return (
        "COMPOSITION_QUERY_V1\n"
        f"workflow={public_workflow}\n"
        f"report={fields}\n"
        "Execute the public workflow in order. Return exactly one canonical JSON "
        f"object after {FINAL_MARKER}; its keys must be the report fields in report order."
    )


def render_composition_state_channel(
    state: SemanticNeuralCompositionDecodeState,
) -> str:
    """Serialize authenticated machine state as data, not a natural-language hint."""

    if not isinstance(state, SemanticNeuralCompositionDecodeState):
        raise TypeError("composition state channel requires typed state")
    payload = {
        "schema": COMPOSITION_STATE_CHANNEL_SCHEMA,
        "objective_sha256": state.objective_sha256,
        "report": list(state.report),
        "semantic_result": state.semantic_result,
        "state_receipt_sha256": state.receipt()["receipt_sha256"],
    }
    return "COMPUTED_TYPED_STATE_V1 " + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_composition_response(
    response: str,
    report: tuple[str, ...],
) -> dict[str, int] | None:
    """Parse the complete canonical answer contract with duplicate-key rejection."""

    if not isinstance(response, str) or response.count(FINAL_MARKER) != 1:
        return None
    prefix, encoded = response.split(FINAL_MARKER, 1)
    if prefix.strip():
        return None

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate composition answer key")
            value[key] = item
        return value

    try:
        parsed = json.loads(encoded.strip(), object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(parsed, dict)
        or tuple(parsed) != report
        or any(type(value) is not int for value in parsed.values())
    ):
        return None
    canonical = json.dumps(parsed, separators=(",", ":"), ensure_ascii=True)
    if encoded.strip() != canonical:
        return None
    return parsed


def render_composition_answer(state: SemanticNeuralCompositionDecodeState) -> str:
    """Return the canonical answer bytes represented by authenticated state."""

    return FINAL_MARKER + json.dumps(
        state.semantic_result,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = [
    "COMPOSITION_DECODE_SCHEMA",
    "COMPOSITION_STATE_CHANNEL_SCHEMA",
    "FINAL_MARKER",
    "SemanticNeuralCompositionDecodeState",
    "execute_composition_decode_state",
    "parse_composition_response",
    "render_composition_answer",
    "render_composition_decode_objective",
    "render_composition_state_channel",
]
