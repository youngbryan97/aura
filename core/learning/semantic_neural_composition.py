"""Answer-blind typed workflows over the qualified semantic neural machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    OP_PAIR_ADD,
    OP_PAIR_COPY,
    OP_PAIR_DIV,
    OP_PAIR_EUCLID_STEP,
    OP_PAIR_MUL_IMMEDIATE,
    OP_PAIR_SET,
    OP_PAIR_SUB_IMMEDIATE,
    OP_RATIO_BAND,
    OP_RATIO_CHOICE,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine

PUBLIC_TYPED_WORKFLOW_SCHEMA: Final = "aura.public_typed_workflow.v1"
SEMANTIC_NEURAL_COMPOSITION_SCHEMA: Final = "aura.semantic_neural_composition.v1"
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1
_PREFIX: Final = "TYPED_WORKFLOW_V1 "
_PAIR_REGISTERS: Final = {f"r{index}": 2 * index for index in range(4)}
_SCALAR_REGISTERS: Final = {"s0": 8}


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


def _document(prompt: str) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.startswith(_PREFIX):
        raise ValueError("public typed workflow prefix is invalid")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("public typed workflow contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(prompt[len(_PREFIX) :], object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError, MemoryError, RecursionError):
        raise ValueError("public typed workflow JSON is invalid") from None
    if not isinstance(value, dict) or set(value) != {"schema", "initial", "steps", "report"}:
        raise ValueError("public typed workflow fields are invalid")
    if value["schema"] != PUBLIC_TYPED_WORKFLOW_SCHEMA:
        raise ValueError("public typed workflow schema is invalid")
    initial = value["initial"]
    steps = value["steps"]
    report = value["report"]
    if (
        not isinstance(initial, dict)
        or set(initial) != set(_PAIR_REGISTERS)
        or any(
            type(item) is not int or not 0 <= item <= MAX_PROCESS_INTEGER
            for item in initial.values()
        )
        or not isinstance(steps, list)
        or not 1 <= len(steps) <= 64
        or not isinstance(report, list)
        or not report
        or len(report) != len(set(report))
        or any(name not in {*_PAIR_REGISTERS, *_SCALAR_REGISTERS} for name in report)
    ):
        raise ValueError("public typed workflow payload is invalid")
    canonical = _PREFIX + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if prompt != canonical:
        raise ValueError("public typed workflow is not canonical")
    return value


def _digits(value: Any) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= MAX_PROCESS_INTEGER:
        raise ValueError("public typed workflow integer is outside the register bank")
    return value % PROCESS_RADIX, value // PROCESS_RADIX


def _pair_slot(value: Any, *, role: str) -> int:
    if not isinstance(value, str) or value not in _PAIR_REGISTERS:
        raise ValueError(f"public typed workflow {role} register is invalid")
    return _PAIR_REGISTERS[value]


def _scalar_slot(value: Any, *, role: str) -> int:
    if not isinstance(value, str) or value not in _SCALAR_REGISTERS:
        raise ValueError(f"public typed workflow {role} register is invalid")
    return _SCALAR_REGISTERS[value]


def _micro(opcode: int, *arguments: int) -> tuple[int, ...]:
    if (
        type(opcode) is not int
        or not 0 <= opcode < ACTION_NULL
        or len(arguments) > 6
        or any(type(value) is not int or not 0 <= value < ACTION_NULL for value in arguments)
    ):
        raise ValueError("public typed workflow instruction is invalid")
    return (opcode, *arguments, *(0 for _index in range(6 - len(arguments))))


def public_typed_workflow_document(prompt: str) -> dict[str, Any]:
    """Return validated public literals without deriving the result."""

    compile_public_typed_workflow(prompt)
    return _document(prompt)


def render_public_typed_workflow(document: dict[str, Any]) -> str:
    """Render one canonical answer-blind workflow after full validation."""

    if not isinstance(document, dict):
        raise TypeError("public typed workflow document is invalid")
    prompt = _PREFIX + json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    compile_public_typed_workflow(prompt)
    return prompt


@dataclass(frozen=True, slots=True)
class PublicTypedWorkflowProgram:
    public_prompt_sha256: str
    values: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.public_prompt_sha256) != 64
            or not self.values
            or any(
                len(row) != 8
                or any(type(value) is not int or not 0 <= value <= ACTION_NULL for value in row)
                for row in self.values
            )
            or self.values[-1][-1] != 1
            or any(row[-1] for row in self.values[:-1])
        ):
            raise ValueError("public typed workflow program is invalid")

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": PUBLIC_TYPED_WORKFLOW_SCHEMA,
            "public_prompt_sha256": self.public_prompt_sha256,
            "steps": len(self.values),
            "program_sha256": _sha(self.values),
            "source": "public_operation_literals_and_order_only",
            "verifier_answer_available": False,
            "private_state_trace_available": False,
            "derived_answer_fields_present": False,
            "correctness_authority": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def compile_public_typed_workflow(prompt: str) -> PublicTypedWorkflowProgram:
    """Compile a literal operation list into the existing recurrent instruction set."""

    document = _document(prompt)
    actions: list[tuple[int, ...]] = []
    for name, value in sorted(document["initial"].items()):
        low, high = _digits(value)
        actions.append(_micro(OP_PAIR_SET, _PAIR_REGISTERS[name], low, high))
    expected_fields = {
        "set": {"op", "dst", "value"},
        "add": {"op", "dst", "left", "right"},
        "mul": {"op", "dst", "factor"},
        "sub": {"op", "dst", "amount"},
        "copy": {"op", "dst", "src"},
        "euclid_step": {"op", "left", "right"},
        "div_exact": {"op", "dst", "numerator", "denominator"},
        "ratio_choice": {"op", "dst", "numerator", "denominator"},
        "ratio_band": {"op", "dst", "numerator", "denominator"},
    }
    for step in document["steps"]:
        if not isinstance(step, dict) or not isinstance(step.get("op"), str):
            raise ValueError("public typed workflow step is invalid")
        operation = step["op"]
        if operation not in expected_fields or set(step) != expected_fields[operation]:
            raise ValueError("public typed workflow operation fields are invalid")
        if operation == "set":
            low, high = _digits(step["value"])
            actions.append(_micro(OP_PAIR_SET, _pair_slot(step["dst"], role="destination"), low, high))
        elif operation == "add":
            actions.append(
                _micro(
                    OP_PAIR_ADD,
                    _pair_slot(step["dst"], role="destination"),
                    _pair_slot(step["left"], role="left"),
                    _pair_slot(step["right"], role="right"),
                )
            )
        elif operation in {"mul", "sub"}:
            immediate_name = "factor" if operation == "mul" else "amount"
            immediate = step[immediate_name]
            if type(immediate) is not int or not 0 <= immediate < ACTION_NULL:
                raise ValueError("public typed workflow immediate is invalid")
            actions.append(
                _micro(
                    OP_PAIR_MUL_IMMEDIATE if operation == "mul" else OP_PAIR_SUB_IMMEDIATE,
                    _pair_slot(step["dst"], role="destination"),
                    immediate,
                )
            )
        elif operation == "copy":
            actions.append(
                _micro(
                    OP_PAIR_COPY,
                    _pair_slot(step["dst"], role="destination"),
                    _pair_slot(step["src"], role="source"),
                )
            )
        elif operation == "euclid_step":
            actions.append(
                _micro(
                    OP_PAIR_EUCLID_STEP,
                    _pair_slot(step["left"], role="left"),
                    _pair_slot(step["right"], role="right"),
                )
            )
        elif operation == "div_exact":
            actions.append(
                _micro(
                    OP_PAIR_DIV,
                    _pair_slot(step["dst"], role="destination"),
                    _pair_slot(step["numerator"], role="numerator"),
                    _pair_slot(step["denominator"], role="denominator"),
                )
            )
        else:
            actions.append(
                _micro(
                    OP_RATIO_CHOICE if operation == "ratio_choice" else OP_RATIO_BAND,
                    _scalar_slot(step["dst"], role="destination"),
                    _pair_slot(step["numerator"], role="numerator"),
                    _pair_slot(step["denominator"], role="denominator"),
                )
            )
    values = tuple(
        (*action, int(index + 1 == len(actions))) for index, action in enumerate(actions)
    )
    return PublicTypedWorkflowProgram(
        public_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        values=values,
    )


@dataclass(frozen=True, slots=True)
class SemanticNeuralCompositionState:
    objective_sha256: str
    states: tuple[tuple[int, ...], ...]
    semantic_result: dict[str, Any]
    program_receipt: dict[str, Any]
    transition_receipts: tuple[dict[str, Any], ...]
    tissue_sha256: str


def execute_public_typed_workflow(
    prompt: str,
    *,
    machine: SemanticNeuralMachine | None = None,
) -> SemanticNeuralCompositionState:
    """Execute a public workflow without changing the qualified serving surface."""

    document = _document(prompt)
    program = compile_public_typed_workflow(prompt)
    active = SemanticNeuralMachine() if machine is None else machine
    states = [(0,) * 11]
    receipts = []
    for action in program.values:
        transition = active.transition(states[-1], action)
        states.append(transition.next_state)
        receipts.append(transition.receipt())
    terminal = states[-1][1:-1]
    result: dict[str, Any] = {}
    for name in document["report"]:
        if name == "s0":
            result[name] = terminal[8]
            continue
        index = int(name[1:])
        result[name] = active.decode_unsigned_pair(terminal[2 * index], terminal[2 * index + 1])
    return SemanticNeuralCompositionState(
        objective_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        states=tuple(states),
        semantic_result=result,
        program_receipt=program.receipt(),
        transition_receipts=tuple(receipts),
        tissue_sha256=active.tissue_sha256,
    )


__all__ = [
    "PUBLIC_TYPED_WORKFLOW_SCHEMA",
    "PublicTypedWorkflowProgram",
    "SemanticNeuralCompositionState",
    "compile_public_typed_workflow",
    "execute_public_typed_workflow",
    "public_typed_workflow_document",
    "render_public_typed_workflow",
]
