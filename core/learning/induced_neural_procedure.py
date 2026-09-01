"""Lower a learned straight-line procedure into the semantic neural machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.procedure_induction import PRIMITIVE_SET_SHA, Program
from core.learning.semantic_neural_composition import (
    MAX_PROCESS_INTEGER,
    PUBLIC_TYPED_WORKFLOW_SCHEMA,
    SemanticNeuralCompositionState,
    execute_public_typed_workflow,
    render_public_typed_workflow,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine

INDUCED_NEURAL_PROCEDURE_SCHEMA: Final = "aura.induced_neural_procedure.v1"
_PHYSICAL_REGISTERS: Final = tuple(f"r{index}" for index in range(4))
_LOWERABLE_PRIMITIVES: Final = frozenset({"add", "idiv"})


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
class LoweredInducedProcedure:
    """One frozen induced program allocated onto the neural register bank."""

    public_workflow: str
    output_register: str
    program_sha: str
    receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InducedNeuralExecution:
    """The induced-program identity and the resulting neural state."""

    lowered: LoweredInducedProcedure
    composition: SemanticNeuralCompositionState


def _validated_inputs(inputs: tuple[Any, ...]) -> tuple[int, ...]:
    if not 1 <= len(inputs) <= len(_PHYSICAL_REGISTERS):
        raise ValueError("induced neural procedure input arity is unsupported")
    if any(
        type(value) is not int or not 0 <= value <= MAX_PROCESS_INTEGER
        for value in inputs
    ):
        raise ValueError("induced neural procedure inputs exceed the register bank")
    return tuple(int(value) for value in inputs)


def _last_uses(program: Program) -> dict[int, int]:
    last: dict[int, int] = {}
    for ordinal, instruction in enumerate(program.instructions):
        output = program.n_inputs + ordinal
        if any(
            type(index) is not int or not 0 <= index < output
            for index in instruction.args
        ):
            raise ValueError("induced procedure is not a forward SSA program")
        for index in instruction.args:
            last[index] = ordinal
    terminal = program.n_inputs + len(program.instructions) - 1
    if not program.instructions:
        terminal = 0
    last[terminal] = len(program.instructions)
    return last


def lower_induced_program(
    program: Program,
    inputs: tuple[Any, ...],
) -> LoweredInducedProcedure:
    """Allocate a family-blind induced program onto the learned register tissue.

    The lowerer receives only a frozen program and its public inputs. It has no
    family, expected output, verifier trace, or natural-language task name.
    Unsupported primitives and register pressure are explicit refusals.
    """

    if not isinstance(program, Program):
        raise TypeError("induced neural procedure requires a frozen Program")
    values = _validated_inputs(inputs)
    if program.n_inputs != len(values):
        raise ValueError("induced neural procedure input arity does not match")
    unsupported = sorted(
        {instruction.op for instruction in program.instructions}
        - _LOWERABLE_PRIMITIVES
    )
    if unsupported:
        raise ValueError(
            "induced procedure uses unsupported neural primitives: "
            + ",".join(unsupported)
        )

    last_use = _last_uses(program)
    allocation = {
        virtual: _PHYSICAL_REGISTERS[virtual]
        for virtual in range(program.n_inputs)
    }
    initial = {name: 0 for name in _PHYSICAL_REGISTERS}
    for virtual, value in enumerate(values):
        initial[allocation[virtual]] = value

    steps: list[dict[str, Any]] = []
    allocation_trace: list[dict[str, Any]] = []
    for ordinal, instruction in enumerate(program.instructions):
        output = program.n_inputs + ordinal
        operands = tuple(allocation[index] for index in instruction.args)

        occupied = set(allocation.values())
        free = [name for name in _PHYSICAL_REGISTERS if name not in occupied]
        reusable = [
            allocation[index]
            for index in instruction.args
            if last_use.get(index) == ordinal
        ]
        candidates = (*free, *reusable)
        if not candidates:
            raise ValueError("induced procedure exceeds the neural register bank")
        destination = candidates[0]

        for virtual in tuple(allocation):
            if last_use.get(virtual, -1) <= ordinal:
                del allocation[virtual]
        allocation[output] = destination

        if instruction.op == "add":
            if len(operands) != 2:
                raise ValueError("induced add arity is invalid")
            step = {
                "op": "add",
                "dst": destination,
                "left": operands[0],
                "right": operands[1],
            }
        elif instruction.op == "idiv":
            if len(operands) != 2:
                raise ValueError("induced division arity is invalid")
            step = {
                "op": "div_exact",
                "dst": destination,
                "numerator": operands[0],
                "denominator": operands[1],
            }
        else:  # pragma: no cover - the closed set is checked above.
            raise AssertionError(instruction.op)
        steps.append(step)
        allocation_trace.append(
            {
                "ordinal": ordinal,
                "virtual_output": output,
                "physical_output": destination,
                "physical_operands": operands,
            }
        )

    output_virtual = (
        program.n_inputs + len(program.instructions) - 1
        if program.instructions
        else 0
    )
    output_register = allocation[output_virtual]
    document = {
        "schema": PUBLIC_TYPED_WORKFLOW_SCHEMA,
        "initial": initial,
        "steps": steps
        or [{"op": "copy", "dst": output_register, "src": output_register}],
        "report": [output_register],
    }
    public_workflow = render_public_typed_workflow(document)
    body = {
        "schema": INDUCED_NEURAL_PROCEDURE_SCHEMA,
        "program_sha": program.sha(),
        "primitive_set_sha": PRIMITIVE_SET_SHA,
        "public_inputs_sha256": _sha(values),
        "public_workflow_sha256": hashlib.sha256(
            public_workflow.encode("utf-8")
        ).hexdigest(),
        "output_register": output_register,
        "allocation": allocation_trace,
        "lowerable_primitives": sorted(_LOWERABLE_PRIMITIVES),
        "family_label_available": False,
        "expected_output_available": False,
        "verifier_trace_available": False,
        "correctness_authority": False,
    }
    return LoweredInducedProcedure(
        public_workflow=public_workflow,
        output_register=output_register,
        program_sha=program.sha(),
        receipt={**body, "receipt_sha256": _sha(body)},
    )


def execute_induced_program(
    program: Program,
    inputs: tuple[Any, ...],
    *,
    machine: SemanticNeuralMachine | None = None,
) -> InducedNeuralExecution:
    """Run one frozen induced procedure through the learned semantic tissue."""

    lowered = lower_induced_program(program, inputs)
    composition = execute_public_typed_workflow(
        lowered.public_workflow,
        machine=machine,
    )
    return InducedNeuralExecution(lowered=lowered, composition=composition)


__all__ = [
    "INDUCED_NEURAL_PROCEDURE_SCHEMA",
    "InducedNeuralExecution",
    "LoweredInducedProcedure",
    "execute_induced_program",
    "lower_induced_program",
]
