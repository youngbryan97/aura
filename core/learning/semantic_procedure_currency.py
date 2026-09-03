"""Carry learned language programs into Aura's common procedure currency.

The neural transducer produces :class:`SemanticProgramIR`, which is correctly
bound to one utterance.  A reusable procedure is the computation beneath that
utterance: source-independent SSA, typed ports, and a cryptographic pointer
back to the observation that proposed it.  This module performs that lowering
without family labels, generated code, or expected answers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    ProceduralValue,
    Procedure,
    ProcedureRegistry,
    Reversibility,
    Signature,
    get_procedure_registry,
)
from core.evidence.packet import EvidencePacket, observe
from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import (
    DEFAULT_SEMANTIC_FLOOR_FUEL,
    SemanticFloorExecution,
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
    semantic_primitive_type_signature,
)
from core.learning.semantic_program_ir import SemanticProgramIR

SEMANTIC_PROCEDURE_PROGRAM_SCHEMA: Final = "aura.semantic_procedure_program.v1"
SEMANTIC_PROCEDURE_EXECUTION_SCHEMA: Final = "aura.semantic_procedure_execution.v1"


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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _program_identity(program: Program) -> str:
    return _sha(
        {
            "n_inputs": program.n_inputs,
            "instructions": [
                [instruction.op, list(instruction.args)] for instruction in program.instructions
            ],
        }
    )


def _program_types(program: Program) -> tuple[tuple[str, ...], str]:
    """Validate one source-independent SSA program and infer all port types."""

    if not isinstance(program, Program) or program.n_inputs < 1 or not program.instructions:
        raise ValueError("semantic procedure needs a non-empty input program")
    register_types: list[str | None] = [None] * program.n_inputs
    for ordinal, instruction in enumerate(program.instructions):
        output = program.n_inputs + ordinal
        signature = semantic_primitive_type_signature(instruction.op)
        if (
            signature is None
            or len(signature[0]) != len(instruction.args)
            or any(
                type(argument) is not int or not 0 <= argument < output
                for argument in instruction.args
            )
        ):
            raise ValueError("semantic procedure is not a typed forward program")
        argument_types, result_type = signature
        for argument, required_type in zip(instruction.args, argument_types, strict=True):
            known = register_types[argument]
            if known is not None and known != required_type:
                raise ValueError("semantic procedure register has incompatible uses")
            register_types[argument] = required_type
        register_types.append(result_type)
    if any(kind is None for kind in register_types[: program.n_inputs]):
        raise ValueError("semantic procedure retains an unused input")
    return tuple(str(kind) for kind in register_types[: program.n_inputs]), str(register_types[-1])


def _typed_projected_program(
    ir: SemanticProgramIR,
) -> tuple[Program, tuple[int, ...], tuple[str, ...], str]:
    """Infer structural types and remove public inputs the graph never reads."""

    register_types: list[str | None] = [None] * ir.n_inputs
    used_inputs: set[int] = set()
    for instruction in ir.instructions:
        signature = semantic_primitive_type_signature(instruction.op)
        if signature is None:
            raise ValueError("semantic procedure primitive has no universal type")
        argument_types, result_type = signature
        if len(argument_types) != len(instruction.args):
            raise ValueError("semantic procedure primitive arity differs")
        for argument, required_type in zip(instruction.args, argument_types, strict=True):
            if argument < ir.n_inputs:
                used_inputs.add(argument)
                known = register_types[argument]
                if known is not None and known != required_type:
                    raise ValueError("semantic procedure input has incompatible uses")
                register_types[argument] = required_type
            elif register_types[argument] != required_type:
                raise ValueError("semantic procedure intermediate has incompatible type")
        register_types.append(result_type)

    source_positions = tuple(sorted(used_inputs))
    if not source_positions:
        raise ValueError("semantic procedure reads no public input")
    input_remap = {source: target for target, source in enumerate(source_positions)}
    instructions: list[Instruction] = []
    for instruction in ir.instructions:
        remapped = []
        for argument in instruction.args:
            if argument < ir.n_inputs:
                remapped.append(input_remap[argument])
            else:
                remapped.append(len(source_positions) + argument - ir.n_inputs)
        instructions.append(Instruction(instruction.op, tuple(remapped)))
    projected = Program(len(source_positions), tuple(instructions))
    input_types = tuple(str(register_types[position]) for position in source_positions)
    output_type = str(register_types[-1])
    return projected, source_positions, input_types, output_type


@dataclass(frozen=True, slots=True)
class SemanticProcedureProgram:
    """A source-independent learned computation with source-bound provenance."""

    program: Program
    source_input_positions: tuple[int, ...]
    input_keys: tuple[str, ...]
    input_types: tuple[str, ...]
    output_key: str
    output_type: str
    semantic_ir_receipt_sha256: str
    source_alpha_normalized_sha256: str
    model_basis_receipt_sha256: str
    transducer_receipt_sha256: str
    schema: str = SEMANTIC_PROCEDURE_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        count = self.program.n_inputs
        inferred_input_types, inferred_output_type = _program_types(self.program)
        if (
            self.schema != SEMANTIC_PROCEDURE_PROGRAM_SCHEMA
            or not self.program.instructions
            or len(self.source_input_positions) != count
            or tuple(sorted(set(self.source_input_positions))) != self.source_input_positions
            or len(self.input_keys) != count
            or len(set(self.input_keys)) != count
            or len(self.input_types) != count
            or self.input_types != inferred_input_types
            or not self.output_key
            or any(
                not key or not kind
                for key, kind in zip(self.input_keys, self.input_types, strict=True)
            )
            or not self.output_type
            or self.output_type != inferred_output_type
            or not all(
                _is_sha256(value)
                for value in (
                    self.semantic_ir_receipt_sha256,
                    self.source_alpha_normalized_sha256,
                    self.model_basis_receipt_sha256,
                    self.transducer_receipt_sha256,
                )
            )
        ):
            raise ValueError("semantic procedure program envelope is invalid")

    @property
    def program_sha256(self) -> str:
        return _program_identity(self.program)

    @property
    def execution_contract_sha256(self) -> str:
        """Identity of execution, excluding which utterance revealed it."""

        return _sha(
            {
                "program_sha256": self.program_sha256,
                "input_keys": list(self.input_keys),
                "input_types": list(self.input_types),
                "output_key": self.output_key,
                "output_type": self.output_type,
            }
        )

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "execution_contract_sha256": self.execution_contract_sha256,
            "source_input_positions": list(self.source_input_positions),
            "input_keys": list(self.input_keys),
            "input_types": list(self.input_types),
            "output_key": self.output_key,
            "output_type": self.output_type,
            "semantic_ir_receipt_sha256": self.semantic_ir_receipt_sha256,
            "source_alpha_normalized_sha256": self.source_alpha_normalized_sha256,
            "model_basis_receipt_sha256": self.model_basis_receipt_sha256,
            "transducer_receipt_sha256": self.transducer_receipt_sha256,
            "source_tokens_retained": False,
            "source_spans_retained": False,
            "family_label_present": False,
            "expected_answer_available": False,
            "correctness_authority": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class SemanticProcedureExecution:
    """One common-currency RLC procedure executed on the universal floor."""

    result: Any
    resulting_state: dict[str, Any]
    floor_execution: SemanticFloorExecution
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        normalized = list(self.result) if isinstance(self.result, tuple) else self.result
        if (
            self.floor_execution.result != self.result
            or body.get("schema") != SEMANTIC_PROCEDURE_EXECUTION_SCHEMA
            or self.receipt.get("receipt_sha256") != _sha(body)
            or body.get("result_sha256") != _sha(normalized)
        ):
            raise ValueError("semantic procedure execution receipt is invalid")


def from_semantic_program(
    ir: SemanticProgramIR,
    *,
    input_keys: tuple[str, ...] | None = None,
    output_key: str = "semantic:result",
    observed_successes: int = 0,
    observed_trials: int = 0,
    value_when_it_works: float = 0.0,
    match_cost: float = 0.0,
    registry: ProcedureRegistry | None = None,
) -> Procedure:
    """Register the computation beneath one accepted neural language parse."""

    if not isinstance(ir, SemanticProgramIR):
        raise TypeError("semantic procedure registration requires validated IR")
    if (
        type(observed_successes) is not int
        or type(observed_trials) is not int
        or not 0 <= observed_successes <= observed_trials
    ):
        raise ValueError("semantic procedure observations are invalid")
    all_input_keys = (
        input_keys
        if input_keys is not None
        else tuple(f"semantic:argument:{index}" for index in range(ir.n_inputs))
    )
    if len(all_input_keys) != ir.n_inputs or len(set(all_input_keys)) != len(all_input_keys):
        raise ValueError("semantic procedure input-key binding is invalid")
    if not output_key:
        raise ValueError("semantic procedure output-key binding is invalid")

    program, source_positions, input_types, output_type = _typed_projected_program(ir)
    selected_keys = tuple(all_input_keys[position] for position in source_positions)
    ir_receipt_sha256 = ir.receipt()["receipt_sha256"]
    stored = SemanticProcedureProgram(
        program=program,
        source_input_positions=source_positions,
        input_keys=selected_keys,
        input_types=input_types,
        output_key=output_key,
        output_type=output_type,
        semantic_ir_receipt_sha256=ir_receipt_sha256,
        source_alpha_normalized_sha256=ir.alpha_normalized_sha256,
        model_basis_receipt_sha256=ir.model_basis_receipt_sha256,
        transducer_receipt_sha256=ir.transducer_receipt_sha256,
    )
    success_rate = observed_successes / observed_trials if observed_trials else 0.5
    evidence: EvidencePacket = observe(
        success_rate,
        origin="core.learning.semantic_procedure_currency",
        ref=ir_receipt_sha256,
        mass=float(observed_trials),
        subject=stored.execution_contract_sha256,
    )
    registry = registry or get_procedure_registry()
    return registry.intern(
        f"semantic:{stored.execution_contract_sha256}",
        stored.execution_contract_sha256,
        f"rlc:{stored.program_sha256}",
        Backend.RLC,
        Signature(
            preconditions=tuple(
                Precondition(key=key, kind=kind)
                for key, kind in zip(selected_keys, input_types, strict=True)
            ),
            effects=(Effect(key=output_key, kind=output_type),),
        ),
        program=stored,
        value=ProceduralValue(
            p_success=success_rate,
            recent_success=success_rate,
            recent_weight=float(observed_trials),
            value_when_it_works=value_when_it_works,
            match_cost=match_cost,
            uses=observed_trials,
            successes=observed_successes,
            transfer_tier="structural_analogue",
        ),
        origin=Origin(
            learner="core.learning.semantic_program_compositional_transducer",
            support_keys=selected_keys,
            rejected_conditions=tuple(
                all_input_keys[index]
                for index in range(ir.n_inputs)
                if index not in source_positions
            ),
        ),
        evidence=evidence,
        reversibility=Reversibility.REVERSIBLE,
    )


def execute_semantic_procedure(
    procedure: Procedure,
    state: dict[str, Any],
    *,
    fuel: int = DEFAULT_SEMANTIC_FLOOR_FUEL,
) -> SemanticProcedureExecution:
    """Execute one registered RLC procedure through its backend contract."""

    if procedure.backend is not Backend.RLC or not isinstance(
        procedure.program, SemanticProcedureProgram
    ):
        raise TypeError("semantic procedure execution requires an RLC procedure")
    if not procedure.signature.matches(state):
        raise ValueError("semantic procedure preconditions do not match state")
    stored = procedure.program
    inputs = tuple(state[key] for key in stored.input_keys)
    floor_program = compile_source_independent_program_to_floor(
        stored.program,
        inputs,
        provenance_receipt_sha256=stored.receipt()["receipt_sha256"],
    )
    floor_execution = execute_semantic_floor_program(floor_program, fuel=fuel)
    resulting_state = dict(state)
    resulting_state[stored.output_key] = floor_execution.result
    body = {
        "schema": SEMANTIC_PROCEDURE_EXECUTION_SCHEMA,
        "procedure_id": procedure.procedure_id,
        "procedure_program_receipt_sha256": stored.receipt()["receipt_sha256"],
        "floor_program_receipt_sha256": floor_program.receipt["receipt_sha256"],
        "floor_execution_receipt_sha256": floor_execution.receipt["receipt_sha256"],
        "result_sha256": _sha(
            list(floor_execution.result)
            if isinstance(floor_execution.result, tuple)
            else floor_execution.result
        ),
        "source_tokens_available": False,
        "family_router_present": False,
        "expected_answer_available": False,
        "correctness_authority": False,
    }
    return SemanticProcedureExecution(
        result=floor_execution.result,
        resulting_state=resulting_state,
        floor_execution=floor_execution,
        receipt={**body, "receipt_sha256": _sha(body)},
    )


__all__ = [
    "SEMANTIC_PROCEDURE_EXECUTION_SCHEMA",
    "SEMANTIC_PROCEDURE_PROGRAM_SCHEMA",
    "SemanticProcedureExecution",
    "SemanticProcedureProgram",
    "execute_semantic_procedure",
    "from_semantic_program",
]
