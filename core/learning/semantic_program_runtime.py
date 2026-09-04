"""Endogenous observed-state execution for a frozen semantic transducer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.runtime_identity import worker_representation_basis
from core.cognition.procedure import Procedure, ProcedureRegistry
from core.learning.semantic_procedure_currency import from_semantic_program
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
)
from core.learning.semantic_program_floor import (
    SemanticFloorExecution,
    compile_semantic_program_to_floor,
    execute_semantic_floor_program,
)
from core.learning.semantic_program_ir import SemanticProgramIR, semantic_value_to_json
from core.learning.semantic_public_inputs import (
    SemanticPublicInputs,
    semantic_public_token_inputs,
)

COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA: Final = (
    "aura.compositional_semantic_runtime.v1"
)


class SemanticProgramObservationError(Exception):
    """The measured source, tokenizer, or neural basis is not admissible."""


class SemanticProgramDecodeRejectedError(Exception):
    """The admitted observation did not produce an executable semantic program."""


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
class CompositionalSemanticRuntimeOutcome:
    """One answer-blind neural parse executed by the universal floor."""

    public_inputs: SemanticPublicInputs
    ir: SemanticProgramIR
    execution: SemanticFloorExecution
    receipt: dict[str, Any]
    procedure: Procedure | None = None

    def __post_init__(self) -> None:
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (
            self.receipt.get("schema") != COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA
            or self.receipt.get("receipt_sha256") != _sha(body)
            or self.receipt.get("result_sha256")
            != _sha(semantic_value_to_json(self.execution.result))
        ):
            raise ValueError("compositional semantic runtime receipt is invalid")


def execute_compositional_semantic_observation(
    *,
    model: CompositionalSemanticProgramTransducer,
    source_text: str,
    source_token_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    hidden_states: Any,
    worker_model_basis: Mapping[str, Any],
    expected_representation_basis_sha256: str,
    procedure_registry: ProcedureRegistry | None = None,
) -> CompositionalSemanticRuntimeOutcome:
    """Decode and execute one resident observation without external semantics."""

    representation = worker_representation_basis(worker_model_basis)
    representation_sha256 = _sha(representation)
    if representation_sha256 != expected_representation_basis_sha256:
        raise SemanticProgramObservationError(
            "compositional semantic representation basis differs"
        )
    tokens = tuple(source_token_ids)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise SemanticProgramObservationError(
            "compositional semantic source tokens are invalid"
        )
    try:
        public_inputs = semantic_public_token_inputs(source_text, offset_mapping)
    except (TypeError, ValueError) as exc:
        raise SemanticProgramObservationError(
            "compositional semantic public input observation is invalid"
        ) from exc
    inference_step_limit = model.inference_step_limit(len(public_inputs.literals))
    if inference_step_limit is None:
        raise SemanticProgramDecodeRejectedError(
            "compositional semantic public input count is unsupported"
        )
    for literal in public_inputs.literals:
        if literal.token_span not in model.input_grounding.candidate_spans(
            tokens,
            literal.value,
        ):
            raise SemanticProgramObservationError(
                "compositional semantic literal differs from token grammar"
            )

    decoded = model.decode(
        source_token_ids=tokens,
        hidden_states=hidden_states,
        public_inputs=public_inputs.values,
        source_text_sha256=public_inputs.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256,
    )
    if decoded.ir is None:
        raise SemanticProgramDecodeRejectedError(
            f"compositional semantic decode rejected:{decoded.refusal}"
        )
    if set(decoded.ir.input_spans) != set(public_inputs.token_spans):
        raise SemanticProgramDecodeRejectedError(
            "compositional semantic decode did not use every public literal"
        )

    try:
        floor_program = compile_semantic_program_to_floor(
            decoded.ir,
            public_inputs.values,
        )
        execution = execute_semantic_floor_program(floor_program)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SemanticProgramDecodeRejectedError(
            "compositional semantic program was not executable"
        ) from exc
    procedure = (
        from_semantic_program(decoded.ir, registry=procedure_registry)
        if procedure_registry is not None
        else None
    )
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA,
        "source_text_sha256": public_inputs.source_text_sha256,
        "source_token_ids_sha256": _sha(list(tokens)),
        "representation_basis_sha256": representation_sha256,
        "training_session_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "public_inputs_receipt_sha256": public_inputs.receipt()["receipt_sha256"],
        "public_input_recovery": "exact_source_parser",
        "observed_max_inputs": model.max_inputs,
        "observed_max_steps": model.max_steps,
        "inference_step_limit": inference_step_limit,
        "geometry_extrapolated": (
            len(public_inputs.literals) > model.max_inputs
            or inference_step_limit > model.max_steps
        ),
        "semantic_ir_receipt": decoded.ir.receipt(),
        "floor_program_receipt": floor_program.receipt,
        "floor_execution_receipt": execution.receipt,
        "result_sha256": _sha(semantic_value_to_json(execution.result)),
        "input_register_order": "source_character_order",
        "representation_rebound_across_session": True,
        "family_router_present": False,
        "oracle_public_values_available": False,
        "expected_answer_available": False,
        "verifier_trace_available": False,
        "generated_text_available": False,
        "correctness_authority": False,
        "procedure_currency_receipt_sha256": (
            procedure.program.receipt()["receipt_sha256"]
            if procedure is not None
            else None
        ),
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    return CompositionalSemanticRuntimeOutcome(
        public_inputs,
        decoded.ir,
        execution,
        receipt,
        procedure,
    )


__all__ = [
    "COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA",
    "CompositionalSemanticRuntimeOutcome",
    "SemanticProgramDecodeRejectedError",
    "SemanticProgramObservationError",
    "execute_compositional_semantic_observation",
]
