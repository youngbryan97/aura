"""Endogenous observed-state execution for a frozen semantic transducer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.runtime_identity import worker_representation_basis
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
)
from core.learning.semantic_program_floor import (
    SemanticFloorExecution,
    compile_semantic_program_to_floor,
    execute_semantic_floor_program,
)
from core.learning.semantic_program_ir import semantic_value_to_json
from core.learning.semantic_public_inputs import (
    SemanticPublicInputs,
    semantic_public_token_inputs,
)

COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA: Final = (
    "aura.compositional_semantic_runtime.v1"
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
class CompositionalSemanticRuntimeOutcome:
    """One answer-blind neural parse executed by the universal floor."""

    public_inputs: SemanticPublicInputs
    execution: SemanticFloorExecution
    receipt: dict[str, Any]

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
) -> CompositionalSemanticRuntimeOutcome:
    """Decode and execute one resident observation without external semantics."""

    representation = worker_representation_basis(worker_model_basis)
    representation_sha256 = _sha(representation)
    if representation_sha256 != expected_representation_basis_sha256:
        raise ValueError("compositional semantic representation basis differs")
    tokens = tuple(source_token_ids)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("compositional semantic source tokens are invalid")
    public_inputs = semantic_public_token_inputs(source_text, offset_mapping)
    if not 1 <= len(public_inputs.literals) <= model.max_inputs:
        raise ValueError("compositional semantic public input count is unsupported")
    for literal in public_inputs.literals:
        if literal.token_span not in model.input_grounding.candidate_spans(
            tokens,
            literal.value,
        ):
            raise ValueError("compositional semantic literal differs from token grammar")

    decoded = model.decode(
        source_token_ids=tokens,
        hidden_states=hidden_states,
        public_inputs=public_inputs.values,
        source_text_sha256=public_inputs.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256,
    )
    if decoded.ir is None:
        raise ValueError(f"compositional semantic decode rejected:{decoded.reason}")
    if set(decoded.ir.input_spans) != set(public_inputs.token_spans):
        raise ValueError("compositional semantic decode did not use every public literal")

    floor_program = compile_semantic_program_to_floor(
        decoded.ir,
        public_inputs.values,
    )
    execution = execute_semantic_floor_program(floor_program)
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA,
        "source_text_sha256": public_inputs.source_text_sha256,
        "source_token_ids_sha256": _sha(list(tokens)),
        "representation_basis_sha256": representation_sha256,
        "training_session_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "public_inputs_receipt_sha256": public_inputs.receipt()["receipt_sha256"],
        "semantic_ir_receipt": decoded.ir.receipt(),
        "floor_program_receipt": floor_program.receipt,
        "floor_execution_receipt": execution.receipt,
        "result_sha256": _sha(semantic_value_to_json(execution.result)),
        "input_register_order": "source_character_order",
        "representation_rebound_across_session": True,
        "family_router_present": False,
        "expected_answer_available": False,
        "verifier_trace_available": False,
        "generated_text_available": False,
        "correctness_authority": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    return CompositionalSemanticRuntimeOutcome(public_inputs, execution, receipt)


__all__ = [
    "COMPOSITIONAL_SEMANTIC_RUNTIME_SCHEMA",
    "CompositionalSemanticRuntimeOutcome",
    "execute_compositional_semantic_observation",
]
