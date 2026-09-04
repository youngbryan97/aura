import hashlib

import numpy as np
import pytest

from core.brain.llm.latent_cortex.runtime_identity import worker_representation_basis
from core.cognition.procedure import Backend, reset_procedure_registry_for_test
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    TokenSpan,
)
from core.learning.semantic_program_runtime import (
    SemanticProgramDecodeRejectedError,
    SemanticProgramObservationError,
    execute_compositional_semantic_observation,
)
from core.learning.semantic_program_transducer import SemanticTransductionOutcome


def _sha(value):
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _Grounding:
    @staticmethod
    def candidate_spans(_tokens, value):
        return {2: (TokenSpan(1, 2),), 3: (TokenSpan(3, 4),)}[value]


class _Model:
    max_inputs = 2
    max_steps = 1
    model_basis_sha256 = "a" * 64
    receipt_sha256 = "b" * 64
    input_grounding = _Grounding()

    @staticmethod
    def inference_step_limit(input_count):
        return 1 if input_count == 2 else None

    @staticmethod
    def decode(**kwargs):
        ir = SemanticProgramIR(
            source_token_ids=tuple(kwargs["source_token_ids"]),
            source_text_sha256=kwargs["source_text_sha256"],
            input_spans=(TokenSpan(1, 2), TokenSpan(3, 4)),
            instructions=(
                SemanticIRInstruction(
                    op="add",
                    args=(0, 1),
                    operation_span=TokenSpan(0, 1),
                    argument_spans=(TokenSpan(1, 2), TokenSpan(3, 4)),
                    depends_on=(),
                ),
            ),
            report_value=2,
            model_basis_receipt_sha256=_Model.model_basis_sha256,
            transducer_receipt_sha256=_Model.receipt_sha256,
        )
        return SemanticTransductionOutcome(ir, "", {}, {})


class _RefusingModel(_Model):
    @staticmethod
    def decode(**_kwargs):
        return SemanticTransductionOutcome(None, "typed_argument_chart_empty", {}, {})


def test_runtime_rejects_a_different_neural_representation_basis():
    with pytest.raises(SemanticProgramObservationError, match="representation basis differs"):
        execute_compositional_semantic_observation(
            model=object(),  # type: ignore[arg-type]
            source_text="Add 2 and 3.",
            source_token_ids=(1, 2, 3),
            offset_mapping=((0, 3), (4, 5), (6, 11)),
            hidden_states=np.ones((3, 1), dtype=np.float32),
            worker_model_basis={"worker_model_path": "/model"},
            expected_representation_basis_sha256="0" * 64,
        )


def test_runtime_representation_projection_ignores_session_only_identity():
    left = {
        "worker_model_path": "/model",
        "worker_model_config_sha256": "a" * 64,
        "worker_boot_id": "first",
        "worker_pid": 12,
    }
    right = {**left, "worker_boot_id": "second", "worker_pid": 91}

    assert worker_representation_basis(left) == worker_representation_basis(right)
    assert _sha(worker_representation_basis(left)) == _sha(
        worker_representation_basis(right)
    )


def test_runtime_executes_learned_ir_on_the_universal_floor():
    basis = {"worker_model_path": "/model"}
    registry = reset_procedure_registry_for_test()

    outcome = execute_compositional_semantic_observation(
        model=_Model(),  # type: ignore[arg-type]
        source_text="Add 2 and 3.",
        source_token_ids=(10, 15, 11, 16, 13),
        offset_mapping=((0, 3), (4, 5), (6, 9), (10, 11), (11, 12)),
        hidden_states=np.ones((5, 1), dtype=np.float32),
        worker_model_basis=basis,
        expected_representation_basis_sha256=_sha(
            worker_representation_basis(basis)
        ),
        procedure_registry=registry,
    )

    assert outcome.execution.result == 5
    assert outcome.public_inputs.values == (2, 3)
    assert outcome.receipt["family_router_present"] is False
    assert outcome.receipt["expected_answer_available"] is False
    assert outcome.receipt["geometry_extrapolated"] is False
    assert outcome.receipt["inference_step_limit"] == 1
    assert outcome.procedure is not None
    assert outcome.procedure.backend is Backend.RLC
    assert registry.report()["by_backend"] == {"rlc": 1}
    assert outcome.receipt["procedure_currency_receipt_sha256"] == (
        outcome.procedure.program.receipt()["receipt_sha256"]
    )


def test_runtime_preserves_a_neural_decode_refusal_as_model_evidence():
    basis = {"worker_model_path": "/model"}

    with pytest.raises(
        SemanticProgramDecodeRejectedError,
        match="typed_argument_chart_empty",
    ):
        execute_compositional_semantic_observation(
            model=_RefusingModel(),  # type: ignore[arg-type]
            source_text="Add 2 and 3.",
            source_token_ids=(10, 15, 11, 16, 13),
            offset_mapping=((0, 3), (4, 5), (6, 9), (10, 11), (11, 12)),
            hidden_states=np.ones((5, 1), dtype=np.float32),
            worker_model_basis=basis,
            expected_representation_basis_sha256=_sha(
                worker_representation_basis(basis)
            ),
        )
