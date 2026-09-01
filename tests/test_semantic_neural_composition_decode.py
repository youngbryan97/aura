from __future__ import annotations

import json
import random

from core.brain.llm.latent_cortex.semantic_neural_composition_decode import (
    execute_composition_decode_state,
    parse_composition_response,
    render_composition_answer,
    render_composition_decode_objective,
    render_composition_state_channel,
)
from core.learning.semantic_neural_composition import render_public_typed_workflow
from tools.run_semantic_neural_composition_canary import _task_document


def _state():
    workflow = render_public_typed_workflow(_task_document(random.Random(71)))
    return workflow, execute_composition_decode_state(workflow)


def test_public_decode_objective_contains_no_result() -> None:
    workflow, state = _state()
    objective = render_composition_decode_objective(workflow)

    assert state.semantic_result
    assert json.dumps(state.semantic_result, separators=(",", ":")) not in objective
    assert "COMPOSITION_QUERY_V1" in objective
    assert workflow in objective


def test_state_channel_binds_result_to_teacher_free_receipt() -> None:
    _workflow, state = _state()
    channel = render_composition_state_channel(state)
    receipt = state.receipt()

    assert channel.startswith("COMPUTED_TYPED_STATE_V1 ")
    assert receipt["teacher_available"] is False
    assert receipt["private_trace_available"] is False
    assert receipt["answer_key_available"] is False
    assert receipt["verifier_available"] is False
    assert receipt["receipt_sha256"] in channel


def test_canonical_answer_round_trips_and_noncanonical_forms_fail() -> None:
    _workflow, state = _state()
    answer = render_composition_answer(state)

    assert parse_composition_response(answer, state.report) == state.semantic_result
    assert parse_composition_response("prose\n" + answer, state.report) is None
    assert parse_composition_response(answer + " trailing", state.report) is None
    assert parse_composition_response(
        'FINAL_ANSWER:{"r0":1,"r0":1,"r1":2,"r2":3,"r3":4,"s0":1}',
        state.report,
    ) is None
