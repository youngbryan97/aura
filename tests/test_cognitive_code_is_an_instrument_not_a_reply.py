"""The cognitive code reads the state. It is never shown to anyone.

The proto-token generator learned this distinction the hard way: "the path
ran" and "its output may be put in front of a person" are two different
questions, and a random projection over thirty-two words was reachable as a
live reply because only the first was being asked.
"""

from __future__ import annotations

import numpy as np

from core.brain.llm.cognitive_code import (
    SPEECH_ACTS,
    CognitiveCode,
    IntentHead,
    read_code,
)
from core.brain.llm.endogenous_state import STATE_DIM, empty_state


def _state(**values: float):
    return empty_state().do(**values)


def test_the_code_is_never_user_presentable():
    code = read_code(_state(**{"uncertainty.confidence": 0.9}), include_organ_lines=False)
    assert code.is_user_presentable is False
    assert code.as_dict()["is_user_presentable"] is False


def test_an_absent_channel_abstains_rather_than_reading_zero():
    code = read_code(empty_state(), include_organ_lines=False)
    assert "UNCERTAINTY" in code.abstained()
    assert "GOAL" in code.abstained()
    assert code.get("UNCERTAINTY") == "absent"


def test_a_channel_that_answered_zero_is_not_an_absence():
    """A goal system reporting no goal is not a goal system that failed."""
    code = read_code(_state(**{"goal.active": 0.0}), include_organ_lines=False)
    assert code.get("GOAL") == "none-held"
    assert "GOAL" not in code.abstained()


def test_uncertainty_is_the_inverse_of_confidence():
    unsure = read_code(_state(**{"uncertainty.confidence": 0.05}), include_organ_lines=False)
    settled = read_code(_state(**{"uncertainty.confidence": 0.95}), include_organ_lines=False)
    assert unsure.get("UNCERTAINTY").startswith("high")
    assert settled.get("UNCERTAINTY").startswith("low")


def test_moving_one_dimension_moves_exactly_one_line():
    """do(U = low) is the experiment; it must not shake the whole readout."""
    base = _state(
        **{
            "uncertainty.confidence": 0.1,
            "goal.active": 1.0,
            "goal.priority": 0.9,
            "memory.recall_hits": 0.8,
        }
    )
    before = read_code(base, include_organ_lines=False)
    after = read_code(
        base.do(**{"uncertainty.confidence": 0.95}), include_organ_lines=False
    )
    assert list(before.diff(after)) == ["UNCERTAINTY"]


def test_the_code_records_that_it_was_intervened_on():
    state = _state(**{"uncertainty.confidence": 0.9})
    code = read_code(state, include_organ_lines=False)
    assert "uncertainty.confidence" in code.interventions


def test_an_untrained_intent_head_abstains():
    code = read_code(_state(**{"affect.valence": 0.5}), include_organ_lines=False)
    assert code.get("INTEND").startswith("abstained")


def test_a_low_confidence_intent_head_abstains_rather_than_guessing():
    head = IntentHead(
        weights=np.zeros((len(SPEECH_ACTS), STATE_DIM)),
        bias=np.zeros(len(SPEECH_ACTS)),
    )
    code = read_code(_state(**{"affect.valence": 0.5}), intent_head=head, include_organ_lines=False)
    assert code.get("INTEND").startswith("abstained"), "a uniform head should decline"


def test_a_confident_intent_head_answers():
    weights = np.zeros((len(SPEECH_ACTS), STATE_DIM))
    bias = np.zeros(len(SPEECH_ACTS))
    bias[SPEECH_ACTS.index("ask")] = 10.0
    head = IntentHead(weights=weights, bias=bias)
    code = read_code(_state(**{"affect.valence": 0.5}), intent_head=head, include_organ_lines=False)
    assert code.get("INTEND").startswith("ask")


def test_the_rendering_is_stable_for_a_state():
    state = _state(**{"uncertainty.confidence": 0.4, "memory.recall_hits": 0.6})
    first = read_code(state, include_organ_lines=False).render()
    second = read_code(state, include_organ_lines=False).render()
    assert first == second


def test_provenance_separates_a_readout_from_an_organ_read():
    code: CognitiveCode = read_code(empty_state(), include_organ_lines=True)
    provenances = {line.field: line.provenance for line in code.lines}
    assert provenances["CONCEPTS"] in {"organ", "abstained"}
    assert provenances["UNCERTAINTY"] in {"state", "abstained"}
    assert "state" not in {provenances["CONCEPTS"]}, (
        "concepts cannot be a readout of 74 floats and must not claim to be"
    )
