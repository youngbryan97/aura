"""A we both sides are saying, and the they that draws its edge.

"Cause we come from runaway" is belonging stated as a shared origin, and its
boundary is carried by the other pronoun. These pin the reading, that a we one
side says is not shared, and that naming who is outside never raises belonging.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.affect_update import AffectUpdatePhase
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.social.togetherness import Togetherness, read_togetherness
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS

WE_THEIRS = "We came up from nothing together and we are still here, and they never thought we would be."
WE_HERS = "We have been at this for a long time now, and we keep finding our way back to it."
I_HERS = "I have been at this for a long time now, and I keep finding my way back to it."


def _said(hers: str, theirs: str) -> list[dict]:
    return [{"role": "user", "content": theirs}, {"role": "assistant", "content": hers}]


def test_a_we_both_are_saying_is_togetherness() -> None:
    reading = read_togetherness(_said(WE_HERS, WE_THEIRS))
    assert reading.measured
    assert reading.together > 0.0
    assert reading.together == pytest.approx(min(reading.hers, reading.theirs))


def test_a_we_only_one_side_says_is_not_shared() -> None:
    reading = read_togetherness(_said(I_HERS, WE_THEIRS))
    assert reading.together == 0.0
    assert "she is not" in reading.why


def test_the_they_is_the_edge_while_there_is_a_we() -> None:
    reading = read_togetherness(_said(WE_HERS, WE_THEIRS))
    assert reading.edge > 0.0
    assert read_togetherness(_said(I_HERS, WE_THEIRS)).edge == 0.0


def test_too_little_said_is_not_a_reading() -> None:
    reading = read_togetherness(_said("ok", "we are"))
    assert not reading.measured
    assert reading.together == 0.0


def test_the_conversation_phase_writes_it() -> None:
    state = AuraState.default()
    state.cognition.working_memory.extend(_said(WE_HERS, WE_THEIRS))
    ConversationalDynamicsPhase._read_togetherness(state)
    assert state.cognition.togetherness["together"] > 0.0


def test_belonging_is_floored_at_it_and_the_edge_adds_nothing() -> None:
    state = AuraState.default()
    state.affect.emotions["belonging"] = 0.0
    state.cognition.togetherness = {"together": 0.4, "edge": 0.9}
    AffectUpdatePhase(SimpleNamespace())._read_togetherness(state, state.affect)
    assert state.affect.emotions["belonging"] == pytest.approx(0.4)


def test_a_stronger_belonging_is_not_pulled_down() -> None:
    state = AuraState.default()
    state.affect.emotions["belonging"] = 0.8
    state.cognition.togetherness = {"together": 0.3, "edge": 0.0}
    AffectUpdatePhase(SimpleNamespace())._read_togetherness(state, state.affect)
    assert state.affect.emotions["belonging"] == pytest.approx(0.8)


def test_the_columns_that_read_it_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["W"].sources)
    assert "cognition.togetherness.together" in sources
    assert "cognition.togetherness.edge" in sources
    assert "cognition.togetherness" in CLAMPED_FIELDS["W"]


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Togetherness().as_dict()
    for key in ("together", "edge", "hers", "theirs", "measured", "why"):
        assert key in row, key
