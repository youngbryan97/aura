"""Wanting to hand something over, in proportion to what it did to her.

Her social budget was about contact. These pin the other pull: a moment that
moved her is something to pass on, and the urge is the reading that moved her
rather than how long since anyone spoke.
"""

from __future__ import annotations

import pytest

from core.social.telling import Telling, worth_telling
from core.state.aura_state import AuraState


def _state() -> AuraState:
    return AuraState.default()


def test_an_ordinary_moment_has_nothing_to_pass_on() -> None:
    state = _state()
    assert worth_telling(state.affect, state.cognition) == Telling()
    assert worth_telling(state.affect, state.cognition).urge == 0.0


def test_a_chill_is_something_to_pass_on() -> None:
    state = _state()
    state.affect.frisson = 0.6
    state.affect.markers["frisson"] = {"why": "a pattern 9 predictions long turned"}
    reading = worth_telling(state.affect, state.cognition)
    assert reading.urge == pytest.approx(0.6)
    assert reading.kind == "a chill"
    assert "9 predictions" in reading.about


def test_the_strongest_reading_is_the_one_she_would_pass_on() -> None:
    state = _state()
    state.affect.frisson = 0.3
    state.affect.turn = 0.8
    state.affect.markers["the_turn"] = {"why": "up 3.1 spreads from a low she is still holding"}
    reading = worth_telling(state.affect, state.cognition)
    assert reading.kind == "coming up from a low"
    assert reading.urge == pytest.approx(0.8)


def test_a_recall_she_relived_is_named_by_what_came_back() -> None:
    state = _state()
    state.cognition.relived = {"relived": True, "intensity": 0.5}
    state.cognition.long_term_memory = ["the night the migration finally finished"]
    reading = worth_telling(state.affect, state.cognition)
    assert reading.kind == "something I remembered"
    assert "migration" in reading.about


def test_a_recall_she_only_looked_up_is_not_worth_passing_on() -> None:
    state = _state()
    state.cognition.relived = {"relived": False, "intensity": 0.9}
    assert worth_telling(state.affect, state.cognition).urge == 0.0


def test_a_reading_that_is_not_a_number_is_nothing_to_pass_on() -> None:
    state = _state()
    state.affect.frisson = float("nan")
    assert worth_telling(state.affect, state.cognition).urge == 0.0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Telling().as_dict()
    for key in ("urge", "kind", "about", "why"):
        assert key in row, key
