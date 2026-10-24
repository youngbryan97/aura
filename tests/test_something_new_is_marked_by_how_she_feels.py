"""The world and feeling meet at deliberation in what she means to look into.

core/agency/marked.py: each thing new to her world, marked at her arousal times
how much it stands out, once.
"""

from __future__ import annotations

import pytest

from core.agency.marked import SEEN, marked
from core.state.aura_state import AuraState
from core.subject.perturbation import perturb

DELTA = 0.15


def _state(arousal: float) -> AuraState:
    state = AuraState.default()
    state.affect.arousal = arousal
    return state


def test_feeling_with_nothing_new_marks_nothing():
    assert marked(_state(0.95)) == []


def test_something_new_met_without_feeling_is_not_pressing():
    state = _state(0.0)
    perturb(state, "W", DELTA)
    assert marked(state) == [("subject_core_probe", 0.0)]


def test_something_new_is_as_pressing_as_she_is_aroused_and_it_stands_out():
    state = _state(0.7)
    perturb(state, "W", DELTA)
    ((name, urgency),) = marked(state)
    assert name == "subject_core_probe"
    assert urgency == pytest.approx(0.7 * (0.5 + DELTA))
    assert state.world.known_entities[name][SEEN]
    assert marked(state) == []


def test_the_world_and_feeling_set_together_mark_more_than_each_adds():
    def urgency(world: bool, feeling: bool) -> float:
        state = _state(0.6)
        if world:
            perturb(state, "W", DELTA)
        if feeling:
            perturb(state, "A", DELTA)
        return sum(value for _name, value in marked(state))

    joint = urgency(True, True) - urgency(True, False) - urgency(False, True) + urgency(False, False)
    assert joint == pytest.approx(DELTA * (0.5 + DELTA))
