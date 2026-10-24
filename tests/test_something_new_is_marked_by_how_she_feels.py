"""The world and feeling meet at deliberation in what a new thing claims of her curiosity.

core/agency/marked.py: each thing new to her world, marked by how far her
arousal stands above where it usually sits, times how much the thing stands
out, once; the mark is a share of her curiosity the thing now claims.
"""

from __future__ import annotations

import pytest

from core.agency.marked import SEEN, marked, stirred
from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState
from core.subject.perturbation import perturb

DELTA = 0.15


def test_her_usual_level_stirs_nothing_and_a_usual_rise_reads_a_half():
    assert stirred(0.7, centre=0.7, scatter=0.05) == 0.0
    assert stirred(0.6, centre=0.7, scatter=0.05) == 0.0
    assert stirred(0.75, centre=0.7, scatter=0.05) == pytest.approx(0.5)
    assert 0.5 < stirred(0.85, centre=0.7, scatter=0.05) < 1.0


def test_being_stirred_with_nothing_new_marks_nothing():
    assert marked(AuraState.default(), stir=0.9) == []


def test_something_new_met_at_her_usual_level_is_noted_and_left():
    state = AuraState.default()
    perturb(state, "W", DELTA)
    assert marked(state, stir=0.0) == [("subject_core_probe", 0.0)]
    assert state.world.known_entities["subject_core_probe"][SEEN]
    assert marked(state, stir=0.9) == []


def test_something_new_is_marked_as_far_as_she_is_stirred_and_it_stands_out():
    state = AuraState.default()
    perturb(state, "W", DELTA)
    assert marked(state, stir=0.6) == [("subject_core_probe", pytest.approx(0.6 * (0.5 + DELTA)))]


def test_the_world_and_feeling_set_together_mark_more_than_each_adds():
    def mark(world: bool, stir: float) -> float:
        state = AuraState.default()
        if world:
            perturb(state, "W", DELTA)
        return sum(value for _name, value in marked(state, stir=stir))

    joint = mark(True, 0.7) - mark(True, 0.2) - mark(False, 0.7) + mark(False, 0.2)
    assert joint == pytest.approx(0.5 * (0.5 + DELTA))


def test_a_mark_takes_its_share_of_what_her_curiosity_holds():
    state = AuraState.default()
    budget = state.motivation.budgets["curiosity"]
    budget["level"] = 80.0
    taken = MotivationUpdatePhase._curiosity_gap(state.motivation, [("thing", 0.25)])
    assert taken == pytest.approx(20.0)
    assert budget["level"] == pytest.approx(60.0)
    assert MotivationUpdatePhase._curiosity_gap(state.motivation, []) == 0.0
