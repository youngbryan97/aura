"""Self and intention meet at recurrent cognition in a rehearsal.

core/consciousness/rehearsal.py: what she has newly set herself to do is
anticipated, fading as it becomes familiar, and rehearsed as far as her self is
in question; the rehearsal enters the substrate as an input current.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.consciousness.rehearsal import ANTICIPATED, anticipating, rehearsal_current, rehearsing
from core.state.aura_state import AuraState
from core.subject.perturbation import perturb

DELTA = 0.15


def _state() -> AuraState:
    state = AuraState.default()
    state.cognition.active_goals = [{"goal": "an old intention", "urgency": 0.9, ANTICIPATED: 0.0}]
    return state


def test_a_new_intention_is_anticipated_at_its_urgency_and_fades_at_her_momentum():
    state = _state()
    state.cognition.active_goals.append({"goal": "something new", "urgency": 0.65})
    assert anticipating(state) == pytest.approx(0.65)
    assert anticipating(state) == pytest.approx(0.65 * state.affect.momentum)


def test_a_settled_self_does_not_rehearse_and_nothing_new_is_not_rehearsed():
    settled = _state()
    perturb(settled, "D", DELTA)
    assert rehearsing(settled) == 0.0
    unsettled = _state()
    perturb(unsettled, "S", DELTA)
    assert rehearsing(unsettled) == 0.0


def test_a_new_intention_and_a_self_in_question_rehearse_more_than_each_adds():
    def rehearsal(self_moved: bool, intention_moved: bool) -> float:
        state = _state()
        if self_moved:
            perturb(state, "S", DELTA)
        if intention_moved:
            perturb(state, "D", DELTA)
        return rehearsing(state)

    joint = rehearsal(True, True) - rehearsal(True, False) - rehearsal(False, True) + rehearsal(False, False)
    assert joint == pytest.approx(0.5 + DELTA)


def test_the_rehearsal_enters_at_the_recurrent_drive_s_strength_along_one_column():
    recurrent = np.full(256, 0.2)
    idle = SimpleNamespace(rehearsal=0.0)
    assert rehearsal_current(idle, recurrent) is None
    one = rehearsal_current(SimpleNamespace(rehearsal=1.0), recurrent)
    half = rehearsal_current(SimpleNamespace(rehearsal=0.5), recurrent)
    assert np.linalg.norm(one) == pytest.approx(np.linalg.norm(recurrent))
    assert np.allclose(half, 0.5 * one)


def test_the_substrate_takes_the_rehearsal_into_its_transition():
    import torch

    from core.consciousness.conjunctive_codes import with_conjunction_current

    drive = torch.full((256,), 0.2)
    quiet = with_conjunction_current(SimpleNamespace(rehearsal=0.0), drive)
    rehearsing_now = with_conjunction_current(SimpleNamespace(rehearsal=0.6), drive)
    assert torch.equal(quiet, drive)
    assert not torch.equal(rehearsing_now, drive)
