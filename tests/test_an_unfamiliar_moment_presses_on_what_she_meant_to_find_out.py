"""Development reaching deliberation: novelty presses on seeking intentions.

Development reached deliberation only through the growth branch, which runs
when a drive is below its line, and N -> D measured exactly zero in seed 7 of
the organ campaign. These pin the pressure: how far novelty lifts a curiosity
or growth intention, that the lift follows novelty down, that what another
reading added survives, and that intentions which keep what she has do not move.
"""

from __future__ import annotations

import pytest

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


def _state(novelty: float, *, drive: str = "curiosity", urgency: float = 0.4) -> AuraState:
    state = AuraState.default()
    state.response_modifiers["ontogenetic_novelty"] = novelty
    state.cognition.pending_initiatives = [
        {"goal": "find out how the tide tables are built", "urgency": urgency, "metadata": {"drive": drive}}
    ]
    return state


def _urgency(state: AuraState) -> float:
    return state.cognition.pending_initiatives[0]["urgency"]


def test_a_novel_moment_lifts_a_seeking_intention_by_its_share_of_the_way_to_one() -> None:
    state = _state(0.5)
    assert MotivationUpdatePhase._explored(state) == 1
    assert _urgency(state) == pytest.approx(0.4 + 0.5 * 0.6, abs=1e-4)


def test_the_lift_follows_novelty_down() -> None:
    state = _state(0.5)
    MotivationUpdatePhase._explored(state)
    state.response_modifiers["ontogenetic_novelty"] = 0.1
    MotivationUpdatePhase._explored(state)
    assert _urgency(state) == pytest.approx(0.4 + 0.1 * 0.6, abs=1e-4)


def test_a_familiar_moment_takes_the_lift_back_out() -> None:
    state = _state(0.5)
    MotivationUpdatePhase._explored(state)
    state.response_modifiers["ontogenetic_novelty"] = 0.0
    MotivationUpdatePhase._explored(state)
    assert _urgency(state) == pytest.approx(0.4, abs=1e-4)


def test_what_another_reading_added_is_kept() -> None:
    state = _state(0.5)
    MotivationUpdatePhase._explored(state)
    state.cognition.pending_initiatives[0]["urgency"] += 0.1  # a reminder, say
    state.response_modifiers["ontogenetic_novelty"] = 0.0
    MotivationUpdatePhase._explored(state)
    assert _urgency(state) == pytest.approx(0.5, abs=1e-4)


def test_repeated_turns_at_the_same_novelty_do_not_climb() -> None:
    state = _state(0.5)
    for _ in range(5):
        MotivationUpdatePhase._explored(state)
    assert _urgency(state) == pytest.approx(0.7, abs=1e-4)


def test_an_intention_that_keeps_what_she_has_does_not_move() -> None:
    state = _state(0.9, drive="integrity")
    assert MotivationUpdatePhase._explored(state) == 0
    assert _urgency(state) == 0.4


def test_an_intention_with_no_drive_does_not_move() -> None:
    state = AuraState.default()
    state.response_modifiers["ontogenetic_novelty"] = 0.9
    state.cognition.pending_initiatives = [{"goal": "tidy the workshop", "urgency": 0.4}]
    assert MotivationUpdatePhase._explored(state) == 0


def test_a_growth_goal_is_lifted_as_well() -> None:
    state = AuraState.default()
    state.response_modifiers["ontogenetic_novelty"] = 0.3
    state.cognition.active_goals = [{"goal": "learn the chord shapes", "urgency": 0.2, "metadata": {"drive": "growth"}}]
    assert MotivationUpdatePhase._explored(state) == 1
    assert state.cognition.active_goals[0]["urgency"] == pytest.approx(0.2 + 0.3 * 0.8, abs=1e-4)
