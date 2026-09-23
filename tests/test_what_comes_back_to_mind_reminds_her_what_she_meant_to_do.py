"""Prospective memory: a recollection raises the intention it bears on, while it is in mind.

Memory reached deliberation only through the growth branch, which runs when a
drive is below its line, so on most turns nothing she recalled could touch an
intention already open. These pin the reminder: how far it moves an intention,
that it moves only intentions the recollection bears on, that holding the same
recollection does not compound it, and that it is taken back when the
recollection passes or the intention is finished. The last two are the
regression: as a ratchet it carried every goal to 1.0, finished ones included,
and `D.goal_urgency` read 1.0 for a whole run.
"""

from __future__ import annotations

import pytest

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState

GOAL = "finish the garden plan before the frost"


def _state(*, recalled: list[str], scores: list[float], urgency: float = 0.4) -> AuraState:
    state = AuraState.default()
    state.cognition.pending_initiatives = [{"goal": GOAL, "urgency": urgency, "source": "motivation_update"}]
    state.cognition.long_term_memory = recalled
    state.cognition.memory_scores = scores
    return state


def _recall(state: AuraState, recalled: list[str], scores: list[float]) -> None:
    state.cognition.long_term_memory = recalled
    state.cognition.memory_scores = scores


def test_a_recollection_that_bears_on_an_intention_raises_it_by_share_times_strength() -> None:
    state = _state(recalled=["we left the garden plan half drawn on the table"], scores=[0.8])
    assert MotivationUpdatePhase._reminded(state) == 1
    # The goal's cues are finish, garden, plan, before and frost; the recollection carries two.
    share = 2 / 5
    expected = 0.4 + share * 0.8 * (1.0 - 0.4)
    assert state.cognition.pending_initiatives[0]["urgency"] == pytest.approx(expected, abs=1e-4)


def test_a_recollection_that_shares_nothing_moves_nothing() -> None:
    state = _state(recalled=["a heron standing in the flooded field"], scores=[0.9])
    assert MotivationUpdatePhase._reminded(state) == 0
    assert state.cognition.pending_initiatives[0]["urgency"] == 0.4


def test_a_stronger_recollection_reminds_harder() -> None:
    weak = _state(recalled=["the garden plan"], scores=[0.3])
    strong = _state(recalled=["the garden plan"], scores=[0.9])
    MotivationUpdatePhase._reminded(weak)
    MotivationUpdatePhase._reminded(strong)
    assert strong.cognition.pending_initiatives[0]["urgency"] > weak.cognition.pending_initiatives[0]["urgency"] > 0.4


def test_the_same_recollection_held_in_mind_does_not_compound() -> None:
    state = _state(recalled=["the garden plan"], scores=[0.9])
    MotivationUpdatePhase._reminded(state)
    once = state.cognition.pending_initiatives[0]["urgency"]
    assert MotivationUpdatePhase._reminded(state) == 0
    assert state.cognition.pending_initiatives[0]["urgency"] == once


def test_a_reminder_that_passes_is_taken_back() -> None:
    state = _state(recalled=["the garden plan"], scores=[0.9])
    MotivationUpdatePhase._reminded(state)
    assert state.cognition.pending_initiatives[0]["urgency"] > 0.4
    _recall(state, [], [])
    assert MotivationUpdatePhase._reminded(state) == 1
    intention = state.cognition.pending_initiatives[0]
    assert intention["urgency"] == pytest.approx(0.4, abs=1e-4)
    assert "reminder_lift" not in intention and "reminded_by" not in intention


def test_what_something_else_added_stays_when_the_reminder_passes() -> None:
    state = _state(recalled=["the garden plan"], scores=[0.9])
    MotivationUpdatePhase._reminded(state)
    state.cognition.pending_initiatives[0]["urgency"] += 0.05
    _recall(state, ["a heron standing in the flooded field"], [0.9])
    MotivationUpdatePhase._reminded(state)
    assert state.cognition.pending_initiatives[0]["urgency"] == pytest.approx(0.45, abs=1e-4)


def test_new_recollections_every_turn_do_not_carry_an_intention_to_the_ceiling() -> None:
    state = _state(recalled=[], scores=[])
    ceiling = 0.0
    for turn in range(60):
        # A different recollection each turn, each carrying two of the five cues.
        _recall(state, [f"the garden plan, page {turn}"], [0.9])
        MotivationUpdatePhase._reminded(state)
        ceiling = max(ceiling, state.cognition.pending_initiatives[0]["urgency"])
    reminded_once = 0.4 + (2 / 5) * 0.9 * (1.0 - 0.4)
    assert ceiling == pytest.approx(reminded_once, abs=1e-4)
    assert ceiling < 1.0


def test_a_finished_intention_is_not_reminded() -> None:
    state = AuraState.default()
    state.cognition.active_goals = [{"goal": GOAL, "status": "done", "urgency": 0.3}]
    _recall(state, ["the garden plan before the frost"], [1.0])
    assert MotivationUpdatePhase._reminded(state) == 0
    assert state.cognition.active_goals[0]["urgency"] == 0.3


def test_an_intention_that_finishes_gives_its_reminder_back() -> None:
    state = AuraState.default()
    state.cognition.active_goals = [{"goal": GOAL, "urgency": 0.3}]
    _recall(state, ["the garden plan before the frost"], [1.0])
    MotivationUpdatePhase._reminded(state)
    assert state.cognition.active_goals[0]["urgency"] > 0.3
    state.cognition.active_goals[0]["status"] = "done"
    assert MotivationUpdatePhase._reminded(state) == 1
    assert state.cognition.active_goals[0]["urgency"] == pytest.approx(0.3, abs=1e-4)


def test_urgency_never_passes_one() -> None:
    state = _state(recalled=["finish the garden plan before the frost"], scores=[1.0], urgency=0.95)
    MotivationUpdatePhase._reminded(state)
    assert 0.95 <= state.cognition.pending_initiatives[0]["urgency"] <= 1.0


def test_an_active_goal_is_reminded_as_well() -> None:
    state = AuraState.default()
    state.cognition.active_goals = [{"goal": "answer the letter from the library", "urgency": 0.2}]
    state.cognition.long_term_memory = ["the library letter is still unanswered"]
    state.cognition.memory_scores = [0.7]
    assert MotivationUpdatePhase._reminded(state) == 1
    assert state.cognition.active_goals[0]["urgency"] > 0.2


def test_nothing_recalled_moves_nothing() -> None:
    state = _state(recalled=[], scores=[])
    assert MotivationUpdatePhase._reminded(state) == 0
