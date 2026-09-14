"""An intention is finished when the need that formed it is met.

Nothing marked an initiative done, and the guard blocks a new intention while
one of hers is open, so the first intention of a run stayed open for the rest
of it. Across the 15,840 frames of run 031 deliberation's goal urgency never
changed and its initiative load took three values, which is why nothing else in
the mind could be shown to reach deliberation.
"""

from __future__ import annotations

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


def _with_intention(level: float, *, drive: str = "growth", kind: str = "motivational_drive") -> AuraState:
    state = AuraState.default()
    state.motivation.budgets[drive]["level"] = level
    state.cognition.pending_initiatives = [
        {
            "goal": f"attend to the {drive} drive",
            "source": "motivation_update",
            "type": kind,
            "status": "pending",
            "metadata": {"drive": drive, "phase": "motivation_update"},
        }
    ]
    return state


def test_an_intention_whose_drive_is_still_depleted_stays_open() -> None:
    state = _with_intention(5.0)
    assert MotivationUpdatePhase._close_met_intentions(state) == 0
    assert MotivationUpdatePhase._own_intention_is_open(state)


def test_an_intention_whose_drive_has_recovered_is_retired() -> None:
    state = _with_intention(95.0)
    assert MotivationUpdatePhase._close_met_intentions(state) == 1
    assert not MotivationUpdatePhase._own_intention_is_open(state)
    assert state.cognition.pending_initiatives == []


def test_the_line_is_the_one_the_need_was_judged_against() -> None:
    state = AuraState.default()
    threshold = MotivationUpdatePhase._need_threshold(state.motivation)
    just_under = _with_intention(threshold - 0.01)
    just_over = _with_intention(threshold + 0.01)
    assert MotivationUpdatePhase._close_met_intentions(just_under) == 0
    assert MotivationUpdatePhase._close_met_intentions(just_over) == 1


def test_something_to_pass_on_closes_once_there_is_nothing_left_to_pass_on() -> None:
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {
            "goal": "Passing on a chill: a pattern 9 predictions long turned",
            "source": "motivation_update",
            "type": "passing_on",
            "metadata": {"kind": "a chill", "phase": "motivation_update"},
        }
    ]
    state.affect.frisson = 0.0
    assert MotivationUpdatePhase._close_met_intentions(state) == 1


def test_something_still_worth_passing_on_stays_open() -> None:
    state = AuraState.default()
    state.affect.frisson = 0.6
    state.cognition.pending_initiatives = [
        {
            "goal": "Passing on a chill: a pattern turned",
            "source": "motivation_update",
            "type": "passing_on",
            "metadata": {"kind": "a chill"},
        }
    ]
    assert MotivationUpdatePhase._close_met_intentions(state) == 0


def test_an_intention_that_says_neither_is_left_as_it_was() -> None:
    """Nothing here can tell whether it was addressed, so it is not touched."""
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "attend to the growth drive", "source": "motivation_update", "status": "pending"}
    ]
    assert MotivationUpdatePhase._close_met_intentions(state) == 0
    assert MotivationUpdatePhase._own_intention_is_open(state)


def test_somebody_else_s_intention_is_never_retired_here() -> None:
    state = AuraState.default()
    state.motivation.budgets["growth"]["level"] = 95.0
    state.cognition.pending_initiatives = [
        {
            "goal": "answer the question",
            "source": "executive_closure",
            "metadata": {"drive": "growth"},
        }
    ]
    assert MotivationUpdatePhase._close_met_intentions(state) == 0
    assert len(state.cognition.pending_initiatives) == 1
