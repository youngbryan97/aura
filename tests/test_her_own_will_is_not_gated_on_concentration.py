"""A guard that tests the wrong thing closes the route it was written to pace.

The motivation phase generates an intention from whichever need is most
depleted, using what she has just recalled to decide which footing is weakest.
It was skipped whenever the cognitive mode was DELIBERATE — and DELIBERATE is
the careful governed route for an ordinary user-facing turn, four turns in five
of a recording. So the route from a depleted need, and from memory, into an
intention was closed on almost every turn she was thinking carefully: the guard
tested a mode that means she is concentrating and read it as meaning she is
already busy with herself.

What it meant to say is readable from the intentions themselves.
"""

from __future__ import annotations

import asyncio

from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


def _state(mode: str) -> AuraState:
    state = AuraState.default()
    for member in type(state.cognition.current_mode):
        if member.value == mode:
            state.cognition.current_mode = member
            break
    return state


def test_concentrating_does_not_stop_her_forming_an_intention() -> None:
    deliberate = _state("deliberate")
    assert deliberate.cognition.current_mode.value == "deliberate"
    assert not MotivationUpdatePhase._own_intention_is_open(deliberate)


def test_an_open_intention_of_her_own_stops_a_second_one() -> None:
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "attend to the growth drive", "source": "motivation_update", "status": "pending"}
    ]
    assert MotivationUpdatePhase._own_intention_is_open(state)


def test_an_intention_she_has_finished_does_not_stop_the_next_one() -> None:
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "attend to the growth drive", "source": "motivation_update", "status": "done"}
    ]
    assert not MotivationUpdatePhase._own_intention_is_open(state)


def test_somebody_else_s_intention_does_not_stop_hers() -> None:
    """The guard is about her own motivational loop, not about the queue."""
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "answer the question", "source": "executive_closure", "status": "pending"}
    ]
    assert not MotivationUpdatePhase._own_intention_is_open(state)


def test_the_phase_still_runs_end_to_end_while_concentrating() -> None:
    phase = MotivationUpdatePhase(None)
    state = _state("deliberate")
    result = asyncio.run(phase.execute(state, objective="something to think about"))
    assert result is not None
