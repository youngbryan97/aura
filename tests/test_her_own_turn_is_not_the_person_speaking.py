"""Whose turn the conversation engine is told this was.

The phase passed `role="user"` on every turn, so `_process_aura_message` was
never reached from it. `turns_since_user_spoke` is set to zero there and
incremented only in the branch that was never taken, so it read zero for every
frame of a three hundred turn recording: a counter that can only be reset. The
same constant also had her own autonomous turns analysed as though the person
had spoken them, which set the floor state, the speech act and the open
question from her own words.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversational.dynamics import ConversationalDynamicsEngine
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.state.aura_state import AuraState


class _Recorder(ConversationalDynamicsEngine):
    """The real engine, remembering what it was told about the speaker."""

    def __init__(self) -> None:
        super().__init__()
        self.roles: list[str] = []

    def update(self, *, message, role, working_memory):
        self.roles.append(role)
        return super().update(
            message=message or "", role=role, working_memory=working_memory or []
        )


def _role_for(origin: str) -> str:
    phase = ConversationalDynamicsPhase(None)
    engine = _Recorder()
    state = AuraState.default()
    state.cognition.current_origin = origin
    asyncio.run(
        phase._execute_compute_dynamics_latest(
            "somebody", engine, state, "an objective", state
        )
    )
    return engine.roles[-1]


@pytest.mark.parametrize("origin", ["user", "voice", "api", "gui", "test"])
def test_a_turn_the_person_opened_is_the_person_speaking(origin):
    assert _role_for(origin) == "user"


@pytest.mark.parametrize("origin", ["system", "motivation", "", "autonomy"])
def test_a_turn_she_opened_is_hers(origin):
    assert _role_for(origin) == "assistant"


# ── what the counter then does ───────────────────────────────────────────


def test_the_counter_rises_while_she_is_the_one_talking():
    engine = ConversationalDynamicsEngine()
    engine.update(message="hello", role="user", working_memory=[])
    assert engine._state.turns_since_user_spoke == 0
    for expected in (1, 2, 3):
        engine.update(message="thinking", role="assistant", working_memory=[])
        assert engine._state.turns_since_user_spoke == expected


def test_the_person_speaking_puts_it_back_to_nothing():
    engine = ConversationalDynamicsEngine()
    for _ in range(4):
        engine.update(message="thinking", role="assistant", working_memory=[])
    assert engine._state.turns_since_user_spoke > 0
    engine.update(message="hello again", role="user", working_memory=[])
    assert engine._state.turns_since_user_spoke == 0


def test_over_a_mixed_run_it_is_not_a_constant():
    """The campaign's eight conditions are five of theirs and three of hers."""
    engine = ConversationalDynamicsEngine()
    origins = ["user", "user", "motivation", "system", "user", "system", "user", "user"]
    seen = set()
    for origin in origins * 4:
        role = "user" if origin in {"user", "voice", "api", "gui", "test"} else "assistant"
        engine.update(message="a turn", role=role, working_memory=[])
        seen.add(engine._state.turns_since_user_spoke)
    assert len(seen) > 1


# ── the phase's own gate ─────────────────────────────────────────────────


def test_a_turn_she_opened_still_reaches_the_engine():
    """The full analysis is for their messages. The count is for both."""
    phase = ConversationalDynamicsPhase(None)
    engine = _Recorder()
    state = AuraState.default()
    engine.update(message="hello", role="user", working_memory=[])
    for expected in (1, 2, 3):
        phase._note_her_own_turn(engine, "a thought of her own", state)
        assert state.cognition.turns_since_user_spoke == expected
    assert engine.roles[-3:] == ["assistant", "assistant", "assistant"]


def test_an_engine_that_raises_leaves_the_count_alone():
    class _Broken(ConversationalDynamicsEngine):
        def update(self, **_kwargs):
            raise RuntimeError("no engine")

    phase = ConversationalDynamicsPhase(None)
    state = AuraState.default()
    state.cognition.turns_since_user_spoke = 5
    phase._note_her_own_turn(_Broken(), "a thought", state)
    assert state.cognition.turns_since_user_spoke == 5


def test_the_phase_gate_notes_her_turn_before_returning():
    """The early return used to say nothing at all."""
    import asyncio

    phase = ConversationalDynamicsPhase(None)
    engine = _Recorder()
    phase._get_engine = lambda: engine
    state = AuraState.default()
    state.cognition.current_origin = "system"
    engine.update(message="hello", role="user", working_memory=[])
    out = asyncio.run(phase.execute(state, objective="keep working", origin="system"))
    assert out is state
    assert engine.roles[-1] == "assistant"
    assert state.cognition.turns_since_user_spoke == 1
