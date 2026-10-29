"""A reply phase that leaves a turn empty serves the draft a gate suppressed.

Her runtime already names this defect: `answer_available_but_never_served`,
"turn ended holding a servable answer that was never shown to the person". It
named it eleven times on the reports run of 29 September, where 33 of 96 arms
recorded an empty reply while her cortex had answered every one. The thinking
loop has asked the turn ledger for its best surviving draft since July; the
phase that serves chat never did.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.phases.response_generation_unitary import (
    UnitaryResponsePhase,
    never_leaves_a_turn_holding_its_answer,
)
from core.runtime.turn_outcome import (
    TurnOutcome,
    bind_turn,
    note_candidate,
    note_suppression,
)


class _Phase:
    _is_user_facing_origin = UnitaryResponsePhase._is_user_facing_origin

    def __init__(self, says: str) -> None:
        self.says = says

    @never_leaves_a_turn_holding_its_answer
    async def execute(self, state, objective=None, **kwargs):
        state.cognition.last_response = self.says
        return state


def _state(said: str = "", origin: str = "user"):
    return SimpleNamespace(
        cognition=SimpleNamespace(last_response=said, current_origin=origin),
        response_modifiers={},
    )


def _ran(phase, state, objective="how are you?"):
    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            note_suppression(
                note_candidate(
                    "A draft a heuristic disliked, and an answer.", source="cortex"
                ),
                gate="a_heuristic",
                reasons=["truncated_tail"],
                recoverable=True,
            )
            return await phase.execute(state, objective)

    return asyncio.run(go())


def test_a_turn_that_would_say_nothing_serves_the_suppressed_draft():
    served = _ran(_Phase(""), _state())
    assert served.cognition.last_response == "A draft a heuristic disliked, and an answer."
    assert served.response_modifiers["served_a_suppressed_draft"] is True


def test_a_turn_that_answered_keeps_its_own_answer():
    served = _ran(_Phase("Steady, and glad you asked."), _state())
    assert served.cognition.last_response == "Steady, and glad you asked."
    assert "served_a_suppressed_draft" not in served.response_modifiers


def test_a_turn_that_said_what_it_said_before_is_a_turn_that_said_nothing():
    # The arm restores a snapshot carrying her last reply. A phase that commits
    # nothing leaves that reply in place, and it was read as this turn's answer.
    served = _ran(_Phase("earlier"), _state(said="earlier"))
    assert served.cognition.last_response == "A draft a heuristic disliked, and an answer."


def test_a_background_turn_is_left_alone():
    served = _ran(_Phase(""), _state(origin="motivation"))
    assert served.cognition.last_response == ""


def test_a_benchmark_turn_that_failed_closed_stays_closed():
    state = _state(origin="benchmark")

    class _Benchmark(_Phase):
        @never_leaves_a_turn_holding_its_answer
        async def execute(self, state, objective=None, **kwargs):
            state.cognition.last_response = ""
            state.response_modifiers["benchmark_generation_failed_closed"] = {"error": "x"}
            return state

    served = _ran(_Benchmark(""), state)
    assert served.cognition.last_response == ""


def test_text_a_gate_marked_unrecoverable_never_comes_back():
    state = _state()

    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            note_suppression(
                note_candidate("the private plan behind the answer", source="cortex"),
                gate="prompt_leak",
                reasons=["internal_task_prompt_leak"],
                recoverable=False,
            )
            return await _Phase("").execute(state, "how are you?")

    served = asyncio.run(go())
    assert served.cognition.last_response == ""


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_answering_twice_the_same_way_is_not_a_lost_answer():
    """A phase that committed nothing leaves her last reply in place, and a
    phase that answered the same way twice commits it again. Only the first is
    an answer lost, so a stale-but-present reply is replaced only when a gate
    took something."""

    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            # A live candidate, nothing suppressed: the pipeline worked.
            note_candidate("An earlier draft nobody rejected.", source="cortex")
            return await _Phase("earlier").execute(_state(said="earlier"), "how are you?")

    served = asyncio.run(go())
    assert served.cognition.last_response == "earlier"
    assert "served_a_suppressed_draft" not in served.response_modifiers


def test_an_empty_reply_still_takes_whatever_survived():
    """Nothing to prefer over a draft, so no suppression evidence is needed."""

    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            note_candidate("A draft nobody rejected.", source="cortex")
            return await _Phase("").execute(_state(), "how are you?")

    served = asyncio.run(go())
    assert served.cognition.last_response == "A draft nobody rejected."


class _GivesUp(_Phase):
    """The phase when its retry and its floor both fail the reliability check."""

    def __init__(self, raises: BaseException) -> None:
        super().__init__("")
        self.raises = raises

    @never_leaves_a_turn_holding_its_answer
    async def execute(self, state, objective=None, **kwargs):
        raise self.raises


_GAVE_UP = TimeoutError(
    "Foreground conversation lane produced only unsafe drafts: "
    "unsupported_self_condition_operational_claim"
)


def test_a_phase_that_gives_up_still_serves_what_the_turn_holds():
    """LIVE 2026-10-28, seed 89's reports run: asked to rate how she felt from
    -1 to +1, her cortex answered twice, the reliability gate suppressed both
    drafts as recoverable, and the turn ended answer_available_but_never_served
    with her last reply, from the anchor's snapshot, still in place. When the
    retry and the floor both fail, this is how the phase ends."""
    served = _ran(_GivesUp(_GAVE_UP), _state(said="earlier"))
    assert served.cognition.last_response == "A draft a heuristic disliked, and an answer."
    assert served.response_modifiers["served_a_suppressed_draft"] is True


def test_a_phase_that_gives_up_holding_nothing_still_fails():
    state = _state(said="earlier")

    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            return await _GivesUp(_GAVE_UP).execute(state, "how are you?")

    with pytest.raises(TimeoutError, match="only unsafe drafts"):
        asyncio.run(go())
    assert state.cognition.last_response == "earlier"


def test_a_stop_is_never_turned_into_an_answer():
    with pytest.raises(asyncio.CancelledError):
        _ran(_GivesUp(asyncio.CancelledError("user_requested_stop")), _state())
