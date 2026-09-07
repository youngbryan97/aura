"""A person's tool call must not queue behind Aura's own unfinished errand.

Rule 5 of the executive defers SPAWN_TASK, TOOL_CALL and REFLECT while an
obligation is open, and it excludes ``IntentSource.USER`` because a person
waiting outranks a background goal. Which intents count as USER was decided by
a literal list of origin spellings, and the list did not contain what the
desktop lane emits.

LIVE, 2026-09-07: "Find out who wrote the novel Solaris and reply with just
the author's name" came back as ``Execution deferred:
temporal_obligation_active`` for a file_operation dispatched from
origin=response_generation_user.
"""

from __future__ import annotations

import pytest

from core.executive.executive_core import IntentSource, _coerce_intent_source
from core.runtime.turn_outcome import TurnOutcome, bind_turn


@pytest.mark.parametrize(
    "origin",
    ["desktop_quick_user", "user_chat", "chat_api", "desktop_ui", "voice_bridge"],
)
def test_a_lane_that_serves_a_person_is_a_person(origin: str) -> None:
    assert _coerce_intent_source(origin) is IntentSource.USER


@pytest.mark.parametrize(
    "phase_origin",
    [
        "response_generation_user",
        "response_generation_amplifier_user",
        "unitary_response_phase",
    ],
)
def test_a_phase_inside_a_persons_turn_is_the_persons_work(phase_origin: str) -> None:
    """A phase name says nothing on its own. The open turn says it."""

    assert _coerce_intent_source(phase_origin) is IntentSource.AUTONOMOUS
    with bind_turn(TurnOutcome(origin="user_chat")):
        assert _coerce_intent_source(phase_origin) is IntentSource.USER


@pytest.mark.parametrize(
    "lane",
    ["curiosity", "dream", "background_reflection", "intention_loop", "research_cycle"],
)
def test_her_own_loop_stays_her_own_while_somebody_waits(lane: str) -> None:
    """An errand running during a person's turn is still an errand."""

    with bind_turn(TurnOutcome(origin="user_chat")):
        assert _coerce_intent_source(lane) is not IntentSource.USER


def test_a_turn_nobody_is_waiting_on_does_not_promote() -> None:
    with bind_turn(TurnOutcome(origin="background_reflection")):
        assert _coerce_intent_source("response_generation_user") is IntentSource.AUTONOMOUS


def test_a_stated_answer_wins_over_the_name() -> None:
    assert (
        _coerce_intent_source("curiosity", person_is_waiting=True) is IntentSource.USER
    )
    with bind_turn(TurnOutcome(origin="user_chat")):
        assert (
            _coerce_intent_source("response_generation_user", person_is_waiting=False)
            is IntentSource.AUTONOMOUS
        )


def test_maintenance_and_research_keep_their_own_kinds() -> None:
    with bind_turn(TurnOutcome(origin="user_chat")):
        assert _coerce_intent_source("system_maintenance:gc") is IntentSource.MAINTENANCE
        assert (
            _coerce_intent_source("research_cycle") is IntentSource.AUTONOMOUS_RESEARCH
        )
        assert _coerce_intent_source("curiosity_web_research") is IntentSource.AUTONOMOUS
        assert _coerce_intent_source("background") is IntentSource.BACKGROUND


@pytest.mark.asyncio
async def test_the_tool_intent_carries_the_persons_priority() -> None:
    from core.executive.executive_core import get_executive_core

    core = get_executive_core()
    with bind_turn(TurnOutcome(origin="user_chat")):
        intent, _record = await core.prepare_tool_intent(
            "file_operation",
            {"operation": "read", "path": "README.md"},
            source="response_generation_user",
        )
    assert intent.source is IntentSource.USER
    assert intent.priority == pytest.approx(0.9)
