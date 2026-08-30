"""Nothing gives up while a loaded model sits idle.

Two paths reach the giving-up reply and only one of them asked the ladder, so
the same question that got a real answer through one route got "I couldn't get
to an answer I'd stand behind" through the other — with the 9B loaded, warm,
not the lane owner, and asked zero times.
"""

from __future__ import annotations

import pytest

import interface.routes.chat as chat


@pytest.mark.asyncio
async def test_the_ladder_is_asked_when_nothing_else_answered(monkeypatch):
    asked: dict[str, object] = {}

    async def ladder(message, *, reason, budget_s=None):
        asked["message"] = message
        asked["reason"] = reason
        asked["budget"] = budget_s
        return "here is what I can tell you"

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    said = await chat._anything_better_than_giving_up(
        "why is my service leaking memory", reason="cortex unavailable", already=""
    )
    assert said == "here is what I can tell you"
    assert asked["message"] == "why is my service leaking memory"


@pytest.mark.asyncio
async def test_the_ladder_gets_the_turn_s_remaining_time_not_a_constant(monkeypatch):
    """Refusing at 190s while a model finishes loading at 60 is the defect."""
    seen: dict[str, object] = {}

    async def ladder(message, *, reason, budget_s=None):
        seen["budget"] = budget_s
        return "answered"

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    await chat._anything_better_than_giving_up(
        "anything", reason="x", already="", budget_s=180.0
    )
    assert seen["budget"] == 180.0


@pytest.mark.asyncio
async def test_evidence_already_in_hand_is_not_replaced(monkeypatch):
    async def ladder(message, *, reason, budget_s=None):  # pragma: no cover - must not run
        raise AssertionError("the ladder was asked over a real reading")

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    said = await chat._anything_better_than_giving_up(
        "anything", reason="x", already="the health channels say this"
    )
    assert said == "the health channels say this"


@pytest.mark.asyncio
async def test_a_ladder_that_cannot_answer_leaves_the_honest_failure(monkeypatch):
    async def ladder(message, *, reason, budget_s=None):
        return ""

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    assert await chat._anything_better_than_giving_up(
        "anything", reason="x", already=""
    ) == ""


@pytest.mark.asyncio
async def test_a_ladder_that_raises_does_not_take_the_turn_with_it(monkeypatch):
    async def ladder(message, *, reason, budget_s=None):
        raise RuntimeError("no router")

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    assert await chat._anything_better_than_giving_up(
        "anything", reason="x", already=""
    ) == ""


def test_every_path_that_would_not_answer_asks_it_first():
    """A rescue that lives at one site and not its twin is the defect here.

    Three of them: the two that serve the honest failure, and the one that
    serves a description of what she is doing instead of an answer.
    """
    import inspect

    source = inspect.getsource(chat)
    assert source.count("_anything_better_than_giving_up(") == 4  # one def, three uses


def test_the_self_process_repair_is_the_second_choice():
    """"I am tracking what I am keeping in memory" is not an answer to anything."""
    import inspect

    source = inspect.getsource(chat)
    at = source.index("# An answer first.")
    block = source[at : at + 1400]
    asked = block.index("_anything_better_than_giving_up(")
    narrated = block.index("_build_grounded_self_process_repair_reply(")
    assert asked < narrated
