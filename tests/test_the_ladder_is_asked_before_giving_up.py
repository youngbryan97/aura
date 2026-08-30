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

    async def ladder(message, *, reason):
        asked["message"] = message
        asked["reason"] = reason
        return "here is what I can tell you"

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    said = await chat._anything_better_than_giving_up(
        "why is my service leaking memory", reason="cortex unavailable", already=""
    )
    assert said == "here is what I can tell you"
    assert asked["message"] == "why is my service leaking memory"


@pytest.mark.asyncio
async def test_evidence_already_in_hand_is_not_replaced(monkeypatch):
    async def ladder(message, *, reason):  # pragma: no cover - must not run
        raise AssertionError("the ladder was asked over a real reading")

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    said = await chat._anything_better_than_giving_up(
        "anything", reason="x", already="the health channels say this"
    )
    assert said == "the health channels say this"


@pytest.mark.asyncio
async def test_a_ladder_that_cannot_answer_leaves_the_honest_failure(monkeypatch):
    async def ladder(message, *, reason):
        return ""

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    assert await chat._anything_better_than_giving_up(
        "anything", reason="x", already=""
    ) == ""


@pytest.mark.asyncio
async def test_a_ladder_that_raises_does_not_take_the_turn_with_it(monkeypatch):
    async def ladder(message, *, reason):
        raise RuntimeError("no router")

    monkeypatch.setattr(chat, "_answer_from_fallback_ladder", ladder)
    assert await chat._anything_better_than_giving_up(
        "anything", reason="x", already=""
    ) == ""


def test_both_giving_up_paths_ask_it():
    """A rescue that lives at one of two sites is the defect being fixed."""
    import inspect

    source = inspect.getsource(chat)
    assert source.count("_anything_better_than_giving_up(") == 3  # one def, two uses
