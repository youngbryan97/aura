"""A question whose answer is read by code does not open a thinking channel.

Measured live on 2026-08-26: the template opened the model's private channel
on a move decision, nothing downstream separated the two streams, and what
came back — "We need answer user. Need decide move for 2048. We must choose
one of up/down/left/right based on screen." — was spoken aloud as her plan.
"""

from __future__ import annotations

import pytest

from core.agency import her_reasoning
from core.brain.llm.chat_format import thinking_enabled_for_generation


class _Router:
    def __init__(self) -> None:
        self.asked: dict[str, object] = {}

    async def think(self, **kwargs):
        self.asked = kwargs
        return "left"


@pytest.mark.asyncio
async def test_a_decision_is_asked_on_a_lane_that_has_no_private_channel(monkeypatch):
    router = _Router()
    monkeypatch.setattr(her_reasoning, "_router", lambda: router)
    generate = her_reasoning.generator()
    assert await generate("which way", 0.3) == "left"
    assert router.asked["cognitive_mode"] == "fast"


def test_that_lane_really_is_a_lane_without_one():
    assert thinking_enabled_for_generation("qwen3-27b", cognitive_mode="fast") is False


@pytest.mark.asyncio
async def test_the_decision_is_still_marked_as_not_answering_a_person(monkeypatch):
    router = _Router()
    monkeypatch.setattr(her_reasoning, "_router", lambda: router)
    await her_reasoning.generator()("which way", 0.3)
    assert router.asked["internal_inference"] is True
