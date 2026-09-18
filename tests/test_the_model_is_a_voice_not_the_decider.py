"""The model is a voice she consults, not the one deciding.

A move her language named was the move she made, and everything she had
worked out about the position — the rule she learned by watching it, the
search through it, the measure she built for it — was consulted only when the
words named nothing. That is the model deciding and the rest of her
commenting, which is the wrong way round.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption, deliberate


def _options(*names: str) -> list[ActionOption]:
    return [ActionOption(name=name, detail=f"press {name}") for name in names]


@pytest.mark.asyncio
async def test_where_she_can_see_where_moves_lead_that_decides():
    async def says_left(_objective, _evidence):
        return "I will press left."

    settled = await deliberate(
        "play until 2048",
        "a board",
        _options("up", "down", "left", "right"),
        think=says_left,
        foresight={"up": (0.9, "it keeps things in order"), "left": (0.1, "it breaks the corner")},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "up"
    assert "order" in f"{settled.rationale} {settled.reason}"


@pytest.mark.asyncio
async def test_with_nothing_of_her_own_the_suggestion_is_the_direction():
    async def says_left(_objective, _evidence):
        return "I will press left."

    settled = await deliberate(
        "play until 2048",
        "a board",
        _options("up", "down", "left", "right"),
        think=says_left,
        foresight={},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "left"


@pytest.mark.asyncio
async def test_the_voice_being_gone_changes_nothing_about_deciding():
    async def no_voice(_objective, _evidence):
        raise RuntimeError("the model is reloading")

    settled = await deliberate(
        "play until 2048",
        "a board",
        _options("up", "down", "left", "right"),
        think=no_voice,
        foresight={"right": (0.8, "it keeps neighbours close")},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "right"
