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


@pytest.mark.asyncio
async def test_among_moves_she_rates_alike_the_voice_is_the_direction():
    """Advice among equals is what advice is for."""
    async def says_left(_objective, _evidence):
        return "I will press left."

    settled = await deliberate(
        "play until 2048",
        "a board",
        _options("up", "down", "left", "right"),
        think=says_left,
        foresight={name: (1.0, "it would show how this moves") for name in ("up", "down", "left", "right")},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "left"


@pytest.mark.asyncio
async def test_a_choice_about_how_she_goes_on_is_not_overruled_by_a_search_of_moves():
    from core.agency.deliberate_action import Expectation

    slow = ActionOption(
        name="slow down",
        detail="wait for the voice",
        expectation=Expectation(changed=False, describes="the voice catching up"),
    )

    async def says_slow(_objective, _evidence):
        return "slow down"

    settled = await deliberate(
        "play until 2048",
        "a board",
        [*_options("up", "left"), slow],
        think=says_slow,
        foresight={"up": (0.9, "it keeps things in order"), "slow down": (0.9, "")},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "slow down"


@pytest.mark.asyncio
async def test_a_half_sentence_from_the_voice_is_not_glued_to_her_own_reason():
    """LIVE 2026-09-18: "Going down — I choose ** down has not been tried yet"."""
    async def cut_off(_objective, _evidence):
        return "I choose **"

    settled = await deliberate(
        "play until 2048",
        "a board",
        _options("up", "down", "left", "right"),
        think=cut_off,
        foresight={"down": (0.7, "it keeps the largest in its corner")},
        control_point="test.voice",
        lived=False,
    )
    assert settled.chosen is not None and settled.chosen.name == "down"
    assert "I choose" not in settled.rationale
    assert settled.rationale == "it keeps the largest in its corner"
