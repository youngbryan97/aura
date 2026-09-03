"""Deliberating over a settled thing is effort spent looking like diligence.

A capture in draughts is compulsory, so where one is available there is
nothing to weigh. It was costing a whole pass of her reasoning, every time,
for a move that could not have come out otherwise.
"""

from __future__ import annotations

import asyncio

from core.agency.deliberate_action import ActionOption, deliberate


def _never_called(*args, **kwargs):
    raise AssertionError("her reasoning should not have been asked")


def test_one_option_is_taken_without_thinking_about_it() -> None:
    got = asyncio.run(
        deliberate(
            "get through the door",
            "one door",
            [ActionOption(name="open it")],
            think=_never_called,
            announce=False,
            lived=False,
        )
    )
    assert got.reached
    assert got.chosen is not None and got.chosen.name == "open it"
    assert got.confidence == 1.0
    assert "only thing available" in got.rationale
    assert not got.spoke, "and she does not claim to have put it into words"


def test_nothing_available_is_still_nothing_available() -> None:
    got = asyncio.run(
        deliberate(
            "get through the door",
            "no doors",
            [],
            think=_never_called,
            announce=False,
            lived=False,
        )
    )
    assert not got.reached
    assert "nothing is available" in got.reason


def test_a_real_choice_still_goes_to_her_reasoning() -> None:
    """The saving must not become a way of never thinking."""
    asked: list[int] = []

    async def think(*args, **kwargs):
        asked.append(1)
        raise RuntimeError("stop here")

    asyncio.run(
        deliberate(
            "pick one",
            "two doors",
            [ActionOption(name="left"), ActionOption(name="right")],
            think=think,
            announce=False,
            lived=False,
        )
    )
    assert asked, "two options is a decision and has to be made"
