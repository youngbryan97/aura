"""Waiting on purpose is not being stuck, and a game she was playing is played on when she cannot think what next.

LIVE 2026-10-10 she read a game's instructions as they played, pressing nothing, and her run was ended as having no
move available three looks in; then, her model busy, the answer to "what next" was refused and she closed the game.
"""
from __future__ import annotations

import pytest

from core.skills.fluid_executor import WAITING, FluidExecutor

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_a_decision_that_waits_does_not_end_the_run_as_stuck():
    looks = {"n": 0}

    async def observe():
        looks["n"] += 1
        return looks["n"]

    async def decide(seen):
        return WAITING if seen <= 8 else None          # eight looks of a lesson playing, then nothing to do

    receipt = await FluidExecutor(stall_window=3).pursue("play", observe=observe, decide=decide,
                                                         is_satisfied=lambda seen: False, max_cycles=50, max_seconds=5)
    assert looks["n"] == 11 and receipt.outcome == "no_move_available"     # stuck only after the waiting was done


def test_a_game_she_has_been_playing_and_not_finished_is_played_on_without_asking_what_next():
    from core.skills import sovereign_browser_drawing as drawing

    url = "https://example.org/details/a-game"
    goal = "Go to it and play the game until you win."
    assert drawing.played_first({"url": url}, goal) is None             # never played here: nothing to go on with
    drawing._PLAYED_HERE.add(drawing._address(url))
    try:
        for _ in range(drawing.PLAY_ON_AT_MOST):
            on = drawing.played_first({"url": url}, goal)
            assert on["resolved_actions"][0]["selector"] == drawing.DRAWING and not on["done"]
        assert drawing.played_first({"url": url}, goal) is None         # gone back to as often as it is
        drawing._PLAYED_ON.clear()
        drawing._PLAY_OVER[drawing._address(url)] = "won"
        assert drawing.played_first({"url": url}, goal)["done"] is True  # finished: said so, not played on
    finally:
        drawing._PLAYED_HERE.discard(drawing._address(url))
        drawing._PLAYED_ON.pop(drawing._address(url), None)
        drawing._PLAY_OVER.pop(drawing._address(url), None)
