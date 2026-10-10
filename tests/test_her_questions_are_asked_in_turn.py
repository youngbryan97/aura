"""Her own questions beside her work are asked one at a time, the one that matters most first.

LIVE 2026-10-10 a game's lesson was never read in five minutes of play: the question of what its rules ask waited
behind questions about what a dog and a cabinet are.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognition.asking_in_turn import A_LOOK, RULES, THE_PLACE, THINGS, in_turn

pytestmark = pytest.mark.unit


def test_the_waiting_question_that_matters_most_goes_next_and_one_is_asked_at_a_time():
    order: list[str] = []
    inside = 0

    async def question(name: str, matters: int, gate: asyncio.Event | None = None) -> None:
        nonlocal inside
        async with in_turn(matters):
            inside += 1
            assert inside == 1
            order.append(name)
            if gate is not None:
                await gate.wait()
            await asyncio.sleep(0)
            inside -= 1

    async def run() -> None:
        gate = asyncio.Event()
        first = asyncio.create_task(question("a dog", THINGS, gate))
        await asyncio.sleep(0)
        rest = [asyncio.create_task(question(n, m)) for n, m in
                (("an odd look", A_LOOK), ("a cabinet", THINGS), ("the rules", RULES), ("the place", THE_PLACE))]
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(first, *rest)

    asyncio.run(run())
    assert order == ["a dog", "the rules", "the place", "a cabinet", "an odd look"]


def test_a_question_given_up_while_waiting_holds_up_nobody():
    async def run() -> list[str]:
        done: list[str] = []
        gate = asyncio.Event()

        async def hold() -> None:
            async with in_turn(THINGS):
                await gate.wait()

        async def ask(name: str, matters: int) -> None:
            async with in_turn(matters):
                done.append(name)

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0)
        dropped = asyncio.create_task(ask("dropped", RULES))
        kept = asyncio.create_task(ask("kept", A_LOOK))
        await asyncio.sleep(0)
        dropped.cancel()
        gate.set()
        await asyncio.wait_for(asyncio.gather(holder, kept), 2)
        async with in_turn(A_LOOK):                      # and the turn is free again after
            done.append("after")
        return done

    assert asyncio.run(run()) == ["kept", "after"]
