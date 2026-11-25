"""What a turn will need is begun with the turn, and waited for where it is used (core/runtime/begun_early.py)."""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.begun_early import begin, begun, in_a_turn


def _counting(answer, log, wait=0.0):
    async def make():
        log.append("made")
        await asyncio.sleep(wait)
        return answer

    return make


def test_what_was_begun_is_what_is_given_and_it_is_made_once():
    log = []

    async def turn():
        with in_a_turn():
            assert begin("recall", "basil", _counting("what she holds", log))
            assert not begin("recall", "basil", _counting("again", log))
            return await begun("recall", "basil", _counting("made here", log)), await begun("recall", "basil", _counting("x", log))

    assert asyncio.run(turn()) == ("what she holds", "what she holds")
    assert log == ["made"]


def test_the_readings_of_a_turn_are_taken_together_and_not_in_a_row():
    async def turn():
        loop = asyncio.get_running_loop()
        with in_a_turn():
            for name in ("recall", "senses", "screen"):
                begin(name, "", _counting(name, [], wait=0.2))
            began = loop.time()
            got = [await begun(name, "", _counting("", [])) for name in ("recall", "senses", "screen")]
            return got, loop.time() - began

    got, took = asyncio.run(turn())
    assert got == ["recall", "senses", "screen"] and took < 0.4


def test_words_that_changed_on_the_way_are_read_again():
    log = []

    async def turn():
        with in_a_turn():
            begin("recall", ("bryan", "what they typed"), _counting("of what they typed", log))
            return await begun("recall", ("bryan", "what they meant"), _counting("of what they meant", log))

    assert asyncio.run(turn()) == "of what they meant"


def test_outside_a_turn_it_is_made_where_it_is_used():
    async def alone():
        assert not begin("recall", "x", _counting("early", []))
        return await begun("recall", "x", _counting("here", []))

    assert asyncio.run(alone()) == "here"


def test_what_nobody_waited_for_stops_with_the_turn():
    stopped = []

    async def long():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.append(True)
            raise

    async def turn():
        with in_a_turn():
            begin("senses", "", long)
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(turn())
    assert stopped == [True]


def test_a_failure_is_the_waiters_own():
    async def fails():
        raise ValueError("no encoder")

    async def turn():
        with in_a_turn():
            begin("recall", "", fails)
            await asyncio.sleep(0)
            await begun("recall", "", fails)

    with pytest.raises(ValueError, match="no encoder"):
        asyncio.run(turn())


def test_a_wait_given_up_leaves_the_reading_for_the_next_who_asks():
    log = []

    async def turn():
        with in_a_turn():
            begin("recall", "", _counting("held", log, wait=0.1))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(begun("recall", "", _counting("", log)), timeout=0.01)
            return await begun("recall", "", _counting("", log))

    assert asyncio.run(turn()) == "held" and log == ["made"]


def test_one_turn_does_not_find_what_another_began():
    log = []

    async def one(name):
        with in_a_turn():
            begin("recall", "same words", _counting(name, log, wait=0.02))
            return await begun("recall", "same words", _counting("", log))

    async def both():
        return await asyncio.gather(one("hers"), one("theirs"))

    assert asyncio.run(both()) == ["hers", "theirs"]
