"""What she gets ready on someone's time stands in only for what their own turn could not get, and never costs it time.

After she answers somebody the turn is theirs, and while they read and write she recalls what bears on the exchange as
it stands (core/cognition/ready_for_their_turn.py). Their words decide what is recalled for their turn; what she got
ready stands in where that comes back empty (suggestions of 23 November 2026, item 21).
"""
from __future__ import annotations

import asyncio

from core.cognition.ready_for_their_turn import ReadyForTheirTurn


def test_what_was_got_ready_is_there_when_their_turn_comes_once() -> None:
    async def run():
        ready = ReadyForTheirTurn()

        async def recall():
            return "what bears on it"

        ready.get_ready("Bryan", recall)
        await asyncio.sleep(0.01)
        ready.their_turn_came("Bryan")
        return ready.take("Bryan"), ready.take("Bryan")

    assert asyncio.run(run()) == ("what bears on it", None)


def test_their_turn_coming_stops_work_still_under_way_for_it() -> None:
    async def run():
        ready, stopped = ReadyForTheirTurn(), []

        async def slow():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                stopped.append(True)
                raise
            return "too late"

        ready.get_ready("Bryan", slow)
        await asyncio.sleep(0.01)
        ready.their_turn_came("Bryan")
        await asyncio.sleep(0.01)
        return ready.take("Bryan"), stopped

    assert asyncio.run(run()) == (None, [True])


def test_what_was_got_ready_before_the_sitting_began_is_not_used() -> None:
    async def run():
        ready = ReadyForTheirTurn()

        async def recall():
            return "from before"

        ready.get_ready("Bryan", recall)
        await asyncio.sleep(0.01)
        return ready.take("Bryan", since=10 ** 12)

    assert asyncio.run(run()) is None


def test_a_turns_recall_that_comes_back_empty_is_stood_in_for_and_one_that_does_not_is_not(monkeypatch) -> None:
    import core.memory.rag_bridge as rag_bridge
    import interface.routes.chat_lane_bookkeeping as bookkeeping

    monkeypatch.setattr(bookkeeping, "_READY", {})
    monkeypatch.setattr(bookkeeping, "_listening_for_her_answers", lambda: None)
    monkeypatch.setattr(bookkeeping._chat_memory_state, "_chat_memory_identity", lambda: ("bryan", "desktop"))
    monkeypatch.setattr(bookkeeping, "_this_sitting_began_at", lambda: 0.0)

    async def no_earlier(_message, _began):
        return None

    monkeypatch.setattr(bookkeeping, "_her_earlier_answers_to", no_earlier)
    live = {"recalled": ""}

    async def fetch(query, **_how):
        return live["recalled"]

    monkeypatch.setattr(rag_bridge, "fetch_deep_context", fetch)

    async def run():
        ready = bookkeeping._ready_for_their_turn()

        async def got_ready():
            return "recalled on their time"

        ready.get_ready("bryan", got_ready)
        await asyncio.sleep(0.01)
        empty = await bookkeeping._fetch_deep_memory_context("and what about the second one?")
        ready.get_ready("bryan", got_ready)
        await asyncio.sleep(0.01)
        live["recalled"] = "recalled for their words"
        full = await bookkeeping._fetch_deep_memory_context("and what about the second one?")
        return empty, full

    assert asyncio.run(run()) == ("recalled on their time", "recalled for their words")
