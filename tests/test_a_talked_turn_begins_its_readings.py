"""A turn that is talked through begins its slow readings together; one that asks for something done does not."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import interface.routes.chat_lane_bookkeeping as bookkeeping
import interface.routes.chat_what_is_asked as asks
from core.runtime.begun_early import begun, in_a_turn
from core.runtime.what_this_turn_asks import Asked
from core.together import taking_it_up
from core.together.taking_it_up import SCORES, SCREEN, SENSES


def _a_client(log):
    async def expect_next(said, candidates):
        log.append(("scores", said))
        return [-1.0] * len(candidates)

    return SimpleNamespace(expect_next=expect_next)


def _taking(monkeypatch, *, together):
    reader = SimpleNamespace(has_a_reference=lambda: True, seconds=lambda: ["On it.", "Sure."])
    taking = SimpleNamespace(
        _reads=lambda: reader, _her_record=lambda: SimpleNamespace(_things={}),
        _withs=lambda: SimpleNamespace(current=lambda who: SimpleNamespace(open=True) if together else None),
    )
    monkeypatch.setattr(taking_it_up, "install", lambda: taking)


def _what_was_begun(monkeypatch, said, asked, *, together=True):
    log = []

    async def deep(text):
        log.append(("recall", text))
        return "what she holds of it"

    async def senses():
        log.append(("senses", ""))
        return {}

    async def screen():
        log.append(("screen", ""))
        return "Notes"

    client = _a_client(log)
    async def of_herself(text):
        log.append(("herself", text))
        return "how she is"

    monkeypatch.setattr("core.conversation.chat_preflight.inject_operational_self_context", of_herself)
    monkeypatch.setattr(bookkeeping, "_fetch_deep_memory_context", deep)
    monkeypatch.setattr(asks, "her_client", lambda: client)
    monkeypatch.setattr(taking_it_up, "_her_senses_now", senses)
    monkeypatch.setattr(taking_it_up, "_the_screen_now", screen)
    _taking(monkeypatch, together=together)

    async def turn():
        async def not_begun():
            return None

        with in_a_turn():
            asks.begin_what_talk_needs(said, asked)
            words = " ".join(said.split())
            return (
                await begun(asks.REMEMBERED, asks.what_is_remembered_of(said), not_begun),
                await begun(SCORES, (id(client), words), not_begun),
                await begun(SENSES, "", not_begun),
                await begun(SCREEN, "", not_begun),
                await begun(asks.OF_HERSELF, said, not_begun),
            )

    return asyncio.run(turn()), log


def test_a_turn_that_is_talked_through_begins_them_all(monkeypatch):
    got, log = _what_was_begun(monkeypatch, "how is  the basil doing", None)
    assert got == ("what she holds of it", [-1.0, -1.0], {}, "Notes", "how she is")
    assert {name for name, _ in log} == {"recall", "scores", "senses", "screen", "herself"}
    assert ("scores", "how is the basil doing") in log


def test_her_senses_are_begun_only_where_she_keeps_somebody_company(monkeypatch):
    got, log = _what_was_begun(monkeypatch, "how is the basil doing", None, together=False)
    assert got[2:4] == (None, None)
    assert {name for name, _ in log} == {"recall", "scores", "herself"}


def test_a_turn_that_asks_for_something_done_leaves_her_worker_to_it(monkeypatch):
    asked = Asked(said="make me a timer", effect="make_program", decision="act", sure=0.99)
    got, log = _what_was_begun(monkeypatch, "make me a timer", asked)
    assert got == (None, None, None, None, None) and log == []
