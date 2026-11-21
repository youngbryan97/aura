"""What she has set in motion is done once, and what can no longer land is not held open.

LIVE 21 November 2026: every desktop request was declared as an intention twice, once by the task and once by the
computer-use call the task made for it; and her loop held 1,176 intentions open, 1,053 of them "Use tool
'web_search'", with a median age of 138 days, from runtimes long stopped. A lesson from watching a strategy game played
(core/cognition/what_she_has_set_in_motion.py): starting again what is already on its way is doing it twice, and a
thing late is different from a thing still coming.
"""
from __future__ import annotations

import asyncio
import contextvars
import sqlite3

import core.agency.intention_loop as intention_loop
from core.agency.intention_loop import NEVER_LANDED, IntentionLoop


def _in_its_own_line(fn):
    """Run ``fn`` as one line of work, as one request's calls are: one task, awaiting its own calls."""
    async def line():
        return fn()

    return contextvars.copy_context().run(asyncio.run, line())


def test_a_call_made_inside_an_open_intention_for_the_same_thing_joins_it(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "i.db"))

    def one_request():
        outer = loop.intend("Open the app and play a game", drive="desktop_ui")
        inner = loop.intend("Open the app and play a game", drive="desktop_ui")
        assert inner == outer
        assert len(loop.get_open_intentions()) == 1
        loop.revise(inner, success=True)  # the inner call is done; the request is not
        assert len(loop.get_open_intentions()) == 1
        loop.revise(outer, success=True)
        assert loop.get_open_intentions() == []

    _in_its_own_line(one_request)
    loop.close()


def test_a_different_call_inside_it_is_an_intention_of_its_own(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "i.db"))

    def one_request():
        outer = loop.intend("Find the paper", drive="desktop_ui")
        inner = loop.intend("Use tool 'web_search'", drive="desktop_ui")
        assert inner != outer
        loop.revise(inner, success=True)
        loop.revise(outer, success=True)
        assert loop.get_open_intentions() == []

    _in_its_own_line(one_request)
    loop.close()


def test_the_same_thing_asked_twice_in_two_requests_is_two_intentions(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "i.db"))
    first = _in_its_own_line(lambda: loop.intend("Use tool 'web_search'", drive="curiosity"))
    second = _in_its_own_line(lambda: loop.intend("Use tool 'web_search'", drive="curiosity"))
    assert first != second
    loop.close()


def test_what_a_stopped_runtime_left_open_is_closed_at_boot(tmp_path, monkeypatch) -> None:
    db = tmp_path / "i.db"
    before = IntentionLoop(db_path=str(db))
    left_open = _in_its_own_line(lambda: before.intend("Use tool 'web_search'", drive="unknown"))
    before.close()
    monkeypatch.setattr(intention_loop, "_THIS_RUNTIME", "the next runtime")
    after = IntentionLoop(db_path=str(db))
    assert after.get_open_intentions() == []
    after.close()
    with sqlite3.connect(db) as conn:
        status, observation = conn.execute("SELECT status, observation FROM intentions WHERE id=?",
                                           (left_open,)).fetchone()
    assert status == "abandoned" and observation == NEVER_LANDED


def test_what_this_runtime_has_open_is_kept_when_it_reads_its_record_again(tmp_path) -> None:
    db = tmp_path / "i.db"
    first = IntentionLoop(db_path=str(db))
    open_now = _in_its_own_line(lambda: first.intend("Write the note", drive="desktop_ui"))
    second = IntentionLoop(db_path=str(db))
    assert [rec.id for rec in second.get_open_intentions()] == [open_now]
    first.close()
    second.close()


def test_calls_outside_any_line_of_work_never_join(tmp_path) -> None:
    """A value left in a thread's context by an intention never closed there is not a line of work."""
    loop = IntentionLoop(db_path=str(tmp_path / "i.db"))
    first = contextvars.copy_context().run(lambda: (loop.intend("Use tool 'web_search'", drive="x"),
                                                    loop.intend("Use tool 'web_search'", drive="x")))
    assert first[0] != first[1]
    loop.close()
