"""A desktop objective runs once a turn, and when the page she worked through goes to her turn, her answer's time
starts then.

LIVE 11 November 2026, asked in her chat what r/nosleep's top thread was about: the browsing took 186 s of the
request's time, her answer was skipped with "remaining budget 2.0s is below 60.0s", and a later door ran the same
request again; the second pursuit's account was served as the reply.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from interface.routes import chat_desktop_objective, chat_desktop_objective_gates as gates

pytestmark = pytest.mark.unit


def _run(state, monkeypatch, status):
    runs: list[str] = []

    async def execute(message, *, cognitive_reply, her_reply_first=False):
        runs.append(message)
        return {"ok": True, "status": status, "response": "", "result": {"ok": True}}

    monkeypatch.setattr(chat_desktop_objective, "_execute_desktop_objective_from_chat", execute)

    async def twice():
        first = await gates._lifted_run_desktop_objective_tracked(
            "look at r/nosleep", cognitive_reply="", _desktop_exec_state=state,
            conversation_only_surface=False, pending_exchange_id=None)
        second = await gates._lifted_run_desktop_objective_tracked(
            "look at r/nosleep", cognitive_reply="later", _desktop_exec_state=state,
            conversation_only_surface=False, pending_exchange_id=None)
        return first, second

    return runs, asyncio.run(twice())


def test_a_later_door_is_given_the_first_run_and_does_not_run_it_again(monkeypatch):
    state = {"attempted": False, "result": None, "answer_clock": {"from": 0.0}}
    runs, (first, second) = _run(state, monkeypatch, "desktop_objective_completed")
    assert runs == ["look at r/nosleep"]
    assert second is first


def test_her_answer_to_the_page_she_was_handed_starts_its_time_then(monkeypatch):
    started = time.monotonic() - 186.0
    state = {"attempted": False, "result": None, "answer_clock": {"from": started}}
    _run(state, monkeypatch, "desktop_objective_page_read")
    assert time.monotonic() - state["answer_clock"]["from"] < 5.0


def test_a_run_that_did_not_hand_her_a_page_leaves_the_clock_where_it_was(monkeypatch):
    started = time.monotonic() - 186.0
    state = {"attempted": False, "result": None, "answer_clock": {"from": started}}
    _run(state, monkeypatch, "desktop_objective_completed")
    assert state["answer_clock"]["from"] == started


def test_once_the_page_is_hers_a_later_door_has_nothing_to_serve_over_her_reply(monkeypatch):
    """LIVE 12 November 2026: her model answered from r/nosleep, and the door after her reply served the first run's
    empty response, so her chat showed nothing."""
    state = {"attempted": False, "result": None, "answer_clock": {"from": 0.0}}
    runs, (first, second) = _run(state, monkeypatch, "desktop_objective_page_read")
    assert runs == ["look at r/nosleep"]
    assert first["status"] == "desktop_objective_page_read" and second is None
