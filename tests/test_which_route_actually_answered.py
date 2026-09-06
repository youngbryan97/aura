"""Fourteen answer routes, and no way to tell which ever fires.

Each returns the reply unchanged when it declines, which is right and also
means a route that CANNOT fire looks exactly like one that rarely applies. An
external review put the general form of it: a channel wired to a consumer is
not a measured downstream effect.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.runtime.what_answered_this_turn import (
    ENOUGH_TO_JUDGE,
    forget_everything,
    how_the_routes_have_gone,
    offer,
    offer_async,
    routes_that_have_never_answered,
    what_answered,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _a_clean_record():
    forget_everything()
    yield
    forget_everything()


def test_a_route_that_changes_the_reply_counts_as_answering():
    out = offer("worked_out_sequence", "I am not sure", lambda: "It becomes 5, 9, 14.")
    assert out == "It becomes 5, 9, 14."
    route = what_answered("worked_out_sequence")
    assert route.offered == 1
    assert route.answered == 1
    assert route.declined == 0
    assert route.last_answered_at > 0


def test_a_route_that_leaves_the_reply_alone_counts_as_declining():
    out = offer("lifetime", "the model's words", lambda: "the model's words")
    assert out == "the model's words"
    route = what_answered("lifetime")
    assert route.answered == 0
    assert route.declined == 1


def test_whitespace_alone_is_not_an_answer():
    offer("tabular_answer", "same words", lambda: "  same words  ")
    assert what_answered("tabular_answer").answered == 0


def test_a_route_that_raises_costs_the_count_and_not_the_turn():
    """An answer route is an improvement on the model's words.

    A broken one must not take the turn down with it, and it must not be
    silently indistinguishable from one that declined.
    """
    def broken():
        raise ValueError("the store was not there")

    out = offer("host_load", "the model's words", broken)
    assert out == "the model's words"
    route = what_answered("host_load")
    assert route.raised == 1
    assert route.declined == 0
    assert "ValueError: the store was not there" in route.why_it_raised


def test_the_same_failure_twice_is_recorded_once():
    def broken():
        raise KeyError("missing")

    for _ in range(5):
        offer("queued_work", "x", broken)
    route = what_answered("queued_work")
    assert route.raised == 5
    assert len(route.why_it_raised) == 1


def test_an_async_route_is_measured_the_same_way():
    async def go():
        return await offer_async("solved_game", "unsure", _answers)

    async def _answers():
        return "The solution is up, left, up."

    out = asyncio.run(go())
    assert out == "The solution is up, left, up."
    assert what_answered("solved_game").answered == 1


def test_an_async_route_that_raises_is_caught_too():
    async def go():
        return await offer_async("solved_game", "unsure", _breaks)

    async def _breaks():
        raise RuntimeError("no board")

    assert asyncio.run(go()) == "unsure"
    assert what_answered("solved_game").raised == 1


# --------------------------------------------------------------- the gate


def test_a_route_offered_enough_turns_that_never_answered_is_named():
    for _ in range(ENOUGH_TO_JUDGE):
        offer("a_route_that_cannot_fire", "x", lambda: "x")
    assert "a_route_that_cannot_fire" in routes_that_have_never_answered()


def test_a_route_with_too_few_turns_to_judge_is_not_named():
    """A route that applies to one turn in a thousand needs a long window."""
    for _ in range(ENOUGH_TO_JUDGE - 1):
        offer("rarely_applies", "x", lambda: "x")
    assert routes_that_have_never_answered() == []


def test_a_route_that_has_answered_once_is_never_named():
    offer("worked_out_sequence", "x", lambda: "worked out")
    for _ in range(ENOUGH_TO_JUDGE * 2):
        offer("worked_out_sequence", "x", lambda: "x")
    assert routes_that_have_never_answered() == []


def test_the_share_is_answers_over_offers():
    for _ in range(3):
        offer("lifetime", "x", lambda: "x")
    offer("lifetime", "x", lambda: "she is three years old")
    assert how_the_routes_have_gone()["lifetime"]["share"] == 0.25


def test_a_route_nobody_offered_has_no_record():
    assert what_answered("never_offered") is None
    assert "never_offered" not in how_the_routes_have_gone()


# ------------------------------------------------------------ the wiring


def test_every_chat_answer_route_goes_through_the_ledger():
    """The finding: fourteen routes, none of them counted."""
    source = (ROOT / "interface" / "routes" / "chat.py").read_text("utf-8")
    for name in (
        "measured_filesystem_count",
        "measured_belief_history",
        "earlier_conversation",
        "host_load",
        "queued_work",
        "recent_activity",
        "saved_artifact",
        "positional_solution",
        "worked_out_sequence",
        "lifetime",
        "tabular_answer",
        "solved_game",
        "repo_diagnosis",
        "built_artifact",
    ):
        assert f'("{name}"' in source, f"{name} is not offered through the ledger"


def test_the_routes_are_not_applied_outside_the_ledger_any_more():
    """The old form called each server directly and dropped the outcome."""
    source = (ROOT / "interface" / "routes" / "chat.py").read_text("utf-8")
    assert "corrected = str(_serve_worked_out_sequence(" not in source
    assert "corrected = str(_serve_positional_solution(" not in source


def test_the_record_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["what_answered_this_turn"]
    assert set(block) >= {"routes", "never_answered"}
