"""Announcing no measurable objective while measuring every future against one.

A request can name a process without naming a finish — "play it" — and she
reads a ceiling off the thing in front of her instead: a laid-out thing that
combines equal pairs cannot exceed one doubling per place it has. Everything
that scores a future uses that.

The sentence naming what kind of problem she is in read the raw request. So
she said "no measurable objective" out loud, on screen, while playing to a
ceiling of sixty-five thousand — LIVE 2026-09-07, with a sixty-four on the
board and a score of five hundred.

Saying a thing about herself that is not true of her is worse than saying
nothing: it is the part a watcher uses to tell a mind that knows its situation
from one that got lucky in it.
"""

from __future__ import annotations

from core.agency.what_kind_of_problem import recognise
from core.perception.how_it_moves import RULES, HowItMoves
from core.perception.where_it_responds import what_is_there
from core.skills.screen_pursuit import _what_there_is_to_aim_at

BOARD = "4 . . .\n2 32 8 2\n16 4 2 .\n4 64 8 ."
ACTS = ["up", "down", "left", "right"]


def _board():
    return what_is_there({"ok": True, "text": BOARD, "layout": [], "bounds": []}, None)


def _one_that_knows() -> HowItMoves:
    knows = HowItMoves()
    for rule in RULES:
        knows.tried[rule.name] = knows.tried_when_it_moved[rule.name] = 20
        got = 20 if "combin" in rule.name else 0
        knows.right[rule.name] = knows.right_when_it_moved[rule.name] = got
    knows.seen = knows.moved = 20
    return knows


def test_she_reads_something_to_aim_at_off_the_thing_itself():
    """Nobody named a finish, and there is one."""
    assert _what_there_is_to_aim_at(_board())


def test_what_she_aims_at_is_countable():
    shape = recognise(
        acts=ACTS,
        knows_how_it_moves=_one_that_knows(),
        state=_board(),
        toward=_what_there_is_to_aim_at(_board()),
    ).shape
    assert shape.countable_goal
    assert "countable objective" in shape.named()


def test_the_raw_request_alone_would_have_her_say_the_opposite():
    """The defect this guards, stated as the thing that used to happen."""
    shape = recognise(
        acts=ACTS, knows_how_it_moves=_one_that_knows(), state=_board(), toward=""
    ).shape
    assert not shape.countable_goal


def test_a_named_finish_is_still_what_she_names():
    """Deriving one must not override being told one."""
    shape = recognise(
        acts=ACTS, knows_how_it_moves=_one_that_knows(), state=_board(), toward="256"
    ).shape
    assert shape.countable_goal


def test_every_caller_hands_it_what_she_is_playing_for():
    """Asked structurally, because the call site keeps moving.

    This read `pursue_on_screen`'s text and the 900 characters after the
    call, so it went red when the pursuit loop was split across modules —
    and it would have gone green again on any edit that kept the two names
    within 900 characters of each other while breaking what they mean. The
    claim is about the ARGUMENT: whoever calls it derives the objective when
    the request named none. One caller did not, and `recognise` was handed
    an empty string on exactly the runs where naming the shape is the point.
    """
    import ast

    from source_contract import module_family_sources

    import core.skills.screen_pursuit as sp

    calls = []
    for name, text in module_family_sources(sp):
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if getattr(called, "id", getattr(called, "attr", "")) != (
                "_say_what_kind_of_problem"
            ):
                continue
            toward = node.args[3] if len(node.args) > 3 else None
            for keyword in node.keywords:
                if keyword.arg == "toward":
                    toward = keyword.value
            calls.append((name, node.lineno, ast.unparse(toward) if toward else ""))

    assert calls, "nothing names the kind of problem any more"
    for name, line, toward in calls:
        assert "_what_there_is_to_aim_at" in toward, (
            f"{name}:{line} names the kind of problem from {toward!r} — the "
            "request rather than what she is playing for"
        )
