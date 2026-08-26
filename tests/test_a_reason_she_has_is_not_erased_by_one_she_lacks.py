"""A move keeps the reason she has when the choice carries none of its own.

Measured live on 2026-08-26: a whole game narrated as "Going up", "Going
down", with every reason discarded one line before it was said, because the
rationale of the chosen option was assigned over the reason unconditionally
and the ordinary option carries no rationale.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.skills import screen_pursuit


class _Chosen:
    def __init__(self, rationale: str = "") -> None:
        self.rationale = rationale
        self.chosen = SimpleNamespace(expectation=SimpleNamespace(describes="the board moves"))
        self.spoke = True


@pytest.fixture
def spoken(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(screen_pursuit, "_tell", lines.append)
    monkeypatch.setattr(screen_pursuit, "_publish_decision", lambda *a, **k: None)
    return lines


def test_a_follow_on_still_says_it_is_the_same_plan(spoken):
    screen_pursuit._say_intent("left", _Chosen(), out_loud=True, following_on=True)
    assert spoken == ["Going left — same plan"]


def test_a_reason_of_its_own_is_preferred(spoken):
    screen_pursuit._say_intent("up", _Chosen("the corner stays put"), out_loud=True, following_on=True)
    assert spoken == ["Going up — the corner stays put"]


def test_a_first_move_without_a_reason_says_only_the_move(spoken):
    screen_pursuit._say_intent("down", _Chosen(), out_loud=True, following_on=False)
    assert spoken == ["Going down"]


def test_no_choice_at_all_still_follows_on(spoken):
    screen_pursuit._say_intent("right", None, out_loud=True, following_on=True)
    assert spoken == ["Going right — same plan"]
