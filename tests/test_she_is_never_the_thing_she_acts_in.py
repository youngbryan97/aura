"""A request to pick a short is a request to watch one, and her own window is never what she plays.

LIVE 8 November 2026, rehearsal 10: "Now Cyanide & Happiness. Pick whichever short you want, as long as it is under ten
minutes." named no page and no act her reader knew ("short" was not a thing watched), "as long as" was read as a cue
to keep at something, and the request became a screen pursuit with nothing named to act in. It took the window in
front, hers, and pressed arrow keys into it 200 times over 14 minutes ("brought 'Aura' back to the front").
"""
from __future__ import annotations

import asyncio

import pytest

from core.intent.asks_to_go_online import acts_asked
from core.runtime.watched_goal import _continuation, read_watched_goal
from core.skills import screen_pursuit

ASKED = "Now Cyanide & Happiness. Pick whichever short you want, as long as it is under ten minutes."


def test_a_short_chosen_is_a_thing_to_watch_and_a_short_answer_is_not():
    assert acts_asked(ASKED) == {"watch"}
    assert acts_asked("Watch a short by them.") == {"watch"}
    for said in ("Give me a short answer.", "In short, no.", "Keep it short and sweet."):
        assert "watch" not in acts_asked(said), said


@pytest.mark.parametrize(("said", "cue"), [
    ("Pick whichever short you want, as long as it is under ten minutes", ""),
    ("watch it as long as it lasts", "as long as it lasts"),
    ("run it for as long as possible", "as long as possible"),
    ("keep playing as long as you can", "keep playing"),
])
def test_as_long_as_keeps_at_a_thing_only_where_it_says_for_how_long(said, cue):
    assert _continuation(said) == cue


def test_the_request_is_not_a_goal_to_keep_at():
    assert read_watched_goal(ASKED) is None


def _her_window_is_up(monkeypatch, up=True):
    from core.perception.ambient_presence import get_ambient_presence

    monkeypatch.setattr(type(get_ambient_presence()), "drawing_surface_attached", lambda _self: up)


def test_a_pursuit_with_nothing_named_and_only_her_window_in_front_does_not_start(monkeypatch):
    async def hers():
        return "Aura"

    monkeypatch.setattr(screen_pursuit, "_frontmost", hers)
    _her_window_is_up(monkeypatch)
    result = asyncio.run(screen_pursuit.pursue_on_screen(goal="pick a short", success_when=""))
    assert result["ok"] is False and result["outcome"] == "needs_person" and result["moves"] == []
    assert "my own window" in result["needs_person"]


def test_another_application_in_front_is_somewhere_to_act(monkeypatch):
    async def a_game():
        return "2048 Game"

    monkeypatch.setattr(screen_pursuit, "_frontmost", a_game)
    _her_window_is_up(monkeypatch)
    assert asyncio.run(screen_pursuit._only_her_own_window_in_front()) is False


def test_with_no_window_of_hers_up_what_is_in_front_is_not_hers(monkeypatch):
    async def hers():
        return "Aura"

    monkeypatch.setattr(screen_pursuit, "_frontmost", hers)
    _her_window_is_up(monkeypatch, up=False)
    assert asyncio.run(screen_pursuit._only_her_own_window_in_front()) is False
