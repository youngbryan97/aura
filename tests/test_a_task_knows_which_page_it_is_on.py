"""She could read a page, act on it, and not know which page it was.

LIVE, 2026-08-18. A dismissal click landed on a tab label instead of a close
button; the browser went to x.com, and later to YouTube. The loop carried on
reading pixels and pressing arrow keys the whole time. Every layer was working
correctly and none of them knew where they were.

An application is not a context. A browser holds the task's page and a dozen
others, so a run bound only to "Google Chrome" sends its keys to whichever tab
is in front — arrow keys meant for a game landing on a video, a form, or
someone's mail, each keystroke legitimately delivered to the wrong world.

Appearance cannot answer "is this still the thing I was working on": two pages
can look alike, and one page can change. Only identity can.

The chain is anchor, verify, restore, refuse:

  * a run records the page it starts on, so a caller never has to remember;
  * every cycle re-reads the real URL and title and compares;
  * drift is repaired by returning to that TAB, not by reloading, which would
    throw away whatever the task had already done;
  * and when it cannot get back it stops, so the policy is never asked and no
    action can reach a foreign page.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills import screen_pursuit as sp


@pytest.fixture
def browser(monkeypatch):
    """A browser whose page can be changed out from under the run."""
    state = {"page": {"url": "https://play2048.co/", "title": "2048"},
             "focused": [], "pressed": []}

    async def identity():
        return dict(state["page"], error="")

    async def focus(match):
        state["focused"].append(match)
        if "play2048" in match.lower():
            state["page"] = {"url": "https://play2048.co/", "title": "2048"}
            return True
        return False

    async def read(app_name=""):
        return {"ok": True, "text": "board", "layout": [], "bounds": []}

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True

    async def frontmost(_app):
        return True

    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "_restore_tab", focus, raising=False)
    return state


async def _always_left(_observation):
    return {"key": "left", "because": "test"}


def test_a_run_anchors_to_the_page_it_started_on(browser):
    """The caller never has to remember; drift is always detectable."""
    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="NEVER", policy=_always_left,
            max_cycles=2, narrate=False, target_app="Google Chrome",
        )
    )

    assert "play2048" in result["anchored_to"]


def test_a_declared_page_wins_over_the_starting_one(browser):
    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="NEVER", policy=_always_left,
            max_cycles=2, narrate=False, target_app="Google Chrome",
            expect_page="play2048",
        )
    )

    assert result["anchored_to"] == "play2048"


def test_navigating_away_stops_the_run_instead_of_acting_there(browser, monkeypatch):
    """The YouTube case: no 2048 keystroke may reach a video page."""
    async def unrecoverable(_match):
        return False

    monkeypatch.setattr(sp, "_ensure_page", unrecoverable)

    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="NEVER", policy=_always_left,
            max_cycles=6, narrate=False, target_app="Google Chrome",
            expect_page="play2048",
        )
    )

    assert result["outcome"] == "navigated_away"
    assert browser["pressed"] == [], "it must not act on a page it did not anchor to"


def test_the_policy_is_never_asked_on_a_foreign_page(browser, monkeypatch):
    """Deciding from the wrong reading is as bad as acting on it."""
    asked: list[str] = []

    async def policy(observation):
        asked.append(observation.get("text", ""))
        return {"key": "left"}

    async def unrecoverable(_match):
        return False

    monkeypatch.setattr(sp, "_ensure_page", unrecoverable)

    asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="NEVER", policy=policy, max_cycles=5,
            narrate=False, target_app="Google Chrome", expect_page="play2048",
        )
    )

    assert asked == []


def test_a_run_without_a_target_app_is_left_alone(browser):
    """Not every pursuit drives a browser; watching one anchors nothing."""
    result = asyncio.run(
        sp.pursue_on_screen(
            goal="g", success_when="NEVER", policy=_always_left,
            max_cycles=2, narrate=False,
        )
    )

    assert result["anchored_to"] == ""
    assert browser["pressed"], "an unanchored run still acts"
