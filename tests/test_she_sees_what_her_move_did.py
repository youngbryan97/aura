"""She is shown what her last move did, and a page that only leads back ends the run.

LIVE 2026-10-02, on her own result page: four "more" links opened her four scores,
she was shown the opened page with nothing to say it had opened, said she still
had to expand it, and pressed "less". Each half of the pair was retired as leading
back somewhere already seen; with every control retired the page was handed back
whole, and the pair ran for twenty minutes.

Nothing here knows what a disclosure control or a result page is. It knows which
lines and controls came and went, and whether anything on the page still leads
somewhere new.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill
from core.skills.sovereign_browser_what_it_did import (
    every_way_on_led_back,
    names_of_the_moves,
    remember_what_was_seen,
    what_the_move_did,
    with_what_was_seen_here,
)

pytestmark = pytest.mark.unit

_URL = "https://example.test/result"
_SHUT = "Introversion-Extroversion is your orientation. [more]"
_OPEN = "Your score on I-E was -0.625, which indicates introverted. [less]"


def _page(text: str, *controls: tuple[str, str], url: str = _URL) -> dict[str, Any]:
    return {
        "url": url,
        "title": "Result",
        "text": f"Your type is INTJ.\n{text}",
        "elements": [
            {"role": "link", "name": name, "selector": selector}
            for selector, name in controls
        ],
    }


def test_lines_a_move_opened_are_named():
    said = what_the_move_did(
        _page(_SHUT, ("#IE_short", "more")),
        _page(_OPEN, ("#IE_long", "less")),
        done=["pressed more"],
        expected="the section expands",
    )
    assert "you pressed more" in said
    assert '"the section expands"' in said
    assert f"+ {_OPEN}" in said
    assert f"- {_SHUT}" in said
    # The line that stayed is not news.
    assert "Your type is INTJ." not in said


def test_a_move_that_changed_nothing_says_so():
    page = _page(_SHUT, ("#IE_short", "more"))
    said = what_the_move_did(page, dict(page), done=["pressed more"])
    assert "Nothing on the page changed." in said


def test_a_move_that_only_changed_a_control_says_which():
    before = _page(_SHUT, ("#q1", "agree"))
    after = _page(_SHUT, ("#q1", "agree"))
    after["elements"][0]["checked"] = True
    said = what_the_move_did(before, after, done=["pressed agree"])
    assert "agree (checked) (was agree)" in said
    assert "The page's text did not change." in said


def test_a_move_to_another_page_says_where_it_went():
    said = what_the_move_did(
        _page(_SHUT, ("#next", "Next")),
        _page(_SHUT, ("#next", "Next"), url="https://example.test/page2"),
        done=["pressed Next"],
    )
    assert "took you from https://example.test/result to https://example.test/page2" in said


def test_a_page_replaced_whole_is_counted_not_repeated():
    """No line survived, and the page below already says what is there."""
    before = {"url": _URL, "text": "\n".join(f"old line {n}" for n in range(30)), "elements": []}
    after = {"url": _URL, "text": "one new line", "elements": []}
    said = what_the_move_did(before, after, done=["pressed Go"])
    assert "old line 7" not in said
    assert "1 line(s) of the page's text appeared and 30 went away" in said


def test_no_move_no_account():
    page = _page(_SHUT)
    assert what_the_move_did({}, page, done=["pressed more"]) == ""
    assert what_the_move_did(page, page, done=[]) == ""


def test_moves_are_named_as_their_controls_were():
    elements = [
        {"name": "more", "selector": "#a"},
        {"name": "Email", "selector": "#e"},
    ]
    moves = [
        (SimpleNamespace(type="click", selector="#a", value=None), ""),
        (SimpleNamespace(type="type", selector="#e", value="x@y"), ""),
        (SimpleNamespace(type="scroll", selector=None, value="down"), ""),
    ]
    assert names_of_the_moves(moves, elements) == [
        "pressed more",
        'typed "x@y" into Email',
        "scrolled down",
    ]


def test_a_page_is_exhausted_only_when_every_control_led_back():
    page = _page(_OPEN, ("#a", "less"), ("#b", "less"))
    assert every_way_on_led_back(page, {"#a", "#b"})
    assert not every_way_on_led_back(page, {"#a"})
    assert not every_way_on_led_back(page, set())
    assert not every_way_on_led_back({"elements": []}, {"#a"})


def test_the_verdict_reads_what_her_moves_opened_and_closed_again():
    seen: dict[str, list[str]] = {}
    remember_what_was_seen(seen, _page(_SHUT))
    remember_what_was_seen(seen, _page(_OPEN))
    shown = with_what_was_seen_here(_page(_SHUT), seen)
    assert _OPEN in shown["text"]
    assert shown["text"].count("Your type is INTJ.") == 1
    # Nothing hidden, nothing added.
    assert with_what_was_seen_here(_page(_OPEN), {_URL: []}) == _page(_OPEN)


class _Toggle:
    """A page with two states and two controls that undo each other."""

    def __init__(self) -> None:
        self.open = False

    async def observe(self, **_kw: Any) -> dict[str, Any]:
        if self.open:
            return _page(_OPEN, ("#long", "less"))
        return _page(_SHUT, ("#short", "more"))

    async def close(self) -> None:
        return None


def _skill(monkeypatch, browser: _Toggle, noticed: list[str], judged: list[str]):
    skill = SovereignBrowserSkill()

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a result", "to_progress": "read it"}

    async def _decided(*_a: Any, **kw: Any) -> dict[str, Any]:
        noticed.append(str(kw.get("noticed") or ""))
        return {
            "actions": [{"index": 0, "type": "click"}],
            "why": "I need to expand it",
            "expect": "the section expands",
            "done": False,
        }

    async def _interacted(*_a: Any, **_k: Any) -> dict[str, Any]:
        browser.open = not browser.open
        return {"ok": True, "results": [{"ok": True}]}

    async def _concluded(goal: str, said_before: str, observation: Any, mind: str) -> str:
        judged.append(str(observation.get("text") or ""))
        return "It says INTJ, as I said."

    async def _held(*_a: Any, **_k: Any) -> None:
        return None

    for name, value in {
        "_understand_page": _understood,
        "_decide_next_actions": _decided,
        "_handle_interact": _interacted,
        "_asks_about_the_one_answering": lambda *_a: False,
        "_narrate": lambda *_a, **_k: None,
        "_narrate_decision": lambda *_a, **_k: "",
        "_say_out_loud": lambda *_a, **_k: None,
        "_remember_the_place": lambda *_a, **_k: None,
        "_recall_about": lambda *_a, **_k: "",
        "_retain_stated_positions": lambda *_a, **_k: None,
        "_hold_the_outcome_against_what_she_said": _concluded,
        "_hold_for_reading": _held,
    }.items():
        monkeypatch.setattr(skill, name, value)
    return skill


def test_a_toggle_pair_ends_the_run_and_she_saw_each_press(monkeypatch):
    browser = _Toggle()
    noticed: list[str] = []
    judged: list[str] = []
    skill = _skill(monkeypatch, browser, noticed, judged)
    result = asyncio.run(
        skill._handle_pursue(browser, None, "take the test", 40, said_before="INTJ")
    )
    # Two states, two controls: it ends on the stall bound, long before forty.
    assert len(noticed) < 8, f"the pair ran {len(noticed)} rounds"
    assert noticed[0] == "", "nothing was done before the first round"
    assert f"+ {_OPEN}" in noticed[1], "she was not shown what pressing it opened"
    assert any(step.get("error") == "no_progress" for step in result["steps"])
    # And the verdict read the opened lines, whichever state the run ended in.
    assert judged and _OPEN in judged[0]


def test_a_control_already_pressed_from_this_state_is_spent_here_only():
    from core.skills.sovereign_browser_what_it_did import the_ones_tried_here

    tried = {("shut", "#short"), ("open", "#long")}
    assert the_ones_tried_here(tried, "shut") == {"#short"}
    assert the_ones_tried_here(tried, "elsewhere") == set()


def test_a_toggle_pair_costs_her_two_decisions_not_four(monkeypatch):
    """Open, close: both ways explored. The closed page offers nothing untried."""
    browser = _Toggle()
    noticed: list[str] = []
    judged: list[str] = []
    skill = _skill(monkeypatch, browser, noticed, judged)
    asyncio.run(skill._handle_pursue(browser, None, "take the test", 40, said_before="INTJ"))
    assert len(noticed) == 2, f"she was asked {len(noticed)} times on a page of two states"


class _Scrolled:
    """A page scrolled past its controls; scrolling back up shows them again."""

    def __init__(self) -> None:
        self.scrolled: list[tuple[str, int]] = []

    async def scroll(self, direction: str = "down", amount: int = 500, *, principal: str = "") -> bool:
        self.scrolled.append((direction, amount))
        return True

    async def observe(self, **_kw: Any) -> dict[str, Any]:
        return _page(_SHUT, ("#short", "more"))


def test_a_page_scrolled_past_its_controls_is_scrolled_back_not_reloaded():
    from core.skills.sovereign_browser_what_it_did import _go_back_to_the_last_good_page

    browser = _Scrolled()
    reloaded: list[str] = []

    class _Skill:
        async def _safe_browse(self, _browser, url):
            reloaded.append(url)
            return True

    seen = asyncio.run(
        _go_back_to_the_last_good_page(
            browser=browser,
            last_good_url=_URL,
            observation={"url": _URL, "text": "Your type is INTJ.", "scroll_y": 1200, "elements": []},
            self=_Skill(),
        )
    )
    assert browser.scrolled == [("up", 1200)]
    assert reloaded == [], "a reload would close what she opened"
    assert seen["elements"]

    # Somewhere else entirely is still a page to go back to.
    asyncio.run(
        _go_back_to_the_last_good_page(
            browser=_Scrolled(),
            last_good_url=_URL,
            observation={"url": "https://elsewhere.test", "scroll_y": 0, "elements": []},
            self=_Skill(),
        )
    )
    assert reloaded == [_URL]


def test_a_scroll_is_said_as_a_scroll(monkeypatch):
    said: list[str] = []
    skill = SovereignBrowserSkill()
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, parts=None: said.append(line))
    decision = {
        "why": "I need to read the rest.",
        "actions": [{"index": 0, "type": "scroll", "value": "down"}],
    }
    skill._narrate_decision(decision, _page(_SHUT, ("#short", "more")), "take the test")
    assert said and "scroll down" in said[0] and "more" not in said[0]
