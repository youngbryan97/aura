"""A new kind of page is taken in once, and the move that follows is not worked out again.

LIVE 2026-10-03 03:31-03:36, the psych test from her chat: her reading of the
front page said "Click the 'Open Jungian Type Scales' link", and the decision
after it spent its whole 806-token private channel and 153 seconds on that one
click. On the next page, the test's introduction, no new reading was made at
all, because only a surprise could prompt one, so she decided what to do there
with a plan written for the front page, through the same full channel.

A person takes in a new place, then acts on what they took in. Working things
out belongs to that reading, and to the moments a page does not do what was
expected; the next page of the same form is not a new place.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill


def _page(url: str, elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {"url": url, "title": "t", "text": "words", "elements": elements}


_LINKS = [{"role": "link", "name": f"Test {n}", "selector": f"#l{n}"} for n in range(6)]
_START = [{"role": "button", "name": "Start", "selector": "#start"}]


class _Browser:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.observed = 0

    async def observe(self, **_k: Any) -> dict[str, Any]:
        page = self._pages[min(self.observed, len(self._pages) - 1)]
        self.observed += 1
        return page


def _skill(understood: list[Any], settled: list[bool], expect: str = "") -> SovereignBrowserSkill:
    """A decision that expects nothing particular, so no round is a surprise."""
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def _decide(goal, observation, history, understanding=None, **kw: Any):
        settled.append(bool(kw.get("settled")))
        if len(settled) > 2:
            return {"done": True, "actions": []}
        return {"actions": [{"index": 0, "type": "click"}], "why": "on", "expect": expect}

    async def _understand(goal, observation, prior, mind, recalled=""):
        understood.append(observation["url"])
        return {"here": observation["url"], "to_progress": f"look {len(understood)}"}

    async def _mind() -> str:
        return ""

    async def _interact(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"ok": True}

    skill._decide_next_actions = _decide
    skill._understand_page = _understand
    skill._assembled_mind = _mind
    skill._handle_interact = _interact
    return skill


@pytest.mark.asyncio
async def test_a_page_of_another_kind_is_taken_in_on_arrival():
    understood: list[Any] = []
    settled: list[bool] = []
    pages = [
        _page("https://a.example/", _LINKS),
        _page("https://a.example/", _LINKS),
        _page("https://a.example/test/", _START),
        _page("https://a.example/test/", _START),
        _page("https://a.example/test/1", _START),
    ]
    await _skill(understood, settled)._handle_pursue(_Browser(pages), None, "take the test", 3)
    assert understood[:2] == ["https://a.example/", "https://a.example/test/"]


@pytest.mark.asyncio
async def test_the_move_after_a_fresh_reading_is_settled():
    understood: list[Any] = []
    settled: list[bool] = []
    pages = [_page("https://a.example/", _LINKS), _page("https://a.example/", _LINKS)]
    await _skill(understood, settled)._handle_pursue(_Browser(pages), None, "take the test", 2)
    assert settled[0] is True
    assert settled[1] is False, "a second decision on the same page is worked out"


@pytest.mark.asyncio
async def test_a_page_that_did_not_do_what_she_expected_is_read_again():
    understood: list[Any] = []
    settled: list[bool] = []
    pages = [_page("https://a.example/", _LINKS), _page("https://a.example/", _LINKS)]
    await _skill(understood, settled, expect="a new page")._handle_pursue(
        _Browser(pages), None, "take the test", 2
    )
    assert len(understood) == 2 and settled[:2] == [True, True]


@pytest.mark.unit
def test_a_settled_decision_does_not_open_her_private_channel(monkeypatch):
    skill = SovereignBrowserSkill()
    asked: list[dict[str, Any]] = []

    class _Router:
        async def think(self, *_a: Any, **_k: Any) -> str:
            return ""

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda *names, default=None: _Router() if "llm_router" in names else default,
    )

    async def _no_mind() -> str:
        return ""

    async def _asked(prompt: str, mind: str = "", **kw: Any) -> tuple[str, str]:
        asked.append(kw)
        return '{"actions": [{"index": 0, "type": "click"}], "why": "x", "done": false}', "Cortex"

    monkeypatch.setattr(skill, "_assembled_mind", _no_mind)
    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    page = {"elements": [{"role": "button", "name": "Start", "selector": "#s"}]}
    for settled in (True, False):
        asyncio.run(skill._decide_next_actions("start", page, [], settled=settled))
    assert [kw.get("worked_out_here") for kw in asked] == [False, True]
