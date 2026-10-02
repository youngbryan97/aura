"""Every decision in a pursuit is narrated, including the ones that do nothing.

Narration fired once per round, after an action report, so a round that produced
no action said nothing: a decision that could not be read, a claim of "done"
before anything was done, an answer naming a control that is not on the page.
Those are exactly the rounds a person watching needs to hear, and from outside
they looked like a browser sitting still.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Page:
    async def observe(self, **_kw: Any) -> dict[str, Any]:
        return {
            "url": "https://example.test/survey",
            "title": "A survey",
            "text": "Question one.",
            "elements": [
                {"role": "button", "name": "Next", "selector": "#next"},
            ],
        }

    async def close(self) -> None:
        return None


def _skill(monkeypatch, decision: dict[str, Any], said: list[str]):
    skill = SovereignBrowserSkill()

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a survey", "to_progress": "answer", "done_when": "submitted"}

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        return dict(decision)

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_asks_about_the_one_answering", lambda *_a: False)
    monkeypatch.setattr(skill, "_narrate", lambda step: said.append(str(step.get("why") or "")))
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, *_a, **_k: said.append(str(line)))
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    return skill


def test_a_decision_that_could_not_be_read_is_still_said(monkeypatch):
    said: list[str] = []
    skill = _skill(monkeypatch, {"error": "empty_decision", "raw": ""}, said)
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 8))
    assert said, "a round that produced no action said nothing at all"
    assert any("empty_decision" in line for line in said)


def test_what_she_actually_said_reaches_the_narration(monkeypatch):
    said: list[str] = []
    skill = _skill(
        monkeypatch,
        {"error": "unparsable_decision", "raw": "The user wants me to take the test."},
        said,
    )
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 8))
    assert any("The user wants me to take the test." in line for line in said)


def test_a_claim_of_done_before_acting_is_said(monkeypatch):
    said: list[str] = []
    skill = _skill(
        monkeypatch, {"done": True, "actions": [], "why": "nothing left here"}, said
    )
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 8))
    assert any("finished" in line for line in said)
    assert any("nothing left here" in line for line in said)


def test_an_acting_decision_names_what_it_will_press(monkeypatch):
    said: list[str] = []
    skill = _skill(
        monkeypatch,
        {"actions": [{"index": 0, "type": "click"}], "why": "this moves to page two.", "done": False},
        said,
    )
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 8))
    assert any("Next" in line for line in said), (
        "a decision must say what it is about to do, not only why"
    )
    assert any("this moves to page two." in line for line in said)


def test_the_narration_fires_before_the_round_is_resolved():
    """A narration after the branch is a narration the failing branch skips."""
    import inspect

    loop = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    narrated = loop.index("self._narrate_decision(decision, observation, goal)")
    branched = loop.index('if decision.get("error"):')
    assert narrated < branched, (
        "every decision must be said before the loop decides what to do with it"
    )
