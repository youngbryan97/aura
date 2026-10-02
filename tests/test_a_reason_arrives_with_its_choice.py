"""Choice, reason, choice, reason — not one bubble listing what already happened.

A screen of questions was answered in one batch and read out afterwards: every
click landed, then a single burst of lines described work that was already over.
A reason that arrives after its choice is a report, not a commentary, and a
person watching cannot follow a decision they are told about once it is done.
Asked for directly on 2026-09-28: "each reason comes as she's making the choice
... not one bubble where she says everything she did".
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
            "elements": [{"role": "button", "name": "Next", "selector": "#next"}],
        }

    async def close(self) -> None:
        return None


def _transcript(monkeypatch) -> tuple[SovereignBrowserSkill, list[str]]:
    """Everything a watcher would see or the page would receive, in order."""
    seen: list[str] = []
    skill = SovereignBrowserSkill()

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a survey", "to_progress": "answer", "done_when": "submitted"}

    rounds = {"n": 0}

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        rounds["n"] += 1
        if rounds["n"] > 1:
            return {"done": True, "actions": [], "why": "finished"}
        return {
            "resolved_actions": [
                {"selector": "#q1c", "name": "agree", "said": "Q1 -> agree. Because one."},
                {"selector": "#q2c", "name": "disagree", "said": "Q2 -> disagree. Because two."},
            ],
            "answered": ["Q1 -> agree. Because one.", "Q2 -> disagree. Because two."],
            "why": "two answers",
            "done": False,
        }

    async def _interacted(_browser, _url, actions, **_k: Any) -> dict[str, Any]:
        for action in actions:
            seen.append(f"click {action.selector}")
        return {"ok": True, "action_report": [{"ok": True} for _ in actions]}

    async def _held(_said: str) -> None:
        return None

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_handle_interact", _interacted)
    monkeypatch.setattr(skill, "_asks_about_the_one_answering", lambda *_a: False)
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_narrate_decision", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, *_a, **_k: seen.append(f"say {line}"))
    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    monkeypatch.setattr(
        skill, "_hold_the_outcome_against_what_she_said",
        lambda *_a, **_k: asyncio.sleep(0, result=""),
    )
    return skill, seen


def test_each_reason_is_said_before_the_choice_it_explains(monkeypatch):
    skill, seen = _transcript(monkeypatch)
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 4))
    order = [line for line in seen if line.startswith(("say Q", "click #q"))]
    assert order[:4] == [
        "say Q1 -> agree. Because one.",
        "click #q1c",
        "say Q2 -> disagree. Because two.",
        "click #q2c",
    ], f"got {order}"


def test_no_reason_is_left_until_after_every_click(monkeypatch):
    skill, seen = _transcript(monkeypatch)
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 4))
    last_click = max(i for i, line in enumerate(seen) if line.startswith("click #q"))
    trailing = [line for line in seen[last_click + 1 :] if line.startswith("say Q")]
    assert not trailing, f"reasons said after the work was done: {trailing}"


def test_each_choice_is_its_own_interaction(monkeypatch):
    """A reason cannot sit in front of a choice that was made in a batch."""
    skill, seen = _transcript(monkeypatch)
    batches: list[int] = []

    async def _interacted(_browser, _url, actions, **_k: Any) -> dict[str, Any]:
        batches.append(len(actions))
        return {"ok": True, "action_report": [{"ok": True} for _ in actions]}

    monkeypatch.setattr(skill, "_handle_interact", _interacted)
    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 4))
    assert batches and all(size == 1 for size in batches), batches


def test_a_round_is_still_judged_as_a_whole(monkeypatch):
    """Nothing downstream needs to know a round was thirty moves."""
    skill, _seen = _transcript(monkeypatch)
    failures = {"n": 0}

    async def _interacted(_browser, _url, actions, **_k: Any) -> dict[str, Any]:
        failures["n"] += 1
        if failures["n"] == 1:
            return {"ok": False, "error": "selector_gone"}
        return {"ok": True, "action_report": [{"ok": True}]}

    monkeypatch.setattr(skill, "_handle_interact", _interacted)
    result = asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 4))
    assert result["landed_total"] >= 1, "one selector going stale lost the whole round"
