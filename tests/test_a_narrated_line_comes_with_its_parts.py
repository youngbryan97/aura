"""A narrated line arrives with its parts, so the chat lays it out.

Bryan, 1 Oct, of the psych-test commentary: "a little ugly when they come in".
Every line was one string — the question, her place on it and her reasons run
together, a decision ending "I am going to Start." — under a badge reading
AUTONOMIC, for a commentary he had asked for. The line still goes everywhere
whole; its parts go beside it for the surfaces that can lay them out.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core.agency.narrator import Narrator
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def test_the_parts_ride_with_the_line(monkeypatch):
    published: list[dict[str, Any]] = []

    class Orchestrator:
        def _publish_telemetry(self, event: dict[str, Any]) -> None:
            published.append(event)

    monkeypatch.setattr("core.agency.narrator.resolve_orchestrator", lambda: Orchestrator())
    Narrator.say_everywhere(
        "q — 4 of 5. Because.", parts={"asks": "q", "chose": "4 of 5", "said": "Because.", "doing": ""}
    )
    meta = published[-1]["metadata"]
    assert published[-1]["message"] == "q — 4 of 5. Because."
    assert meta["narration"] is True
    assert meta["narrated"] == {"asks": "q", "chose": "4 of 5", "said": "Because."}


def test_a_line_without_parts_is_published_as_before(monkeypatch):
    published: list[dict[str, Any]] = []

    class Orchestrator:
        def _publish_telemetry(self, event: dict[str, Any]) -> None:
            published.append(event)

    monkeypatch.setattr("core.agency.narrator.resolve_orchestrator", lambda: Orchestrator())
    Narrator.say_everywhere("Board: Up")
    assert "narrated" not in published[-1]["metadata"]


def test_an_answer_is_said_as_its_question_its_place_and_its_reason(monkeypatch):
    spoken: list[tuple[str, Any]] = []
    skill = SovereignBrowserSkill()
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, parts=None: spoken.append((line, parts)))

    async def _hold(_line: str) -> None:
        return None

    async def _interact(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"ok": True, "action_report": [{"ok": True}]}

    async def _think() -> tuple[str, dict[str, str]]:
        return "q — 4 of 5. Because.", {"asks": "q", "chose": "4 of 5", "said": "Because."}

    monkeypatch.setattr(skill, "_hold_for_reading", _hold)
    monkeypatch.setattr(skill, "_handle_interact", _interact)
    from core.skills.sovereign_browser import BrowserAction

    asyncio.run(skill._make_each_move(None, [(BrowserAction(type="click", selector="#a"), _think)]))
    assert spoken == [("q — 4 of 5. Because.", {"asks": "q", "chose": "4 of 5", "said": "Because."})]


def test_a_decision_is_her_reason_with_the_act_beneath_it(monkeypatch):
    spoken: list[tuple[str, Any]] = []
    skill = SovereignBrowserSkill()
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, parts=None: spoken.append((line, parts)))
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    observation = {
        "title": "t",
        "elements": [{"role": "button", "name": "Start", "selector": "#s"}],
    }
    skill._narrate_decision(
        {"actions": [{"index": 0, "type": "click"}], "why": "This is the test."}, observation, "g"
    )
    line, parts = spoken[-1]
    assert line == "This is the test. I am going to Start."
    assert parts == {"said": "This is the test.", "doing": "Start"}


def test_the_chat_lays_a_narrated_line_out_and_does_not_call_it_autonomic():
    source = (ROOT / "interface/static/aura.js").read_text(encoding="utf-8")
    assert "function narrationCardHtml(" in source
    badge = source.split("function messageBadgeHtml", 1)[1].split("\n}\n", 1)[0]
    assert badge.index("metadata.narration") < badge.index("metadata.autonomic")


def test_the_card_keeps_the_whole_of_what_she_said(monkeypatch):
    """The bubble is bounded by reading time; the card stays, so it is not.

    LIVE 2026-10-02 the forecast's card stopped at 599 characters, before the
    type she expected.
    """
    from core.agency.reading_pace import as_much_as_can_be_read

    published: list[dict[str, Any]] = []

    class Orchestrator:
        def _publish_telemetry(self, event: dict[str, Any]) -> None:
            published.append(event)

    monkeypatch.setattr("core.agency.narrator.resolve_orchestrator", lambda: Orchestrator())
    forecast = " ".join(f"This is reason number {n} for what I expect." for n in range(60))
    forecast += " So I expect INTJ."
    SovereignBrowserSkill._say_out_loud(forecast, {"label": "What I expect", "said": forecast})
    event = published[-1]
    assert event["metadata"]["narrated"]["said"].endswith("So I expect INTJ.")
    assert event["message"] == as_much_as_can_be_read(forecast)
    assert len(event["message"]) < len(forecast)
