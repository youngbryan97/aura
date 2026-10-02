"""A forecast about an instrument is made from what the instrument says it is.

LIVE 2026-09-28: asked to say what the test would give her before she started,
she said nothing about it at all. The forecast was written before any page
opened — before she knew what the thing measured — so there was nothing to
forecast from, and the run began with no claim for the result to be held
against.

General to any instrument that will report on her, and to one that names no
scores: she infers what it may say about her from what it says it measures,
which is what a person does with a test they have not taken.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Instrument:
    """A page that says what it measures and offers a scale."""

    async def observe(self, **_kw: Any) -> dict[str, Any]:
        return {
            "url": "https://example.test/instrument",
            "title": "An instrument",
            "text": "This measures four dichotomies and reports a four-letter type.",
            "elements": [
                {"role": "radio", "name": name, "group": group, "selector": f"#{group}{name}"}
                for group in ("q1", "q2")
                for name in ("a", "b", "c")
            ],
        }

    async def close(self) -> None:
        return None


def _skill(monkeypatch, said: list[str], forecast: str = "I expect it to call me reflective."):
    skill = SovereignBrowserSkill()
    asked: list[str] = []

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "an instrument", "to_progress": "answer", "done_when": "submitted"}

    async def _expects(goal: str, observation: Any, mind: str, measured: Any = None) -> str:
        asked.append(str(observation.get("text") or ""))
        return forecast

    async def _answered(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "resolved_actions": [{"selector": "#q1a", "name": "a", "said": "Q1 -> a."}],
            "answered": ["Q1 -> a."],
            "why": "one",
            "done": False,
        }

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"done": True, "actions": [], "why": "finished"}

    async def _interacted(_b, _u, actions, **_k: Any) -> dict[str, Any]:
        return {"ok": True, "action_report": [{"ok": True} for _ in actions]}

    async def _held(_said: str) -> None:
        return None

    held_against: list[str] = []

    async def _concluded(goal: str, said_before: str, observation: Any, mind: str) -> str:
        held_against.append(said_before)
        return "It matches."

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_what_she_expects_it_to_say", _expects)
    monkeypatch.setattr(skill, "_answer_each_question", _answered)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_handle_interact", _interacted)
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_narrate_decision", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_say_out_loud", lambda line, *_a, **_k: said.append(str(line)))
    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_hold_the_outcome_against_what_she_said", _concluded)
    return skill, asked, held_against


def test_she_says_what_she_expects_before_the_first_answer(monkeypatch):
    said: list[str] = []
    skill, _asked, _held = _skill(monkeypatch, said)
    asyncio.run(skill._handle_pursue(_Instrument(), None, "take it", 3))
    assert said, "nothing was said at all"
    assert said[0] == "I expect it to call me reflective."
    assert said.index("I expect it to call me reflective.") < said.index("Q1 -> a."), (
        "the forecast must come before the first answer, or it is a report"
    )


def test_the_forecast_is_made_from_what_the_page_says_it_measures(monkeypatch):
    said: list[str] = []
    skill, asked, _held = _skill(monkeypatch, said)
    asyncio.run(skill._handle_pursue(_Instrument(), None, "take it", 3))
    assert asked and "four dichotomies" in asked[0]


def test_the_forecast_becomes_what_the_result_is_held_against(monkeypatch):
    said: list[str] = []
    skill, _asked, held = _skill(monkeypatch, said)
    asyncio.run(skill._handle_pursue(_Instrument(), None, "take it", 3))
    assert held and held[0] == "I expect it to call me reflective."


def test_a_forecast_read_from_the_page_replaces_one_made_before_arriving(monkeypatch):
    """A reply given before opening anything is not a forecast about this.

    Gating the page-read forecast on an empty `said_before` meant the reply she
    gave the person before opening anything counted as one, so the informed
    forecast never ran and there was nothing for the result to be held against:
    LIVE 2026-09-29, `page_forecast` appears zero times in a run that answered
    thirty-two items and reached its results.
    """
    said: list[str] = []
    skill, asked, held = _skill(monkeypatch, said)
    asyncio.run(
        skill._handle_pursue(
            _Instrument(), None, "take it", 3, said_before="I said ENTP earlier."
        )
    )
    assert asked, "she never read the page before forecasting"
    assert held and held[0] == "I expect it to call me reflective."


def test_what_she_said_before_arriving_stands_when_the_page_says_nothing(monkeypatch):
    said: list[str] = []
    skill, asked, held = _skill(monkeypatch, said, forecast="")
    asyncio.run(
        skill._handle_pursue(
            _Instrument(), None, "take it", 3, said_before="I said ENTP earlier."
        )
    )
    assert held and held[0] == "I said ENTP earlier."


def test_a_page_that_asks_nothing_about_her_gets_no_forecast(monkeypatch):
    class _Plain:
        async def observe(self, **_kw: Any) -> dict[str, Any]:
            return {
                "url": "https://example.test/plain",
                "title": "plain",
                "text": "nothing here",
                "elements": [{"role": "button", "name": "Next", "selector": "#next"}],
            }

        async def close(self) -> None:
            return None

    said: list[str] = []
    skill, asked, _held = _skill(monkeypatch, said)
    asyncio.run(skill._handle_pursue(_Plain(), None, "press next", 2))
    assert not asked
