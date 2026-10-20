"""A run over a page that stopped short says so in words, and a round she looked past did not stop it.

LIVE 2026-10-20 13:27, asked to find an Allen Pan video: her reply began "Worked through
https://www.youtube.com/results?search_query=Allen+Pan over 3 round(s), then stopped:
browser_interaction_incomplete. Here is what I wrote:", and what followed was her choices on the page, which she had
not written.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills.how_a_pursuit_ended import how_it_ended

RESULTS = "https://www.youtube.com/results?search_query=Allen+Pan"
ASKED = ("Let's watch something together. Go to YouTube and find a random video from Allen Pan, under 15 minutes. Before "
         "it starts, tell me what you think it'll be about, think out loud while we watch, and tell me what you thought "
         "after.")


def test_the_reason_a_run_stopped_is_said_as_what_happened():
    said = how_it_ended(RESULTS, 3, "browser_interaction_incomplete")
    assert said == f"I got as far as {RESULTS} and stopped: what I pressed did nothing."
    assert how_it_ended(RESULTS, 3, "no_progress:browser_interaction_incomplete").endswith("the page stopped changing.")
    # A reason that is already a sentence is said as it is.
    assert how_it_ended(RESULTS, 1, "the browser task ended before it could act").endswith("before it could act.")


def _run(monkeypatch, report: dict) -> dict:
    import core.container as container
    from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill

    class _Engine:
        async def execute(self, skill, params, context=None):
            return {"final_url": RESULTS, **report}

    monkeypatch.setattr(
        container.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
    )
    # Whether the request wants a page used is her reader's call, which a test does not load; it said yes live.
    monkeypatch.setattr("core.conversation.page_interaction.page_interaction_target", lambda _text: "https://www.youtube.com")
    params = DesktopTaskParams(objective=ASKED)
    return asyncio.run(DesktopTaskSkill()._delegate_page_objective(params, {}))


def test_a_run_that_stopped_on_a_click_that_found_nothing_says_so(monkeypatch):
    result = _run(monkeypatch, {"ok": True, "completed": False, "steps": [
        {"asked": "Search", "chose": ["Search"], "why": "his channel first", "ok": True, "landed": 1},
        {"asked": "", "chose": ["I bought an Exoskeleton"], "why": "under 15 minutes",
         "error": "browser_interaction_incomplete"},
    ]})
    assert result["summary"] == f"I got as far as {RESULTS} and stopped: what I pressed did nothing."
    assert "browser_interaction_incomplete" not in result["summary"]


def test_a_round_looked_past_did_not_stop_a_run_that_finished(monkeypatch):
    result = _run(monkeypatch, {"ok": True, "completed": True, "steps": [
        {"asked": "", "chose": ["#thumbnail"], "why": "the first video", "error": "browser_interaction_incomplete"},
        {"asked": "", "chose": ["I bought an Exoskeleton"], "why": "under 15 minutes", "ok": True, "landed": 1},
    ]})
    assert "stopped_because" not in result and "error" not in result
    assert "stopped" not in result["summary"]


@pytest.mark.asyncio
async def test_her_choices_on_a_page_are_not_introduced_as_something_she_wrote(monkeypatch):
    from interface.routes import chat as chat_routes

    class _Engine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True, "status": "completed", "planner": "browser_pursuit", "url": RESULTS,
                "summary": f"I got as far as {RESULTS} and stopped: what I pressed did nothing.",
                "steps_requested": 1, "steps_completed": 1,
                "receipts": [{"index": 0, "action": "browse_pursue", "ok": True, "effect_verified": True,
                              "effect_evidence": f"pressed Search on {RESULTS}", "result": {"ok": True}}],
                "narration": [{"asked": "Search", "chose": ["Search"], "why": "his channel first"}],
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
    )
    monkeypatch.setattr("interface.routes.chat_preflight._looks_like_desktop_objective", lambda _text: True)
    result = await chat_routes._execute_desktop_objective_from_chat(
        ASKED, cognitive_reply="",
    )
    assert result is not None
    response = str(result.get("response") or "")
    assert "Here is what I wrote" not in response
    assert response.startswith("I got as far as")
    assert "his channel first" in response
