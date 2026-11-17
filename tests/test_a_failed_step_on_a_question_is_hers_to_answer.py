"""A desktop step that did not complete, on a question, goes to her turn as a reading; she answers it.

LIVE 14 November 2026: asked "What is on my screen right now?" with a private window up, her screen read was refused
and the person was given the desktop lane's own sentence, "This did not complete: read screen text failed: ... I am
not claiming it was done." A question is hers to answer, as when the step succeeds.
"""
from __future__ import annotations

import pytest

from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    turn_grounding_evidence,
    turn_world_evidence,
)


def _engine(result):
    class _Engine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "desktop_task"
            # As her capability engine's dispatcher records every call it makes (core/transparency/dev_mode.py).
            from core.conversation.surface_disposition import record_tool_receipt

            record_tool_receipt(skill_name, ok=bool(result.get("ok")), evidence=str(result.get("error") or ""))
            return dict(result)

    return _Engine()


REFUSED = {
    "ok": False,
    "status": "failed",
    "error": "read_screen_text_failed:private_window_on_screen",
    "steps_requested": 1,
    "steps_completed": 0,
}


@pytest.mark.asyncio
async def test_a_refused_screen_read_is_her_failed_call_and_the_lane_says_nothing(monkeypatch):
    """Her model reads the step as the call she made and what it came back with, after the person's message; her
    reply check reads it as grounding. No reading of the runtime's says it again in her prompt (17 November 2026)."""
    import json

    from core.conversation.surface_disposition import her_calls_this_turn
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _engine(REFUSED) if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        result = await chat_routes._execute_desktop_objective_from_chat("What is on my screen right now?",
                                                                        cognitive_reply="")
        grounding = turn_grounding_evidence()
        beside_her_turn = turn_world_evidence()
        calls = her_calls_this_turn()
    assert result["status"] == "desktop_objective_failure_read" and result["response"] == ""
    assert result["ok"] is False
    assert not beside_her_turn
    assert [m["role"] for m in calls] == ["assistant", "tool"]
    assert json.loads(calls[1]["content"]) == {"ok": False, "came_to": "read_screen_text_failed:private_window_on_screen"}
    reading = "\n".join(grounding)
    assert "did not complete: read screen text failed: private window on screen" in reading
    assert "I am not claiming" not in reading  # a reading, not words of hers


@pytest.mark.asyncio
async def test_after_her_reply_the_lane_still_gives_its_account(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _engine(REFUSED) if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        result = await chat_routes._execute_desktop_objective_from_chat(
            "What is on my screen right now?", cognitive_reply="Let me look.", her_reply_first=True)
    assert result["status"] != "desktop_objective_failure_read"


@pytest.mark.asyncio
async def test_her_answer_passed_by_her_gates_stands_over_a_step_that_failed_after_it(monkeypatch):
    """LIVE 17 November 2026, 08:55, a private window in front: her answer said nothing was read from the screen, the
    step after it was refused, and its "This did not complete: ... I am not claiming it was done." was served in her
    answer's place. A draft her gates did not pass is not served this way: it keeps the lane's account."""
    from interface.routes import chat as chat_routes

    hers = "There's a private window in front, so I haven't read your screen."
    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _engine(REFUSED) if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id="t-stands"):
        stood = await chat_routes._execute_desktop_objective_from_chat(
            "What is on my screen right now?", cognitive_reply=hers, her_reply_first=True, her_reply_stands=True)
    with bind_turn_evidence_custody(session_id="s", turn_id="t-salvaged"):
        salvaged = await chat_routes._execute_desktop_objective_from_chat(
            "What is on my screen right now?", cognitive_reply=hers, her_reply_first=True)
    assert stood["response"] == hers and stood["ok"] is False
    assert salvaged["response"].startswith("This did not complete")
