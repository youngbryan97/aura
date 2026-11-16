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
async def test_a_refused_screen_read_is_beside_her_turn_and_the_lane_says_nothing(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _engine(REFUSED) if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        result = await chat_routes._execute_desktop_objective_from_chat("What is on my screen right now?",
                                                                        cognitive_reply="")
        grounding = turn_grounding_evidence()
        beside_her_turn = turn_world_evidence()
    assert result["status"] == "desktop_objective_failure_read" and result["response"] == ""
    assert result["ok"] is False
    # Where her model reads it (what_she_brings_to_a_turn), not only where her reply check does: turn grounding alone
    # never reached her prompt (15 November 2026).
    assert any("did not complete" in block for block in beside_her_turn)
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
