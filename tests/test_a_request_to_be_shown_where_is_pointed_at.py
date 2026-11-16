"""A request to be shown where something is on the screen is pointed at, and what the pointing came to is hers to say.

15 November 2026: "Can you highlight the back button?" went to chat and nothing was drawn. Pointing was attached only
to a screen question answered from her ambient reading, and a request to highlight is not one.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.turn_evidence_custody import bind_turn_evidence_custody, turn_world_evidence
from core.perception import screen_highlight
from core.perception.screen_highlight import HighlightResult, Rect


def _drawn(monkeypatch, shown: bool):
    async def highlight(needle, **kwargs):
        assert kwargs.get("requested") is True
        if shown:
            return HighlightResult(shown=True, reason="drawn", rect=Rect(795, 287, 36, 36), matched_text=needle)
        return HighlightResult(shown=False, reason="could not locate it in the accessibility tree; describing it instead")

    monkeypatch.setattr(screen_highlight, "highlight", highlight)


def test_a_request_to_highlight_is_a_desktop_step_and_a_world_question_is_not():
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert looks_like_desktop_objective("Can you highlight the back button?")
    assert looks_like_desktop_objective("Show me where Downloads is.")
    assert not looks_like_desktop_objective("Where is Paris?")
    assert not looks_like_desktop_objective("Show me a piece of your own code that you find interesting.")


def test_pointing_is_a_step_with_its_receipt(monkeypatch):
    _drawn(monkeypatch, True)
    done = asyncio.run(screen_highlight.point_as_asked("Can you highlight the back button?"))
    assert done["ok"] is True and done["status"] == "pointed_on_screen" and done["steps_completed"] == 1
    assert done["receipts"][0]["effect_verified"] is True and "back button" in done["receipts"][0]["effect_evidence"]
    assert asyncio.run(screen_highlight.point_as_asked("What time is it?")) is None


@pytest.mark.parametrize("shown,said", [(True, "A rectangle was drawn on the screen around 'back button'"),
                                        (False, "Nothing was drawn: could not locate it")])
@pytest.mark.asyncio
async def test_what_the_pointing_came_to_is_beside_her_turn_and_the_lane_says_nothing(monkeypatch, shown, said):
    from interface.routes import chat as chat_routes

    _drawn(monkeypatch, shown)
    pointed = await screen_highlight.point_as_asked("Can you highlight the back button?")

    class _Engine:
        async def execute(self, skill_name, params, context=None):
            assert skill_name == "desktop_task"
            return dict(pointed)

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _Engine() if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id=f"t-{shown}"):
        result = await chat_routes._execute_desktop_objective_from_chat("Can you highlight the back button?",
                                                                        cognitive_reply="")
        beside = "\n".join(turn_world_evidence())
    assert result["status"] == "desktop_objective_pointing_read" and result["response"] == ""
    assert said in beside


@pytest.mark.asyncio
async def test_the_desktop_task_points_where_nothing_else_is_asked(monkeypatch):
    """Pointing is the step when the only thing planned is a look; a request that also acts keeps its plan
    (tests/test_desktop_task_skill.py, the squid wallpaper and where it was found)."""
    from core.container import ServiceContainer
    from core.skills.desktop_task import DesktopTaskSkill

    class _Engine:
        async def execute(self, skill_name, params, context=None):
            raise AssertionError(f"nothing but the pointing was asked: {skill_name} {params}")

    _drawn(monkeypatch, True)
    monkeypatch.setattr(ServiceContainer, "get",
                        lambda name, default=None: _Engine() if name == "capability_engine" else default)
    result = await DesktopTaskSkill().execute({"objective": "Can you highlight the back button?", "steps": []},
                                              {"origin": "desktop_ui"})
    assert result["status"] == "pointed_on_screen" and result["ok"] is True
