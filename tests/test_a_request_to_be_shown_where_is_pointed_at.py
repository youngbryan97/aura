"""A request to be shown where something is on the screen is pointed at, and what the pointing came to is hers to say.

15 November 2026: "Can you highlight the back button?" went to chat and nothing was drawn. Pointing was attached only
to a screen question answered from her ambient reading, and a request to highlight is not one.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.surface_disposition import what_she_did_this_turn
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


DREW = "drew a rectangle around 'back button' at x 795, y 287, 36 by 36"
DID_NOT = "did not draw a rectangle around 'back button': could not locate it"


@pytest.mark.parametrize("shown", [True, False])
@pytest.mark.asyncio
async def test_a_pointing_hands_the_reply_to_her_and_the_lane_says_nothing(monkeypatch, shown):
    """No reading of the route's own: what the pointing came to reaches her as its receipt (the tests below)."""
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
    assert "rectangle" not in beside


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
    with bind_turn_evidence_custody(session_id="s", turn_id="t-task"):
        result = await DesktopTaskSkill().execute({"objective": "Can you highlight the back button?", "steps": []},
                                                  {"origin": "desktop_ui"})
        told = what_she_did_this_turn()
    assert result["status"] == "pointed_on_screen" and result["ok"] is True
    assert f"done: {DREW}" in told, told


@pytest.mark.parametrize("shown,said", [(True, f"done: {DREW}"), (False, f"failed: {DID_NOT}")])
@pytest.mark.asyncio
async def test_through_her_own_capability_engine_what_it_came_to_is_still_hers_to_say(monkeypatch, tmp_path, shown,
                                                                                      said):
    """LIVE 16 November 2026, 12:30 local: the ring was drawn around Chrome's back button and Bryan saw it, and he was
    given "Done - the desktop steps completed and their effects verified." Her capability engine writes its own verdict
    over a result's "status" ("success_verified"), and the stand-in engine in the test above did not."""
    from core.capabilities.permission_model import PermissionRiskModel
    from core.capability_engine import CapabilityEngine
    from core.container import ServiceContainer
    from interface.routes import chat as chat_routes

    from core.agency.intention_loop import IntentionLoop

    _drawn(monkeypatch, shown)
    monkeypatch.setattr("core.runtime.runtime_settings.runtime_approval_mode", lambda: "none")  # as Bryan runs her
    # Its intentions kept apart: in the suite's shared data they became the "real traffic" a later test's routing
    # surface is seeded from (tests/test_routing_consults_what_actually_ran.py, 16 November 2026).
    intentions = IntentionLoop(db_path=str(tmp_path / "intention_loop.db"))
    monkeypatch.setattr("core.agency.intention_loop._instance", intentions)
    ServiceContainer.register_instance("intention_loop", intentions, required=False)
    ServiceContainer.register_instance("permission_model", PermissionRiskModel(), required=False)
    ServiceContainer.register_instance("capability_engine", CapabilityEngine(), required=False)
    try:
        with bind_turn_evidence_custody(session_id="s", turn_id=f"engine-{shown}"):
            result = await chat_routes._execute_desktop_objective_from_chat("Can you highlight the back button?",
                                                                            cognitive_reply="")
            told = what_she_did_this_turn()
    finally:
        ServiceContainer.unregister("capability_engine")
        ServiceContainer.unregister("permission_model")
        ServiceContainer.unregister("intention_loop")
        intentions.close()

    assert result["status"] == "desktop_objective_pointing_read" and result["response"] == "", result
    assert said in told, told
    # One call: the dispatcher's receipt for the task is not a second one beside its step's (02:41, 17 November).
    assert told.count("- desktop_task") == 1, told


@pytest.mark.asyncio
async def test_the_window_she_pointed_in_is_named_with_what_came_of_it(monkeypatch):
    """LIVE 16 November 2026, 20:18: drawn around Finder's back button, her reply put it "in the browser toolbar"."""
    monkeypatch.setattr(screen_highlight, "_suppressed", lambda: False)
    monkeypatch.setattr(screen_highlight, "rate_limited", lambda: False)
    monkeypatch.setattr("core.senses.screen_context.frontmost_window_hint", lambda: ("Finder", "bryan"))
    monkeypatch.setattr(screen_highlight, "_whose_window_is_meant_by", lambda app, title: (app, title))
    monkeypatch.setattr(screen_highlight, "locate_on_screen", lambda needle: Rect(824, 316, 36, 36))
    monkeypatch.setattr(screen_highlight, "_screen_area", lambda: 1_930_176.0)

    async def drawn(rect, seconds):
        return True

    monkeypatch.setattr(screen_highlight, "_draw", drawn)
    result = await screen_highlight.highlight("back button", requested=True)
    assert result.shown and result.window == "Finder: bryan" and result.to_dict()["window"] == "Finder: bryan"


def test_pointing_runs_before_her_cognition_and_her_cognition_still_answers():
    """LIVE 16 November 2026, 20:23: marked self-sufficient, the pointing request skipped her cognition, and the turn
    ended on a stand-in model's "I couldn't get to an answer I'd stand behind"."""
    from core.runtime.desktop_objective_intent import asks_to_be_shown_where
    from interface.routes.chat_desktop_evidence import (
        _desktop_objective_self_sufficient_without_cognitive_text,
        _screen_perception_needs_her_answer,
    )

    for asked in ("Can you highlight the back button for me?", "Show me where Downloads is."):
        assert _screen_perception_needs_her_answer(asked), asked
        assert not _desktop_objective_self_sufficient_without_cognitive_text(asked), asked
    assert asks_to_be_shown_where("Can you highlight the back button for me?") == "back button"


@pytest.mark.asyncio
async def test_a_ring_she_drew_is_drawn_again_as_her_reply_is_served(monkeypatch):
    """LIVE 17 November 2026, 02:30: drawn 0.80 s after the request and gone 3 s later, and her reply, 28 s after the
    request, said "it should be visible now". A pointing that was not drawn is not drawn at her reply either."""
    from fastapi.responses import JSONResponse

    from core.epistemics.turn_effects import record_verified_effects
    from core.runtime.turn_outcome import TurnOutcome
    from interface.routes.chat_http_shapes import _mark_http_turn_served

    looked_for = []

    async def highlight(needle, **kwargs):
        looked_for.append((needle, kwargs.get("requested")))
        return HighlightResult(shown=True, reason="drawn", rect=Rect(824, 316, 36, 36), matched_text=needle)

    monkeypatch.setattr(screen_highlight, "highlight", highlight)
    with bind_turn_evidence_custody(session_id="s", turn_id="t-land"):
        record_verified_effects([screen_highlight.pointing_receipt(
            "back button", HighlightResult(shown=True, reason="drawn", rect=Rect(824, 316, 36, 36)))])
        record_verified_effects([screen_highlight.pointing_receipt(
            "submit button", HighlightResult(shown=False, reason="could not locate it"))])
        _mark_http_turn_served(TurnOutcome(turn_id="t-land"), JSONResponse(
            {"response": "Done. I drew a rectangle around the back button in your Finder window."}))
        await asyncio.gather(*screen_highlight._AS_HER_WORDS_LAND)
    assert looked_for == [("back button", True)]
