"""At the desk, pointing at something on the person's screen is a call she makes in her own turn.

LIVE 16 November 2026, 01:04, in the bubble: "I can't highlight or interact with any UI elements on your screen - I
don't have tools for that." Her turn at the desk was composed with no tools in its prompt, so she said she had none,
and her pointing was chosen for her afterwards by matching her words against the screen. Bryan, 15 November 2026:
"she's allowed to physically act and react to things that are going on. highlight, show me something ... full
companion. and yes im giving you permission to work with whatever permissions she needs".
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain import her_unprompted_turn as turn
from core.capabilities.permission_model import PermissionRiskModel
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope
from core.executive.standing_authority import DESK_COMPANY_GRANT, DESK_COMPANY_ORIGIN, DESK_COMPANY_TOOLS
from core.skills.at_the_desk import PointAtOnScreenSkill, tool_definitions
from core.skills.catalog_policy import NOT_OFFERED_TO_A_TURN
from core.skills.catalog_policy import DESK_COMPANY_TOOLS as CATALOGUED_DESK_TOOLS
from tests.test_standing_authority import _manager

POINT = {"words": "Back"}


def _rated(tool, args):
    scope = resolve_execution_effect_scope(tool, args)
    return scope, classify_execution_risk(tool, args, effect_scope=scope)


def test_the_grant_names_exactly_her_desk_tools_and_her_desk_turn():
    assert DESK_COMPANY_TOOLS == CATALOGUED_DESK_TOOLS == set(tool_definitions())
    assert turn.DESK_ORIGIN == DESK_COMPANY_ORIGIN
    assert DESK_COMPANY_TOOLS <= NOT_OFFERED_TO_A_TURN, "a conversational turn was offered a desk tool"


@pytest.mark.asyncio
async def test_her_desk_turn_holds_a_lease_to_point():
    scope, risk = _rated("point_at_on_screen", POINT)
    decision = await _manager().issue_child_lease("point_at_on_screen", POINT, origin=DESK_COMPANY_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert decision.approved, decision.reason
    assert decision.grant_id == DESK_COMPANY_GRANT


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "args", "origin"), [
    ("point_at_on_screen", POINT, "autonomous_initiative"),
    ("point_at_on_screen", {"words": "Back", "x": 10}, DESK_COMPANY_ORIGIN),
    ("point_at_on_screen", {"words": "  "}, DESK_COMPANY_ORIGIN),
    ("os_manipulation", {"action": "click", "x": 10, "y": 10}, DESK_COMPANY_ORIGIN),
    ("computer_use", {"action": "type", "text": "hello"}, DESK_COMPANY_ORIGIN),
])
async def test_the_grant_covers_nothing_else(tool, args, origin):
    scope, risk = _rated(tool, args)
    decision = await _manager().issue_child_lease(tool, args, origin=origin, effect_scope=scope, risk_level=risk)
    assert not (decision.approved and decision.grant_id == DESK_COMPANY_GRANT)


@pytest.mark.asyncio
async def test_pointing_within_the_grant_needs_no_person_and_without_it_still_does(monkeypatch):
    manager = _manager()
    monkeypatch.setattr("core.executive.standing_authority.get_standing_authority_manager", lambda: manager)
    tool, args = "point_at_on_screen", POINT
    scope, risk = _rated(tool, args)
    assert (scope, risk) == ("foreground_desktop_control", "high")
    lease = await manager.issue_child_lease(tool, args, origin=DESK_COMPANY_ORIGIN, effect_scope=scope,
                                            risk_level=risk)
    model = PermissionRiskModel()

    def asked(context):
        return model.check_permission("tool_execution", f"tool:{tool}", context, effect_scope=scope,
                                      execution_risk=risk)

    held = asked({**lease.context, "tool": tool, "origin": DESK_COMPANY_ORIGIN, "authority_arguments": args})
    assert held.approved and not held.requires_confirmation
    assert DESK_COMPANY_GRANT in held.reason
    bare = asked({"tool": tool, "origin": DESK_COMPANY_ORIGIN})
    assert not bare.approved and bare.requires_confirmation
    forged = asked({"tool": tool, "origin": DESK_COMPANY_ORIGIN, "standing_authority_grant_id": DESK_COMPANY_GRANT})
    assert not forged.approved and forged.requires_confirmation


def test_the_skill_draws_what_she_names_as_hers_to_draw(monkeypatch):
    from core.perception import screen_highlight

    drawn: dict[str, object] = {}

    async def highlight(needle, **kwargs):
        drawn.update(needle=needle, **kwargs)
        return screen_highlight.HighlightResult(shown=True, reason="drawn", matched_text=needle)

    monkeypatch.setattr(screen_highlight, "highlight", highlight)

    result = asyncio.run(PointAtOnScreenSkill().execute({"words": " Back "}, {}))

    assert drawn == {"needle": "Back", "invited": True}
    assert result["ok"] is True and "Back" in result["summary"]


def test_a_ring_that_was_not_drawn_says_why(monkeypatch):
    from core.perception import screen_highlight

    async def highlight(needle, **kwargs):
        return screen_highlight.HighlightResult(shown=False, reason="a private window is in the foreground")

    monkeypatch.setattr(screen_highlight, "highlight", highlight)

    result = asyncio.run(PointAtOnScreenSkill().execute({"words": "Back"}, {}))

    assert result["ok"] is False and result["error"] == "a private window is in the foreground"


class _HerCortex:
    def __init__(self, result):
        self.result, self.asked = result, {}

    async def think_and_act(self, objective, **kwargs):
        self.asked = {"objective": objective, **kwargs}
        return self.result


def _her_desk_turn(monkeypatch, result, conversation):
    cortex = _HerCortex(result)
    monkeypatch.setattr("core.together.remarks.said_since", lambda began: conversation)
    gate = type("Gate", (), {"_mlx_client": cortex})()
    monkeypatch.setattr("core.runtime.service_access.resolve_inference_gate", lambda default=None: gate)
    monkeypatch.setattr("core.brain.llm_health_router.a_person_is_waiting_for_her_model", lambda: False)

    async def lease(**_kwargs):
        return object()

    monkeypatch.setattr("core.brain.llm_health_router.acquire_external_generation_gate_lease", lease)
    monkeypatch.setattr("core.brain.llm_health_router.release_external_generation_gate_lease", lambda _lease: None)
    said = asyncio.run(turn.her_turn_at_the_desk("WHAT IS ON THE SCREEN", "ChatGPT\n< Back", least_s=1.0))
    return said, cortex.asked


def test_her_desk_turn_is_her_conversation_and_the_screen_with_her_desk_tools(monkeypatch):
    conversation = [(1.0, "user", "Can you highlight the back button?"), (2.0, "assistant", "Give me a moment.")]
    said, asked = _her_desk_turn(monkeypatch, {
        "content": "There it is, top left.",
        "tool_calls": [{"tool": "point_at_on_screen", "args": POINT, "result": "pointed at 'Back'"}],
    }, conversation)

    assert said == "There it is, top left."
    assert asked["objective"] == ""
    assert [m["role"] for m in asked["conversation"]] == ["user", "assistant", "runtime_evidence"]
    assert "< Back" in asked["conversation"][-1]["content"]
    assert set(asked["tools"]) == DESK_COMPANY_TOOLS
    assert asked["origin"] == asked["context"]["origin"] == DESK_COMPANY_ORIGIN
    assert asked["cognitive_mode"] == turn._HER_MODE and asked["native_tools_only"] is True
    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR

    assert asked["max_tokens"] == A_CLOSED_QUESTIONS_FLOOR, "her time was worked out for this allowance"


def test_with_nobody_in_her_conversation_her_desk_turn_is_not_taken(monkeypatch):
    said, asked = _her_desk_turn(monkeypatch, {"content": "Anything."}, [(1.0, "assistant", "Hello.")])

    assert said == "" and asked == {}


def test_the_loop_follows_a_conversation_in_place_of_a_request():
    from core.brain.llm.mlx_client_tool_results import _how_the_loop_generates, _what_the_loop_answers

    conversation = [{"role": "user", "content": "Hi."}, {"role": "runtime_evidence", "content": "## SCREEN"}]
    assert _what_the_loop_answers("", conversation) == conversation
    assert _what_the_loop_answers("Read the file.", None) == [{"role": "user", "content": "Read the file."}]
    assert _how_the_loop_generates({"work_in_background": True, "origin": "x", "cognitive_mode": "reflective"}) == {
        "origin": "x", "is_background": True, "cognitive_mode": "reflective"}
    assert _how_the_loop_generates({}) == {}


def test_only_her_own_passes_carry_bryans_standing_yes():
    from core.executive.standing_authority import OWN_GOAL_ORIGIN, carries_bryans_standing_yes

    assert carries_bryans_standing_yes(DESK_COMPANY_ORIGIN, "point_at_on_screen")
    assert carries_bryans_standing_yes(OWN_GOAL_ORIGIN, "edit_draft")
    assert not carries_bryans_standing_yes(DESK_COMPANY_ORIGIN, "edit_draft")
    assert not carries_bryans_standing_yes(OWN_GOAL_ORIGIN, "point_at_on_screen")
    assert not carries_bryans_standing_yes("autonomous_initiative", "point_at_on_screen")


@pytest.fixture
def her_permission_model():
    """Her permission model where her capability engine asks for it, as in her running app.

    Without one registered, the engine skips the permission gate, and on 16 November 2026 a test of the whole chain
    passed while her live runtime refused the same call there, before any lease existed.
    """
    from core.capabilities.permission_model import PermissionRiskModel
    from core.container import ServiceContainer

    model = PermissionRiskModel()
    ServiceContainer.register_instance("permission_model", model, required=False)
    yield model
    ServiceContainer.unregister("permission_model")


@pytest.mark.asyncio
async def test_her_capability_engine_runs_her_call_from_the_desk_and_from_nowhere_else(monkeypatch, her_permission_model):
    """16 November 2026: with the grant and the permission model answering, the constitution still held the call
    for a fresh confirmation (runtime_setting_user_confirmation_required:approval_mode_destructive)."""
    from core.capability_engine import CapabilityEngine
    from core.perception import screen_highlight

    async def highlight(needle, **kwargs):
        return screen_highlight.HighlightResult(shown=True, reason="drawn", matched_text=needle)

    monkeypatch.setattr(screen_highlight, "highlight", highlight)
    engine = CapabilityEngine()

    at_the_desk = await engine.execute("point_at_on_screen", dict(POINT),
                                       context={"origin": DESK_COMPANY_ORIGIN, "requested_by": "aura"})
    elsewhere = await engine.execute("point_at_on_screen", dict(POINT),
                                     context={"origin": "autonomous_initiative", "requested_by": "aura"})

    assert at_the_desk.get("ok") is True, at_the_desk
    assert elsewhere.get("ok") is not True, elsewhere
