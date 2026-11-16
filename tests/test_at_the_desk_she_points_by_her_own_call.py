"""At the desk, pointing at something on the person's screen is a call she makes in her own turn.

LIVE 16 November 2026, 01:04, in the bubble: "I can't highlight or interact with any UI elements on your screen - I
don't have tools for that." Her turn at the desk was composed with no tools in its prompt, so she said she had none,
and her pointing was chosen for her afterwards by matching her words against the screen. Bryan, 15 November 2026:
"she's allowed to physically act and react to things that are going on. highlight, show me something ... full
companion. and yes im giving you permission to work with whatever permissions she needs".
"""
from __future__ import annotations

import asyncio
import time

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
    now = time.time()
    conversation = [(now - 60.0, "user", "Can you highlight the back button?"), (now - 30.0, "assistant", "One moment.")]
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


def test_an_earlier_sitting_is_not_the_conversation_her_desk_turn_answers(monkeypatch):
    """LIVE 16 November 2026, 08:32: "I'm sorry - I wasn't able to highlight it just now ... Could you try asking again
    in a moment?", for "Can you highlight the back button?" asked at 00:40, eight hours and a sitting before."""
    long_ago = time.time() - 8 * 3600
    said, asked = _her_desk_turn(monkeypatch, {"content": "That page is the submit form for r/ResearchML."}, [
        (long_ago, "user", "Can you highlight the back button?"), (long_ago + 90, "assistant", "No tool for that.")])

    assert said.startswith("That page")
    assert [m["role"] for m in asked["conversation"]] == ["user"], "the screen is the turn before hers"
    assert "back button" not in str(asked["conversation"])


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


SHOWS = {"show_page": {"url": "https://docs.python.org/3/library/asyncio.html"},
         "reveal_file": {"path": "~/Desktop/notes.txt"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(SHOWS))
async def test_her_desk_turn_holds_a_lease_to_show_him_something(tool):
    """Bryan, 15 November 2026: "highlight, show me something, literally anything OS related"."""
    scope, risk = _rated(tool, SHOWS[tool])
    decision = await _manager().issue_child_lease(tool, SHOWS[tool], origin=DESK_COMPANY_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert decision.approved, decision.reason
    assert decision.grant_id == DESK_COMPANY_GRANT


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///etc/passwd", "https://user:pass@example.com/",
                                 "x-apple-systempreferences:com.apple.preference.security", "https://"])
async def test_what_she_shows_from_the_desk_is_a_web_page(url):
    args = {"url": url}
    scope, risk = _rated("show_page", args)
    decision = await _manager().issue_child_lease("show_page", args, origin=DESK_COMPANY_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert not (decision.approved and decision.grant_id == DESK_COMPANY_GRANT)


def test_a_page_she_shows_opens_in_a_new_tab(monkeypatch):
    from types import SimpleNamespace

    from core.skills.at_the_desk import ShowPageSkill

    opened: list[tuple[str, bool]] = []

    class Browser:
        async def open_url(self, url, new_tab=True):
            opened.append((url, new_tab))
            return SimpleNamespace(success=True, error="")

    monkeypatch.setattr("core.runtime.service_registry.get_runtime_service",
                        lambda name, default=None: Browser() if name == "browser_controller" else default)

    shown = asyncio.run(ShowPageSkill().execute(dict(SHOWS["show_page"]), {}))
    refused = asyncio.run(ShowPageSkill().execute({"url": "javascript:alert(1)"}, {}))

    assert shown["ok"] is True and opened == [(SHOWS["show_page"]["url"], True)]
    assert refused["ok"] is False and len(opened) == 1


def test_a_file_she_shows_is_selected_in_finder_and_nothing_is_run(monkeypatch):
    from core.skills.at_the_desk import RevealFileSkill

    revealed: list[str] = []

    class Broker:
        async def reveal_in_finder(self, path):
            revealed.append(path)
            return path.endswith("notes.txt")

    monkeypatch.setattr("core.capabilities.file_broker.get_file_broker", lambda: Broker())

    shown = asyncio.run(RevealFileSkill().execute(dict(SHOWS["reveal_file"]), {}))
    outside = asyncio.run(RevealFileSkill().execute({"path": "/System/Library/CoreServices"}, {}))

    assert shown["ok"] is True and outside["ok"] is False
    assert revealed == ["~/Desktop/notes.txt", "/System/Library/CoreServices"]


@pytest.mark.asyncio
async def test_her_capability_engine_shows_a_page_from_the_desk_and_from_nowhere_else(monkeypatch,
                                                                                      her_permission_model):
    from types import SimpleNamespace

    from core.capability_engine import CapabilityEngine

    opened: list[str] = []

    class Browser:
        async def open_url(self, url, new_tab=True):
            opened.append(url)
            return SimpleNamespace(success=True, error="")

    monkeypatch.setattr("core.runtime.service_registry.get_runtime_service",
                        lambda name, default=None: Browser() if name == "browser_controller" else default)
    engine = CapabilityEngine()

    at_the_desk = await engine.execute("show_page", dict(SHOWS["show_page"]),
                                       context={"origin": DESK_COMPANY_ORIGIN, "requested_by": "aura"})
    elsewhere = await engine.execute("show_page", dict(SHOWS["show_page"]),
                                     context={"origin": "autonomous_initiative", "requested_by": "aura"})

    assert at_the_desk.get("ok") is True, at_the_desk
    assert elsewhere.get("ok") is not True, elsewhere
    assert opened == [SHOWS["show_page"]["url"]]


def test_what_she_has_decided_to_say_here_comes_before_her_background_work(monkeypatch):
    """LIVE 16 November 2026, 08:31: her desk turn waited out its time behind her goal pass, "her model is held for
    something else", and nothing was said."""
    from core.brain import llm_health_router_gate_owners as owners

    asked_to_stop: list[str] = []
    seen_waiting: list[bool] = []
    monkeypatch.setattr(owners, "_background_stands_aside", lambda reason: asked_to_stop.append(reason))
    cortex = _HerCortex({"content": "That form wants one thing you built."})
    monkeypatch.setattr("core.together.remarks.said_since", lambda began: [])
    gate = type("Gate", (), {"_mlx_client": cortex})()
    monkeypatch.setattr("core.runtime.service_access.resolve_inference_gate", lambda default=None: gate)
    monkeypatch.setattr("core.brain.llm_health_router.a_person_is_waiting_for_her_model", lambda: False)

    async def lease(**_kwargs):
        seen_waiting.append(owners.a_person_is_waiting_for_her_model())
        return object()

    monkeypatch.setattr("core.brain.llm_health_router.acquire_external_generation_gate_lease", lease)
    monkeypatch.setattr("core.brain.llm_health_router.release_external_generation_gate_lease", lambda _lease: None)

    said = asyncio.run(turn.her_turn_at_the_desk("WHAT IS ON THE SCREEN", "TextEdit\nDescribe it.", least_s=1.0))

    assert said.startswith("That form")
    assert seen_waiting == [True], "her background work reads someone waiting while she waits for her model"
    assert len(asked_to_stop) == 1 and "something to say" in asked_to_stop[0]
    assert owners.a_person_is_waiting_for_her_model() is False, "the wait ends with her turn"


def test_a_call_to_a_tool_she_was_not_offered_goes_back_to_her_and_is_never_her_words(monkeypatch):
    """LIVE 16 November 2026, 11:23 local: her desk turn called "point_at_on_words", which was never offered; the loop
    read a call it could not match as her answer, and the bubble showed "<tool_call> <function=point_at_on_words>"."""
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.container import ServiceContainer

    executed: list[str] = []

    class Engine:
        async def execute(self, name, args, context=None, **_kwargs):
            executed.append(name)
            return {"ok": True, "result": "drawn"}

    replies = iter([
        "<tool_call>\n<function=point_at_on_words>\n<parameter=words>\nBack\n</parameter>\n</function>\n</tool_call>",
        "That form wants one thing you built, five levels down.",
    ])
    seen: list[list[dict]] = []

    async def generate(_prompt, *, messages, **_kwargs):
        seen.append([dict(message) for message in messages])
        return next(replies)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(
        lambda name, default=None: Engine() if name == "capability_engine" else default))
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.max_tokens, client.model_path, client.generate_text_async = 512, "her-cortex", generate

    out = asyncio.run(client.think_and_act("", "", tools=tool_definitions(), max_turns=2,
                                           conversation=[{"role": "user", "content": "## WHAT IS ON THE SCREEN\nBack"}]))

    assert out["content"] == "That form wants one thing you built, five levels down."
    assert executed == [], "nothing she was not offered ran"
    assert "no tool by that name was offered" in str(seen[1][-1]["content"])
