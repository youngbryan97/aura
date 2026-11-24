"""Which tool a loop calls next, and when the work is done, are hers; her model writes the one call's arguments.

Bryan, 24 November 2026: "the 27b is an organ. not the central hub. aura should be driving things. not her model."
Handed every tool a turn offered, her model chose which to call, with what, in what order, and when to stop.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.brain.llm.her_steps import UNWRITTEN_TRIES, HerSteps, came_back_done

pytestmark = pytest.mark.unit

TOOLS = {
    "find": {"description": "find where something is", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
    "change": {"description": "change it", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}},
    "check": {"description": "check the change", "parameters": {"type": "object", "properties": {}}},
}


def test_her_steps_follow_her_order_and_what_each_waits_on() -> None:
    steps = HerSteps.for_loop(TOOLS, "make it right", order=["find", "change", "check"], needs={"check": "change"})
    assert steps.next() == "find"
    steps.called("find", True)
    assert steps.next() == "change"
    steps.called("change", False)
    assert steps.next() == "change"  # tried once more with what came back in view
    steps.called("change", False)
    # The change never came back done: the check that waits on it is let go, and her steps are done.
    assert steps.next() is None


def test_a_step_her_model_writes_no_call_for_is_let_go() -> None:
    steps = HerSteps.for_loop(TOOLS, "make it right")
    for _ in range(UNWRITTEN_TRIES):
        assert steps.next() == "find"
        steps.not_filled("find")
    assert steps.next() == "change"


def test_steps_already_taken_in_this_piece_of_work_are_not_taken_again() -> None:
    steps = HerSteps.for_loop(TOOLS, "make it right", already=[("find", True), ("change", True)])
    assert steps.next() == "check"


def test_a_choice_among_tools_is_her_reading_by_meaning(monkeypatch) -> None:
    import core.brain.requests_by_meaning as meaning

    asked: list[tuple[str, dict]] = []

    def nearest(text, described):
        asked.append((text, dict(described)))
        return sorted(described, key=lambda name: name != "check")

    monkeypatch.setattr(meaning, "nearest_by_meaning", nearest)
    steps = HerSteps.for_loop(TOOLS, "see whether it holds", choose_by_meaning=True)
    assert steps.next("it was changed") == "check"
    assert asked and "see whether it holds" in asked[0][0] and "it was changed" in asked[0][0]
    assert asked[0][1]["find"] == "find where something is"


def test_without_her_reading_a_choice_keeps_her_callers_order(monkeypatch) -> None:
    import core.brain.requests_by_meaning as meaning

    monkeypatch.setattr(meaning, "nearest_by_meaning", lambda text, described: [])
    assert HerSteps.for_loop(TOOLS, "anything", choose_by_meaning=True).next() == "find"


def test_what_came_back_says_whether_it_was_done() -> None:
    assert came_back_done({"ok": True}) and not came_back_done({"ok": False})
    assert came_back_done({"status": "completed"}) and not came_back_done({"status": "error"})


def test_each_pass_offers_her_model_her_step_alone_and_a_call_to_another_is_not_run(monkeypatch) -> None:
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.container import ServiceContainer

    ran: list[str] = []
    offered: list[list[str]] = []

    class Engine:
        async def execute(self, name, args, context=None, **_kwargs):
            ran.append(name)
            return {"ok": True, "result": name}

    async def her_model(*_args, tools=None, writes_no_call=False, messages=None, **_kwargs):
        # What the pass offers, by either protocol: native definitions, or the block in the system message.
        system = str((messages or [{}])[0].get("content") or "")
        offered.append(sorted(t.get("function", t).get("name", "") for t in (tools or []))
                       or sorted(name for name in TOOLS if f"• {name}:" in system))
        if writes_no_call:
            return "All done."
        # Her model would rather jump to the check every time.
        return "```json\n" + json.dumps({"tool": "check", "args": {}}) + "\n```"

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(
        lambda name, default=None: Engine() if name == "capability_engine" else default))
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.max_tokens, client.model_path, client.generate_text_async = 512, "her-cortex", her_model
    out = asyncio.run(client.think_and_act("make it right", "", tools=TOOLS, max_turns=8,
                                           her_plan=["find", "change", "check"], her_needs={"check": "change"}))
    # Every pass offered one tool, hers; the check her model kept asking for was never run before its turn, and,
    # with the change never made, never at all.
    assert all(len(names) <= 1 for names in offered), offered
    assert offered[0] == ["find"]
    assert "check" not in ran
    assert out["content"] in ("", "All done.")


def test_steps_let_go_in_this_piece_of_work_are_not_offered_again() -> None:
    steps = HerSteps.for_loop(TOOLS, "make it right", let_go=["find"])
    assert steps.next() == "change"
    steps.not_filled("change")
    assert steps.let_go() == ["find", "change"]
