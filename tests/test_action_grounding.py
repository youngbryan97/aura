"""Tests for action grounding + user-intent detection.

Ensures the regression Bryan caught live — LLM emitting
`[SKILL_RESULT:computer_use] I opened Notes` while nothing happens — is
caught, executed, or explicitly labelled as unverified.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from core.phases.action_grounding import (
    check_unverified_action_claims,
    ground_response,
    receipts_from_context,
)
from core.phases.action_intent import (
    apply_intent_to_context,
    detect_action_intent,
)

# ---------------------------------------------------------------------------
# Marker parser
# ---------------------------------------------------------------------------


class _StubCapabilityEngine:
    def __init__(self, ok: bool = True, summary: str = "Did the thing.") -> None:
        self.ok = ok
        self.summary = summary
        self.calls: list[dict[str, Any]] = []

    async def execute(self, name: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "params": dict(params), "context": dict(context)})
        return {"ok": self.ok, "summary": self.summary, "error": "" if self.ok else "denied"}


@pytest.mark.asyncio
async def test_marker_gets_executed_and_replaced_on_success():
    engine = _StubCapabilityEngine(ok=True, summary="Notes opened and the text was typed.")
    text = "[SKILL_RESULT:computer_use] I opened the Notes app and created a new note."
    result = await ground_response(text, capability_engine=engine)

    assert result.had_markers is True
    assert result.dispatched == 1
    assert result.dispatched_ok == 1
    assert result.marker_hits[0]["ok"] is True
    assert result.marker_hits[0]["summary"] == "Notes opened and the text was typed."
    assert "Notes opened and the text was typed." in result.grounded_text
    # The fabricated suffix must be gone — the real summary replaces the whole marker line.
    assert "[SKILL_RESULT:computer_use]" not in result.grounded_text


@pytest.mark.asyncio
async def test_marker_reports_failure_honestly_instead_of_pretending():
    engine = _StubCapabilityEngine(ok=False)
    text = "[ACTION:computer_use] terminal: echo hi"
    result = await ground_response(text, capability_engine=engine)

    assert result.dispatched == 1
    assert result.dispatched_ok == 0
    assert "did not complete" in result.grounded_text
    assert "not pretending" in result.grounded_text


@pytest.mark.asyncio
async def test_marker_without_engine_marks_unverified():
    text = "[SKILL:computer_use] I opened Notes."
    result = await ground_response(text, capability_engine=None)

    assert result.had_markers is True
    assert result.dispatched == 0
    assert "was not available" in result.grounded_text
    assert "intent, not completed action" in result.grounded_text


@pytest.mark.asyncio
async def test_marker_with_omitted_engine_resolves_container_engine(monkeypatch):
    from core.container import ServiceContainer

    engine = _StubCapabilityEngine(ok=True, summary="Container engine opened Notes.")
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: engine if name == "capability_engine" else default),
    )

    text = "[SKILL:computer_use] I opened Notes."
    result = await ground_response(text)

    assert result.had_markers is True
    assert result.dispatched == 1
    assert result.dispatched_ok == 1
    assert engine.calls[0]["name"] == "computer_use"
    assert "Container engine opened Notes." in result.grounded_text


@pytest.mark.asyncio
async def test_multiple_markers_all_processed():
    engine = _StubCapabilityEngine(ok=True, summary="done")
    text = (
        "First [SKILL:computer_use] open Notes.\n"
        "Then [ACTION:computer_use] type a message."
    )
    result = await ground_response(text, capability_engine=engine)

    assert result.dispatched == 2
    assert result.replaced == 2
    assert "[SKILL" not in result.grounded_text
    assert "[ACTION" not in result.grounded_text


@pytest.mark.asyncio
async def test_structured_desktop_task_marker_dispatches_json_plan():
    engine = _StubCapabilityEngine(ok=True, summary="Desktop task completed 2/2 governed computer-use steps.")
    plan = {
        "objective": "Open Notes and write a receipt.",
        "steps": [
            {"action": "open_app", "target": "Notes"},
            {"action": "write_text_file", "target": {"path": "Aura Proof/receipt.txt", "content": "done"}},
        ],
    }
    text = f"[ACTION:desktop_task] {json.dumps(plan)}"

    result = await ground_response(text, capability_engine=engine)

    assert result.dispatched == 1
    assert result.dispatched_ok == 1
    assert engine.calls[0]["name"] == "desktop_task"
    assert engine.calls[0]["params"]["objective"] == "Open Notes and write a receipt."
    assert engine.calls[0]["params"]["steps"][1]["target"]["content"] == "done"
    assert "Desktop task completed" in result.grounded_text


@pytest.mark.asyncio
async def test_structured_computer_use_marker_dispatches_json_payload():
    engine = _StubCapabilityEngine(ok=True, summary="Clipboard set.")
    text = '[ACTION:computer_use] {"action":"set_clipboard","target":"hello"}'

    result = await ground_response(text, capability_engine=engine)

    assert result.dispatched == 1
    assert engine.calls[0]["name"] == "computer_use"
    assert engine.calls[0]["params"] == {"action": "set_clipboard", "target": "hello"}
    assert "Clipboard set." in result.grounded_text


@pytest.mark.asyncio
async def test_freeform_computer_use_type_marker_uses_target_schema():
    engine = _StubCapabilityEngine(ok=True, summary="Typed text.")
    text = "[ACTION:computer_use] type the timestamped summary into Notes"

    result = await ground_response(text, capability_engine=engine)

    assert result.dispatched == 1
    assert engine.calls[0]["name"] == "computer_use"
    assert engine.calls[0]["params"]["action"] == "type"
    assert "target" in engine.calls[0]["params"]
    assert "text" not in engine.calls[0]["params"]
    assert "timestamped summary" in engine.calls[0]["params"]["target"]
    assert "Typed text." in result.grounded_text


def test_unverified_action_claim_detector():
    hallucinated = (
        "I just opened Notes. I typed 'hi'. I clicked the send button. "
        "The note is there."
    )
    flagged = check_unverified_action_claims(hallucinated, skill_receipts=[])
    assert any("opened" in f or "typed" in f or "clicked" in f or "note is there" in f for f in flagged)


def test_unverified_claim_detector_respects_receipts():
    text = "I just opened Notes."
    flagged = check_unverified_action_claims(
        text, skill_receipts=[{"skill": "computer_use", "summary": "Notes opened"}]
    )
    assert flagged == []


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


def test_imperative_counts_as_permission():
    intent = detect_action_intent("Open Notes and type a message about your status")
    assert intent.has_action_request is True
    assert intent.has_permission_grant is True
    assert intent.should_execute is True


def test_explicit_permission_phrase():
    intent = detect_action_intent("I'm giving you permission. Do it")
    assert intent.has_permission_grant is True


def test_question_without_permission_is_not_should_execute():
    intent = detect_action_intent("Can you open the browser for me?")
    assert intent.should_execute is False


def test_apply_intent_stamps_context():
    ctx: dict[str, object] = {}
    apply_intent_to_context("Please open Notes and type hello", ctx)
    assert ctx.get("user_granted_permission") is True
    assert ctx.get("user_explicit_action_request") is True
    assert ctx.get("user_requested_action") is True


def test_receipts_from_context_helper():
    ctx = {"skill_receipts": [{"skill": "computer_use", "at": 1.0}]}
    out = receipts_from_context(ctx)
    assert out and out[0]["skill"] == "computer_use"
