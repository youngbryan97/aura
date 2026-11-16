"""A gate that cannot name a violation has not proven one.

LIVE, 2026-08-10. A turn died with this failure class:

    reply_reliability_gate_failed:

The separator, and nothing after it. assessment.reasons was empty, so the
reliability gate rejected a complete reply without naming a single violation,
the turn was recorded as exhausted, and the person got "I couldn't get to an
answer I'd stand behind on that one."

The runtime already refuses the mirror image of this — absence of a check must
never be reported as a passed check, which is written into five subsystems. The
same rule holds pointing the other way: absence of a finding must not be
reported as a failure. A reply in hand beats a canned refusal justified by
nothing.

The gate keeps every power it had. It simply has to say what is wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interface.routes.chat import _named_gate_failure, _reply_gate_proved_a_violation
import interface.routes.chat_memory_state as _chat_memory_state


@pytest.mark.parametrize(
    "reasons",
    [(), [], None, ("",), ("   ",), ["", "  "]],
)
def test_an_empty_or_blank_reason_list_proves_nothing(reasons) -> None:
    assessment = SimpleNamespace(reasons=reasons)

    assert _reply_gate_proved_a_violation(assessment) is False


@pytest.mark.parametrize(
    "reasons",
    [
        ("truncated_tail",),
        ("fabricated_shared_history", "reply_abandons_thread"),
        ["runtime_boilerplate"],
    ],
)
def test_a_named_violation_still_rejects(reasons) -> None:
    """The gate keeps every power it had."""
    assessment = SimpleNamespace(reasons=reasons)

    assert _reply_gate_proved_a_violation(assessment) is True


def test_an_assessment_without_a_reasons_attribute_proves_nothing() -> None:
    assert _reply_gate_proved_a_violation(object()) is False


def test_the_failure_class_carries_its_reasons() -> None:
    assessment = SimpleNamespace(reasons=("truncated_tail", "runtime_boilerplate"))

    assert _named_gate_failure(assessment) == (
        "reply_reliability_gate_failed:truncated_tail,runtime_boilerplate"
    )


def test_an_unnamed_failure_class_says_so_instead_of_trailing_a_colon() -> None:
    """"reply_reliability_gate_failed:" told an operator nothing at all."""
    assessment = SimpleNamespace(reasons=())

    assert _named_gate_failure(assessment) == (
        "reply_reliability_gate_failed:unnamed_violation"
    )


@pytest.mark.asyncio
async def test_the_rejection_path_serves_the_draft_when_nothing_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.conversation import response_reliability
    from interface.routes import chat

    prompt = "Tell me one interesting fact about octopus cognition."
    draft = (
        "An octopus distributes much of its nervous system through its arms, "
        "so local sensing and movement do not wait on a single central controller."
    )
    calls: list[str] = []

    class _FakeCognitiveEngine:
        async def think(self, objective, **_kwargs):
            calls.append(str(objective))
            return SimpleNamespace(content=draft)

    async def _no_recall(*_args, **_kwargs):
        return None

    assessment = SimpleNamespace(ok=False, reasons=(), retryable=True)
    monkeypatch.setattr(
        chat.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _FakeCognitiveEngine()
            if name == "cognitive_engine"
            else default
        ),
    )
    monkeypatch.setattr(
        response_reliability,
        "assess_user_facing_reply",
        lambda *_args, **_kwargs: assessment,
    )
    monkeypatch.setattr(
        chat,
        "_reply_assessment_requires_repair_with_memory_evidence",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        chat,
        "_desktop_secondary_model_repair_allowed",
        lambda **_kwargs: (False, "single_owner_test"),
    )
    monkeypatch.setattr(_chat_memory_state, "_what_the_conversation_holds_for", _no_recall)
    trace: dict[str, object] = {}

    reply = await chat._run_cognitive_engine_chat_turn(
        prompt,
        visible_user_message=prompt,
        origin="user",
        timeout_s=60.0,
        lane={"conversation_ready": True, "state": "ready"},
        source="desktop_ui",
        require_engine=True,
        turn_trace=trace,
    )

    assert calls == [prompt]
    assert reply == draft
    assert reply != prompt
    assert trace["cognitive_engine_reply_accepted"] is True
    assert trace["response_path"] == "cognitive_engine_reply_gate_unnamed"
