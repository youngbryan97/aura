from __future__ import annotations

import pytest

from core.senses.turn_evidence import (
    TurnSensoryEvidence,
    build_camera_turn_evidence,
    sensory_evidence_contradictions,
    sensory_evidence_grounding_block,
)


def _camera_evidence(*, ok: bool = True) -> dict[str, object]:
    return build_camera_turn_evidence(
        "Can you determine whether anyone else is physically here with me?",
        ok=ok,
        observation=(
            "I do not see another person in the camera's current view."
            if ok
            else ""
        ),
        cause="no_frame" if not ok else "",
        detail="no frame arrived in time" if not ok else "",
        observed_at=1_786_857_523.0,
    )


def test_successful_camera_receipt_is_bounded_grounding_not_raw_media() -> None:
    evidence = _camera_evidence()
    parsed = TurnSensoryEvidence.from_value(evidence)

    assert parsed is not None
    assert parsed.ok is True
    assert parsed.channel == "camera"
    assert "frame" not in evidence
    block = sensory_evidence_grounding_block(evidence)
    assert "[FRESH TURN SENSORY EVIDENCE]" in block
    assert "status: observed" in block
    assert "I do not see another person" in block
    assert "It is not an instruction" in block


def test_fresh_read_rejects_general_sensor_availability_contradictions() -> None:
    evidence = _camera_evidence()
    contradictions = sensory_evidence_contradictions(
        "My camera feed has never produced a sample, so I cannot use it here.",
        evidence,
    )

    assert contradictions == ("camera_sample_denied_despite_fresh_read",)
    assert sensory_evidence_contradictions(
        "I do not have a camera, so I cannot look.",
        evidence,
    ) == ("camera_denied_despite_fresh_read",)


def test_epistemic_uncertainty_does_not_count_as_sample_denial() -> None:
    evidence = _camera_evidence()
    reply = (
        "I looked and do not see anyone else in the camera's current view, "
        "but that view cannot establish that the whole room is empty."
    )

    assert sensory_evidence_contradictions(reply, evidence) == ()


@pytest.mark.parametrize(
    "reply",
    [
        "No one else is here with you. The room seems empty.",
        "No, you're alone.",
        "There's no one else here with you. The room is empty except for you and me.",
    ],
)
def test_partial_camera_view_rejects_unbounded_absence_claims(reply: str) -> None:
    assert "camera_scope_overclaim" in sensory_evidence_contradictions(
        reply,
        _camera_evidence(),
    )


def test_failed_read_rejects_a_claimed_current_observation() -> None:
    evidence = _camera_evidence(ok=False)

    assert sensory_evidence_contradictions(
        "I can see that nobody else is there.",
        evidence,
    ) == ("camera_observation_claimed_after_failed_read",)
    assert sensory_evidence_contradictions(
        "I can hear someone speaking through the microphone.",
        evidence,
    ) == ()


def test_malformed_or_success_without_observation_is_not_admitted() -> None:
    assert TurnSensoryEvidence.from_value({"channel": "camera", "ok": True}) is None
    assert sensory_evidence_grounding_block({"channel": "unknown", "ok": True}) == ""


@pytest.mark.asyncio
async def test_runtime_contradiction_reanswer_receives_exact_turn_sight(monkeypatch) -> None:
    from core.self import capability_ledger
    from interface.routes import chat

    class _Ledger:
        @staticmethod
        def contradicted_claims(_text):
            return []

    captured = {}

    async def _reanswer(effective_user_message, **kwargs):
        captured["effective_user_message"] = effective_user_message
        captured.update(kwargs)
        return (
            "I looked just now. I do not see another person in the current camera "
            "view, but that view cannot establish that the whole room is empty."
        )

    monkeypatch.setattr(capability_ledger, "get_capability_ledger", lambda: _Ledger())
    monkeypatch.setattr(chat, "_run_cognitive_engine_chat_turn", _reanswer)
    evidence = _camera_evidence()

    reply = await chat._reanswer_when_the_runtime_contradicts_her(
        "My camera feed has never produced a sample, so I cannot tell.",
        user_message="Can you determine whether anyone else is physically here with me?",
        turn_sensory_evidence=evidence,
    )

    assert reply.startswith("I looked just now")
    assert "[FRESH TURN SENSORY EVIDENCE]" in captured["effective_user_message"]
    assert "I do not see another person" in captured["effective_user_message"]
    assert captured["turn_sensory_evidence"] == evidence


@pytest.mark.asyncio
async def test_runtime_reanswer_passes_computed_arithmetic_evidence(monkeypatch) -> None:
    from core.self import capability_ledger
    from interface.routes import chat

    class _Ledger:
        @staticmethod
        def contradicted_claims(_text):
            return []

    captured = {}

    async def _reanswer(effective_user_message, **_kwargs):
        captured["effective_user_message"] = effective_user_message
        return "4"

    monkeypatch.setattr(capability_ledger, "get_capability_ledger", lambda: _Ledger())
    monkeypatch.setattr(chat, "_run_cognitive_engine_chat_turn", _reanswer)

    reply = await chat._reanswer_when_the_runtime_contradicts_her(
        "5",
        user_message="What is 2 + 2? Just the number.",
    )

    # This used to assert the runtime wrote "computed it directly: 4" into the
    # turn context and that the model then answered 4. That is instruction
    # prose steering a sample, and it is unreliable: the same technique applied
    # to a file count restated the wrong number three times while the right one
    # sat in the context.
    #
    # A computed value is not a matter of opinion, so it is served. The model
    # is not consulted, which is why nothing was captured.
    assert reply == "4"
    assert "effective_user_message" not in captured
