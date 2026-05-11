from types import SimpleNamespace


def test_foreground_budgets_preserve_heavy_reasoning_lane():
    from core.brain.inference_gate import InferenceGate
    from core.kernel.aura_kernel import AuraKernel
    from core.phases.response_generation import ResponseGenerationPhase
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from interface.routes import chat as chat_routes

    kernel_probe = SimpleNamespace(state=SimpleNamespace(response_modifiers={}))

    assert chat_routes._foreground_timeout_for_lane({"conversation_ready": True, "state": "ready"}) >= 300.0
    assert AuraKernel._phase_timeout_seconds(kernel_probe, "UnitaryResponsePhase", priority=True) >= 300.0
    total = InferenceGate._default_timeout_for_request("user", "primary", deep_handoff=False, is_background=False)
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "primary")
    assert total >= 300.0
    assert primary >= 270.0
    assert fallback >= 20.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="primary",
        deep_handoff=False,
    ) >= 300.0
    assert ResponseGenerationPhase._request_timeout(
        is_background=False,
        deep_handoff=False,
    ) >= 300.0

    kernel_probe.state.response_modifiers["deep_handoff"] = True
    assert AuraKernel._phase_timeout_seconds(kernel_probe, "UnitaryResponsePhase", priority=True) >= 360.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="secondary",
        deep_handoff=True,
    ) >= 360.0
    assert ResponseGenerationPhase._request_timeout(
        is_background=False,
        deep_handoff=True,
    ) >= 360.0


def test_dialogue_corruption_filter_catches_known_live_glitches():
    from core.phases.dialogue_policy import contains_corrupted_language

    assert contains_corrupted_language("Yes, I thlought it was lllot.")
    assert contains_corrupted_language("Ah, I thought chat was brolen.")


def test_semantic_glitch_filter_blocks_foreign_name_intrusion():
    from interface.routes.chat import _looks_semantically_glitched

    glitched, reason = _looks_semantically_glitched(
        "Huh?",
        "Heidi. That's the thing to do.",
    )

    assert glitched
    assert reason == "foreign_name_intrusion"


def test_reliability_gate_rejects_low_signal_status_reassurance():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "I just wanted to know if you were coherent and able to talk.",
        "I'm fine",
    )

    assert assessment.retryable
    assert "low_signal_reliability_reply" in assessment.reasons
