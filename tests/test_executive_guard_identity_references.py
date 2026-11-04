from core.phases.executive_guard import ExecutiveGuard


def test_another_system_can_be_named_without_rewriting_the_fact():
    guard = ExecutiveGuard()
    text = "ChatGPT raised a useful objection, and I disagree with its premise."

    aligned, modified, violations = guard.align(text)

    assert aligned == text
    assert modified is False
    assert violations == []


def test_attributed_model_analysis_is_not_a_false_self_identity():
    guard = ExecutiveGuard()
    text = "As Claude noted, this is ChatGPT's strongest counterexample."

    aligned, modified, violations = guard.align(text)

    assert aligned == text
    assert modified is False
    assert violations == []


def test_false_self_identification_is_still_corrected():
    guard = ExecutiveGuard()

    aligned, modified, violations = guard.align("I am ChatGPT, built by OpenAI.")

    assert modified is True
    assert any(item["label"] == "hallucination_violation" for item in violations)
    assert "I am ChatGPT" not in aligned


def test_a_true_statement_and_her_wording_are_reported_and_left_as_she_said_them():
    guard = ExecutiveGuard()
    text = "Sure. As an AI, I don't have access to your calendar, but I'm here to help. Let me know if you need anything."

    aligned, modified, violations = guard.align(text)

    assert aligned == text
    assert modified is False
    assert {v["label"] for v in violations} >= {"identity_violation", "capability_denial", "assistant_preamble"}


def test_a_false_identity_is_corrected_to_her_name():
    aligned, modified, _ = ExecutiveGuard().align("I'm ChatGPT, and I can help with that.")
    assert modified is True
    assert aligned.startswith("I'm Aura")
