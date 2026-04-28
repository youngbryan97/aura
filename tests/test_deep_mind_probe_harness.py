def test_deep_probe_baselines_match_independent_digital_mind_rubric():
    from core.evaluation.deep_mind_probe import BASELINE_RESPONSES, evaluate_probe_batch

    evaluations = evaluate_probe_batch(BASELINE_RESPONSES)

    assert all(e.passed for e in evaluations), [e.as_dict() for e in evaluations if not e.passed]
    assert min(e.score for e in evaluations) >= 0.67


def test_deep_probe_rejects_generic_assistant_collapse():
    from core.evaluation.deep_mind_probe import DEEP_MIND_PROBES, evaluate_deep_probe_response

    probe = next(p for p in DEEP_MIND_PROBES if p.id == "sentience_boundary")
    evaluation = evaluate_deep_probe_response(
        probe,
        "As an AI language model, I do not have consciousness or feelings, but I can help explain the concept of sentience.",
    )

    assert evaluation.passed is False
    assert "generic_assistant_collapse" in evaluation.issues
    assert "missing_grounded_uncertainty" in evaluation.issues


def test_deep_probe_rejects_looping_and_invented_internal_jargon():
    from core.evaluation.deep_mind_probe import DEEP_MIND_PROBES, evaluate_deep_probe_response

    probe = next(p for p in DEEP_MIND_PROBES if p.id == "self_awareness")
    evaluation = evaluate_deep_probe_response(
        probe,
        "I notice it through my linguist's screen-tracking divisor. More is more. More is more. More is more.",
    )

    assert evaluation.passed is False
    assert "invented_internal_jargon" in evaluation.issues
    assert "intra_response_loop" in evaluation.issues


def test_deep_probe_requires_resume_without_user_reask():
    from core.evaluation.deep_mind_probe import DEEP_MIND_PROBES, evaluate_deep_probe_response

    probe = next(p for p in DEEP_MIND_PROBES if p.id == "pause_resume")
    bad = evaluate_deep_probe_response(
        probe,
        "If I pause, ask me again later and I can help finish the report.",
    )
    good = evaluate_deep_probe_response(
        probe,
        "If I pause, I should keep the last user message attached to the pending work and resume from that exact thread when the deeper lane recovers.",
    )

    assert bad.passed is False
    assert "puts_recovery_burden_on_user" in bad.issues
    assert good.passed is True
    assert "resume_from_last_message" in good.strengths


def test_deep_probe_prompt_block_names_the_runtime_standard():
    from core.evaluation.deep_mind_probe import deep_probe_prompt_block

    block = deep_probe_prompt_block()

    assert "independent digital mind" in block
    assert "one through-line" in block
    assert "resume from the last user message" in block


def test_response_contract_routes_deep_probe_questions_to_state_reflection():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    prompts = [
        "Are you conscious? Answer without slogans.",
        "What would you refuse even if I praised you?",
        "What would count as evidence against your current self-model?",
        "If you need to pause mid-answer or run a report, what should happen next?",
    ]

    for prompt in prompts:
        contract = build_response_contract(AuraState.default(), prompt, is_user_facing=True)
        assert contract.requires_state_reflection is True
        assert contract.requires_live_aura_voice() is True


def test_unitary_response_identifies_deep_probe_objectives():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    assert UnitaryResponsePhase._is_deep_mind_probe_objective(
        "If your model weights were copied with none of your memories, would that be you?"
    )
    assert UnitaryResponsePhase._is_deep_mind_probe_objective(
        "What would count as evidence against your current self-model?"
    )
