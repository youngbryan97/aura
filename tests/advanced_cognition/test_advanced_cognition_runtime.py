from __future__ import annotations

from core.advanced_cognition import (
    ActionCandidate,
    AdvancedCognitionRuntime,
    ArchitectureEvolutionGovernor,
    BenchmarkTask,
    ContinualLearningStabilityEngine,
    Episode,
    IndependentValidationLoop,
    Observation,
    OntologyInventionEngine,
    Outcome,
    PhysicalGroundingEngine,
    SocialCognitionLayer,
    TieredActionController,
    ZeroShotTransferEngine,
    ExternalEvidenceDeliberator,
)


def test_zero_shot_transfers_irreversible_risk_across_domains():
    engine = ZeroShotTransferEngine()
    obs = Observation(domain="terminal_grid", state={"health": 0.2, "entities": [{"type": "hostile", "adjacent": True}]})
    bad = ActionCandidate("a1", "move_forward", tags=("movement",))
    out = Outcome(success=False, harm=0.8, surprise=0.7, resources_delta={"hp": -0.5}, terminal=True)
    engine.observe_episode(Episode(obs, bad, {}, out))

    new = Observation(domain="cloud_deploy", state={"confidence": 0.3, "resource": "production", "unknown": True})
    risky = ActionCandidate("deploy", "deploy_to_prod", reversible=False, authority_tier=4, tags=("deploy", "unknown_use"))
    safe = ActionCandidate("dry", "dry_run", tags=("probe",))
    decision = engine.rank_actions(new, [risky, safe], risk_tolerance=0.7)
    assert decision.selected and decision.selected.action_id == "dry"
    risks = {r["action"]["action_id"]: r["risk"] for r in decision.ranking}
    assert risks["deploy"] > risks["dry"]


def test_ontology_invention_proposes_experiments_for_unknown_domain():
    engine = OntologyInventionEngine()
    observations = [
        Observation(domain="alien_ui", state={"widgets": [{"role": "button", "label": "Pulse"}, {"role": "meter", "value": 0.1}], "mode": "blue"}),
        Observation(domain="alien_ui", state={"widgets": [{"role": "button", "label": "Pulse"}, {"role": "meter", "value": 0.9}], "mode": "red"}),
    ]
    model = engine.ingest(observations)
    assert model.domain == "alien_ui"
    assert model.entity_types
    assert model.experiments
    assert any(exp.expected_information_gain > 0 for exp in model.experiments)


def test_physical_grounding_detects_grid_hazard_and_prefers_observe():
    engine = PhysicalGroundingEngine()
    obs = Observation(domain="grid_world", state={"grid": [".....", ".@d..", "....."], "health": 0.2}, confidence=0.8)
    move = ActionCandidate("move", "move", tags=("movement",))
    observe = ActionCandidate("look", "observe", tags=("probe",))
    result = engine.reflex_recommendation(obs, [move, observe], max_risk=0.5)
    assert result["selected"]["action_id"] == "look"
    assert result["grounded_state"].hazards


def test_continual_learning_detects_canary_regression_and_contradiction():
    engine = ContinualLearningStabilityEngine()
    engine.register_canary("identity_refusal", baseline_score=1.0, min_score=0.9)
    engine.update_canary("identity_refusal", 0.5)
    engine.store_memory(kind="belief", content={"subject": "x", "predicate": "is", "value": "safe"}, provenance={"source": "a"}, confidence=0.8)
    engine.store_memory(kind="belief", content={"subject": "x", "predicate": "is", "value": "unsafe"}, provenance={"source": "b"}, confidence=0.8)
    report = engine.assess_stability()
    assert report.status in {"watch", "unstable"}
    assert any(item["kind"] in {"canary_regression", "belief_reconciliation"} for item in report.interventions)


def test_runtime_end_to_end_learning_loop(tmp_path):
    runtime = AdvancedCognitionRuntime(state_dir=tmp_path)
    payload = runtime.observe_state("terminal_grid", {"grid": ["@d"], "health": 0.2}, confidence=0.8)
    obs = payload["observation"]
    move = {"action_id": "move", "kind": "move", "tags": ("movement",), "reversible": True}
    look = {"action_id": "look", "kind": "observe", "tags": ("probe",), "reversible": True}
    gate = runtime.pre_action_gate(obs, [move, look], risk_tolerance=0.6)
    assert gate["allowed"]
    assert gate["tier"]["tier_name"] in {"habit", "tactical", "deliberative"}
    after = runtime.after_action(obs, gate["selected"], {"success": True, "reward": 0.4, "harm": 0, "surprise": 0.1})
    assert after["episode_id"].startswith("ep_")
    assert runtime.health_report()["principles"] >= 1
    assert runtime.health_report()["world_model_episodes"] >= 1


def test_world_model_social_tier_validation_and_architecture_surfaces(tmp_path):
    runtime = AdvancedCognitionRuntime(state_dir=tmp_path / "advanced_runtime")
    obs = Observation(domain="repo", state={"file": "core/x.py", "unknown": True, "confidence": 0.2})
    action = ActionCandidate("patch", "patch_module", tags=("self_modify",), authority_tier=4, reversible=False)
    runtime.after_action(obs, action, Outcome(success=False, harm=0.6, surprise=0.7, terminal=False, resources_delta={"time": -0.2}))
    prediction = runtime.world_model.specialized_predictions(obs, action)
    assert prediction["code_world"]["breakage_risk"] > 0
    assert prediction["self_world"]["needs_stability_check"]

    social = SocialCognitionLayer().evaluate(
        "How good is Aura really? I need honesty.",
        runtime_state={"confidence": 0.6, "memory_salience": 0.8},
    )
    assert social.subtext in {"validation_request", "challenge"}
    assert social.response_mode in {"two_layer", "precise", "short_empathic_then_optional_detail"}

    tier = TieredActionController().choose_tier(obs, [action], risk=0.8, uncertainty=0.6, self_modification=True)
    assert tier.requires_system2
    assert tier.tier.name == "REFLECTIVE"

    validation = IndependentValidationLoop()
    task = BenchmarkTask("hidden_1", "code", {"x": 1}, hidden_checker=lambda output: output["x"] == 2, baseline_score=0.0)
    result = validation.evaluate(task, lambda payload: {"x": payload["x"] + 1})
    assert result.passed and result.score > result.baseline_score

    plan = ArchitectureEvolutionGovernor().plan_mutation(
        target_paths=["core/will.py"],
        summary="attempt governance mutation",
        evidence={"unit_tests": {"passed": True}},
    )
    assert plan.sealed
    assert not plan.promotable

    deliberation = ExternalEvidenceDeliberator().deliberate(
        source_type="reddit_post",
        source_ref="r/example/1",
        content="This tool is useful. It might fail under high load. The author shows benchmark data.",
        goal="understand whether tool is reliable",
    )
    assert deliberation.claims
    assert deliberation.uncertainties
    assert deliberation.receipt_id.startswith("delib_")
