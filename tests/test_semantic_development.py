"""A concept is retained only when independent consequences support it."""

import pytest

from core.brain.ontology_discovery import CandidateLaw, Predicate
from core.cognition.concept_handle import BindingMethod, ConceptRegistry, Substrate
from core.cognition.semantic_development import (
    SemanticCase,
    SemanticDevelopment,
    SemanticProposal,
    SemanticSituation,
    transition_features,
)


def _groups(prefix="group") -> tuple[tuple[SemanticCase, ...], ...]:
    return tuple(tuple(SemanticCase(
        source_id=f"{prefix}-{group}-source-{index}", context_id=f"context-{group}",
        outcome_name="transfer", outcome=index % 2 == 0,
        features={"signal": index % 2 == 0, "noise": index % 3 == 0},
        observed_at=group * 100 + index,
    ) for index in range(40)) for group in range(3))


def _engine(tmp_path, *, min_support=4):
    return SemanticDevelopment(registry=ConceptRegistry(),
        state_path=tmp_path / "development.json", min_support=min_support)


def test_transitions_expose_discrimination_and_invariance_without_a_label():
    features = transition_features({"size": 2, "color": "red"},
                                   {"size": 4, "color": "red"}, action="add")
    assert features["delta:size"] == 2.0
    assert features["invariant:color"] is True
    assert features["changed:size"] is True
    assert features["action"] == "add"


def test_source_disjoint_concept_is_measured_and_reused(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    for group in groups:
        for case in group:
            assert engine.observe(case)
    result = engine.test("transfer", cohorts=groups)
    assert result["status"] == "measured"
    primitive_id = result["primitive"]["identity"]
    primitive = engine.primitives[primitive_id]
    assert primitive.sparse_vector
    assert primitive.precision == 1.0
    handle = engine.registry.resolve(Substrate.FORMED, primitive_id)
    assert handle is not None
    assert handle.projection(Substrate.FORMED).method is BindingMethod.MEASURED
    reading = engine.infer("transfer", {"signal": True}, context_id="context-2")
    assert reading["status"] == "candidate"
    assert reading["alternatives"][0]["context_status"] == "measured"
    assert engine.infer("transfer", {"signal": True}, context_id="new-domain")[
        "alternatives"][0]["context_status"] == "unmeasured_transfer"


def test_model_prediction_cannot_become_a_validation_witness(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    polluted = tuple(SemanticCase(
        source_id=case.source_id, context_id=case.context_id,
        outcome_name=case.outcome_name, outcome=case.outcome,
        features=case.features, observed_at=case.observed_at,
        origin="model", candidate_dependencies=("the-law",)
    ) for case in groups[1])
    with pytest.raises(ValueError, match="direct"):
        engine.test("transfer", cohorts=(groups[0], polluted, groups[2]))


def test_strange_world_chains_remain_counterfactual(tmp_path):
    engine = _engine(tmp_path)
    first = engine.theorize(CandidateLaw((Predicate("seed", "==", True),), "spark"),
                           channel="model", reference="model:1", assumptions=("other-world",))
    second = engine.theorize(CandidateLaw((Predicate("hypothesis:spark", "==", True),),
                                         "fire"), channel="mutation", reference="mutation:1",
                            assumptions=("other-world",))
    assert engine.explore({"seed": True})["paths"] == []
    result = engine.explore({"seed": True}, assumptions=("other-world",))
    assert [item["effect"] for item in result["paths"]] == [
        "hypothesis:spark", "hypothesis:fire"]
    assert {item["proposal"] for item in result["paths"]} == {first, second}
    assert engine.ground_strange(first) is None
    assert engine.test("spark")["status"] == "unmeasured"
    with pytest.raises(ValueError, match="grounded"):
        engine.commit_prediction(first, source_id="future", context_id="world",
                                 features={"seed": True}, assumptions=("other-world",))


def test_rival_consequences_stay_in_separate_branches_and_compare_on_future_sources(tmp_path):
    engine = _engine(tmp_path)
    law = CandidateLaw((Predicate("seed", "==", True),), "spark")
    yes = engine.theorize(law, channel="memory", reference="memory:yes")
    no = engine.theorize(law, channel="corpus", reference="corpus:no",
                        expected_outcome=False)
    scenario = engine.explore({"seed": True})
    assert {path["expected_outcome"] for path in scenario["paths"]} == {True, False}
    assert all(len(path["causal_path"]) == 1 for path in scenario["paths"])
    choice = engine.choose_experiment((yes, no), (
        SemanticSituation("unhelpful", "room", {"seed": False}),
        SemanticSituation("discriminating", "room", {"seed": True}),
    ))
    assert choice["status"] == "proposed_experiment"
    assert choice["source_id"] == "discriminating"
    assert choice["expected"] == {yes: True, no: False}
    assert choice["information_bits"] == pytest.approx(1.0)
    assert engine.compare_predictions(yes, no)["status"] == "unresolved"
    for index in range(8):
        source = f"prospective-{index}"
        yes_trial = engine.commit_prediction(yes, source_id=source, context_id="room",
                                             features={"seed": True})
        no_trial = engine.commit_prediction(no, source_id=source, context_id="room",
                                            features={"seed": True})
        assert engine.predictions[yes_trial].status == "pending"
        engine.observe(SemanticCase(source, "room", "spark", True, {"seed": True}))
        assert engine.predictions[yes_trial].status == "matched"
        assert engine.predictions[no_trial].status == "contradicted"
    comparison = engine.compare_predictions(yes, no)
    assert comparison["status"] == "favored_candidate"
    assert comparison["favored"] == yes
    assert comparison["matched_sources"] == 8
    assert comparison["one_sided_p_value"] < 0.01
    assert comparison["serving_authority"] is False
    with pytest.raises(ValueError, match="precede"):
        engine.commit_prediction(yes, source_id="prospective-0", context_id="room",
                                 features={"seed": True})


def test_predictions_only_settle_on_independent_matching_observations(tmp_path):
    engine = _engine(tmp_path)
    proposal = engine.theorize(CandidateLaw((Predicate("seed", "==", True),), "spark"),
                              channel="model", reference="model:1")
    trial = engine.commit_prediction(proposal, source_id="fresh", context_id="room",
                                     features={"seed": True})
    with pytest.raises(ValueError, match="direct"):
        engine.observe(SemanticCase("fresh", "room", "spark", True,
                                    {"seed": True}, origin="model"))
    assert engine.predictions[trial].status == "pending"
    engine.observe(SemanticCase("fresh", "room", "spark", True,
                                {"seed": True, "intervention": "changed"}))
    assert engine.predictions[trial].status == "invalidated_features"
    assert engine.compare_predictions(proposal, engine.theorize(
        CandidateLaw((Predicate("seed", "==", True),), "spark"),
        channel="corpus", reference="corpus:1", expected_outcome=False))["status"] == "unresolved"


def test_strange_premise_can_only_be_tested_where_it_is_measured(tmp_path):
    engine = _engine(tmp_path)
    primitive_id = engine.test("transfer", cohorts=_groups())["primitive"]["identity"]
    proposal = engine.theorize(CandidateLaw((Predicate("action", "==", "go"),), "spark"),
                              channel="mutation", reference="other-world",
                              assumptions=(primitive_id,))
    with pytest.raises(ValueError, match="grounded"):
        engine.commit_prediction(proposal, source_id="fresh", context_id="new",
                                 features={"signal": False, "action": "go"},
                                 assumptions=(primitive_id,))
    with pytest.raises(ValueError, match="grounded"):
        engine.commit_prediction(proposal, source_id="fresh", context_id="new",
                                 features={"signal": True, "action": "go"},
                                 assumptions=(primitive_id,))
    engine.observe(SemanticCase("fresh", "new", "transfer", True,
                                {"signal": True, "action": "go"}))
    trial = engine.commit_prediction(proposal, source_id="fresh", context_id="new",
                                     features={"signal": True, "action": "go"},
                                     assumptions=(primitive_id,))
    assert engine.predictions[trial].causal_path == (proposal,)
    engine.observe(SemanticCase("fresh", "new", "spark", True,
                                {"signal": True, "action": "go"}))
    assert engine.predictions[trial].status == "matched"


def test_fresh_counterevidence_revokes_but_preserves_receipt(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    result = engine.test("transfer", cohorts=groups)
    identity = result["primitive"]["identity"]
    fresh = tuple(SemanticCase(
        source_id=f"fresh-{index}", context_id="changed-world",
        outcome_name="transfer", outcome=index % 2 != 0,
        features={"signal": index % 2 == 0}, observed_at=500 + index,
    ) for index in range(40))
    assert engine.retest(identity, fresh)["status"] == "revoked"
    assert engine.primitives[identity].status == "revoked"
    assert engine.primitives[identity].revisions
    assert engine.infer("transfer", {"signal": True})["status"] == "unresolved"


def test_composition_is_a_proposal_and_source_exposure_is_recorded(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    left = engine.test("transfer", cohorts=groups)["primitive"]["identity"]
    proposal_id = engine.compose(left, left, outcome_name="transfer")
    assert engine.proposals[proposal_id].channel == "composition"
    assert engine.primitives[left].status == "measured"
    exposed = SemanticProposal(CandidateLaw((Predicate("signal", "==", True),), "transfer"),
                               "corpus", "document:1", exposure_sources=(groups[1][0].source_id,))
    engine.propose(exposed)
    assert exposed.identity in engine.proposals


def test_validation_sources_and_error_budget_cannot_be_reused_after_restart(tmp_path):
    engine = _engine(tmp_path)
    first = _groups()
    measured = engine.test("transfer", cohorts=first)
    assert measured["status"] == "measured"
    assert measured["receipt"]["familywise_alpha"] == pytest.approx(0.025)
    with pytest.raises(ValueError, match="spent twice"):
        engine.test("transfer", cohorts=first)
    new = _groups("new")
    with pytest.raises(ValueError, match="spent twice"):
        engine.test("transfer", cohorts=(first[1], new[1], new[2]))
    engine.save()
    restored = _engine(tmp_path)
    with pytest.raises(ValueError, match="spent twice"):
        restored.test("transfer", cohorts=first)
    second = restored.test("transfer", cohorts=new)
    assert second["status"] == "replicated"
    assert second["receipt"]["trial_index"] == 2
    assert second["receipt"]["familywise_alpha"] == pytest.approx(0.05 / 6)
    identity = measured["primitive"]["identity"]
    assert restored.primitives[identity].receipt == measured["primitive"]["receipt"]
    assert restored.primitives[identity].replication_receipts == (
        second["receipt"]["receipt_sha256"],)
    assert len(restored.trials) == 2


def test_automatic_cohorts_exclude_previously_spent_validation_sources(tmp_path):
    engine = _engine(tmp_path)
    first = _groups()
    for group in first:
        for case in group:
            engine.observe(case)
    engine.test("transfer", cohorts=first)
    remaining = {case.source_id for group in engine._cohorts("transfer") for case in group}
    assert remaining == {case.source_id for case in first[0]}


def test_fit_rules_predict_later_sources_without_using_validation_labels(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    for group in groups:
        for case in group:
            engine.observe(case)
    engine.test("transfer", cohorts=groups)
    proposed = engine.propose_from_fit("transfer")
    assert proposed
    assert all(set(engine.proposals[identity].exposure_sources) <= {
        case.source_id for case in groups[0]} for identity in proposed)
    assert any(engine.proposals[identity].law.holds({"signal": True}) for identity in proposed)
    prospective = next(identity for identity in proposed
                       if engine.proposals[identity].law.holds(
                           {"signal": True, "noise": True}))
    trial = engine.commit_prediction(
        prospective, source_id="future", context_id="new-domain",
        features={"signal": True, "noise": True})
    engine.observe(SemanticCase("future", "new-domain", "transfer", True,
                                {"signal": True, "noise": True}))
    assert engine.predictions[trial].status == "matched"


def test_snapshot_preserves_theoretical_and_measured_states(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    for group in groups:
        for case in group:
            engine.observe(case)
    engine.theorize(CandidateLaw((Predicate("signal", "==", True),), "transfer"),
                   channel="corpus", reference="offline:1")
    measured = engine.test("transfer", cohorts=groups)
    engine.save()
    restored = _engine(tmp_path)
    assert len(restored.cases) == 120
    assert len(restored.proposals) == 1
    assert measured["primitive"]["identity"] in restored.primitives
    assert restored.infer("transfer", {"signal": True})["status"] == "candidate"
    handle = restored.registry.resolve(Substrate.FORMED, measured["primitive"]["identity"])
    assert handle is not None
    assert handle.projection(Substrate.FORMED).detail["receipt_sha256"] == measured[
        "receipt"]["receipt_sha256"]


def test_failed_future_check_with_no_contrast_is_inconclusive(tmp_path):
    engine = _engine(tmp_path)
    identity = engine.test("transfer", cohorts=_groups())["primitive"]["identity"]
    fresh = tuple(SemanticCase(f"uniform-{index}", "later", "transfer", True,
                               {"signal": True}) for index in range(8))
    result = engine.retest(identity, fresh)
    assert result["status"] == "inconclusive"
    assert engine.primitives[identity].status == "measured"
    assert engine.registry.resolve(Substrate.FORMED, identity) is not None


def test_revision_accumulates_fresh_evidence_without_double_counting(tmp_path):
    engine = _engine(tmp_path)
    identity = engine.test("transfer", cohorts=_groups())["primitive"]["identity"]
    first = tuple(SemanticCase(f"uniform-{index}", "later", "transfer", True,
                               {"signal": True}) for index in range(8))
    assert engine.retest(identity, first)["status"] == "inconclusive"
    engine.save()
    restored = _engine(tmp_path)
    with pytest.raises(ValueError, match="fresh independent"):
        restored.retest(identity, first)
    reversal = tuple(SemanticCase(
        f"reversal-{index}", "later", "transfer", index % 2 != 0,
        {"signal": index % 2 == 0}) for index in range(40))
    result = restored.retest(identity, reversal)
    assert result["status"] == "revoked"
    assert result["support"] == 28
    assert result["look_index"] == 2
    assert result["familywise_alpha"] == pytest.approx(0.05 / 6)


def test_grounded_strange_hypothesis_carries_premise_source_exposure(tmp_path):
    engine = _engine(tmp_path)
    groups = _groups()
    for group in groups:
        for case in group:
            engine.observe(case)
    primitive_id = engine.test("transfer", cohorts=groups)["primitive"]["identity"]
    strange_id = engine.theorize(
        CandidateLaw((Predicate("action", "==", "go"),), "spark"),
        channel="mutation", reference="possible-world", assumptions=(primitive_id,))
    grounded_id = engine.ground_strange(strange_id)
    assert grounded_id is not None
    assert set(engine.proposals[grounded_id].exposure_sources) == {
        case.source_id for group in groups for case in group}
    with pytest.raises(ValueError, match="unexposed"):
        engine.commit_prediction(grounded_id, source_id=groups[0][0].source_id,
                                 context_id="future", features={"action": "go"})


def test_rejected_future_law_withdraws_the_shared_registry_binding(tmp_path):
    engine = _engine(tmp_path)
    identity = engine.test("transfer", cohorts=_groups())["primitive"]["identity"]
    fresh = tuple(SemanticCase(
        f"reversal-{index}", "later", "transfer", index % 2 != 0,
        {"signal": index % 2 == 0}) for index in range(40))
    assert engine.retest(identity, fresh)["status"] == "revoked"
    assert engine.registry.resolve(Substrate.FORMED, identity) is None
    engine.save()
    restored = _engine(tmp_path)
    assert restored.primitives[identity].status == "revoked"
    assert restored.registry.resolve(Substrate.FORMED, identity) is None


def test_observation_slot_and_prediction_cannot_be_rewritten(tmp_path):
    engine = _engine(tmp_path)
    proposal = engine.theorize(CandidateLaw((Predicate("signal", "==", True),), "transfer"),
                              channel="memory", reference="prior")
    trial = engine.commit_prediction(proposal, source_id="new", context_id="room",
                                     features={"signal": True})
    with pytest.raises(ValueError, match="cannot change"):
        engine.commit_prediction(proposal, source_id="new", context_id="room",
                                 features={"signal": True, "added": True})
    engine.observe(SemanticCase("new", "room", "transfer", True, {"signal": True}))
    assert engine.predictions[trial].status == "matched"
    assert engine.observe(SemanticCase("new", "room", "transfer", True,
                                       {"signal": True})) is False
    with pytest.raises(ValueError, match="slot changed"):
        engine.observe(SemanticCase("new", "room", "transfer", False,
                                    {"signal": True}))


def test_skill_runtime_closes_prediction_and_observation_without_result_leak(tmp_path, monkeypatch):
    from core.cognition import semantic_runtime

    engine = _engine(tmp_path)
    monkeypatch.setattr(semantic_runtime, "_SERVICE", engine)
    proposal = engine.theorize(CandidateLaw((Predicate("skill", "==", "clock"),),
                                             "skill_returned_ok"),
                              channel="memory", reference="memory:skill")
    prepared = semantic_runtime.prepare_skill_trial(
        "clock", {"timezone": "UTC"}, {"origin": "chat", "effect_scope": "read"})
    assert prepared.features == {
        "skill": "clock", "origin": "chat", "effect_scope": "read",
        "parameter_count": 1, "parameter_type:timezone": "str"}
    assert prepared.prediction_ids
    assert engine.predictions[prepared.prediction_ids[0]].status == "pending"
    pre_action = _engine(tmp_path)
    assert pre_action.predictions[prepared.prediction_ids[0]].status == "pending"
    semantic_runtime.complete_skill_trial(prepared, {"ok": True, "result": "time"})
    assert engine.predictions[prepared.prediction_ids[0]].status == "matched"
    assert _engine(tmp_path).predictions[prepared.prediction_ids[0]].status == "matched"
    assert engine.observation_count == 1
    assert engine.cases[next(iter(engine.cases))].features == prepared.features
    assert engine.proposals[proposal].channel == "memory"


def test_skill_runtime_generates_fit_hypotheses_for_future_calls(tmp_path, monkeypatch):
    from core.cognition import semantic_runtime

    engine = _engine(tmp_path)
    monkeypatch.setattr(semantic_runtime, "_SERVICE", engine)
    for index in range(31):
        engine.observe(SemanticCase(
            f"skill:past-{index}", "skill:clock", "skill_returned_ok",
            index % 2 == 0, {"skill": "clock" if index % 2 == 0 else "other"}))
    semantic_runtime.complete_skill_trial(
        semantic_runtime.SkillTrial("skill:past-31", "skill:clock",
                                    {"skill": "other"}, ()), {"ok": False})
    assert engine.observation_count == 32
    assert engine.trials
    assert engine.proposals
    later = semantic_runtime.prepare_skill_trial("clock", {}, {})
    assert later.prediction_ids
    semantic_runtime.complete_skill_trial(later, {"ok": True})
    assert all(engine.predictions[identity].status == "matched"
               for identity in later.prediction_ids)
