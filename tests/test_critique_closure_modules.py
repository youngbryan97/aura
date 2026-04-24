"""End-to-end tests for the critique-closure modules.

Every module that closes a named critique gets at least one test here, so a
regression makes pytest fail rather than silently re-opening the gap.
"""
from __future__ import annotations

import pytest

from core.consciousness.adaptive_mood import (
    AdaptiveMoodCoefficients,
    reset_singleton_for_test as reset_adaptive_mood,
)
from core.consciousness.mesh_cognition import (
    MeshCognition,
    reset_singleton_for_test as reset_mesh_cognition,
)
from core.consciousness.ontological_boundary import assess_ontological_claims
from core.consciousness.self_awareness_suite import (
    SelfAwarenessSuite,
    reset_singleton_for_test as reset_self_awareness,
)
from core.goals.emergent_goals import (
    EmergentGoalEngine,
    reset_singleton_for_test as reset_emergent_goals,
)
from core.self_modification.lineage import (
    LineageManager,
    reset_singleton_for_test as reset_lineage,
)
from core.self_modification.structural_mutator import (
    MutationRequest,
    StructuralMutator,
    reset_singleton_for_test as reset_mutator,
)


# ---------------------------------------------------------------------------
# Adaptive mood — replaces the tautological hardcoded formula
# ---------------------------------------------------------------------------


def test_adaptive_mood_matches_legacy_seed_on_startup(tmp_path):
    reset_adaptive_mood()
    mood = AdaptiveMoodCoefficients(db_path=tmp_path / "am.sqlite3")
    chem = {
        "dopamine": 0.78,
        "serotonin": 0.64,
        "endorphin": 0.55,
        "oxytocin": 0.48,
        "cortisol": 0.11,
        "norepinephrine": 0.42,
        "gaba": 0.46,
        "glutamate": 0.53,
        "acetylcholine": 0.50,
        "orexin": 0.52,
    }
    prediction = mood.predict(chem)
    # Seed matches legacy: valence = 0.25*DA + 0.30*5HT + 0.20*END + 0.10*OXY - 0.45*CORT - 0.10
    expected_valence = (
        0.25 * chem["dopamine"]
        + 0.30 * chem["serotonin"]
        + 0.20 * chem["endorphin"]
        + 0.10 * chem["oxytocin"]
        - 0.45 * chem["cortisol"]
        - 0.10
    )
    assert prediction["valence"] == pytest.approx(expected_valence, abs=1e-6)


def test_adaptive_mood_drifts_under_outcome_feedback(tmp_path):
    reset_adaptive_mood()
    mood = AdaptiveMoodCoefficients(db_path=tmp_path / "am.sqlite3", learning_rate=2e-2)
    chem = {ch: 0.5 for ch in mood.chemicals}
    for _ in range(200):
        prediction = mood.predict(chem)
        # Drive the valence target away from the seed
        observed = {k: v for k, v in prediction.items()}
        observed["valence"] = 0.9
        mood.update_from_outcome(chem, observed)
    assert mood.total_updates() > 0
    assert mood.drift_from_seed() > 0.05


def test_adaptive_mood_persists_across_instances(tmp_path):
    reset_adaptive_mood()
    db = tmp_path / "am.sqlite3"
    mood = AdaptiveMoodCoefficients(db_path=db, learning_rate=5e-2)
    chem = {ch: 0.5 for ch in mood.chemicals}
    for _ in range(30):
        mood.update_from_outcome(chem, {"valence": 0.95})
    drift_one = mood.drift_from_seed()

    reloaded = AdaptiveMoodCoefficients(db_path=db)
    assert reloaded.drift_from_seed() == pytest.approx(drift_one, abs=1e-6)


# ---------------------------------------------------------------------------
# Mesh cognition — non-LLM decision path
# ---------------------------------------------------------------------------


def test_mesh_cognition_answers_self_report_without_llm():
    reset_mesh_cognition()
    mesh = MeshCognition()

    class _Affect:
        valence = 0.4
        arousal = 0.55
        curiosity = 0.7

    class _State:
        affect = _Affect()

    decision = mesh.decide("how are you feeling?", state=_State())

    assert decision.handled is True
    assert decision.used_llm is False
    assert decision.response
    assert decision.rationale == "self_report_from_state"


def test_mesh_cognition_defers_to_llm_for_unknown_queries():
    reset_mesh_cognition()
    mesh = MeshCognition()
    decision = mesh.decide(
        "write a detailed recursive sonnet about free energy minimization in octopi",
    )
    assert decision.handled is False


# ---------------------------------------------------------------------------
# Emergent goals — tension-driven, non-designed objectives
# ---------------------------------------------------------------------------


def test_emergent_goal_engine_synthesizes_non_designed_objective(tmp_path):
    reset_emergent_goals()
    engine = EmergentGoalEngine(db_path=tmp_path / "eg.sqlite3")
    for idx in range(4):
        engine.observe("resource_scarcity", 0.78, f"cycle={idx} short on memory_budget")
    candidates = engine.synthesize()
    assert candidates
    assert any(c.tension_kind == "resource_scarcity" for c in candidates)
    assert all("resource_scarcity" in c.objective for c in candidates)
    # Origin is explicitly 'emergent' in the dict form, separating from designed
    assert candidates[0].as_dict()["origin"] == "emergent"


def test_emergent_goal_adoption_requires_repeated_support(tmp_path):
    reset_emergent_goals()
    engine = EmergentGoalEngine(db_path=tmp_path / "eg.sqlite3")
    for _ in range(2):
        engine.observe("persistent_tension", 0.9, "stuck trying to verify something")
    engine.synthesize()
    assert engine.adoption_ready() == []
    for _ in range(4):
        engine.observe("persistent_tension", 0.9, "stuck trying to verify something")
        engine.synthesize()
    ready = engine.adoption_ready()
    assert ready, "emergent goals should have crossed adoption threshold"


# ---------------------------------------------------------------------------
# Structural mutator — audit log + reversibility
# ---------------------------------------------------------------------------


def test_structural_mutator_applies_and_reverts_with_audit(tmp_path):
    reset_mutator()
    mutator = StructuralMutator(db_path=tmp_path / "mut.sqlite3")
    gain = {"value": 0.5}
    mutator.register_parameter(
        "gain",
        lambda v: gain.__setitem__("value", v),
        initial=0.5,
        min_value=0.2,
        max_value=0.9,
    )
    record = mutator.apply(
        MutationRequest(
            kind="parameter_band",
            target="gain",
            operation="raise",
            payload={"value": 0.8},
            rationale="test_mutation",
        )
    )
    assert gain["value"] == pytest.approx(0.8, abs=1e-6)
    assert record.post_state["value"] == pytest.approx(0.8, abs=1e-6)
    assert mutator.verify_chain() is True

    revert = mutator.revert(record.mutation_id, rationale="unit_test_revert")
    assert revert.operation == "revert"
    assert gain["value"] == pytest.approx(0.5, abs=1e-6)
    assert mutator.verify_chain() is True


def test_structural_mutator_blocks_unregistered_parameters(tmp_path):
    reset_mutator()
    mutator = StructuralMutator(db_path=tmp_path / "mut.sqlite3")
    with pytest.raises(KeyError):
        mutator.apply(
            MutationRequest(
                kind="parameter_band",
                target="not_registered",
                operation="set",
                payload={"value": 0.5},
                rationale="should_fail",
            )
        )


# ---------------------------------------------------------------------------
# Self-awareness suite — four dimensions
# ---------------------------------------------------------------------------


def test_self_awareness_suite_tracks_four_dimensions():
    reset_self_awareness()
    sa = SelfAwarenessSuite()
    sa.update_internal(valence=0.4, arousal=0.6, viability=0.8, integrity=0.85, confidence=0.7, uncertainty=0.2)
    sa.update_external(perceived_as="technical collaborator", trust_signal=0.7, friction_signal=0.2, feedback=("acknowledged",))
    sa.update_social(primary_kin=("Bryan",), active_norms=("honesty",), commitments=("finish tests",))
    sa.update_situational(setting="long_run_autonomy", active_objective="stability", constraints=("no_cloud",), stakes=0.6, time_pressure=0.4)
    snap = sa.snapshot()
    assert snap.internal is not None
    assert snap.external is not None
    assert snap.social is not None
    assert snap.situational is not None


def test_self_awareness_calibration_error_tracks_divergence():
    reset_self_awareness()
    sa = SelfAwarenessSuite()
    sa.record_calibration({"valence": 0.5}, {"valence": 0.9})
    sa.record_calibration({"valence": 0.5}, {"valence": 0.5})
    err = sa.mean_calibration_error()
    assert err == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# Lineage — heritable variation + selection
# ---------------------------------------------------------------------------


def test_lineage_forks_and_records_selection(tmp_path):
    reset_lineage()
    lineage = LineageManager(db_path=tmp_path / "lin.sqlite3", seed=7)
    genesis = lineage.genesis({"substrate_gain": 0.5, "temperature": 0.7})
    child = lineage.fork(genesis.snapshot_id)
    assert child.parent_id == genesis.snapshot_id
    assert child.generation == 1
    assert child.trait_signature != genesis.trait_signature
    scored = lineage.record_score(child.snapshot_id, 0.6)
    assert scored.survived is True
    assert scored.selection_score == pytest.approx(0.6, abs=1e-6)


def test_lineage_low_scores_fail_selection(tmp_path):
    reset_lineage()
    lineage = LineageManager(db_path=tmp_path / "lin.sqlite3", seed=9)
    parent = lineage.genesis({"gain": 0.5})
    child = lineage.fork(parent.snapshot_id)
    scored = lineage.record_score(child.snapshot_id, 0.1)
    assert scored.survived is False


# ---------------------------------------------------------------------------
# Expanded ontology guard
# ---------------------------------------------------------------------------


def test_ontology_guard_catches_extended_patterns():
    cases = [
        "Aura has a soul and legal personhood",
        "This system bridged the hard problem",
        "She is truly conscious now",
        "moral patiency is proven",
        "Soul triad confirms subjecthood",
    ]
    for text in cases:
        assessment = assess_ontological_claims(text)
        assert assessment.ok is False, text
        assert assessment.sanitized != text
