"""Concept formation: repeated similar prediction errors abstract into a named primitive."""
from __future__ import annotations

import pytest

from core.cognition.concept_formation import (
    ConceptFormationEngine,
    get_concept_formation_engine,
)


@pytest.fixture
def engine(tmp_path):
    return ConceptFormationEngine(storage_path=tmp_path / "concepts.json", autosave=False,
                                  min_support=3, error_threshold=0.5, similarity_threshold=0.5)


# ── formation from repeated similar surprise ─────────────────────────────────

def test_repeated_similar_error_forms_concept(engine):
    sig = ["heat", "metal", "expands"]
    result = None
    for _ in range(3):
        result = engine.observe_prediction_error(sig, magnitude=0.8)
    assert result.formed is not None
    assert "heat" in "+".join(result.formed.defining_features)
    assert engine.get_health()["concepts"] == 1


def test_single_error_does_not_form_concept(engine):
    result = engine.observe_prediction_error(["a", "b", "c"], magnitude=0.9)
    assert result.formed is None
    assert result.reason == "accumulating"


def test_low_error_events_never_form_concept(engine):
    for _ in range(6):
        r = engine.observe_prediction_error(["x", "y", "z"], magnitude=0.2)
    assert r.formed is None
    assert r.reason == "below_error_threshold"
    assert engine.get_health()["concepts"] == 0


def test_dissimilar_errors_do_not_merge(engine):
    for _ in range(2):
        engine.observe_prediction_error(["heat", "metal", "expands"], magnitude=0.8)
    r = engine.observe_prediction_error(["cold", "water", "freezes"], magnitude=0.8)
    # Different signature → its own cluster, not enough support yet → no concept.
    assert r.formed is None
    assert engine.get_health()["concepts"] == 0


# ── recognition: the closed loop ─────────────────────────────────────────────

def test_formed_concept_recognizes_future_instances(engine):
    sig = ["pressure", "volume", "inverse"]
    for _ in range(3):
        engine.observe_prediction_error(sig, magnitude=0.8)
    # The concept now explains a matching event rather than treating it as novel.
    follow_up = engine.observe_prediction_error(["pressure", "volume", "inverse"], magnitude=0.9)
    assert follow_up.recognized is not None
    assert follow_up.formed is None


def test_recognize_returns_concept_for_matching_signature(engine):
    for _ in range(3):
        engine.observe_prediction_error(["gravity", "mass", "attracts"], magnitude=0.8)
    c = engine.recognize(["gravity", "mass", "attracts"])
    assert c is not None and "gravity" in "+".join(c.defining_features)
    assert engine.recognize(["unrelated", "tokens", "here"]) is None


def test_repeated_recognition_does_not_certify_a_concept(engine):
    sig = ["light", "speed", "constant"]
    for _ in range(3):
        engine.observe_prediction_error(sig, magnitude=0.8)
    for _ in range(3):
        engine.observe_prediction_error(sig, magnitude=0.8)
    c = engine.recognize(sig)
    assert c.status == "provisional"
    assert c.validation_provenance == ""


def test_independent_discovery_certifies_matching_concept(engine):
    from core.brain.ontology_discovery import Observation, OntologyDiscovery

    for _ in range(3):
        result = engine.observe_prediction_error(["light"], magnitude=0.8)
    concept_id = result.formed.concept_id
    groups = []
    for group in range(3):
        groups.append(tuple(
            Observation({"cue:light": index % 2 == 0}, index % 2 == 0,
                        source_id=f"{group}-{index}")
            for index in range(40)
        ))
    discovered = OntologyDiscovery(outcome_name="surprise", min_support=4).discover_partitioned(
        *groups
    ).discovered
    assert discovered is not None
    concept = engine.certify(concept_id, discovered)
    assert concept.status == "consolidated"
    assert concept.validation_provenance == discovered.provenance()


def test_counterevidence_reopens_a_certified_concept_and_survives_reload(engine, tmp_path):
    from core.cognition.concept_handle import Substrate
    from core.cognition.semantic_development import SemanticCase, SemanticDevelopment

    for _ in range(3):
        engine.observe_prediction_error(["light"], magnitude=0.8)
    cohorts = tuple(tuple(SemanticCase(
        f"{group}-{index}", f"context-{group}", "surprise", index % 2 == 0,
        {"cue:light": index % 2 == 0}) for index in range(40)) for group in range(3))
    path = tmp_path / "semantic.json"
    semantic = SemanticDevelopment(concept_engine=engine, state_path=path, min_support=4)
    primitive_id = semantic.test("surprise", cohorts=cohorts)["primitive"]["identity"]
    assert engine.concepts()[0]["status"] == "consolidated"
    fresh = tuple(SemanticCase(
        f"later-{index}", "later", "surprise", index % 2 != 0,
        {"cue:light": index % 2 == 0}) for index in range(40))
    assert semantic.retest(primitive_id, fresh)["status"] == "revoked"
    assert engine.concepts()[0]["status"] == "provisional"
    assert engine.concepts()[0]["validation_revoked_by"]
    semantic.save()
    restored = SemanticDevelopment(concept_engine=engine, state_path=path, min_support=4)
    assert restored.registry.resolve(Substrate.FORMED, primitive_id) is None


def test_prior_unverified_consolidation_is_reopened(tmp_path):
    path = tmp_path / "concepts.json"
    path.write_text('{"concepts":[{"concept_id":"old","name":"old",'
                    '"defining_features":["old"],"support":3,"status":"consolidated"}]}')
    engine = ConceptFormationEngine(storage_path=path, autosave=False)
    assert engine.recognize(["old"]).status == "provisional"


# ── bounded clusters + retrieval + persistence ──────────────────────────────

def test_open_clusters_are_bounded(tmp_path):
    eng = ConceptFormationEngine(storage_path=tmp_path / "c.json", autosave=False,
                                 max_clusters=5, min_support=10)
    for i in range(40):
        eng.observe_prediction_error([f"feat_{i}_a", f"feat_{i}_b"], magnitude=0.8)
    assert eng.get_health()["open_clusters"] <= 5


def test_retrieve_returns_relevant_concepts(engine):
    for _ in range(3):
        engine.observe_prediction_error(["thermal", "conductivity", "metal"], magnitude=0.8)
    hits = engine.retrieve("thermal metal", limit=5)
    assert hits and "concept" in hits[0]["content"].lower()


def test_concepts_persist(tmp_path):
    path = tmp_path / "c.json"
    a = ConceptFormationEngine(storage_path=path, autosave=False, min_support=3)
    for _ in range(3):
        a.observe_prediction_error(["entropy", "increases", "always"], magnitude=0.8)
    a.save()
    b = ConceptFormationEngine(storage_path=path, autosave=False)
    assert b.get_health()["concepts"] == 1
    assert b.recognize(["entropy", "increases", "always"]) is not None


# ── singleton ────────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_concept_formation_engine() is get_concept_formation_engine()
