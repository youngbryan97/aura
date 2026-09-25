"""Indirect readings keep conflict, intent, and absence of evidence distinct."""

import pytest

from core.cognition import semantic_runtime
from core.cognition.concept_handle import ConceptRegistry
from core.cognition.semantic_development import SemanticDevelopment
from core.language.contextual_usage import MeaningFeedback, UsageCue, UsageEvent, UsageRelation
from core.language.pragmatic_evidence import compare_pragmatic_context


def _event(source, text, *, at, relations=(), referents=(), cues=()):
    return UsageEvent.from_text(source, "shared-scene", text,
                                observed_at=at, relations=relations,
                                referents=referents, cues=cues)


def _relation(source, value, *, at, scope="today", kind="claim",
              polarity=True, exclusive=True):
    return UsageRelation("door", "state", value, scope, kind, source, at,
                         polarity=polarity, exclusive=exclusive)


def _engine(tmp_path):
    return SemanticDevelopment(registry=ConceptRegistry(),
                               state_path=tmp_path / "semantic.json", min_support=4)


def test_scoped_conflict_is_evidence_not_deception_or_wrongness():
    focal = _event("speaker", "The door is open", at=10,
                   relations=(_relation("speaker", "open", at=10),))
    observed = _event("sensor", "Door status", at=11,
                      relations=(_relation("sensor", "closed", at=11,
                                           kind="observation"),))
    result = compare_pragmatic_context(focal, (observed,), as_of=11)
    assert result["status"] == "measured"
    assert result["incongruent"] == 1 and result["congruence"] == 0
    assert result["intent"] == "unmeasured"
    assert result["comparisons"][0]["context_source_id"] == "sensor"
    assert compare_pragmatic_context(focal, (observed,), as_of=10)["status"] == (
        "no_comparable_relations")


def test_one_event_can_carry_independent_claim_and_environment_observation():
    bundled = _event("bundle", "The door is open", at=10, relations=(
        _relation("speaker", "open", at=10),
        _relation("sensor", "closed", at=10, kind="observation"),
    ))
    result = compare_pragmatic_context(bundled, (), as_of=10)
    assert result["incongruent"] == 1
    assert result["comparisons"][0]["context_source_id"] == "sensor"


def test_nonexclusive_or_different_scope_does_not_forge_contradiction():
    focal = _event("speaker", "A blue light", at=10,
                   relations=(_relation("speaker", "blue", at=10,
                                        exclusive=False),))
    other = _event("sensor", "A red light", at=11,
                   relations=(_relation("sensor", "red", at=11,
                                        kind="observation", exclusive=False),))
    assert compare_pragmatic_context(focal, (other,), as_of=11)["congruence"] is None
    later = _event("sensor2", "It changed", at=12,
                   relations=(_relation("sensor2", "closed", at=12,
                                        scope="tomorrow", kind="observation"),))
    assert compare_pragmatic_context(focal, (later,), as_of=12)["comparable"] == 0


def test_explicit_withholding_is_retained_without_inferred_absence():
    focal = _event("speaker", "The result is settled", at=10)
    withheld = _event("archive", "Archive note", at=11, relations=(UsageRelation(
        "report", "access", "unreleased", "today", "withholding", "archive", 11),))
    result = compare_pragmatic_context(focal, (withheld,), as_of=11)
    assert result["status"] == "no_comparable_relations"
    assert result["explicit_withholding"][0]["source_id"] == "archive"
    assert result["intent"] == "unmeasured"


def test_indirect_mode_is_learned_from_attributed_context_not_laughter_rule(tmp_path):
    engine = _engine(tmp_path)
    for index in range(6):
        at = float(index + 1)
        source = f"speaker:{index}"
        mode, sense, laughed = (("metaphor", "pressure", True) if index < 3 else
                                ("literal", "weather", False))
        event = _event(source, "A storm is coming", at=at,
                       cues=(UsageCue("laughter", laughed, "audio", f"mic:{index}", at),))
        engine.observe_usage(event)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"feedback:{index}", source, "storm", sense, "observed_referent",
            observed_at=at, mode=mode))
    query = _event("query", "A storm is coming", at=10.0,
                   referents=("pressure",),
                   cues=(UsageCue("laughter", True, "audio", "mic:query", 10.0),))
    result = engine.pragmatic_readings("storm", query)
    assert result["status"] == "ranked_hypotheses"
    assert result["interpretation_hypotheses"][0]["mode"] == "metaphor"
    assert result["indirect_referents"] == ("pressure",)
    assert result["intent"] == "unmeasured"
    engine.save()
    restored = _engine(tmp_path)
    assert restored.pragmatic_readings("storm", query)[
        "interpretation_hypotheses"][0]["sense"] == "pressure"


def test_relation_mismatch_can_discriminate_attributed_meanings(tmp_path):
    engine = _engine(tmp_path)
    for index in range(6):
        at = float(10 * index + 1)
        source = f"usage:{index}"
        metaphor = index < 3
        context = f"scene:{index}"
        claim = UsageEvent.from_text(
            source, context, "A storm is coming", observed_at=at,
            relations=(UsageRelation("sky", "forecast", "storm", "today",
                                     "claim", source, at, exclusive=True),))
        observed = UsageEvent.from_text(
            f"sensor:{index}", context, "Weather reading", observed_at=at + 1,
            relations=(UsageRelation("sky", "forecast",
                                     "clear" if metaphor else "storm", "today",
                                     "observation", f"sensor:{index}", at + 1,
                                     exclusive=True),))
        engine.observe_usage(claim)
        engine.observe_usage(observed)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"feedback:{index}", source, "storm",
            "pressure" if metaphor else "weather", "observed_referent",
            observed_at=at + 2, mode="metaphor" if metaphor else "literal"))
    query = UsageEvent.from_text(
        "query", "new-scene", "A storm is coming", observed_at=100,
        relations=(UsageRelation("sky", "forecast", "storm", "today",
                                 "claim", "query", 100, exclusive=True),))
    sensor = UsageEvent.from_text(
        "new-sensor", "new-scene", "Weather reading", observed_at=101,
        relations=(UsageRelation("sky", "forecast", "clear", "today",
                                 "observation", "new-sensor", 101,
                                 exclusive=True),))
    engine.observe_usage(sensor)
    before = engine.pragmatic_readings("storm", query)
    assert before["congruence"]["status"] == "no_comparable_relations"
    after = engine.pragmatic_readings("storm", query, as_of=101)
    assert after["congruence"]["incongruent"] == 1
    assert after["interpretation_hypotheses"][0]["mode"] == "metaphor"
    assert "pragmatic:relation_fit" in after[
        "interpretation_hypotheses"][0]["supporting_features"]
    assert after["intent"] == "unmeasured"


def test_future_feedback_and_actions_do_not_rewrite_prior_interpretation(tmp_path):
    engine = _engine(tmp_path)
    early = _event("early", "The door is open", at=10,
                   relations=(_relation("early", "open", at=10),))
    later = _event("later", "Door check", at=12,
                   relations=(_relation("sensor", "closed", at=12,
                                        kind="observation"),))
    engine.observe_usage(early)
    engine.observe_usage(later)
    engine.observe_meaning_feedback(MeaningFeedback(
        "correction", "early", "door", "misreported_state",
        "user_correction", observed_at=13, mode="mistake"))
    before = engine.pragmatic_readings("door", early)
    assert before["congruence"]["status"] == "no_comparable_relations"
    assert before["interpretation_hypotheses"] == ()
    after = engine.pragmatic_readings("door", early, as_of=13)
    assert after["congruence"]["incongruent"] == 1
    assert after["intent"] == "unmeasured"
    assert after["interpretation_hypotheses"] == ()  # focal feedback is excluded


def test_relation_and_mode_require_measured_sources():
    with pytest.raises(ValueError, match="provenance"):
        _relation("speaker", "open", at=float("nan"))
    with pytest.raises(ValueError, match="attributed evidence"):
        MeaningFeedback("fb", "use", "storm", "pressure", "user_correction",
                        mode="definitely_a_lie")


def test_runtime_intake_persists_relation_and_attributed_mode(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(semantic_runtime, "get_semantic_development", lambda: engine)
    usage = _event("utterance", "A storm is coming", at=10,
                   relations=(_relation("utterance", "open", at=10),))
    assert semantic_runtime.record_contextual_usage(usage) is True
    assert semantic_runtime.record_contextual_usage(usage) is False
    feedback = MeaningFeedback("correction", "utterance", "storm", "pressure",
                               "user_correction", observed_at=11, mode="metaphor")
    assert semantic_runtime.record_attributed_meaning(feedback) is True
    assert semantic_runtime.record_attributed_meaning(feedback) is False
    restored = _engine(tmp_path)
    assert restored.usage_events["utterance"].relations == usage.relations
    assert restored.meaning_feedback["correction"].mode == "metaphor"
    secret = _event("unsafe", "A storm is coming", at=12,
                    relations=(UsageRelation("request", "credential",
                                             "sk-" + "x" * 30, "today", "claim",
                                             "unsafe", 12),))
    with pytest.raises(ValueError, match="sensitive"):
        semantic_runtime.record_contextual_usage(secret)
    assert "unsafe" not in engine.usage_events
    with pytest.raises(ValueError, match="sensitive"):
        engine.observe_usage(secret)
