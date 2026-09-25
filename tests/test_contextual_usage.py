"""Context changes the evidence for a meaning without defining it by fiat."""

import pytest

from core.cognition import semantic_runtime
from core.cognition.concept_handle import ConceptRegistry
from core.cognition.semantic_development import SemanticDevelopment
from core.knowledge.atomspace import INHERITANCE, AtomSpace, Link, TruthValue, concept
from core.language.contextual_usage import MeaningFeedback, UsageCue, UsageEvent


def _engine(tmp_path):
    return SemanticDevelopment(registry=ConceptRegistry(),
                               state_path=tmp_path / "semantic.json", min_support=4)


def test_surface_form_delivery_and_indirect_reference_are_observed_not_interpreted():
    cue = UsageCue("laughter", True, "audio", "microphone:1", 100.0)
    event = UsageEvent.from_text(
        "utterance:1", "room", "Soooo, the dragon again?", observed_at=100.0,
        setting="game", cues=(cue,), referents=("secret",))
    assert event.stretched_terms == ("soooo",)
    assert event.features_for("soooo")["usage:stretched"] is True
    assert event.features_for("dragon")["cue:audio:laughter"] is True
    assert event.features_for("secret")["usage:indirect_referent"] is True
    assert "secret" not in event.terms
    with pytest.raises(ValueError, match="window"):
        UsageEvent.from_text("late", "room", "dragon", observed_at=100.0,
                              cues=(UsageCue("gaze", "away", "camera", "camera:1", 1000.0),))


def test_story_association_is_not_an_asserted_taxonomy(tmp_path):
    engine = _engine(tmp_path)
    for index in range(8):
        engine.observe_usage(UsageEvent.from_text(
            f"story:{index}", "tale", "sword dragon medieval",
            observed_at=float(index), setting="storybook"))
    for index in range(8):
        engine.observe_usage(UsageEvent.from_text(
            f"zoo:{index}", "biology", "dragon reptile crocodile",
            observed_at=float(index + 10), setting="biology"))
    story = engine.usage_associations("dragon", setting="storybook")
    biology = engine.usage_associations("dragon", setting="biology")
    assert {row["term"] for row in story["associations"]} == {"sword", "medieval"}
    assert {row["term"] for row in biology["associations"]} == {"reptile", "crocodile"}
    atoms = AtomSpace()
    relation = engine.relation_evidence("dragon", "reptile", atomspace=atoms)
    assert relation["co_use_sources"] == 8
    assert relation["taxonomic_relation"] is None
    atoms.add(Link(INHERITANCE, (concept("dragon"), concept("reptile"))),
              TruthValue(0.9, 8), source="biology:verified")
    relation = engine.relation_evidence("dragon", "reptile", atomspace=atoms)
    assert relation["taxonomic_relation"]["strength"] == pytest.approx(0.9)
    assert engine.relation_evidence("dragon", "sword", atomspace=atoms)[
        "taxonomic_relation"] is None
    assert engine.associative_analogy("sword", "dragon", setting="storybook")[
        "relation"] == "shared_observed_use_not_taxonomy"
    assert engine.usage_associations("dragon", setting="unseen")["status"] == "unexposed"


def test_contextual_abduction_requires_exposure_and_attributed_feedback(tmp_path):
    engine = _engine(tmp_path)
    query = UsageEvent.from_text("future", "new", "nova tonight", setting="festival",
                                 community="community-a")
    assert engine.contextual_senses("nova", query)["status"] == (
        "unexposed_to_grounded_sense")
    for index in range(4):
        event = UsageEvent.from_text(f"a:{index}", "gathering", "nova party tonight",
                                     observed_at=float(index), setting="festival",
                                     community="community-a")
        assert engine.observe_usage(event)
        assert engine.observe_meaning_feedback(MeaningFeedback(
            f"feedback:a:{index}", event.source_id, "nova", "celebration",
            "user_correction", observed_at=float(index + 1)))
    for index in range(4):
        event = UsageEvent.from_text(f"b:{index}", "observatory", "nova star tonight",
                                     observed_at=float(index + 10), setting="astronomy",
                                     community="community-b")
        engine.observe_usage(event)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"feedback:b:{index}", event.source_id, "nova", "stellar_event",
            "verified_source", observed_at=float(index + 11)))
    result = engine.contextual_senses("nova", query)
    assert result["status"] == "ranked_hypotheses"
    assert result["candidates"][0]["sense"] == "celebration"
    assert result["serving_authority"] is False
    alien = UsageEvent.from_text("alien", "new", "nova tonight", setting="unknown",
                                 community="community-c")
    assert engine.contextual_senses("nova", alien)["status"] == "unmeasured_context_transfer"
    assert len(engine.grounded_sense_cases("nova", "celebration")) == 4
    assert all(case.outcome is True for case in engine.grounded_sense_cases(
        "nova", "celebration"))
    engine.save()
    restored = _engine(tmp_path)
    assert restored.contextual_senses("nova", query)["candidates"][0]["sense"] == (
        "celebration")


def test_multiple_readings_do_not_imply_refutation(tmp_path):
    engine = _engine(tmp_path)
    event = UsageEvent.from_text("turn:1", "chat", "nova")
    engine.observe_usage(event)
    first = MeaningFeedback("correction:1", event.source_id, "nova", "star",
                            "user_correction")
    engine.observe_meaning_feedback(first)
    assert engine.observe_meaning_feedback(first) is False
    with pytest.raises(ValueError, match="changed its claim"):
        engine.observe_meaning_feedback(MeaningFeedback(
            "correction:1", event.source_id, "nova", "festival", "user_correction"))
    engine.observe_meaning_feedback(MeaningFeedback(
        "correction:2", event.source_id, "nova", "festival", "verified_source"))
    query = UsageEvent.from_text("future", "chat", "nova")
    assert engine.contextual_senses("nova", query)["status"] == (
        "no_discriminating_evidence")
    assert {row["sense"] for row in engine.contextual_senses(
        "nova", query)["candidates"]} == {"star", "festival"}
    assert len(engine.grounded_sense_cases("nova", "star")) == 1
    assert engine.grounded_sense_cases("nova", "festival")[0].outcome is True
    with pytest.raises(ValueError, match="changed its observation"):
        engine.observe_usage(UsageEvent.from_text("turn:1", "chat", "new nova"))


def test_explicit_refutation_is_not_an_implied_negative(tmp_path):
    engine = _engine(tmp_path)
    event = UsageEvent.from_text("turn:1", "chat", "nova party", observed_at=10.0)
    engine.observe_usage(event)
    engine.observe_meaning_feedback(MeaningFeedback(
        "feedback:yes", event.source_id, "nova", "festival", "user_correction",
        observed_at=11.0))
    engine.observe_meaning_feedback(MeaningFeedback(
        "feedback:no", event.source_id, "nova", "star", "user_correction",
        observed_at=11.0, stance="refutes"))
    cases = engine.grounded_sense_cases("nova", "star")
    assert len(cases) == 1 and cases[0].outcome is False
    assert engine.grounded_sense_cases("nova", "festival")[0].outcome is True
    engine.save()
    restored = _engine(tmp_path)
    assert restored.grounded_sense_cases("nova", "star")[0].outcome is False
    restored.observe_meaning_feedback(MeaningFeedback(
        "feedback:contradiction", event.source_id, "nova", "festival",
        "verified_source", observed_at=12.0, stance="refutes"))
    assert restored.grounded_sense_cases("nova", "festival") == ()


def test_multilabel_heldout_result_is_not_reported_as_exact_sense(tmp_path):
    engine = _engine(tmp_path)
    for index, senses in enumerate((("star",), ("festival",),
                                    ("star", "festival"))):
        event = UsageEvent.from_text(f"turn:{index}", "chat", "nova tonight",
                                     observed_at=float(index + 1))
        engine.observe_usage(event)
        for sense in senses:
            engine.observe_meaning_feedback(MeaningFeedback(
                f"feedback:{index}:{sense}", event.source_id, "nova", sense,
                "verified_source", observed_at=float(index + 1)))
    result = engine.evaluate_contextual_senses("nova", ("turn:2",))
    assert result["n"] == 1
    assert result["unambiguous_n"] == 0
    assert result["unambiguous_correct"] == 0


def test_cues_can_inform_a_later_reading_without_hardcoded_affect(tmp_path):
    engine = _engine(tmp_path)
    for index in range(4):
        timestamp = float(index + 1)
        laughing = UsageCue("laughter", True, "audio", f"mic:{index}", timestamp)
        event = UsageEvent.from_text(f"joke:{index}", "chat", "bananas again",
                                     observed_at=timestamp, cues=(laughing,))
        engine.observe_usage(event)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"explain:joke:{index}", event.source_id, "bananas", "playful_usage",
            "user_correction"))
    quiet = UsageEvent.from_text("query", "chat", "bananas again",
                                 observed_at=10.0, cues=(UsageCue(
                                     "laughter", True, "audio", "mic:query", 10.0),))
    assert engine.contextual_senses("bananas", quiet)["candidates"][0]["sense"] == (
        "playful_usage")
    cue_counts = engine.usage_associations("bananas")["delivery_cues"]
    assert cue_counts[0]["name"] == "laughter" and cue_counts[0]["sources"] == 4


def test_bounded_long_message_keeps_middle_and_late_exposure(tmp_path):
    text = "opening " * 100 + "middlemarker " + "ordinary " * 100 + "closingmarker"
    event = UsageEvent.from_text("long:1", "chat", text)
    assert len(event.terms) == 128
    assert event.original_token_count == 202
    assert {"opening", "middlemarker", "ordinary", "closingmarker"} <= set(event.terms)
    assert event.features_for("closingmarker")["usage:spoken"] is True
    assert event.features_for("closingmarker")["usage:sampled"] is True
    engine = _engine(tmp_path)
    engine.observe_usage(event)
    assert engine.usage_associations("not-in-sample")["status"] == (
        "unmeasured_due_to_sampling")
    with pytest.raises(ValueError, match="bounded measured source"):
        UsageCue("laughter", True, "audio", "x" * 129, 1.0)


def test_chat_intake_retains_exposure_without_inventing_a_sense(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(semantic_runtime, "get_semantic_development", lambda: engine)
    assert semantic_runtime.record_chat_usage(
        "Soooo that was funny", session_id="a session", turn_id="turn:1")
    assert engine.usage_observation_count == 1
    event = next(iter(engine.usage_events.values()))
    assert event.context_id.startswith("chat:")
    assert event.stretched_terms == ("soooo",)
    assert event.community == "" and event.cues == ()
    assert engine.meaning_feedback == {}
    assert engine.contextual_senses("soooo", event)["status"] == (
        "unexposed_to_grounded_sense")
    assert not semantic_runtime.record_chat_usage(
        "Soooo that was funny", session_id="a session", turn_id="turn:1")
    assert engine.usage_observation_count == 1
    with pytest.raises(ValueError, match="changed its observation"):
        semantic_runtime.record_chat_usage(
            "This turn says something different", session_id="a session", turn_id="turn:1")
    assert not semantic_runtime.record_chat_usage("...", session_id="a session")
    assert semantic_runtime.record_chat_usage(
        "A second observation survives", session_id="a session", turn_id="turn:2")
    restored = _engine(tmp_path)
    assert len(restored.usage_events) == 2
    assert restored.usage_observation_count == 2


def test_chat_intake_cold_start_creates_real_service(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_runtime, "_SERVICE", None)
    monkeypatch.setattr(semantic_runtime, "SemanticDevelopment",
                        lambda **_kwargs: _engine(tmp_path))
    service = semantic_runtime.get_semantic_development()
    assert isinstance(service, SemanticDevelopment)
    assert semantic_runtime.record_chat_usage(
        "one grounded exposure", session_id="session", turn_id="turn")
    assert service.usage_observation_count == 1
    assert not semantic_runtime.record_chat_usage(
        "my key is sk-abcdefghijklmnopqrstuvwxyz012345",
        session_id="session", turn_id="secret-turn")
    assert service.usage_observation_count == 1


def test_chat_intake_retry_flushes_after_write_failure(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(semantic_runtime, "get_semantic_development", lambda: engine)
    original_save = engine.save
    attempts = 0

    def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient write failure")
        original_save()

    monkeypatch.setattr(engine, "save", fail_once)
    with pytest.raises(OSError, match="transient write failure"):
        semantic_runtime.record_chat_usage(
            "A remembered phrase", session_id="session", turn_id="one")
    assert not semantic_runtime.record_chat_usage(
        "A remembered phrase", session_id="session", turn_id="one")
    restored = _engine(tmp_path)
    assert len(restored.usage_events) == 1


def test_contextual_rank_uses_contrast_not_shared_background(tmp_path):
    engine = _engine(tmp_path)
    for index in range(6):
        playful = UsageEvent.from_text(
            f"playful:{index}", "shared-room", "nova blue night",
            setting="shared", community="mixed", observed_at=float(index + 1),
            cues=(UsageCue("laughter", True, "audio", f"mic:{index}",
                           float(index + 1)),))
        literal = UsageEvent.from_text(
            f"literal:{index}", "shared-room", "nova blue night",
            setting="shared", community="mixed", observed_at=float(index + 20),
            cues=(UsageCue("laughter", False, "audio", f"mic:literal:{index}",
                           float(index + 20)),))
        engine.observe_usage(playful)
        engine.observe_usage(literal)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"meaning:playful:{index}", playful.source_id, "nova", "joke",
            "observed_referent"))
        engine.observe_meaning_feedback(MeaningFeedback(
            f"meaning:literal:{index}", literal.source_id, "nova", "star",
            "observed_referent"))
    query = UsageEvent.from_text(
        "new", "shared-room", "nova blue night", setting="shared",
        community="mixed", cues=(UsageCue("laughter", True, "audio", "mic:new",
                                          100.0),), observed_at=100.0)
    ranking = engine.contextual_senses("nova", query)
    assert ranking["candidates"][0]["sense"] == "joke"
    assert "cue:audio:laughter" in ranking["candidates"][0]["supporting_features"]
    assert ranking["serving_authority"] is False
    no_audio = UsageEvent.from_text(
        "no-audio", "shared-room", "nova blue night", setting="shared",
        community="mixed")
    assert engine.contextual_senses("nova", no_audio)["status"] == (
        "no_discriminating_evidence")


def test_heldout_meaning_evaluation_excludes_all_test_sources(tmp_path):
    engine = _engine(tmp_path)
    heldout = []
    for index in range(6):
        for sense, laughed in (("joke", True), ("star", False)):
            source = f"{sense}:{index}"
            event = UsageEvent.from_text(
                source, "scene", "nova blue night", setting="shared",
                community="mixed", observed_at=float(index + 1),
                cues=(UsageCue("laughter", laughed, "audio", "mic:" + source,
                               float(index + 1)),))
            engine.observe_usage(event)
            engine.observe_meaning_feedback(MeaningFeedback(
                "feedback:" + source, source, "nova", sense,
                "observed_referent"))
            if index >= 4:
                heldout.append(source)
    result = engine.evaluate_contextual_senses("nova", tuple(heldout))
    assert result["status"] == "measured_development_only"
    assert result["n"] == result["answered"] == result["correct"] == 4
    assert result["majority_baseline_correct"] == 2
    assert result["serving_authority"] is False
    with pytest.raises(ValueError, match="distinct bounded"):
        engine.evaluate_contextual_senses("nova", (heldout[0], heldout[0]))
    inquiry = engine.discriminating_usage_observations(
        "nova", UsageEvent.from_text("unseen", "scene", "nova blue night",
                                     setting="shared", community="mixed"))
    assert inquiry["status"] == "candidate_observations"
    assert inquiry["observations"][0]["feature"] == "cue:audio:laughter"
    assert inquiry["observations"][0]["separation"] == 1.0
    assert inquiry["serving_authority"] is False


def test_an_unmeasured_cue_cannot_be_a_discriminator(tmp_path):
    engine = _engine(tmp_path)
    for index, (sense, has_audio) in enumerate((
            ("joke", True), ("star", False), ("joke", True), ("star", False))):
        timestamp = float(index + 1)
        cues = ((UsageCue("laughter", True, "audio", f"mic:{index}", timestamp),)
                if has_audio else ())
        event = UsageEvent.from_text(
            f"source:{index}", "chat", "nova tonight", observed_at=timestamp,
            cues=cues)
        engine.observe_usage(event)
        engine.observe_meaning_feedback(MeaningFeedback(
            f"feedback:{index}", event.source_id, "nova", sense,
            "observed_referent"))
    inquiry = engine.discriminating_usage_observations(
        "nova", UsageEvent.from_text("future", "chat", "nova tonight"))
    assert not any(item["feature"] == "cue:audio:laughter"
                   for item in inquiry["observations"])
    query = UsageEvent.from_text(
        "query", "chat", "nova tonight", observed_at=100.0,
        cues=(UsageCue("laughter", True, "audio", "mic:query", 100.0),))
    assert engine.contextual_senses("nova", query)["status"] == (
        "no_discriminating_evidence")
