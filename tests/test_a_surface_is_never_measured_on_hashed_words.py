"""A meaning surface gets the model's vectors or none, never hashed words of the same width.

`EmbeddingEngine.embed_batch` answers a missing model with hashed words, and the hashed vector has the model's length.
Recall can use that: a weaker match beats none. A LearnedMatcher cannot. It measures its boundary once and keeps it,
so a boundary measured on hashed words and then asked with model vectors (or the reverse) compares two spaces that
share nothing but their length. In her app the shared runtime held no model while evidence_relevance's encoder did
(13 November 2026), so every surface on `embed_sentences` was measured on hashed words there. And a surface that got
no vectors at all settled with no boundary, and stayed silent for the life of the process after the encoder came up.
"""

from __future__ import annotations

from contextlib import contextmanager

from core.cognition import evidence_relevance
from core.language import learned_matcher
from core.language.learned_matcher import LearnedMatcher, embed_sentences
from core.memory import embedding_runtime
from core.memory.vector_memory_engine import EmbeddingEngine


def _without_a_model() -> EmbeddingEngine:
    engine = EmbeddingEngine()
    engine._checkout_model = lambda: None  # refused, loading, or its lane being yielded
    return engine


def test_without_its_model_the_engine_gives_no_meaning_and_recall_still_gets_words():
    engine = _without_a_model()

    assert engine.embed_batch_by_meaning(["all flights were grounded on Friday"]) is None
    words = engine.embed_batch(["all flights were grounded on Friday"])
    assert words.shape == (1, engine.VECTOR_DIM), "recall keeps its fallback"


def test_the_shared_runtime_without_a_model_gives_a_surface_nothing(monkeypatch):
    engine = _without_a_model()

    @contextmanager
    def lease(_owner):
        yield engine

    monkeypatch.setattr(evidence_relevance, "vectors_if_ready", lambda _texts: [])
    monkeypatch.setattr(embedding_runtime, "acquire_shared_embedding_engine", lease)

    assert embed_sentences(["You told me last week that your sister was moving to Leeds."]) == []


def test_the_encoder_her_app_holds_comes_first(monkeypatch):
    def lease(_owner):
        raise AssertionError("the shared runtime is not asked while her app's encoder answers")

    monkeypatch.setattr(evidence_relevance, "vectors_if_ready", lambda texts: [[1.0, 0.0]] * len(texts))
    monkeypatch.setattr(embedding_runtime, "acquire_shared_embedding_engine", lease)

    assert embed_sentences(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_vectors_if_ready_never_hands_back_hashed_words(monkeypatch):
    engine = _without_a_model()
    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: True)
    monkeypatch.setattr(evidence_relevance, "_embedder", lambda **_kw: engine)

    assert evidence_relevance.vectors_if_ready(["NSA outage today"]) == []


def test_a_surface_with_no_vectors_yet_is_measured_when_they_come(monkeypatch):
    monkeypatch.setattr(learned_matcher.LearnedMatcher, "load", lambda self: False)
    up = {"now": False}

    def features(texts):
        if not up["now"]:
            return []
        return [[1.0, 0.0] if "yes" in text else [0.0, 1.0] for text in texts]

    surface = LearnedMatcher(
        name="probe_measured_when_vectors_come",
        positives=("yes one here", "yes two here"),
        negatives=("no one here", "no two here"),
        features=features,
    )

    assert surface.decide("yes, this one is") is None, "nothing to measure with: no verdict"
    up["now"] = True
    assert surface.decide("yes, this one is") is True, "the encoder came up and the surface measured itself"
    assert surface.decide("no, not this one") is False, "and it still refuses what it was declared to refuse"

