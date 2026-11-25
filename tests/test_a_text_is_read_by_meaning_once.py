"""A text her encoder has read is not read again until the engine that read it is changed (evidence_relevance)."""
from __future__ import annotations

import numpy as np
import pytest

import core.cognition.evidence_relevance as relevance


class _Engine:
    """Reads a text as its length, and keeps what it was asked to read."""

    def __init__(self):
        self.read: list[list[str]] = []
        self._model = object()

    def embed_batch_by_meaning(self, texts):
        self.read.append(list(texts))
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def engine(monkeypatch):
    made = _Engine()
    monkeypatch.setattr(relevance, "semantic_routing_ready", lambda: True)
    monkeypatch.setattr(relevance, "_embedder", lambda *, initialize=True: made)
    relevance._MEANT.clear()
    yield made
    relevance._MEANT.clear()


def test_only_what_was_not_read_before_is_read(engine):
    first = relevance.vectors_if_ready(["one", "three"])
    again = relevance.vectors_if_ready(["three", "fourth", "one", "three"])
    assert engine.read == [["one", "three"], ["fourth"]]
    assert again == [first[1], [6.0, 1.0], first[0], first[1]]


def test_texts_already_read_are_given_without_her_encoder(engine):
    relevance.vectors_if_ready(["one"])
    engine.embed_batch_by_meaning = lambda texts: None
    assert relevance.vectors_if_ready(["one"]) == [[3.0, 1.0]]
    assert relevance.vectors_if_ready(["one", "new"]) == []


def test_another_engine_reads_them_for_itself(engine, monkeypatch):
    relevance.vectors_if_ready(["one"])
    monkeypatch.setattr(relevance, "_CACHE_ENGINE_TOKEN", ("another", "", 0, 0))
    relevance._bind_cache_to_engine(engine)
    relevance.vectors_if_ready(["one"])
    assert engine.read == [["one"], ["one"]]


def test_they_are_kept_within_their_bound(engine, monkeypatch):
    monkeypatch.setattr(relevance, "_MEANT_MAX", 3)
    relevance.vectors_if_ready([str(n) for n in range(5)])
    assert len(relevance._MEANT) == 3
    assert list(relevance.document_cache_is_bounded()) == []
