"""A memory is put before her where it bears on what is said, and is left out where it is further from it than any that does."""
from __future__ import annotations

import asyncio

import pytest

from core.memory import bears_on_it

_ABOUT = {"knee": 0, "trail": 0, "chess": 1, "queen": 1, "basil": 2, "herb": 2}


def _vectors(texts):
    out = []
    for text in texts:
        row = [0.02] * 5
        for word, at in _ABOUT.items():
            if word in str(text).lower():
                row[at] = 1.0
        out.append(row if max(row) == 1.0 else [0.02, 0.02, 0.02, 0.6, 0.4])
    return out


@pytest.fixture
def by_meaning(monkeypatch):
    monkeypatch.setattr(bears_on_it, "_vectors", _vectors)
    monkeypatch.setattr(bears_on_it, "BEARS", (("how's the knee?", "I fell on the trail"), ("another game of chess?", "you lost your queen")))
    bears_on_it._BOUND.clear()
    yield
    bears_on_it._BOUND.clear()


def test_what_bears_on_it_is_kept_and_what_does_not_is_not(by_meaning):
    assert bears_on_it.bear_on("how do I keep basil alive?", ["you lost your queen", "that herb wants light", "I fell on the trail"]) == [False, True, False]
    assert bears_on_it.bear_on("how do I keep basil alive?", []) == []


def test_where_it_cannot_be_read_every_memory_is_kept():
    assert bears_on_it.bear_on("anything", ["a", "b"], vectors=lambda texts: []) == [True, True]


def test_a_turn_is_given_only_what_bears_on_it(by_meaning, monkeypatch):
    """LIVE 25 November 2026: a question about basil was given three thousand characters about chess."""
    from core.memory import rag_bridge

    class Facade:
        async def search(self, query, limit=10, **_scope):
            return [{"text": "you lost your queen", "score": 0.9, "timestamp": 1.79e9}, {"text": "that herb wants light", "score": 0.4, "timestamp": 1.78e9}]

    monkeypatch.setattr(rag_bridge.ServiceContainer, "get", staticmethod(lambda name, default=None: Facade() if name == "memory_facade" else default))
    told = asyncio.run(rag_bridge.fetch_deep_context("how do I keep basil alive indoors?"))
    assert "that herb wants light" in told and "queen" not in told
    assert asyncio.run(rag_bridge.fetch_deep_context("what is the capital of Peru then?")) == "", "nothing bears on it, and nothing is put before her"
