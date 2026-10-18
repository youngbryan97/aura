"""A surprise source that has never scored a prediction does not set novelty.

The registered predictive engine answers 0.0 until a prediction is scored, and
nothing in the runtime asks it to predict. Episodic memory read that 0.0 as
"nothing new" for 34,992 of 36,138 live episodes (26 July to 17 October).
"""
from __future__ import annotations

from types import SimpleNamespace

from core.consciousness.predictive_engine import PredictiveEngine
from core.memory.episodic_memory import EpisodicMemory


def _memory_with(monkeypatch, services: dict) -> EpisodicMemory:
    mem = EpisodicMemory.__new__(EpisodicMemory)
    monkeypatch.setattr(mem, "_service", lambda name: services.get(name))
    monkeypatch.setattr(mem, "_peek_recent", lambda limit=5: [])
    return mem


def test_an_engine_that_never_predicted_is_passed_over(monkeypatch):
    engine = PredictiveEngine()
    assert engine.surprise_measured is False
    mem = _memory_with(monkeypatch, {"predictive_engine": engine})

    # Nothing recent to resemble, so the word-overlap fallback calls it new.
    assert mem._estimate_novelty("a heron on the roof", "looked", "it flew") == 1.0


def test_a_scored_surprise_is_used(monkeypatch):
    engine = PredictiveEngine()
    engine.surprise_history.append(0.3)
    engine.total_surprise = 0.3
    assert engine.surprise_measured is True
    mem = _memory_with(monkeypatch, {"predictive_engine": engine})

    assert mem._estimate_novelty("a heron on the roof", "looked", "it flew") == 0.3


def test_a_source_that_does_not_say_is_still_read(monkeypatch):
    self_prediction = SimpleNamespace(get_surprise_signal=lambda: 0.4)
    mem = _memory_with(
        monkeypatch,
        {"predictive_engine": PredictiveEngine(), "self_prediction": self_prediction},
    )

    assert mem._estimate_novelty("a heron on the roof", "looked", "it flew") == 0.4
