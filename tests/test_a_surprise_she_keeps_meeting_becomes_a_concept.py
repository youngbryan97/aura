"""Concept formation is shown the world model's surprise, on a scale it can use.

The engine clusters repeated, similar prediction errors into a named concept,
and nothing ever showed it an error. The world model's surprise is
reconstruction error plus a KL term clipped at ten, which has no scale a fixed
line can be held against, so it is ranked against the model's own recent
surprises. These pin the rank and the feed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.phases.consciousness_phase import form_concepts
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept
from core.world_model.learned_world_model import LearnedWorldModel
from core.world_model.unified_world_model import UnifiedWorldModel


def _learned(history: list[float], latest: float | None) -> LearnedWorldModel:
    model = object.__new__(LearnedWorldModel)
    from collections import deque

    model._surprise_history = deque(history, maxlen=100)
    model._last_prediction = None if latest is None else SimpleNamespace(surprise=latest)
    return model


def test_the_rank_is_the_share_of_recent_surprises_no_larger_than_the_latest() -> None:
    model = _learned([0.2, 0.4, 3.0, 0.4, 7.5], latest=0.4)
    assert model.surprise_rank() == pytest.approx(3 / 5)


def test_the_largest_surprise_in_the_window_ranks_one() -> None:
    assert _learned([0.2, 0.4, 7.5], latest=7.5).surprise_rank() == pytest.approx(1.0)


def test_nothing_seen_ranks_zero() -> None:
    assert _learned([], latest=None).surprise_rank() == 0.0


def test_the_unified_model_passes_the_rank_through(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _learned([1.0, 2.0], latest=1.0)
    monkeypatch.setattr(UnifiedWorldModel, "learned", property(lambda self: stub))
    unified = object.__new__(UnifiedWorldModel)
    assert unified.surprise_rank() == pytest.approx(0.5)


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float, str]] = []

    def observe_prediction_error(self, features, magnitude, *, context=""):  # noqa: ANN001, ANN202
        self.calls.append((list(features), magnitude, context))
        return SimpleNamespace(to_dict=lambda: {"reason": "accumulating", "formed": None})


def _state() -> AuraState:
    state = AuraState.default()
    state.cognition.current_objective = "repair the greenhouse vent"
    state.cognition.attention_focus = "the vent hinge"
    emit_percept(state.world, "discovery", content="condensation pooling under the broken hinge", intensity=0.7)
    return state


def test_a_ranked_surprise_and_the_situations_cues_reach_the_engine() -> None:
    engine = _Engine()
    state = _state()
    reading = asyncio.run(form_concepts(state, SimpleNamespace(surprise_rank=lambda: 0.9), engine))
    ((features, magnitude, context),) = engine.calls
    assert magnitude == pytest.approx(0.9)
    assert {"greenhouse", "vent", "hinge", "condensation"} <= set(features)
    assert context == "repair the greenhouse vent"
    assert state.response_modifiers["concept_formation"] == reading == {"reason": "accumulating", "formed": None}


def test_no_world_model_shows_the_engine_nothing() -> None:
    engine = _Engine()
    assert asyncio.run(form_concepts(_state(), None, engine)) is None
    assert engine.calls == []


def test_a_situation_with_nothing_to_name_it_by_shows_the_engine_nothing() -> None:
    engine = _Engine()
    state = AuraState.default()
    assert asyncio.run(form_concepts(state, SimpleNamespace(surprise_rank=lambda: 0.9), engine)) is None
    assert engine.calls == []


def test_repeated_surprise_in_the_same_situation_forms_a_concept(tmp_path: Any) -> None:
    from core.cognition.concept_formation import ConceptFormationEngine

    engine = ConceptFormationEngine(storage_path=tmp_path / "concepts.json", autosave=False)
    world = SimpleNamespace(surprise_rank=lambda: 0.9)
    readings = [asyncio.run(form_concepts(_state(), world, engine)) for _ in range(3)]
    assert readings[-1]["formed"] is not None
    assert readings[-1]["reason"] == "formed_new_concept"
