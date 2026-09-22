"""One world model, reached by every caller under one name.

Two objects were registered as `world_model`: the unified model, whose facets
are the forward dynamics, the causal graph and the outcome model, and at
orchestrator boot the belief engine, which replaced it. The prompt assembler
asked the winner for `get_context_injection`, so on any path that had not
booted the orchestrator her beliefs were missing from every system prompt, and
on the desktop the callers that asked for surprise and the causal graph got the
belief engine instead. And dreaming asked `world_model` for `update_belief`,
which only the epistemic state has, so no pattern a dream found ever became a
belief.
"""

from __future__ import annotations

import pytest

from core.world_model.unified_world_model import UnifiedWorldModel

pytestmark = pytest.mark.unit


class _Store:
    def __init__(self) -> None:
        self.beliefs: dict = {}

    def add_belief(self, claim, confidence, source_id=None, tags=None):
        self.beliefs[claim] = confidence

    def get_context_injection(self) -> str:
        return "[WORLD MODEL BELIEFS]\n" + "\n".join(f"- {claim}" for claim in self.beliefs)


def test_the_unified_model_carries_her_beliefs() -> None:
    world = UnifiedWorldModel(belief_store=_Store())
    world.add_belief("the kettle is slow", 0.8)
    assert "the kettle is slow" in world.beliefs
    assert "the kettle is slow" in world.get_context_injection()
    assert world.status()["facets"]["beliefs"]["detail"]["count"] == 1


def test_the_engine_boot_loaded_becomes_the_belief_facet() -> None:
    world = UnifiedWorldModel(belief_store=_Store())
    loaded = _Store()
    loaded.add_belief("she was here yesterday", 0.6)
    world.adopt_beliefs(loaded)
    assert world.belief_store is loaded
    assert "she was here yesterday" in world.get_context_injection()


def test_boot_registers_the_model_with_every_facet(monkeypatch, tmp_path) -> None:
    import core.final_engines as final_engines
    import core.world_model.unified_world_model as unified

    registered: dict = {}
    monkeypatch.setattr(unified, "_instance", UnifiedWorldModel(belief_store=_Store()))
    monkeypatch.setattr(final_engines, "WorldModelEngine", lambda: _Store())
    monkeypatch.setattr(
        final_engines, "register_runtime_service", lambda name, value, **_: registered.setdefault(name, value)
    )
    monkeypatch.setattr(final_engines, "NarrativeIdentityEngine", lambda: object())
    monkeypatch.setattr(final_engines, "MetacognitiveCalibrator", lambda: object())
    engines = final_engines.register_final_engines()
    world = registered["world_model"]
    assert isinstance(world, UnifiedWorldModel)
    assert world.belief_store is engines["world"]
    for method in ("get_context_injection", "add_belief", "causal_effects", "surprise"):
        assert hasattr(world, method), method


@pytest.mark.asyncio
async def test_a_dream_writes_its_patterns_as_beliefs(service_container) -> None:
    from types import SimpleNamespace

    from core.consciousness.dreaming import DreamingProcess

    written: list[tuple] = []

    class _Epistemic:
        def update_belief(self, subject, predicate, obj, confidence=1.0):
            written.append((subject, predicate, obj, confidence))

    dreamer = DreamingProcess(SimpleNamespace(_last_user_interaction_time=0), interval=0.1)
    dreamer._identity = SimpleNamespace(record_evolution=lambda **kwargs: None)
    dreamer._narrator = object()

    async def _recent_summary():
        return (
            "Context: code code code | Action: tested repair | Outcome: code stable (Valence: 0.2)\n"
            "Context: code code code | Action: tested repair | Outcome: code stable (Valence: 0.2)"
        )

    dreamer._get_recent_summary = _recent_summary
    service_container.register_instance("epistemic_state", _Epistemic(), required=False)
    await dreamer.dream()
    assert written, "a dream found patterns and none of them became a belief"
    assert all(row[0] == "aura_pattern" for row in written)
