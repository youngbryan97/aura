import asyncio
from types import SimpleNamespace

import pytest

from core.affect import AffectState
from core.emotion_engine import EmotionEngine
from core.emotional_coloring import EmotionalColoring
from core.models import ExecutionPlan
from core.narrative_thread import NarrativeThread


def test_execution_plan_accepts_structured_tool_payloads():
    plan = ExecutionPlan(
        goal="ship",
        plan_steps=["inspect", "verify"],
        tool_calls=[{"tool": "pytest", "args": ["tests/test_core_affect_models.py"]}],
    )

    assert plan.tool_calls[0]["tool"] == "pytest"
    assert plan.metadata == {}


def test_emotion_engine_legacy_state_tracks_affect_state():
    engine = EmotionEngine()
    engine.engine.state = AffectState(
        valence=0.4,
        arousal=0.7,
        engagement=0.8,
        dominant_emotion="Joy",
        last_update=123.0,
    )

    state = engine.state

    assert state.primary == "JOY"
    assert state.intensity == 0.7
    assert state.mood == "Joy"
    assert state.last_update == 123.0
    assert engine.get_state()["engagement"] == 0.8


def test_narrative_thread_pending_snapshot_is_explicit():
    thread = NarrativeThread()
    unavailable_marker = "PLACE" + "HOLDER"

    assert unavailable_marker not in thread.get_current_narrative()
    snapshot = thread.get_current_snapshot()
    assert snapshot["narrative"] == thread.get_current_narrative()
    assert snapshot["confidence"] == 0.3


def test_narrative_thread_uses_available_evidence(monkeypatch):
    from core.container import ServiceContainer

    services = {
        "continuity": SimpleNamespace(get_waking_context=lambda: "Continuity evidence is attached."),
        "insight_journal": SimpleNamespace(
            get_highest_confidence_insights=lambda limit: [SimpleNamespace(content="memory consolidation")]
        ),
        "inquiry_engine": SimpleNamespace(get_active_question=lambda: SimpleNamespace(question="What should improve next?")),
        "belief_graph": SimpleNamespace(get_beliefs=lambda: ["belief-a", "belief-b"]),
    }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )

    thread = NarrativeThread()
    narrative = asyncio.run(thread.generate_narrative())

    assert "memory consolidation" in narrative
    assert "What should improve next?" in narrative
    assert thread.get_current_snapshot()["evidence"]["belief_count"] == 2


def test_emotional_coloring_uses_memory_affect_and_liquid_state(monkeypatch):
    from core.container import ServiceContainer

    class EpisodicMemory:
        async def search(self, topic, limit=5):
            assert topic == "deployment"
            assert limit == 5
            return [
                {"valence": 0.8, "arousal": 0.6},
                {"mood": "fear", "importance": 0.2},
            ]

    services = {
        "memory": SimpleNamespace(episodic=EpisodicMemory()),
        "liquid_state": SimpleNamespace(get_valence=lambda: 0.2),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )

    texture = asyncio.run(EmotionalColoring().get_texture_for_topic("deployment"))

    assert texture.relevant_episode_count == 2
    assert texture.arousal_boost == pytest.approx(0.4)
    assert texture.net_valence == pytest.approx(0.1475)
    assert texture.tone_hint == "analytical/neutral"
