"""What the fourth start of day run 4 (9 and 10 November 2026) found, each measured where it happens.

Her journal recorded no entry in 591 attempts across both log files: composed through the full cognitive cycle, it
was read as a task, wrapped in about 100,000 characters and judged by the gates for a reply to a person. Each failure
was a loss her resilience took, and from the minute of the first one her cortisol held above its crisis line, blocking
her initiative and expression for six hours.

The background brainstem was loaded and shed eight times in 71 minutes: admission asked for its footprint and half
again, and the inference gate sheds background workers when free memory falls below the cortex's load reserve plus
the fallback reserve, a higher line.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain import llm_background_deferral as deferral
from core.brain import narrative_memory


@pytest.mark.asyncio
async def test_her_journal_is_composed_directly_and_recorded(monkeypatch):
    episode = SimpleNamespace(timestamp=0.0, action="read a post", outcome="found five posts", episode_id="e1")
    deleted, written, asked = [], [], []

    class Episodic:
        async def recall_recent_async(self, limit=20):
            return [episode]

        async def delete_episodes_async(self, ids):
            deleted.extend(ids)

    class Memory:
        async def add_memory(self, text, metadata):
            written.append((text, metadata))
            return True

    class Brain:
        async def think(self, *args, **kwargs):
            raise AssertionError("a journal is not a turn of the full cognitive cycle")

        async def generate(self, prompt, **kwargs):
            asked.append(kwargs)
            return "I read a post and found five more. A plain afternoon."

    monkeypatch.setattr(narrative_memory, "get_episodic_memory", lambda: Episodic())
    monkeypatch.setattr(narrative_memory, "get_runtime_service", lambda name, default=None: Memory())
    engine = narrative_memory.NarrativeEngine(SimpleNamespace(cognitive_engine=Brain()))
    await engine.consolidate_episodes()

    assert asked and asked[0]["origin"] == "narrative_journal" and asked[0]["is_background"] is True
    assert "prefer_tier" not in asked[0], "background work takes the background tier; asking for her cortex was ignored"
    assert written and written[0][0].startswith("I read a post")
    assert written[0][1]["type"] == "narrative_journal"


def test_the_brainstem_is_not_loaded_where_the_shed_would_take_it(monkeypatch):
    import core.brain.llm.how_big_is_the_checkpoint as size
    import core.brain.llm.model_registry as registry
    import core.runtime.service_access as access

    monkeypatch.setattr(registry, "get_brainstem_path", lambda: "/models/brainstem")
    monkeypatch.setattr(size, "_projected_model_footprint_gb", lambda path: 7.0)
    monkeypatch.setattr(access, "resolve_inference_gate",
                        lambda: SimpleNamespace(_primary_load_required_gb=lambda: 20.0))
    monkeypatch.setenv("AURA_FALLBACK_RESIDENT_RESERVE_GB", "8.0")
    # Room for it above the line the shed keeps: 7 + (20 + 8), not 7 x 1.5.
    assert deferral._floor_for_the_model_on_this_lane() == pytest.approx(35.0)


def test_without_a_gate_the_footprint_floor_holds(monkeypatch):
    import core.brain.llm.how_big_is_the_checkpoint as size
    import core.brain.llm.model_registry as registry
    import core.runtime.service_access as access

    monkeypatch.setattr(registry, "get_brainstem_path", lambda: "/models/brainstem")
    monkeypatch.setattr(size, "_projected_model_footprint_gb", lambda path: 7.0)
    monkeypatch.setattr(access, "resolve_inference_gate", lambda: None)
    monkeypatch.setenv("AURA_FALLBACK_RESIDENT_RESERVE_GB", "8.0")
    assert deferral._floor_for_the_model_on_this_lane() == pytest.approx(15.0)
