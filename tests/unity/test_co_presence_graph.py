from __future__ import annotations

import time

from core.unity.co_presence_graph import CoPresenceGraphBuilder
from core.unity.unity_state import BoundContent


def _content(content_id: str, modality: str, source: str, summary: str, *, charge: float = 0.0) -> BoundContent:
    return BoundContent(
        content_id=content_id,
        modality=modality,
        source=source,
        summary=summary,
        salience=0.75,
        confidence=0.85,
        timestamp=time.time(),
        ownership="self" if modality != "world" else "world",
        action_relevance=0.7 if modality in {"goal", "tool"} else 0.4,
        affective_charge=charge,
    )


def test_related_memory_binds_to_active_focus():
    graph = CoPresenceGraphBuilder().build(
        [
            _content("goal", "goal", "planner", "finish the memory routing fix"),
            _content("memory", "memory", "memory_retrieval", "previous memory routing failure in response generation"),
            _content("tool", "tool", "tool_runner", "recent tool failure in the memory routing path"),
        ],
        focus_hint="memory routing fix",
    )

    assert graph.focus_id == "goal"
    assert graph.metrics["largest_connected_component_ratio"] >= 0.66
    assert graph.metrics["focus_periphery_binding_strength"] > 0.0


def test_unrelated_memory_stays_peripheral():
    graph = CoPresenceGraphBuilder().build(
        [
            _content("goal", "goal", "planner", "deploy the unity patch"),
            _content("memory", "memory", "memory_retrieval", "favorite coffee order from last week"),
        ],
        focus_hint="deploy the unity patch",
    )

    assert graph.focus_id == "goal"
    assert "memory" in graph.peripheral_ids


def test_conflicting_charges_raise_conflict_density():
    graph = CoPresenceGraphBuilder().build(
        [
            _content("goal", "goal", "planner", "publish the release note", charge=0.7),
            _content("affect", "affect", "affect_engine", "publish the release note feels unsafe", charge=-0.7),
            _content("world", "world", "world_state", "release window is active"),
        ],
        focus_hint="publish the release note",
    )

    assert graph.metrics["conflict_density"] > 0.0
    assert graph.metrics["cross_modal_edge_density"] > 0.0
