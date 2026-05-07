from __future__ import annotations

import time

from core.state.aura_state import AuraState
from core.unity.co_presence_graph import CoPresenceGraphBuilder
from core.unity.draft_reconciliation import DraftReconciliationEngine
from core.unity.self_world_binding import SelfWorldBindingModel
from core.unity.temporal_binding import TemporalBindingField
from core.unity.unity_monitor import UnityMonitor
from core.unity.unity_state import BoundContent


def _content(content_id: str, modality: str, source: str, summary: str, *, charge: float = 0.0) -> BoundContent:
    return BoundContent(
        content_id=content_id,
        modality=modality,
        source=source,
        summary=summary,
        salience=0.8,
        confidence=0.85,
        timestamp=time.time(),
        ownership="self" if modality != "world" else "world",
        action_relevance=0.7,
        affective_charge=charge,
    )


def test_monitor_detects_fragmentation_causes():
    state = AuraState()
    state.cognition.coherence_score = 0.25
    state.cognition.working_memory.append({"role": "user", "content": "you chose this"})

    contents = [
        _content("goal", "goal", "planner", "publish the external update"),
        _content("affect", "affect", "affect_engine", "publishing this feels unsafe", charge=-0.8),
        _content("world", "world", "world_state", "stale external signal"),
    ]

    temporal = TemporalBindingField().bind_now(
        "tick_monitor",
        contents,
        previous_content_ids=["old_goal"],
        now_ts=time.time(),
    )
    graph = CoPresenceGraphBuilder().build(contents, focus_hint="publish the external update")
    drafts = DraftReconciliationEngine().reconcile(
        [
            {"draft_id": "a", "content": "publish now", "coherence": 0.7},
            {"draft_id": "b", "content": "do not publish now", "coherence": 0.7},
        ]
    )
    binding = SelfWorldBindingModel().bind(state, contents)

    unity_state, report = UnityMonitor().compute(state, temporal, graph, drafts, binding)

    assert unity_state.level in {"fragmented", "dissociated", "strained"}
    cause_names = [name for name, _weight, _text in report.top_causes]
    assert any(name in cause_names for name in {"draft_conflict", "ownership_ambiguity", "workspace_conflict"})
    if unity_state.level in {"fragmented", "dissociated"}:
        assert report.safe_to_act is False
