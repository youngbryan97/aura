from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from core.state.aura_state import AuraState
from core.unity.runtime import UnityRuntime
from core.will import ActionDomain, UnifiedWill, WillOutcome
from interface.routes.inner_state import get_unity_state


def test_end_to_end_unity_changes_tool_decision():
    state = AuraState()
    state.cognition.current_objective = "publish the release note externally"
    state.cognition.current_origin = "user"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "publish the release note externally"},
            {"role": "assistant", "content": "I am unsure whether I authored that choice."},
        ]
    )
    state.cognition.active_goals = [
        {"objective": "publish the release note externally", "priority": 0.9},
        {"objective": "hold until the uncertainty is resolved", "priority": 0.8},
    ]
    state.cognition.coherence_score = 0.22
    state.world.recent_percepts = [{"summary": "stale deployment signal", "timestamp": 1.0}]

    runtime = UnityRuntime()
    with patch.object(
        runtime,
        "_draft_inputs",
        return_value=[
            {"draft_id": "a", "content": "publish now", "coherence": 0.7},
            {"draft_id": "b", "content": "do not publish now", "coherence": 0.72},
        ],
    ):
        runtime.apply_to_state(state, objective=state.cognition.current_objective, tick_id="tick_e2e")

    will = UnifiedWill()
    with patch.object(will, "_consult_substrate", return_value=(0.85, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            decision = will.decide(
                content="publish the release note externally",
                source="tool_runner",
                domain=ActionDomain.TOOL_EXECUTION,
                context={"external_action": True},
            )

    assert state.cognition.unity_state is not None
    assert state.response_modifiers["unity_summary"]["level"] in {"fragmented", "strained", "dissociated"}
    if state.cognition.unity_state.level in {"fragmented", "dissociated"}:
        assert decision.outcome == WillOutcome.REFUSE


def test_unity_route_exposes_current_summary():
    state = AuraState()
    runtime = UnityRuntime()
    runtime.apply_to_state(state, objective="answer the user", tick_id="tick_api")

    response = __import__("asyncio").run(get_unity_state())
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["unity_id"]
    assert "unity_score" in payload
    assert "fragmentation_score" in payload
