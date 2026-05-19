from types import SimpleNamespace
from unittest.mock import patch

from core.mind_tick import MindTick


@patch("core.runtime.background_policy.background_activity_reason", return_value="")
def test_mind_tick_background_reasoning_pauses_on_event_loop_lag(mock_bg_policy):
    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace(
        _flow_controller=SimpleNamespace(
            snapshot=lambda _orch: SimpleNamespace(
                lag_seconds=0.22,
                load=0.20,
                overloaded=False,
                governor_mode="FULL",
            )
        ),
        _last_user_interaction_time=0.0,
    )

    reason = MindTick._background_reasoning_pause_reason(
        tick,
        SimpleNamespace(cognition=SimpleNamespace(current_objective="Hold continuity", active_goals=[])),
    )

    assert reason == "event_loop_lag"


@patch("core.runtime.background_policy.background_activity_reason", return_value="")
def test_mind_tick_background_reasoning_requires_context(mock_bg_policy):
    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace(
        _flow_controller=SimpleNamespace(
            snapshot=lambda _orch: SimpleNamespace(
                lag_seconds=0.0,
                load=0.10,
                overloaded=False,
                governor_mode="FULL",
            )
        ),
        _last_user_interaction_time=0.0,
    )

    reason = MindTick._background_reasoning_pause_reason(
        tick,
        SimpleNamespace(cognition=SimpleNamespace(current_objective="", active_goals=[])),
    )

    assert reason == "no_reasoning_context"

