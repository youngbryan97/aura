import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.consciousness.closed_loop import notify_closed_loop_output
from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.state.aura_state import AuraState
from core.utils.output_gate import AutonomousOutputGate
from core.brain.llm.mlx_client import _notify_closed_loop_output as notify_mlx_closed_loop


@pytest.mark.asyncio
async def test_executive_closure_engine_integrates_runtime_signals(service_container):
    state = AuraState()
    state.loop_cycle = 24
    state.motivation.budgets["energy"]["level"] = 15.0
    state.motivation.budgets["integrity"]["level"] = 35.0
    state.world.recent_percepts.append({"type": "screen", "summary": "CPU pressure spike"})
    state.soma.hardware["cpu_usage"] = 88.0

    service_container.register_instance(
        "closed_causal_loop",
        SimpleNamespace(
            get_status=lambda: {
                "loop": {"cycle_count": 42},
                "free_energy": {"current": 0.28},
                "phi": {"estimate": 0.17},
            }
        ),
    )
    service_container.register_instance(
        "global_workspace",
        SimpleNamespace(
            get_snapshot=lambda: {
                "last_winner": "self_prediction",
                "last_content": "Prediction surprise in thermal load",
                "last_priority": 0.72,
            }
        ),
    )
    service_container.register_instance(
        "homeostasis",
        SimpleNamespace(
            pulse=AsyncMock(
                return_value={
                    "integrity": 0.91,
                    "persistence": 0.88,
                    "curiosity": 0.62,
                    "metabolism": 0.82,
                    "sovereignty": 0.95,
                    "will_to_live": 0.79,
                }
            ),
            get_status=lambda: {
                "integrity": 0.91,
                "persistence": 0.88,
                "curiosity": 0.62,
                "metabolism": 0.82,
                "sovereignty": 0.95,
                "will_to_live": 0.79,
            },
        ),
    )
    goal_hierarchy = SimpleNamespace(
        get_next_goal=lambda: SimpleNamespace(description="Protect continuity"),
        add_goal=MagicMock(),
    )
    service_container.register_instance("goal_hierarchy", goal_hierarchy)
    self_model = SimpleNamespace(update_belief=AsyncMock())
    service_container.register_instance("self_model", self_model)
    service_container.register_instance("volition_engine", SimpleNamespace(tick=AsyncMock(return_value=None)))

    engine = ExecutiveClosureEngine()
    result = await engine.integrate(state)
    await asyncio.sleep(0)

    assert result.free_energy == pytest.approx(0.28, abs=1e-6)
    assert result.phi_estimate == pytest.approx(0.17, abs=1e-6)
    assert result.loop_cycle == 42
    assert result.cognition.attention_focus == "Prediction surprise in thermal load"
    assert result.cognition.current_objective is None
    assert result.cognition.pending_initiatives[0]["goal"] == "Protect continuity"
    assert result.response_modifiers["executive_closure"]["dominant_need"] in {"stability", "integrity"}
    assert result.response_modifiers["executive_closure"]["workspace_source"] == "self_prediction"
    assert result.response_modifiers["executive_closure"]["selected_objective"] == "Protect continuity"
    assert result.cognition.active_goals
    assert engine.get_status()["closure_score"] > 0.0
    self_model.update_belief.assert_awaited_once()
    assert goal_hierarchy.add_goal.called


def test_notify_closed_loop_output_routes_only_to_running_loop(service_container):
    loop = SimpleNamespace(is_running=True, on_inference_output=MagicMock())
    service_container.register_instance("closed_causal_loop", loop)

    notify_closed_loop_output("Aura is thinking.")
    loop.on_inference_output.assert_called_once_with("Aura is thinking.")

    loop.is_running = False
    loop.on_inference_output.reset_mock()
    notify_closed_loop_output("Aura is still thinking.")
    loop.on_inference_output.assert_not_called()


@pytest.mark.asyncio
async def test_output_gate_emits_closed_loop_feedback(service_container):
    loop = SimpleNamespace(is_running=True, on_inference_output=MagicMock())
    service_container.register_instance("closed_causal_loop", loop)

    gate = AutonomousOutputGate(orchestrator=SimpleNamespace(conversation_history=[]))

    await gate.emit(
        "A real output reached the communication layer.",
        origin="system",
        target="secondary",
        metadata={"suppress_bus": True},
    )

    loop.on_inference_output.assert_called_once()


def test_mlx_closed_loop_notification_helper_forwards_text():
    with patch("core.consciousness.closed_loop.notify_closed_loop_output") as mock_notify:
        notify_mlx_closed_loop("Local model response")
    mock_notify.assert_called_once_with("Local model response")
