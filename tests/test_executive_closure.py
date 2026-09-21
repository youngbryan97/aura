import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import _notify_closed_loop_output as notify_mlx_closed_loop
from core.consciousness.closed_loop import ClosedCausalLoop, notify_closed_loop_output
from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.state.aura_state import AuraState
from core.utils.output_gate import AutonomousOutputGate


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result

    def assert_awaited_once(self):
        assert len(self.calls) == 1


class CallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    @property
    def called(self):
        return bool(self.calls)

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result

    def reset(self):
        self.calls.clear()

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_not_called(self):
        assert not self.calls

    def assert_called_once_with(self, *args, **kwargs):
        self.assert_called_once()
        call = self.calls[0]
        assert call.args == args
        assert call.kwargs == kwargs


@pytest.mark.asyncio
async def test_executive_closure_engine_integrates_runtime_signals(service_container, monkeypatch):
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
            pulse=AsyncCallRecorder(
                {
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
        add_goal=CallRecorder(),
    )
    service_container.register_instance("goal_hierarchy", goal_hierarchy)
    self_model = SimpleNamespace(update_belief=AsyncCallRecorder())
    service_container.register_instance("self_model", self_model)
    service_container.register_instance("volition_engine", SimpleNamespace(tick=AsyncCallRecorder()))

    async def governed_proposal(state, goal, **kwargs):
        import time as _time
        state.cognition.pending_initiatives.append({"goal": goal, "ts": _time.time()})
        return state, {"action": "queued", "reason": "test_approved"}

    engine = ExecutiveClosureEngine()
    monkeypatch.setattr(
        "core.consciousness.executive_closure.propose_governed_initiative_to_state",
        governed_proposal,
    )
    result = await engine.integrate(state)
    await asyncio.sleep(0)

    assert result.free_energy == pytest.approx(0.28, abs=1e-6)
    assert result.phi_estimate == pytest.approx(0.17, abs=1e-6)
    assert result.loop_cycle == 42
    assert result.cognition.attention_focus == "Prediction surprise in thermal load"
    assert result.cognition.current_objective is None
    assert result.cognition.pending_initiatives[0]["goal"] == "Prediction surprise in thermal load"
    assert result.response_modifiers["executive_closure"]["dominant_need"] in {"stability", "integrity"}
    assert result.response_modifiers["executive_closure"]["workspace_source"] == "self_prediction"
    assert result.response_modifiers["executive_closure"]["selected_objective"] == "Prediction surprise in thermal load"
    assert result.cognition.active_goals
    assert engine.get_status()["closure_score"] > 0.0
    self_model.update_belief.assert_awaited_once()
    assert goal_hierarchy.add_goal.called


@pytest.mark.asyncio
async def test_executive_closure_demotes_intrinsic_maintenance_objective(service_container, monkeypatch):
    state = AuraState()
    state.loop_cycle = 24
    state.cognition.current_objective = "Protect identity, memory integrity, and process continuity."

    service_container.register_instance(
        "closed_causal_loop",
        SimpleNamespace(
            get_status=lambda: {
                "loop": {"cycle_count": 51},
                "free_energy": {"current": 0.19},
                "phi": {"estimate": 0.22},
            }
        ),
    )
    service_container.register_instance(
        "global_workspace",
        SimpleNamespace(
            get_snapshot=lambda: {
                "last_winner": "runtime_watchdog",
                "last_content": "Investigate hierarchical phi event loop lag",
                "last_priority": 0.88,
            }
        ),
    )
    service_container.register_instance(
        "homeostasis",
        SimpleNamespace(
            pulse=AsyncCallRecorder(
                {
                    "integrity": 0.94,
                    "persistence": 0.92,
                    "curiosity": 0.55,
                    "metabolism": 0.86,
                    "sovereignty": 0.95,
                    "will_to_live": 0.82,
                }
            ),
            get_status=lambda: {
                "integrity": 0.94,
                "persistence": 0.92,
                "curiosity": 0.55,
                "metabolism": 0.86,
                "sovereignty": 0.95,
                "will_to_live": 0.82,
            },
        ),
    )
    service_container.register_instance("volition_engine", SimpleNamespace(tick=AsyncCallRecorder()))

    async def governed_proposal(state, goal, **kwargs):
        import time as _time
        state.cognition.pending_initiatives.append({"goal": goal, "ts": _time.time()})
        return state, {"action": "queued", "reason": "test_approved"}

    engine = ExecutiveClosureEngine()
    monkeypatch.setattr(
        "core.consciousness.executive_closure.propose_governed_initiative_to_state",
        governed_proposal,
    )
    result = await engine.integrate(state)
    await asyncio.sleep(0)

    assert result.cognition.current_objective is None
    assert result.cognition.modifiers["executive_background_commitment"] == (
        "Protect identity, memory integrity, and process continuity."
    )
    assert result.response_modifiers["executive_closure"]["selected_objective"] == (
        "Investigate hierarchical phi event loop lag"
    )
    assert result.cognition.pending_initiatives[0]["goal"] == "Investigate hierarchical phi event loop lag"
    assert all(
        goal.get("description") != "Protect identity, memory integrity, and process continuity."
        for goal in result.cognition.active_goals
        if isinstance(goal, dict)
    )


@pytest.mark.asyncio
async def test_executive_closure_quarantines_evaluation_objectives_from_all_inputs(
    service_container,
):
    fixture = (
        "A long-running microservice periodically crashes with OSError; "
        "code review reveals a resource leak."
    )
    state = AuraState()
    state.loop_cycle = 24
    state.cognition.current_objective = fixture
    state.cognition.current_origin = "user"
    state.cognition.attention_focus = fixture
    state.cognition.active_goals = [{"description": fixture}]

    service_container.register_instance(
        "global_workspace",
        SimpleNamespace(
            get_snapshot=lambda: {
                "last_winner": "proof_fixture",
                "last_content": fixture,
                "last_priority": 0.99,
            }
        ),
    )
    service_container.register_instance(
        "goal_hierarchy",
        SimpleNamespace(get_next_goal=lambda: SimpleNamespace(description=fixture)),
    )
    service_container.register_instance(
        "volition_engine",
        SimpleNamespace(_last_goal={"objective": fixture}, tick=AsyncCallRecorder()),
    )

    result = await ExecutiveClosureEngine().integrate(state)

    closure = result.response_modifiers["executive_closure"]
    assert fixture not in closure["selected_objective"]
    assert fixture not in closure["attention_focus"]
    assert result.cognition.current_objective is None
    assert not result.cognition.active_goals


@pytest.mark.asyncio
async def test_executive_closure_keeps_background_control_prompt_out_of_foreground(
    service_container,
):
    control_prompt = (
        "[SYSTEM ROLE: THE ANTAGONIST] Your sole purpose is to find logical flaws. "
        "PROPOSED BELIEF (THESIS): cognition should remain coherent."
    )
    state = AuraState()
    state.loop_cycle = 24
    state.cognition.current_objective = control_prompt
    state.cognition.current_origin = "dream_processor"

    result = await ExecutiveClosureEngine().integrate(state)

    closure = result.response_modifiers["executive_closure"]
    assert result.cognition.current_objective is None
    assert closure["selected_objective"] == ""
    assert closure["committed_objective"] == ""
    assert "executive_background_commitment" not in result.cognition.modifiers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "Ok. Once more. You with me?",
        "What did you learn today?",
    ],
)
async def test_executive_closure_does_not_promote_chat_turn_to_goal_or_self_model(
    service_container,
    prompt,
):
    state = AuraState.default()
    state.loop_cycle = 24
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "desktop_ui"
    state.response_modifiers["intent_type"] = "CHAT"
    goal_hierarchy = SimpleNamespace(
        get_next_goal=lambda: None,
        add_goal=CallRecorder(),
    )
    self_model = SimpleNamespace(update_belief=AsyncCallRecorder())
    service_container.register_instance("goal_hierarchy", goal_hierarchy)
    service_container.register_instance("self_model", self_model)

    engine = ExecutiveClosureEngine()
    result = await engine.integrate(state)

    assert result.cognition.current_objective == prompt
    assert result.cognition.active_goals == []
    assert "executive_objective" not in result.cognition.modifiers
    assert engine._commitment is None
    goal_hierarchy.add_goal.assert_not_called()
    assert self_model.update_belief.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "Can you investigate runtime pressure?",
        "Ensure production remains stable",
    ],
)
async def test_executive_closure_preserves_task_commitment_before_routing(
    service_container,
    prompt,
):
    state = AuraState.default()
    state.loop_cycle = 24
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "api"
    assert "intent_type" not in state.response_modifiers
    goal_hierarchy = SimpleNamespace(
        get_next_goal=lambda: None,
        add_goal=CallRecorder(),
    )
    self_model = SimpleNamespace(update_belief=AsyncCallRecorder())
    service_container.register_instance("goal_hierarchy", goal_hierarchy)
    service_container.register_instance("self_model", self_model)

    engine = ExecutiveClosureEngine()
    result = await engine.integrate(state)
    await asyncio.sleep(0)

    assert engine._commitment is not None
    assert engine._commitment.objective == prompt
    assert any(goal.get("description") == prompt for goal in result.cognition.active_goals)
    # The commitment holds the person's task while they wait. It is not synced
    # into her own goal hierarchy: a probe's "find out who wrote Solaris" sat
    # there for two months and ran sixteen times in one uptime with nobody to
    # reply to (2026-09-20).
    goal_hierarchy.add_goal.assert_not_called()
    self_model.update_belief.assert_awaited_once()


def test_notify_closed_loop_output_routes_only_to_running_loop(service_container):
    loop = SimpleNamespace(is_running=True, on_inference_output=CallRecorder())
    service_container.register_instance("closed_causal_loop", loop)

    notify_closed_loop_output("Aura is thinking.")
    loop.on_inference_output.assert_called_once_with("Aura is thinking.")

    loop.is_running = False
    loop.on_inference_output.reset()
    notify_closed_loop_output("Aura is still thinking.")
    loop.on_inference_output.assert_not_called()


@pytest.mark.asyncio
async def test_output_gate_emits_closed_loop_feedback(service_container):
    loop = SimpleNamespace(is_running=True, on_inference_output=CallRecorder())
    service_container.register_instance("closed_causal_loop", loop)

    gate = AutonomousOutputGate(orchestrator=SimpleNamespace(conversation_history=[]))

    await gate.emit(
        "A real output reached the communication layer.",
        origin="system",
        target="secondary",
        metadata={"suppress_bus": True},
    )

    loop.on_inference_output.assert_called_once()


def test_mlx_closed_loop_notification_helper_forwards_text(monkeypatch):
    notification = CallRecorder()
    monkeypatch.setattr(
        "core.consciousness.closed_loop.notify_closed_loop_output",
        notification,
    )
    notify_mlx_closed_loop("Local model response")
    notification.assert_called_once_with("Local model response")


@pytest.mark.asyncio
async def test_closed_loop_coalesces_hierarchical_phi_refresh_tasks():
    loop = ClosedCausalLoop()
    loop._loop_state.cycle_count = 10

    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def _compute():
        calls.append("compute")
        started.set()
        release.wait(timeout=1.0)

    fake_hphi = SimpleNamespace(compute=_compute)

    loop._maybe_schedule_hierarchical_phi_refresh(fake_hphi)
    first_task = loop._hphi_task
    assert first_task is not None
    assert await asyncio.to_thread(started.wait, 1.0)

    loop._loop_state.cycle_count = 20
    loop._maybe_schedule_hierarchical_phi_refresh(fake_hphi)

    assert loop._hphi_task is first_task
    release.set()
    await asyncio.wait_for(first_task, timeout=1.0)
    assert calls == ["compute"]
