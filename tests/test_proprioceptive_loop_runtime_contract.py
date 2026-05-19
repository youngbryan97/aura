from pathlib import Path
from types import SimpleNamespace

import pytest


class DictContainer:
    def __init__(self, services=None, *, fail_names=None):
        self.services = services or {}
        self.fail_names = set(fail_names or ())

    def get(self, name, default=None):
        if name in self.fail_names:
            raise RuntimeError(f"{name} lookup failed")
        return self.services.get(name, default)


class HealthyFeedbackProcessor:
    def get_unhealthy_limbs(self, threshold=0.5):
        return []

    def detect_action_stagnation(self, window=10):
        return {"stagnant": False}


class FakePsutil:
    @staticmethod
    def cpu_percent(interval=0):
        return 17.5

    @staticmethod
    def virtual_memory():
        return SimpleNamespace(percent=42.0)

    @staticmethod
    def sensors_temperatures():
        return {"core": [SimpleNamespace(current=39.5)]}

    @staticmethod
    def sensors_battery():
        return SimpleNamespace(percent=88.0)


def test_proprioceptive_loop_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/phases/proprioceptive_loop.py")) == []


@pytest.mark.asyncio
async def test_proprioception_marks_partial_sensor_loss_without_losing_body_schema(monkeypatch):
    import core.phases.proprioceptive_loop as proprioceptive_module
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    class PartialPsutil(FakePsutil):
        @staticmethod
        def sensors_temperatures():
            reason = "thermal bus unavailable"
            raise OSError(reason)

    class BrokenRouter:
        def get_stats(self):
            reason = "router metrics timed out"
            raise TimeoutError(reason)

    monkeypatch.setattr(proprioceptive_module, "psutil", PartialPsutil)
    loop = ProprioceptiveLoop(
        DictContainer(
            {
                "feedback_processor": HealthyFeedbackProcessor(),
                "llm_router": BrokenRouter(),
            }
        )
    )

    state = await loop.execute(AuraState.default())

    assert state.soma.hardware["cpu_usage"] == 17.5
    assert state.soma.hardware["vram_usage"] == 42.0
    assert state.soma.hardware["temperature_available"] is False
    assert state.soma.latency["token_velocity_available"] is False
    assert state.soma.hardware["proprioceptive_status"] == "degraded"
    assert set(state.soma.hardware["proprioceptive_degraded_channels"]) >= {
        "thermal_sensor",
        "token_velocity",
    }


@pytest.mark.asyncio
async def test_proprioception_surfaces_motor_and_stagnation_causally(monkeypatch):
    import core.phases.proprioceptive_loop as proprioceptive_module
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    class MotorCortex:
        def drain_pending_reports(self):
            return [
                SimpleNamespace(
                    success=False,
                    latency_ms=12.0,
                    handler_name="screen",
                    result_summary="capture timeout",
                )
            ]

    class StagnantFeedbackProcessor(HealthyFeedbackProcessor):
        def get_unhealthy_limbs(self, threshold=0.5):
            return ["screen_capture"]

        def detect_action_stagnation(self, window=10):
            return {
                "stagnant": True,
                "failure_rate": 0.8,
                "loop_detected": True,
                "loop_length": 3,
                "degraded_limbs": [{"name": "screen_capture"}],
                "recent_outcomes": [
                    {"action": "screenshot", "outcome": "timeout"},
                    {"action": "screenshot", "outcome": "timeout"},
                ],
            }

    monkeypatch.setattr(proprioceptive_module, "psutil", FakePsutil)
    loop = ProprioceptiveLoop(
        DictContainer(
            {
                "feedback_processor": StagnantFeedbackProcessor(),
                "motor_cortex": MotorCortex(),
            }
        )
    )

    state = await loop.execute(AuraState.default())

    assert state.soma.hardware["motor_cortex_actions"] == 1
    assert state.soma.hardware["motor_cortex_failures"] == 1
    assert state.soma.hardware["unhealthy_limb_count"] == 1
    assert state.soma.hardware["action_stagnation"] is True
    assert state.soma.hardware["action_stagnation_available"] is True
    assert any(
        isinstance(item, dict)
        and item.get("metadata", {}).get("type") == "proprioceptive_percept"
        for item in state.cognition.working_memory
    )


@pytest.mark.asyncio
async def test_autonomic_reflex_failures_are_causal_not_fatal():
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    class BrokenInhibition:
        async def inhibit(self, subsystem, *, duration, reason):
            failure = f"{subsystem} cannot be inhibited"
            raise RuntimeError(failure)

    state = AuraState.default()
    state.soma.hardware["cpu_usage"] = 95.0
    loop = ProprioceptiveLoop(
        DictContainer(
            {
                "inhibition_manager": BrokenInhibition(),
                "qualia_synthesizer": SimpleNamespace(q_vector=[0, 0, 0, 0, 0, 0.95]),
                "homeostasis": SimpleNamespace(integrity=0.1),
            }
        )
    )

    await loop._autonomic_reflex_check(state)

    assert state.soma.hardware["autonomic_reflex_degraded"] is True
    assert "autonomic_reflex" in state.soma.hardware["proprioceptive_degraded_channels"]
