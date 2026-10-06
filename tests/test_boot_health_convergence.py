from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

from core.brain import llm_health_router as router_module
from core.brain.llm_health_router import HealthAwareLLMRouter
from core.fictional_ai_synthesis import DistributedResilienceCore
from core.orchestrator.main import RobustOrchestrator
from core.runtime.control_plane import (
    AdmissionPriority,
    AdmissionRequest,
    ResourceAdmissionController,
    WorkClass,
)
from core.runtime.resource_observation import SimulatedResourceObserver
from core.runtime.runtime_pressure import UnifiedRuntimePressure
from core.utils.concurrency import EventLoopMonitor


class _RunningTask:
    def done(self) -> bool:
        return False


def test_event_loop_incident_does_not_claim_the_monitor_task_died(monkeypatch):
    monkeypatch.delenv("AURA_EVENT_LOOP_MONITOR_FAILURE_RECOVERY_S", raising=False)
    monitor = EventLoopMonitor(threshold=0.5, interval=1.0)
    monitor._task = _RunningTask()
    monitor._started_at = time.perf_counter()
    monitor._capture_lag_sample(0.01)
    monitor._last_failure_at = time.time()
    monitor._last_failure_reason = "hard event-loop lag 7.0294s exceeded 5.00s"
    monitor._last_incident_at = monitor._last_failure_at
    monitor._last_incident_reason = monitor._last_failure_reason
    monitor._incident_count = 1

    status = monitor.get_status()

    assert status["alive"] is True
    assert status["running"] is True
    assert status["healthy"] is False
    assert status["incident_active"] is True
    assert status["last_incident_reason"] == monitor._last_failure_reason
    assert monitor.failure_recovery_window_s <= 30.0


def test_runtime_pressure_uses_fresh_signal_while_retaining_incident(monkeypatch):
    monitor = type(
        "Monitor",
        (),
        {
            "get_status": staticmethod(
                lambda: {
                    "alive": True,
                    "running": True,
                    "healthy": False,
                    "incident_active": True,
                    "last_lag_s": 0.012,
                    "last_sample_at_unix": time.time(),
                    "sample_age_s": 0.02,
                    "sample_fresh": True,
                }
            )
        },
    )()
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: monitor if name == "event_loop_monitor" else default,
    )
    pressure = UnifiedRuntimePressure(
        observer=SimulatedResourceObserver(
            scenario_id="fresh-signal-active-incident",
            memory_percent=40.0,
        )
    )

    snapshot = pressure.runtime_pressure_snapshot()

    assert snapshot["loop_monitor_running"] is True
    assert snapshot["loop_monitor_healthy"] is False
    assert snapshot["loop_monitor_incident_active"] is True
    assert snapshot["loop_lag_s"] == 0.012
    assert "loop_monitor_unavailable" not in snapshot["red_zones"]
    assert snapshot["pressure_ok"] is True


def test_background_model_admission_recovers_before_incident_history_expires():
    controller = ResourceAdmissionController(
        pressure_provider=lambda: {
            "loop_lag_s": 0.012,
            "loop_lag_sample_age_s": 0.02,
            "loop_lag_sample_fresh": True,
            "loop_monitor_alive": True,
            "loop_monitor_running": True,
            "loop_monitor_healthy": False,
            "loop_monitor_incident_active": True,
        }
    )

    async def scenario() -> None:
        decision = await controller.acquire(
            AdmissionRequest(
                owner="mlx.model_load:cortex-recovery",
                work_class=WorkClass.MODEL_LOAD,
                lane="cortex",
                priority=AdmissionPriority.BACKGROUND,
                timeout_s=0.0,
            )
        )
        assert decision.admitted is True
        await controller.release(decision.lease_id)

    asyncio.run(scenario())


def test_background_gate_wait_never_force_aborts_serving_worker(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(router_module, "_GENERATION_GATE", gate)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FORCED_LEASES", set())
    monkeypatch.setattr(router_module, "_GENERATION_GATE_NEXT_LEASE_ID", 0)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_WAIT_S", 0.02)
    monkeypatch.setattr(router_module, "_BACKGROUND_GENERATION_GATE_WAIT_S", 0.01)
    assert gate.acquire(False) is True
    lease_id = router_module._mark_generation_gate_acquired("system:active_serving_work")
    router = HealthAwareLLMRouter()
    monkeypatch.setattr(router, "_background_suppression_result", lambda **_kwargs: None)

    async def no_adapter(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(router, "_maybe_route_expert_adapter", no_adapter)
    monkeypatch.setattr(
        router,
        "force_abort_active_generation",
        lambda reason="": (_ for _ in ()).throw(
            AssertionError(f"background work attempted force-abort: {reason}")
        ),
    )

    try:
        result = asyncio.run(
            router.generate_with_metadata(
                "repair analysis",
                origin="system",
                purpose="healing_shard",
                is_background=True,
            )
        )
        assert result["ok"] is False
        assert result["endpoint"] == "generation_gate_background_deferred"
        assert result["deferred"] is True
        assert result["retryable"] is True
        assert gate.acquire(False) is False
    finally:
        router_module._release_generation_gate_after_call(lease_id)


def test_a_background_request_from_a_kernel_phase_does_not_wait_with_the_lock_held(monkeypatch):
    """LIVE 2026-10-06: a background UnitaryResponsePhase waited 5.16 s for the gate, kernel lock held."""
    from core.runtime.kernel_phase_context import running_a_kernel_phase

    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(router_module, "_GENERATION_GATE", gate)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FORCED_LEASES", set())
    monkeypatch.setattr(router_module, "_GENERATION_GATE_NEXT_LEASE_ID", 0)
    waits = []
    real_acquire = router_module._acquire_generation_gate_slot

    async def recording_acquire(wait_s):
        waits.append(wait_s)
        return await real_acquire(0.0)

    monkeypatch.setattr(router_module, "_acquire_generation_gate_slot", recording_acquire)
    assert gate.acquire(False) is True
    lease_id = router_module._mark_generation_gate_acquired("stream_narrative:unknown")
    router = HealthAwareLLMRouter()
    monkeypatch.setattr(router, "_background_suppression_result", lambda **_kwargs: None)

    async def no_adapter(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(router, "_maybe_route_expert_adapter", no_adapter)

    async def ask():
        return await router.generate_with_metadata(
            "what to say next", origin="system", purpose="healing_shard", is_background=True
        )

    async def ask_from_a_phase():
        with running_a_kernel_phase("UnitaryResponsePhase"):
            return await ask()

    try:
        inside = asyncio.run(ask_from_a_phase())
        outside = asyncio.run(ask())
        assert inside["endpoint"] == outside["endpoint"] == "generation_gate_background_deferred"
        assert waits[0] == 0.0
        assert waits[1] == min(
            router_module._GENERATION_GATE_WAIT_S, router_module._BACKGROUND_GENERATION_GATE_WAIT_S
        )
    finally:
        router_module._release_generation_gate_after_call(lease_id)


def test_generation_gate_snapshot_reports_true_oldest_lease(monkeypatch):
    now = time.time()
    monkeypatch.setattr(
        router_module,
        "_GENERATION_GATE_ACTIVE_LEASES",
        {
            1: (now - 90.0, "oldest:work"),
            2: (now - 1.0, "newest:work"),
        },
    )
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})

    snapshot = router_module.generation_gate_snapshot()

    assert snapshot["oldest"]["lease_id"] == 1
    assert snapshot["oldest"]["owner"] == "oldest:work"


def test_resilience_core_reports_transitions_not_every_poll(caplog):
    core = DistributedResilienceCore()
    core._failure_threshold = 2
    core._reminder_interval_s = 600.0
    core.register_subsystem("orchestrator")

    with caplog.at_level(logging.INFO, logger="Aura.FictionalSynthesis"):
        core._report_failure("orchestrator", "runtime_control_plane not ready")
        core._report_failure("orchestrator", "runtime_control_plane not ready")
        core._report_failure("orchestrator", "runtime_control_plane not ready")
        core._report_success("orchestrator")

    messages = [record.getMessage() for record in caplog.records]
    assert sum("became UNHEALTHY" in message for message in messages) == 1
    assert sum("remains UNHEALTHY" in message for message in messages) == 0
    assert sum("RECOVERED" in message for message in messages) == 1
    assert core._subsystems["orchestrator"].healthy is True


def test_orchestrator_preserves_canonical_failed_dependency_reason(monkeypatch):
    from core.runtime import health_contract

    report = {
        "status": "critical",
        "healthy": False,
        "operational": False,
        "failures": {
            "critical": [
                {
                    "container_key": "runtime_control_plane",
                    "error": "is_ready() returned False",
                }
            ],
            "important": [
                {
                    "container_key": "event_loop_monitor",
                    "error": "hard event-loop lag 7.0294s",
                }
            ],
            "optional": [],
        },
        "probe_blockers": ["runtime_required_probes"],
    }
    monkeypatch.setattr(health_contract, "runtime_health_report", lambda: report)
    monkeypatch.setattr(health_contract, "required_probe_status", lambda _report: {})
    monkeypatch.setattr(
        health_contract,
        "required_probe_groups_pass",
        lambda _probes: False,
    )
    orchestrator = object.__new__(RobustOrchestrator)
    orchestrator.status = SimpleNamespace(
        running=True,
        initialized=True,
        last_error="",
        healthy=True,
    )
    orchestrator.stats = {}

    assert orchestrator.health_check() is False
    assert orchestrator._last_runtime_health_phase == "critical"
    assert "critical:runtime_control_plane:is_ready() returned False" in (
        orchestrator._last_health_reason
    )
    assert "important:event_loop_monitor:hard event-loop lag 7.0294s" in (
        orchestrator._last_health_reason
    )
