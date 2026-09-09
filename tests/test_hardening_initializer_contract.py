from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator.initializers import hardening
from core.runtime.control_plane import reset_runtime_control_plane
from core.runtime.errors import get_degradation_tracker
from core.runtime.health_contract import RUNTIME_CONTRACT, ServiceTier
from core.runtime.resource_observation import ProcessObservation


class _Validator:
    async def run_all(self) -> bool:
        return True


class _Supervisor:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        alive_after_start: bool = True,
    ) -> None:
        self.error = error
        self.alive_after_start = alive_after_start
        self.start_calls = 0
        self.stop_calls = 0
        self._alive = False

    async def start(self) -> None:
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self._alive = self.alive_after_start

    async def stop(self) -> None:
        self.stop_calls += 1
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _EventLoopMonitor:
    threshold = 0.25
    active_threshold = 5.0

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        alive_after_start: bool = True,
    ) -> None:
        self.error = error
        self.alive_after_start = alive_after_start
        self.start_calls = 0
        self._alive = False
        self._running = False

    def start(self) -> None:
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self._alive = self.alive_after_start
        self._running = True

    def is_alive(self) -> bool:
        return self._alive

    def is_running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        self._alive = False
        self._running = False


@pytest.fixture(autouse=True)
def isolated_contract_state():
    ServiceContainer.clear()
    reset_runtime_control_plane()
    get_degradation_tracker().reset()
    yield
    ServiceContainer.clear()
    reset_runtime_control_plane()
    get_degradation_tracker().reset()


def _patch_dependencies(monkeypatch, *, reaper, hypervisor, monitor) -> None:
    import core.ops.hypervisor as hypervisor_module
    import core.ops.lymphatic_reaper as reaper_module
    import core.startup.validator as validator_module
    import core.utils.concurrency as concurrency_module

    monkeypatch.setattr(validator_module, "get_validator", lambda: _Validator())
    monkeypatch.setattr(reaper_module, "get_reaper", lambda: reaper)
    monkeypatch.setattr(hypervisor_module, "get_hypervisor", lambda: hypervisor)
    monkeypatch.setattr(concurrency_module, "EventLoopMonitor", lambda: monitor)


def test_hardening_initializer_leaves_failed_supervisor_unregistered_in_dev(monkeypatch):
    monkeypatch.setattr(hardening.config, "env", hardening.Environment.DEV)
    reaper = _Supervisor(error=RuntimeError("reaper spawn failed"))
    hypervisor = _Supervisor()
    monitor = _EventLoopMonitor()
    _patch_dependencies(monkeypatch, reaper=reaper, hypervisor=hypervisor, monitor=monitor)

    orchestrator = SimpleNamespace()
    asyncio.run(hardening.init_hardening_layer(orchestrator))

    assert ServiceContainer.get("reaper", default=None) is None
    assert ServiceContainer.get("hypervisor", default=None) is hypervisor
    assert ServiceContainer.get("event_loop_monitor", default=None) is monitor
    assert orchestrator.hardening_status["reaper"]["state"] == "failed"
    assert orchestrator.hardening_status["hypervisor"]["state"] == "online"
    assert orchestrator.hardening_status["runtime_control_plane"]["registered"] is True
    assert ServiceContainer.get("runtime_control_plane", default=None) is not None
    assert ServiceContainer.get("resource_admission", default=None) is not None
    assert "unregistered" in get_degradation_tracker().recent(subsystem="hardening")[-1].action


def test_hardening_initializer_fails_closed_on_dead_monitor_in_prod(monkeypatch):
    monkeypatch.setattr(hardening.config, "env", hardening.Environment.PROD)
    reaper = _Supervisor()
    hypervisor = _Supervisor()
    monitor = _EventLoopMonitor(alive_after_start=False)
    _patch_dependencies(monkeypatch, reaper=reaper, hypervisor=hypervisor, monitor=monitor)

    with pytest.raises(RuntimeError, match="event_loop_monitor"):
        asyncio.run(hardening.init_hardening_layer(SimpleNamespace()))

    assert ServiceContainer.get("reaper", default=None) is reaper
    assert ServiceContainer.get("hypervisor", default=None) is hypervisor
    assert ServiceContainer.get("event_loop_monitor", default=None) is None
    assert get_degradation_tracker().recent(subsystem="hardening")[-1].severity == "critical"


def test_control_plane_does_not_restart_running_monitor_during_lag_recovery(monkeypatch):
    monkeypatch.setattr(hardening.config, "env", hardening.Environment.DEV)
    reaper = _Supervisor()
    hypervisor = _Supervisor()
    monitor = _EventLoopMonitor()
    _patch_dependencies(monkeypatch, reaper=reaper, hypervisor=hypervisor, monitor=monitor)

    async def _exercise():
        await hardening.init_hardening_layer(SimpleNamespace())
        monitor._alive = False
        plane = ServiceContainer.get("runtime_control_plane")
        return await plane.reconcile_once()

    report = asyncio.run(_exercise())

    assert monitor.is_alive() is False
    assert monitor.is_running() is True
    assert monitor.start_calls == 1
    assert report["services"]["event_loop_monitor"]["observed_state"] == "ready"
    assert not [
        action
        for action in report["actions"]
        if action["service"] == "event_loop_monitor"
    ]


def test_long_run_supervisors_are_part_of_runtime_health_contract():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key in {"reaper", "hypervisor", "event_loop_monitor"}
    }

    assert set(required) == {"reaper", "hypervisor", "event_loop_monitor"}
    assert all(requirement.tier == ServiceTier.IMPORTANT for requirement in required.values())
    assert all(requirement.liveness_check == "is_alive" for requirement in required.values())


def test_control_plane_retains_hypervisor_during_measured_lag_recovery(monkeypatch):
    from core.ops.hypervisor import Hypervisor

    monkeypatch.setattr(hardening.config, "env", hardening.Environment.DEV)
    hypervisor = Hypervisor()
    _patch_dependencies(
        monkeypatch, reaper=_Supervisor(), hypervisor=hypervisor,
        monitor=_EventLoopMonitor(),
    )

    async def exercise():
        await hardening.init_hardening_layer(SimpleNamespace())
        original_task = hypervisor._task
        hypervisor._last_severe_lag_at = time.time()
        hypervisor._last_failure_reason = "severe event-loop lag 15.708s"
        try:
            assert hypervisor.is_alive() is False
            plane = ServiceContainer.get("runtime_control_plane")
            report = await plane.reconcile_once()
            assert hypervisor._task is original_task
            assert not original_task.done()
            assert hypervisor.is_alive() is False
            assert not [a for a in report["actions"] if a["service"] == "hypervisor"]
        finally:
            await hypervisor.stop()

    asyncio.run(exercise())


def test_hypervisor_lifecycle_probe_distinguishes_running_from_recovered():
    from core.ops.hypervisor import Hypervisor

    async def exercise():
        hypervisor = Hypervisor()
        assert hypervisor.is_running() is False
        await hypervisor.start()
        try:
            hypervisor._last_severe_lag_at = time.time()
            assert hypervisor.is_running() is True
            assert hypervisor.is_alive() is False
            hypervisor._task.cancel()
            await asyncio.gather(hypervisor._task, return_exceptions=True)
            assert hypervisor.is_running() is False
            dead_task = hypervisor._task
            await hypervisor.start()
            assert hypervisor._task is not dead_task
            assert hypervisor.is_running() is True
            assert hypervisor.is_alive() is False
        finally:
            await hypervisor.stop()

    asyncio.run(exercise())


class _ChildProcess:
    def __init__(self, status: str, *, pid: int = 1001) -> None:
        self._status = status
        self.pid = pid
        self.terminated = False
        self.waited = False
        self.wait_timeout = object()

    def status(self) -> str:
        return self._status

    def create_time(self) -> float:
        return 0.0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> None:
        self.waited = True
        self.wait_timeout = timeout


def test_lymphatic_reaper_retains_long_lived_children_without_opt_in(
    monkeypatch,
    tmp_path,
    resource_observer,
):
    import core.ops.lymphatic_reaper as reaper_module

    monkeypatch.delenv("AURA_REAPER_TERMINATE_LONG_CHILDREN", raising=False)
    child = _ChildProcess("sleeping")
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=child.pid,
                ppid=os.getpid(),
                create_time=0.0,
                status="sleeping",
                name="test-child",
                cmdline=("test-child",),
                rss_bytes=0,
                ancestor_pids=(os.getpid(),),
            )
        ]
    )
    monkeypatch.setattr(reaper_module.psutil, "Process", lambda _pid: child)
    monkeypatch.setattr(reaper_module.time, "time", lambda: reaper_module.LONG_CHILD_AGE_S + 60.0)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._hunt_orphans() == 0
    assert child.terminated is False
    assert child.waited is False


def test_lymphatic_reaper_reaps_zombie_children(monkeypatch, tmp_path, resource_observer):
    import core.ops.lymphatic_reaper as reaper_module

    child = _ChildProcess(reaper_module.psutil.STATUS_ZOMBIE)
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=child.pid,
                ppid=os.getpid(),
                create_time=0.0,
                status="zombie",
                name="test-zombie",
                cmdline=("test-zombie",),
                rss_bytes=0,
                ancestor_pids=(os.getpid(),),
            )
        ]
    )
    monkeypatch.setattr(reaper_module.psutil, "Process", lambda _pid: child)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._hunt_orphans() == 1
    assert child.waited is True
    assert child.wait_timeout == 0
    assert child.terminated is False


def test_lymphatic_reaper_unlinks_stale_symlink_without_touching_target(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    link_path = tmp_dir / "old-target-link"
    link_path.symlink_to(target_dir, target_is_directory=True)
    os.utime(link_path, (1.0, 1.0), follow_symlinks=False)
    monkeypatch.setattr(reaper_module.time, "time", lambda: reaper_module.STALE_TMP_AGE_S + 2.0)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._filesystem_sweep() >= 0
    assert not link_path.exists()
    assert target_dir.exists()


def test_lymphatic_reaper_sweep_preserves_step_errors(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    marker = {"called": False}

    def failing_hunt() -> int:
        marker["called"] = True
        assert marker["called"] is True
        raise RuntimeError("process scan unavailable")

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)
    monkeypatch.setattr(reaper, "_hunt_orphans", failing_hunt)
    monkeypatch.setattr(reaper, "_filesystem_sweep", lambda: 128)
    monkeypatch.setattr(reaper, "_defragment_memory", lambda: True)

    result = asyncio.run(reaper.sweep())

    assert result["processes_reaped"] == 0
    assert result["storage_reclaimed_bytes"] == 128
    assert "hunt_orphans" in result["step_errors"]
    assert reaper.get_status()["last_sweep_status"]["storage_reclaimed_bytes"] == 128


def test_metabolic_monitor_dispatches_pressure_mitigation(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module
    import core.resource.resource_governor as governor_module

    actions = []

    class _Process:
        def cpu_percent(self) -> float:
            return 145.0

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=2048 * 1024 * 1024)

    class _Governor:
        def execute_eviction(self, tier) -> int:
            actions.append(tier.value)
            return 1

    monkeypatch.setattr(metabolic_module.psutil, "Process", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(
        metabolic_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=96.5, total=64 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(metabolic_module.psutil, "disk_usage", lambda _path: SimpleNamespace(percent=72.0))
    monkeypatch.setattr(governor_module, "get_resource_governor", lambda: _Governor())

    monitor = metabolic_module.MetabolicMonitor(ram_threshold_mb=1024, cpu_threshold=80.0)
    monkeypatch.setattr(monitor, "_sync_registry", lambda _snapshot: None)

    snapshot = monitor.get_current_metabolism()

    assert snapshot.pressure_state == "critical"
    assert actions == ["aggressive"]
    assert monitor.get_status_report()["pressure_actions_total"] == 1


def test_metabolic_monitor_default_ram_threshold_scales_for_primary_model_lane(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module

    monkeypatch.delenv("AURA_METABOLIC_RAM_THRESHOLD_MB", raising=False)
    monkeypatch.setattr(
        metabolic_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=62.0, total=64 * 1024 * 1024 * 1024),
    )

    monitor = metabolic_module.MetabolicMonitor()

    assert monitor.ram_threshold_mb == 55705
    assert (
        monitor._classify_pressure(
            cpu_percent=20.0,
            rss_mb=40960.0,
            ram_percent=62.0,
            disk_percent=50.0,
            health_score=0.90,
        )
        == "nominal"
    )


def test_metabolic_monitor_cpu_only_pressure_does_not_dispatch_eviction(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module
    import core.resource.resource_governor as governor_module

    actions = []

    class _Governor:
        def execute_eviction(self, tier) -> int:
            actions.append(tier.value)
            return 1

    monkeypatch.setattr(governor_module, "get_resource_governor", lambda: _Governor())

    monitor = metabolic_module.MetabolicMonitor(ram_threshold_mb=45_875, cpu_threshold=80.0)
    snapshot = metabolic_module.MetabolismSnapshot(
        cpu_percent=91.0,
        ram_rss_mb=20.0,
        ram_percent=68.0,
        disk_usage_percent=3.0,
        llm_latency_avg=0.5,
        health_score=0.84,
        pressure_state="stressed",
    )

    monitor._apply_pressure_controls(snapshot)

    assert actions == []
    assert monitor.get_status_report()["pressure_actions_total"] == 0


def test_metabolic_monitor_explicit_ram_threshold_still_escalates(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module

    monkeypatch.setattr(
        metabolic_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=62.0, total=64 * 1024 * 1024 * 1024),
    )

    monitor = metabolic_module.MetabolicMonitor(ram_threshold_mb=1024)

    assert monitor.ram_threshold_mb == 1024
    assert (
        monitor._classify_pressure(
            cpu_percent=20.0,
            rss_mb=2048.0,
            ram_percent=62.0,
            disk_percent=50.0,
            health_score=0.90,
        )
        == "critical"
    )


def test_metabolic_monitor_loop_records_failure_without_exiting_silently(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module

    monitor = metabolic_module.MetabolicMonitor()
    calls = {"count": 0}

    def failing_sample():
        calls["count"] += 1
        monitor._running = False
        assert calls["count"] == 1
        raise RuntimeError("sample unavailable")

    monkeypatch.setattr(monitor, "get_current_metabolism", failing_sample)
    monitor._interval = 0.01
    monitor._running = True

    monitor._run_loop()

    assert monitor._consecutive_failures == 1
    recent = get_degradation_tracker().recent(subsystem="metabolic_monitor")
    assert recent[-1].severity == "degraded"
    assert "backed off" in recent[-1].action


def test_compute_cost_tracker_handles_corrupt_state_and_clamps_invalid_inputs(tmp_path):
    import core.ops.metabolic_monitor as metabolic_module

    state_path = tmp_path / "metabolic_state.json"
    state_path.write_text("{not json")

    tracker = metabolic_module.PersistentComputeCostTracker(state_path=state_path)
    ergs = tracker.record_operation("invalid", -10, float("nan"))

    assert tracker.total_ergs == 0.0
    assert ergs == 0.0
    assert tracker.cost_history[-1]["tokens"] == 0
    assert tracker.cost_history[-1]["duration"] == 0.0
    assert get_degradation_tracker().count("metabolic_monitor") >= 2


def test_aegis_pulse_restores_missing_mycelial_network(monkeypatch):
    import core.mycelium as mycelium_module
    from core.orchestrator.handlers import aegis

    class _MycelialNetwork:
        def __init__(self) -> None:
            self._aegis_locked = True

    monkeypatch.setattr(mycelium_module, "MycelialNetwork", _MycelialNetwork)
    orchestrator = SimpleNamespace()

    status = asyncio.run(aegis._aegis_pulse(orchestrator))

    assert status["state"] == "restored"
    assert status["locked"] is True
    assert ServiceContainer.get("mycelial_network")._aegis_locked is True


def test_aegis_pulse_marks_integrity_failure_when_lock_restore_fails(monkeypatch):
    import core.mycelium as mycelium_module
    from core.orchestrator.handlers import aegis

    class _MycelialNetwork:
        @classmethod
        async def restore_from_vault(cls) -> bool:
            return False

    monkeypatch.setattr(mycelium_module, "MycelialNetwork", _MycelialNetwork)
    ServiceContainer.register_instance("mycelial_network", SimpleNamespace(_aegis_locked=False))
    orchestrator = SimpleNamespace()

    status = asyncio.run(aegis._aegis_pulse(orchestrator))

    assert status["state"] == "failed_closed"
    assert status["reason"] == "true_lock_restore_failed"
    assert orchestrator._aegis_integrity_failed is True
    recent = get_degradation_tracker().recent(subsystem="aegis")
    assert recent[-1].severity == "critical"


@pytest.mark.parametrize("receipt", (False, None))
def test_aegis_pulse_reports_failed_vault_sync_and_keeps_retry_eligible(
    monkeypatch, receipt
):
    from core.orchestrator.handlers import aegis

    class _Mycelium:
        _aegis_locked = True

        @staticmethod
        async def vault_sync():
            return receipt

    ServiceContainer.register_instance("mycelial_network", _Mycelium())
    orchestrator = SimpleNamespace(_last_vault_sync=0.0)

    status = asyncio.run(
        aegis._aegis_pulse(orchestrator, vault_sync_interval_s=1.0)
    )

    assert status["state"] == "degraded"
    assert status["reason"] == "root_vault_sync_failed"
    assert status["vault_synced"] is False
    assert status["vault_sync_retry_eligible"] is True
    assert orchestrator._last_vault_sync == 0.0


def test_aegis_pulse_advances_sync_clock_only_after_truthy_receipt(monkeypatch):
    from core.orchestrator.handlers import aegis

    class _Mycelium:
        _aegis_locked = True

        @staticmethod
        async def vault_sync():
            return True

    ServiceContainer.register_instance("mycelial_network", _Mycelium())
    orchestrator = SimpleNamespace(_last_vault_sync=0.0)

    status = asyncio.run(
        aegis._aegis_pulse(orchestrator, vault_sync_interval_s=1.0)
    )

    assert status["state"] == "healthy"
    assert status["vault_synced"] is True
    assert orchestrator._last_vault_sync > 0.0


@pytest.mark.parametrize(
    "invalid_clock",
    (float("nan"), float("inf"), -1.0, "not-a-clock"),
)
def test_aegis_pulse_invalid_sync_clock_cannot_suppress_immediate_retry(
    monkeypatch, invalid_clock
):
    from core.orchestrator.handlers import aegis

    calls = []

    class _Mycelium:
        _aegis_locked = True

        @staticmethod
        async def vault_sync():
            calls.append("sync")
            return True

    ServiceContainer.register_instance("mycelial_network", _Mycelium())
    orchestrator = SimpleNamespace(_last_vault_sync=invalid_clock)

    status = asyncio.run(
        aegis._aegis_pulse(orchestrator, vault_sync_interval_s=60.0)
    )

    assert calls == ["sync"]
    assert status["state"] == "healthy"
    assert status["vault_synced"] is True
    assert orchestrator._last_vault_sync > 0.0


def test_aegis_loop_records_degraded_pulse_and_exits_on_stop(monkeypatch):
    from core.orchestrator.handlers import aegis

    class _StopEvent:
        def __init__(self) -> None:
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

    orchestrator = SimpleNamespace(_stop_event=_StopEvent())

    async def failing_pulse(_orch, *, vault_sync_interval_s: float) -> dict:
        assert vault_sync_interval_s == 60.0
        orchestrator._stop_event.set()
        raise RuntimeError("aegis pulse unavailable")

    monkeypatch.setattr(aegis, "_aegis_pulse", failing_pulse)

    asyncio.run(aegis.aegis_sentinel_loop(orchestrator, interval_s=0.01))

    assert orchestrator.aegis_status["state"] == "degraded"
    assert orchestrator.aegis_status["consecutive_failures"] == 1
    recent = get_degradation_tracker().recent(subsystem="aegis")
    assert recent[-1].severity == "degraded"
    assert "backed off" in recent[-1].action


def test_orchestrator_background_task_handler_records_and_routes_failure(monkeypatch):
    from core.orchestrator import orchestrator_types

    orchestrator_types._BACKGROUND_TASK_FAILURES.clear()
    routed = []

    def route_immune(task_name, exc) -> None:
        routed.append(("immune", task_name, type(exc).__name__))

    def route_morphogenesis(task_name, exc) -> None:
        routed.append(("morphogenesis", task_name, type(exc).__name__))

    async def run_task() -> None:
        async def failing_task() -> None:
            assert True
            raise RuntimeError("background failure")

        task = asyncio.create_task(failing_task(), name="unit-background")
        await asyncio.sleep(0)
        assert task.done()
        orchestrator_types._bg_task_exception_handler(task)

    monkeypatch.setattr(orchestrator_types, "_notify_immune_system", route_immune)
    monkeypatch.setattr(orchestrator_types, "_notify_morphogenesis", route_morphogenesis)

    asyncio.run(run_task())

    status = orchestrator_types.get_background_task_failure_status()
    assert status["recent_failure_count"] == 1
    assert status["recent_failures"][-1]["task"] == "unit-background"
    assert routed == [
        ("immune", "unit-background", "RuntimeError"),
        ("morphogenesis", "unit-background", "RuntimeError"),
    ]
    recent = get_degradation_tracker().recent(subsystem="orchestrator_types")
    assert recent[-1].severity == "degraded"
    assert "routed" in recent[-1].action


def test_legacy_orchestrator_types_bridge_uses_canonical_handler():
    import core.orchestrator_types as legacy_types
    from core.orchestrator import orchestrator_types

    assert legacy_types.SystemStatus is orchestrator_types.SystemStatus
    assert legacy_types._bg_task_exception_handler is orchestrator_types._bg_task_exception_handler


def test_affect_update_records_substrate_telemetry_failure_without_losing_affect():
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    class _Substrate:
        def update(self, **kwargs):
            assert isinstance(kwargs["valence"], float)
            raise RuntimeError("substrate unavailable")

        def get_state_summary_nowait(self):
            # The affect phase reads the substrate back as well as writing to
            # it. A stale snapshot is skipped rather than blended, so this
            # leaves the telemetry write as the only failing path and the test
            # keeps testing what it was written to test.
            return {"valence": 0.0, "arousal": 0.0, "snapshot_stale": True}

    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    ServiceContainer.register_instance("liquid_substrate", _Substrate())
    phase = AffectUpdatePhase(SimpleNamespace(organs={}))

    result = asyncio.run(phase.execute(state))

    assert result is state
    degraded = state.cognition.modifiers["affect_update_degraded"]
    assert degraded["stage"] == "substrate_telemetry"
    # By name, not by position. The phase records more than one thing about the
    # substrate now — a stale snapshot is noted at info severity — and asserting
    # on the last record makes this test about ordering rather than about the
    # telemetry failure it was written for.
    recent = get_degradation_tracker().recent(subsystem="affect_update")
    telemetry = [
        item
        for item in recent
        if item.severity == "warning" and "telemetry" in (item.action or "")
    ]
    assert telemetry, [item.action for item in recent]
    assert "substrate" in telemetry[-1].action


def test_affect_update_keeps_physiology_when_empathy_audit_fails():
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    class _Prober:
        def audit(self, state):
            assert state.affect is not None
            raise RuntimeError("audit unavailable")

    state = AuraState.default()
    kernel = SimpleNamespace(organs={"prober": SimpleNamespace(instance=_Prober())})
    phase = AffectUpdatePhase(kernel)

    phase._update_physiology(state.affect, state)

    assert state.affect.physiology["heart_rate"] >= 60
    degraded = state.cognition.modifiers["affect_update_degraded"]
    assert degraded["stage"] == "vk_empathy_audit"


def test_inference_phase_normalizes_router_payload():
    from core.phases.inference_phase import InferencePhase
    from core.service_names import ServiceNames
    from core.state.aura_state import AuraState

    class _Router:
        async def think(self, *_args, **_kwargs) -> str:
            return (
                '{"implicit_intent":"needs help","affective_subtext":"anxious",'
                '"momentum":"LOUD","conversation_hooks":"runtime health"}'
            )

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == ServiceNames.LLM_ROUTER else default

    state = AuraState.default()
    state.cognition.current_origin = "user"
    phase = InferencePhase(_Container())

    asyncio.run(phase.execute(state, objective="Can you check this?"))

    assert state.cognition.modifiers["inferred_intent"] == "needs help"
    assert state.cognition.modifiers["user_subtext"] == "anxious"
    assert state.cognition.modifiers["momentum"] == "flowing"
    assert state.cognition.modifiers["conversation_hooks"] == ["runtime health"]
    assert state.cognition.modifiers["deep_inference_status"]["status"] == "ok"


def test_inference_phase_marks_parse_failure_explicitly():
    from core.phases.inference_phase import InferencePhase
    from core.service_names import ServiceNames
    from core.state.aura_state import AuraState

    class _Router:
        async def think(self, *_args, **_kwargs) -> str:
            return "no structured object"

    class _Container:
        @staticmethod
        def get(name, default=None):
            return _Router() if name == ServiceNames.LLM_ROUTER else default

    state = AuraState.default()
    state.cognition.current_origin = "user"
    phase = InferencePhase(_Container())

    asyncio.run(phase.execute(state, objective="Can you check this?"))

    status = state.cognition.modifiers["deep_inference_status"]
    assert status["status"] == "degraded"
    recent = get_degradation_tracker().recent(subsystem="inference_phase")
    assert recent[-1].severity == "warning"
    assert "explicit degraded inference status" in recent[-1].action
