import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace

from core import process_manager as process_module
from core.process_manager import ManagedProcess, ProcessConfig, ProcessManager, ProcessState
from core.runtime.errors import get_degradation_tracker


def _target():
    return None


def test_start_failure_marks_process_failed_with_degradation(monkeypatch):
    class ProcessFactoryUnavailable:
        def __init__(self, *_args, **_kwargs):
            self.created = False
            raise RuntimeError("process factory unavailable")

    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        monkeypatch.setattr(process_module.mp, "Process", ProcessFactoryUnavailable)
        managed = ManagedProcess(ProcessConfig(name="factory_down", target=_target))

        started = await managed.start()

        assert started is False
        assert managed.state == ProcessState.FAILED
        assert any(
            "startup exception" in record.action
            for record in tracker.recent(subsystem="process_manager")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_restart_terminal_state_survives_incident_and_metric_failures(monkeypatch):
    incident_module = ModuleType("core.resilience.incident_manager")

    class IncidentSeverity:
        CRITICAL = "critical"

    def get_incident_manager():
        get_incident_manager.calls += 1
        raise RuntimeError("incident manager unavailable")

    get_incident_manager.calls = 0
    incident_module.IncidentSeverity = IncidentSeverity
    incident_module.get_incident_manager = get_incident_manager
    monkeypatch.setitem(sys.modules, "core.resilience.incident_manager", incident_module)

    metrics_module = ModuleType("core.observability.metrics")

    def get_metrics():
        get_metrics.calls += 1
        raise RuntimeError("metrics unavailable")

    get_metrics.calls = 0
    metrics_module.get_metrics = get_metrics
    monkeypatch.setitem(sys.modules, "core.observability.metrics", metrics_module)

    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        managed = ManagedProcess(
            ProcessConfig(name="terminal_worker", target=_target, max_restarts=0)
        )
        managed.last_restart_attempt = time.time()
        managed.stats.restarts = 0
        managed.stats.restart_timestamps = [time.time()]

        restarted = await managed.restart()

        actions = [record.action for record in tracker.recent(subsystem="process_manager")]
        assert restarted is False
        assert managed.state == ProcessState.PERMANENTLY_FAILED
        assert get_incident_manager.calls >= 1
        assert get_metrics.calls >= 1
        assert any("incident report failed" in action for action in actions)
        assert any("restart metric failed" in action for action in actions)
        tracker.reset()

    asyncio.run(scenario())


def test_check_all_processes_continues_after_status_probe_failure():
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = ProcessManager()

    class StatusProbeUnavailable:
        def __init__(self):
            self.calls = 0

        def get_status(self):
            self.calls += 1
            raise RuntimeError("status probe unavailable")

    first = StatusProbeUnavailable()
    second = SimpleNamespace(
        stats=SimpleNamespace(restarts=0),
        config=SimpleNamespace(max_restarts=0),
        state=ProcessState.RUNNING,
        get_status=lambda: {"state": "running", "alive": False},
    )
    manager.processes["first"] = first
    manager.processes["second"] = second

    manager._check_all_processes()

    assert first.calls == 1
    assert second.state == ProcessState.FAILED
    assert any(
        "remaining processes" in record.action
        for record in tracker.recent(subsystem="process_manager")
    )
    manager.processes.clear()
    manager.shutdown_event.set()
    tracker.reset()
