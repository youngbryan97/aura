import pytest

from core.resilience.sovereign_watchdog import SovereignWatchdog
from core.runtime.errors import get_degradation_tracker


class _Bus:
    def __init__(self):
        self.events = []

    def publish_threadsafe(self, topic, payload):
        self.events.append((topic, payload))


class _Orchestrator:
    def __init__(self, *, fail_reset=False):
        self.fail_reset = fail_reset
        self.reset_count = 0

    async def reset_internal_state(self):
        self.reset_count += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")


@pytest.fixture(autouse=True)
def _reset_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_watchdog_recovery_continues_when_gpu_sentinel_fails(monkeypatch):
    bus = _Bus()
    orchestrator = _Orchestrator()
    watchdog = SovereignWatchdog(orchestrator)

    monkeypatch.setattr(
        "core.utils.gpu_sentinel.get_gpu_sentinel",
        lambda: (_ for _ in ()).throw(RuntimeError("sentinel unavailable")),
    )
    monkeypatch.setattr("core.event_bus.get_event_bus", lambda: bus)

    result = await watchdog._execute_recovery()

    assert result["ok"] is False
    assert "gpu_sentinel" in result["degraded_steps"]
    assert "telemetry_notification" in result["completed_steps"]
    assert "orchestrator_reset" in result["completed_steps"]
    assert orchestrator.reset_count == 1
    assert bus.events
    last = get_degradation_tracker().recent(subsystem="sovereign_watchdog")[-1]
    assert last.action == "continued recovery sequence after GPU sentinel recovery failed"


@pytest.mark.asyncio
async def test_watchdog_recovery_reports_orchestrator_reset_failure(monkeypatch):
    bus = _Bus()
    orchestrator = _Orchestrator(fail_reset=True)
    watchdog = SovereignWatchdog(orchestrator)

    class _Sentinel:
        def __init__(self):
            import threading

            self._lock = threading.RLock()

    monkeypatch.setattr("core.utils.gpu_sentinel.get_gpu_sentinel", lambda: _Sentinel())
    monkeypatch.setattr("core.event_bus.get_event_bus", lambda: bus)

    result = await watchdog._execute_recovery()

    assert result["ok"] is False
    assert result["failure_streak"] == 1
    assert "orchestrator_reset" in result["degraded_steps"]
    assert watchdog.get_status()["last_recovery_result"]["degraded_steps"] == [
        "orchestrator_reset"
    ]
    last = get_degradation_tracker().recent(subsystem="sovereign_watchdog")[-1]
    assert last.action == "completed watchdog recovery with orchestrator reset still degraded"
