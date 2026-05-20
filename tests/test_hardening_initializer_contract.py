from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator.initializers import hardening
from core.runtime.errors import get_degradation_tracker
from core.runtime.health_contract import RUNTIME_CONTRACT, ServiceTier


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

    def start(self) -> None:
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self._alive = self.alive_after_start

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture(autouse=True)
def isolated_contract_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    yield
    ServiceContainer.clear()
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


def test_long_run_supervisors_are_part_of_runtime_health_contract():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key in {"reaper", "hypervisor", "event_loop_monitor"}
    }

    assert set(required) == {"reaper", "hypervisor", "event_loop_monitor"}
    assert all(requirement.tier == ServiceTier.IMPORTANT for requirement in required.values())
    assert all(requirement.liveness_check == "is_alive" for requirement in required.values())
