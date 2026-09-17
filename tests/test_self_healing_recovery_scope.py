"""Process lifetime and component restart have different owners."""

import time

import pytest

from core.container import ServiceContainer
from core.runtime.self_healing import SelfHealing


class _Lifecycle:
    def __init__(self):
        self.calls = []

    async def stop(self):
        self.calls.append("stop")

    async def start(self):
        self.calls.append("start")


@pytest.mark.asyncio
async def test_process_root_cannot_be_recycled_or_report_a_false_heartbeat(monkeypatch):
    root = _Lifecycle()
    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: root)
    healer = SelfHealing()
    healer.watch("root", container_key="root", recovery_scope="process_root")
    watch = healer._watches["root"]
    watch.last_heartbeat_at = time.time() - 100
    before = watch.last_heartbeat_at
    records = []

    async def append(record):
        records.append(record)

    monkeypatch.setattr(healer, "_append_record_async", append)
    await healer._heal(watch, 100)
    assert await healer._restart_watch(watch) is False
    assert root.calls == []
    assert watch.last_heartbeat_at == before
    assert records[0]["result"] == "process_root_requires_external_recovery"
    assert healer.get_status()["watches"]["root"]["heartbeat_age_s"] >= 100


@pytest.mark.asyncio
async def test_boot_watch_is_armed_by_its_first_real_heartbeat(monkeypatch):
    healer = SelfHealing()
    healer.watch("root", expected_interval_s=0.01, wait_for_heartbeat=True)
    watch = healer._watches["root"]
    watch.last_heartbeat_at = time.time() - 100
    repairs = []

    async def heal(*args):
        repairs.append(args)

    monkeypatch.setattr(healer, "_heal", heal)
    monkeypatch.setattr(healer, "_healing_defer_reason", lambda: "")
    await healer._tick()
    assert repairs == []
    assert healer.get_status()["watches"]["root"]["armed"] is False
    healer.heartbeat("root")
    assert watch.armed is True
    await healer._tick()
    assert repairs == []
    watch.last_heartbeat_at = time.time() - 100
    await healer._tick()
    assert len(repairs) == 1


@pytest.mark.asyncio
async def test_component_lifecycle_can_still_restart(monkeypatch):
    component = _Lifecycle()
    monkeypatch.setattr(ServiceContainer, "get", lambda *args, **kwargs: component)
    healer = SelfHealing()
    healer.watch("component", container_key="component")
    assert await healer._restart_watch(healer._watches["component"]) is True
    assert component.calls == ["stop", "start"]


def test_invalid_recovery_scope_is_rejected():
    with pytest.raises(ValueError, match="Unknown recovery scope"):
        SelfHealing().watch("root", recovery_scope="typo")
