from __future__ import annotations

import asyncio

import pytest

from core.cognitive.autopoiesis import (
    AutopoiesisEngine,
    ComponentSnapshot,
    RepairStrategy,
)


def test_governance_unavailable_denies_repair(monkeypatch):
    import core.will as will_module

    def unavailable_will():
        if will_module:
            raise RuntimeError("will offline")
        return None

    monkeypatch.setattr(will_module, "get_will", unavailable_will)

    engine = AutopoiesisEngine()

    assert engine._request_governance_approval("memory", RepairStrategy.HEAL) is False


@pytest.mark.asyncio
async def test_repair_handler_failure_records_result_and_error():
    engine = AutopoiesisEngine()
    engine._request_governance_approval = lambda _component, _strategy: True
    engine.register_component("memory", lambda: 0.4)
    engine._component_snapshots["memory"].append(ComponentSnapshot("memory", 0.4, 0))

    def failing_handler():
        if engine:
            raise RuntimeError("repair failed")
        return True

    engine.register_repair_handler(RepairStrategy.HEAL, "memory", failing_handler)

    result = await engine.request_repair("memory", RepairStrategy.HEAL)

    assert result.success is False
    assert result.governance_approved is True
    assert "RuntimeError" in result.error_message
    assert engine._repair_attempts["memory"] == 1
    assert any(component == "memory" for component, _, _ in engine._error_buffer)


@pytest.mark.asyncio
async def test_repair_handler_timeout_is_bounded():
    engine = AutopoiesisEngine()
    engine._request_governance_approval = lambda _component, _strategy: True
    engine._REPAIR_HANDLER_TIMEOUT = 0.01
    engine.register_component("search", lambda: 0.2)
    engine._component_snapshots["search"].append(ComponentSnapshot("search", 0.2, 0))

    async def slow_handler():
        await asyncio.sleep(1.0)
        return True

    engine.register_repair_handler(RepairStrategy.RESTART_COMPONENT, "search", slow_handler)

    result = await engine.request_repair("search", RepairStrategy.RESTART_COMPONENT)

    assert result.success is False
    assert "TimeoutError" in result.error_message
    assert engine._repair_attempts["search"] == 1


@pytest.mark.asyncio
async def test_successful_repair_requires_actual_health_improvement():
    health = {"value": 0.2}
    engine = AutopoiesisEngine()
    engine._request_governance_approval = lambda _component, _strategy: True
    engine.register_component("memory", lambda: health["value"])
    engine._component_snapshots["memory"].append(ComponentSnapshot("memory", 0.2, 0))

    def heal_memory():
        health["value"] = 0.9
        return True

    engine.register_repair_handler(RepairStrategy.HEAL, "memory", heal_memory)

    result = await engine.request_repair("memory", RepairStrategy.HEAL)

    assert result.success is True
    assert result.health_before == 0.2
    assert result.health_after == 0.9
    assert engine._repair_attempts["memory"] == 0
