from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def test_task_tracker_singleton_is_not_split_brain():
    from core.utils.task_tracker import get_task_tracker, task_tracker

    assert get_task_tracker() is task_tracker


def test_atomic_writer_is_self_contained_and_schema_named(tmp_path: Path):
    from core.runtime.atomic_writer import atomic_write_json, read_json_envelope

    target = tmp_path / "state_snapshot.json"
    atomic_write_json(target, {"ok": True}, schema_version=3)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "state_snapshot"
    assert payload["schema_name"] == "state_snapshot"
    assert payload["schema_version"] == 3
    assert read_json_envelope(target)["payload"] == {"ok": True}
    assert not list(tmp_path.glob(".aura_atomic_*"))


def test_governed_decorator_fails_closed_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")

    from core.governance_context import GovernanceViolation, governed

    @governed
    def mutate_without_receipt():
        return "mutated"

    with pytest.raises(GovernanceViolation):
        mutate_without_receipt()


def test_loop_lag_monitor_has_bounded_shutdown_contract():
    from core.runtime.loop_guard import LoopLagMonitor

    async def scenario():
        monitor = LoopLagMonitor(threshold_s=5.0, sample_interval_s=0.01)
        await monitor.run_for(0.03)

        stop_event = asyncio.Event()
        task = asyncio.create_task(monitor.start(stop_event))
        await asyncio.sleep(0.02)
        monitor.stop()
        await asyncio.wait_for(task, timeout=0.25)
        assert task.done()

    asyncio.run(scenario())
