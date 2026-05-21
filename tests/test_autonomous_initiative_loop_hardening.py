from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.autonomous_initiative_loop import AutonomousInitiativeLoop


async def _hold_until_cancelled(marker: list[str], name: str) -> None:
    try:
        await asyncio.Event().wait()
    finally:
        marker.append(name)


def _install_held_loops(loop: AutonomousInitiativeLoop, marker: list[str]) -> None:
    loop._world_watcher_loop = lambda: _hold_until_cancelled(marker, "world")
    loop._knowledge_gap_monitor_loop = lambda: _hold_until_cancelled(marker, "knowledge")
    loop._self_development_loop = lambda: _hold_until_cancelled(marker, "self_development")
    loop._social_interaction_loop = lambda: _hold_until_cancelled(marker, "social")


@pytest.mark.asyncio
async def test_start_keeps_core_loops_when_event_subscription_fails(monkeypatch):
    class BrokenBus:
        async def subscribe(self, _topic: str):
            self.topic = _topic
            if self.topic:
                raise RuntimeError("event bus offline")
            return asyncio.Queue()

    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: BrokenBus(),
    )

    status = await loop.start()
    await asyncio.sleep(0)

    assert status == {
        "ok": True,
        "already_running": False,
        "core_tasks": {
            "world": True,
            "knowledge": True,
            "self_development": True,
            "social": True,
        },
        "event_subscription": False,
    }
    assert all(getattr(task, "_aura_supervised", False) for task in loop._core_tasks())

    await loop.stop()
    assert set(marker) == {"world", "knowledge", "self_development", "social"}


@pytest.mark.asyncio
async def test_start_is_idempotent_while_core_tasks_are_alive(monkeypatch):
    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: None,
    )

    first = await loop.start()
    await asyncio.sleep(0)
    first_world_task = loop._world_task
    second = await loop.start()

    assert first["already_running"] is False
    assert second["already_running"] is True
    assert loop._world_task is first_world_task

    await loop.stop()
    assert set(marker) == {"world", "knowledge", "self_development", "social"}


@pytest.mark.asyncio
async def test_stop_awaits_background_task_cancellation(monkeypatch):
    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: None,
    )

    await loop.start()
    await asyncio.sleep(0)
    await loop.stop()

    assert loop.running is False
    assert all(task.done() for task in loop._core_tasks())
    assert set(marker) == {"world", "knowledge", "self_development", "social"}
