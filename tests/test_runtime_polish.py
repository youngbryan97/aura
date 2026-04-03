import asyncio
import time
from pathlib import Path

import pytest

from core.affect import heartstone_values as heartstone_module
from core.world_model import user_model as user_model_module
from interface.server import MessageBroadcastBus, Response, WebSocketManager, _cache_policy_for_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cache_policy_keeps_live_shell_uncached():
    assert _cache_policy_for_path("/")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/aura.js")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/aura.css")["Cache-Control"].startswith("no-store")
    assert _cache_policy_for_path("/static/icon-192.png")["Cache-Control"] == "public, max-age=31536000, immutable"


def test_cache_policy_middleware_dependencies_are_imported():
    assert Response is not None


def test_gui_actor_exits_after_extended_kernel_loss():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "Kernel API unavailable for too long" in gui_actor
    assert "os._exit(1)" in gui_actor


@pytest.mark.asyncio
async def test_message_broadcast_bus_replaces_lowest_priority_when_full():
    bus = MessageBroadcastBus(maxsize=2)
    queue = await bus.subscribe()

    await bus.publish("low", priority=20)
    await bus.publish("mid", priority=10)
    await bus.publish("high", priority=0)

    first = await queue.get()
    second = await queue.get()

    assert [first[0], second[0]] == [0, 10]
    assert {first[2], second[2]} == {"high", "mid"}


@pytest.mark.asyncio
async def test_websocket_manager_replaces_lowest_priority_when_full():
    manager = WebSocketManager()
    queue = asyncio.PriorityQueue(maxsize=2)
    manager.active_connections = {object(): queue}

    await manager.broadcast({"type": "telemetry", "message": "low"})
    await manager.broadcast({"type": "chat_response", "message": "high"})
    await manager.broadcast({"type": "aura_message", "message": "critical"})

    first = await queue.get()
    second = await queue.get()

    assert [first[0], second[0]] == [0, 0]


def test_bryan_model_save_is_debounced(monkeypatch, tmp_path):
    monkeypatch.setattr(user_model_module, "_USER_MODEL_PATH", tmp_path / "user_model.json")
    monkeypatch.setattr(user_model_module, "_SAVE_DEBOUNCE_SECONDS", 10.0)

    engine = user_model_module.BryanModelEngine()
    writes: list[str] = []
    monkeypatch.setattr(engine, "_write_now", lambda: writes.append("write"))

    engine._last_saved = time.time()
    engine.save()
    engine.save()

    assert writes == []
    assert engine._save_timer is not None

    engine._save_timer.cancel()
    engine._flush_pending_save()

    assert writes == ["write"]


def test_heartstone_save_is_debounced(monkeypatch, tmp_path):
    monkeypatch.setattr(heartstone_module, "_PERSIST_PATH", tmp_path / "heartstone_values.json")
    monkeypatch.setattr(heartstone_module, "_SAVE_DEBOUNCE_SECONDS", 10.0)

    values = heartstone_module.HeartstoneValues()
    writes: list[str] = []
    monkeypatch.setattr(values, "_write_now", lambda: writes.append("write"))

    values._last_saved = time.time()
    values._save()
    values._save()

    assert writes == []
    assert values._save_timer is not None

    values._save_timer.cancel()
    values._flush_pending_save()

    assert writes == ["write"]
