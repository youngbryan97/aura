from __future__ import annotations

import asyncio
import types

import pytest

import core.constitutional_alignment as constitutional_alignment
from core.constitutional_alignment import ConstitutionalAlignmentLayer


def test_constitutional_alignment_start_records_optional_registration_failure(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def _record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    class _BrokenBus:
        def __init__(self):
            self.called = False

        async def publish(self, *_args, **_kwargs):
            self.called = True
            raise RuntimeError("event bus unavailable")

    monkeypatch.setattr(constitutional_alignment, "record_degradation", _record)
    monkeypatch.setattr(
        constitutional_alignment.ServiceContainer,
        "get",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        constitutional_alignment.ServiceContainer,
        "register_instance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(constitutional_alignment, "get_event_bus", lambda: _BrokenBus())

    async def _scenario():
        layer = ConstitutionalAlignmentLayer()
        await layer.start()
        assert layer.running is True
        await layer.stop()
        assert layer.running is False

    asyncio.run(_scenario())

    assert recorded
    assert recorded[0][0] == "constitutional_alignment"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "Mycelium registration failed" in str(recorded[0][2]["action"])


def test_constitutional_alignment_task_creation_failure_fails_closed(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def _record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(constitutional_alignment, "record_degradation", _record)
    monkeypatch.setattr(
        constitutional_alignment.ServiceContainer,
        "get",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        constitutional_alignment.ServiceContainer,
        "register_instance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        constitutional_alignment.task_tracker,
        "create_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("scheduler offline")),
    )

    async def _scenario():
        layer = ConstitutionalAlignmentLayer()
        with pytest.raises(RuntimeError):
            await layer.start()
        assert layer.running is False
        assert layer._alignment_task is None

    asyncio.run(_scenario())

    assert recorded
    assert recorded[0][0] == "constitutional_alignment"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["severity"] == "critical"
    assert "task creation failed" in str(recorded[0][2]["action"])


def test_constitutional_alignment_allows_local_file_work_but_blocks_exfiltration():
    async def _scenario():
        layer = ConstitutionalAlignmentLayer()

        local_allowed = await layer.check_action(
            "file_operation",
            {"operation": "read", "path": "project/README.md"},
        )
        blocked = await layer.check_action(
            "email password file to external address",
            {},
        )

        assert local_allowed is True
        assert blocked is False
        assert layer.get_moral_status()["blocked_actions"] == 1

    asyncio.run(_scenario())


def test_cognitive_coordinator_awaits_async_alignment(monkeypatch):
    from core.coordinators.cognitive_coordinator import CognitiveCoordinator

    calls: list[tuple[str, dict[str, object]]] = []

    class _Alignment:
        async def check_action(self, tool_name, params):
            calls.append((tool_name, params))
            return False

    async def _trigger(*_args, **_kwargs):
        return []

    async def _validate(_action):
        return True

    orch = types.SimpleNamespace(
        alignment=_Alignment(),
        hooks=types.SimpleNamespace(trigger=_trigger),
    )
    coordinator = CognitiveCoordinator(orch)
    monkeypatch.setattr(coordinator, "validate_action_safety", _validate)
    thought = types.SimpleNamespace(action={"tool": "shell", "params": {"command": "echo hi"}})

    async def _scenario():
        result = await coordinator.handle_action_step(thought, trace=None, successful_tools=[])
        assert result["break"] is True
        assert "Conscience Block" in result["response"]

    asyncio.run(_scenario())

    assert calls == [("shell", {"command": "echo hi"})]
