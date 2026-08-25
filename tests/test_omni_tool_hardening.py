from __future__ import annotations

import asyncio
import sys

import pytest

from core.cybernetics import omni_tool as omni_module
from core.cybernetics.omni_tool import OmniTool


class _Verdict:
    """A stand-in for the Will's answer about one command."""

    def __init__(self, approved: bool, reason: str = "test") -> None:
        self.approved = approved
        self.reason = reason

    def receipt(self) -> dict:
        return {"approved": self.approved, "reason": self.reason}


@pytest.fixture
def approving_authority(monkeypatch):
    """Approve the spawn, so this file measures supervision, not the gate.

    A daemon is general execution that outlives its caller, so it goes through
    the same standing-authority gate as the terminal and MCP. That gate is
    tested where it lives; here it must be out of the way and still be CALLED,
    which the returned list is for.
    """
    asked: list[tuple] = []

    async def _approve(kind, argv, **kwargs):
        asked.append((kind, tuple(argv), kwargs.get("source")))
        return _Verdict(True)

    monkeypatch.setattr(omni_module, "authorize_execution", _approve)
    return asked


def test_omni_tool_awaits_handler_returned_coroutine():
    tool = OmniTool()

    async def inner():
        await asyncio.sleep(0)
        return {"ok": True}

    def handler():
        return inner()

    result = asyncio.run(tool.execute_action("async_return", handler, cooldown=0))

    assert result == {"ok": True}
    assert tool._execution_logs["async_return"][-1]["status"] == "success"


def test_omni_tool_action_failure_records_receipt(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    def handler():
        handler.attempted = True
        raise RuntimeError("field action failed")

    monkeypatch.setattr(omni_module, "record_degradation", record_degradation)

    result = asyncio.run(OmniTool().execute_action("field_action", handler, cooldown=0))

    assert result == {"error": "field action failed"}
    assert recorded
    assert recorded[0][0] == "omni_tool"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert recorded[0][2]["extra"]["action_name"] == "field_action"


def test_omni_tool_spawn_daemon_runs_real_supervised_process(approving_authority):
    async def scenario():
        tool = OmniTool()
        command = f"{sys.executable} -c \"print('daemon-ok')\""
        result = await tool.spawn_daemon("smoke", command, timeout_s=5)
        metadata = result["daemon"]
        for _ in range(100):
            if metadata["status"] in {"completed", "failed", "timed_out", "watch_failed"}:
                break
            await asyncio.sleep(0.02)
        return result, metadata, tool.get_status()

    result, metadata, status = asyncio.run(scenario())

    assert result["status"] == "spawned"
    assert metadata["status"] == "completed"
    assert metadata["returncode"] == 0
    assert "daemon-ok" in metadata["stdout_tail"]
    assert status["daemons"]["smoke"]["status"] == "completed"
    assert approving_authority, "the daemon spawn must ask the Will first"
    kind, argv, source = approving_authority[0]
    assert kind == "shell"
    assert argv[0] == sys.executable
    assert source == "core.cybernetics.omni_tool.daemon"


def test_a_refused_daemon_spawns_nothing(monkeypatch):
    """A denial is an answer, not a traceback, and no process starts."""
    spawned: list = []

    async def _refuse(kind, argv, **kwargs):
        return _Verdict(False, "standing_authority_denied:no_matching_standing_grant")

    async def _never(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("a refused daemon must not reach the process spawn")

    monkeypatch.setattr(omni_module, "authorize_execution", _refuse)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _never)

    async def scenario():
        tool = OmniTool()
        result = await tool.spawn_daemon("blocked", f"{sys.executable} -c pass")
        return result, tool.get_status()

    result, status = asyncio.run(scenario())
    assert result["status"] == "error"
    assert "no_matching_standing_grant" in result["message"]
    assert not spawned
    assert "blocked" not in status["daemons"]
