from __future__ import annotations

import asyncio
import sys

from core.cybernetics import omni_tool as omni_module
from core.cybernetics.omni_tool import OmniTool


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


def test_omni_tool_spawn_daemon_runs_real_supervised_process():
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
