from __future__ import annotations

import asyncio
import sqlite3
import types

from core.coordinators import tool_executor as tool_executor_module
from core.coordinators.tool_executor import ToolExecutor


def test_tool_executor_returns_structured_failure_after_router_sqlite_error(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []
    crash_records: list[dict[str, object]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    class Router:
        skills = {"persist": object()}

        async def execute(self, _goal, _context):
            self.attempted = True
            raise sqlite3.OperationalError("database locked")

    class Memory:
        async def commit_interaction(self, **kwargs):
            crash_records.append(kwargs)

    orch = types.SimpleNamespace(
        _current_objective="persist conversation",
        status=types.SimpleNamespace(mode="test"),
        router=Router(),
        hephaestus=None,
        memory=Memory(),
    )

    monkeypatch.setattr(tool_executor_module, "record_degradation", record_degradation)
    monkeypatch.setattr(
        ToolExecutor,
        "_record_coding_tool_event",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        ToolExecutor,
        "_emit_action_feedback",
        staticmethod(lambda *_args, **_kwargs: None),
    )

    result = asyncio.run(ToolExecutor(orch).execute_tool("persist", {"path": "db"}))

    assert result["ok"] is False
    assert result["error"] == "execution_jolt"
    assert result["message"].startswith("OperationalError:")
    assert crash_records
    assert crash_records[0]["success"] is False
    assert recorded
    assert recorded[0][0] == "tool_executor"
    assert recorded[0][1] == "OperationalError"
    assert recorded[0][2]["receipt_required"] is True
    assert recorded[0][2]["extra"]["tool_name"] == "persist"


def test_tool_executor_handles_unavailable_router_without_crashing(monkeypatch):
    coding_events: list[dict[str, object]] = []
    orch = types.SimpleNamespace(
        _current_objective="notify",
        status=types.SimpleNamespace(mode="test"),
        router=None,
        hephaestus=None,
        memory=None,
    )

    def record_event(_orch, **kwargs):
        coding_events.append(kwargs)

    monkeypatch.setattr(
        ToolExecutor,
        "_record_coding_tool_event",
        staticmethod(record_event),
    )

    result = asyncio.run(ToolExecutor(orch).execute_tool("web_search", {"query": "aura"}))

    assert result == {
        "ok": False,
        "error": "tool_router_unavailable",
        "message": "Tool router is unavailable.",
    }
    assert coding_events[0]["success"] is False
    assert coding_events[0]["error"] == "tool_router_unavailable"


def test_tool_executor_sanitizes_invalid_plan_calls(monkeypatch):
    monkeypatch.setattr(
        ToolExecutor,
        "_record_coding_tool_event",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    orch = types.SimpleNamespace(
        _current_objective="",
        status=types.SimpleNamespace(mode="test"),
        router=None,
        hephaestus=None,
        memory=None,
    )

    results = asyncio.run(
        ToolExecutor(orch).execute_plan(
            {"tool_calls": ["bad-call", {"tool": "", "args": None}]}
        )
    )

    assert results[0] == {"ok": False, "error": "invalid_tool_call"}
    assert results[1] == {"ok": False, "error": "invalid_tool_name"}
