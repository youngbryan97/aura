"""asyncio.run's close waits forever for a task that will not end; ours does not.

LIVE, 2026-09-16: the root coroutine had returned and the process sat in
``_cancel_all_tasks`` for fifteen minutes with the port closed and no line
saying why. The bounded runner waits the shutdown budget, names the task
and its frame, and closes the loop underneath it.
"""
from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from core.runtime import bounded_run

ROOT = Path(__file__).resolve().parent.parent


def test_a_task_that_ignores_cancellation_is_named_and_the_process_exits(monkeypatch, caplog):
    monkeypatch.setattr(bounded_run, "_cancellation_budget_s", lambda: 0.3)

    async def never_ends():
        # Catches the cancellation and keeps going: the shape that held the
        # live process in _cancel_all_tasks.
        while True:
            try:
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                continue

    async def main():
        asyncio.create_task(never_ends(), name="stuck_forever")
        await asyncio.sleep(0.01)
        return "done"

    with caplog.at_level(logging.WARNING, logger="Aura.BoundedRun"):
        result = bounded_run.run(main())
    assert result == "done"
    line = next(r.getMessage() for r in caplog.records if "did not end within" in r.getMessage())
    assert "stuck_forever" in line and "never_ends" in line


def test_a_clean_run_closes_quietly(caplog):
    async def main():
        async def child():
            await asyncio.sleep(0.01)

        asyncio.create_task(child())
        return 7

    with caplog.at_level(logging.WARNING, logger="Aura.BoundedRun"):
        assert bounded_run.run(main()) == 7
    assert not [r for r in caplog.records if "did not end" in r.getMessage()]


def test_the_runner_refuses_to_nest():
    async def outer():
        rejected = asyncio.sleep(0)
        try:
            with pytest.raises(RuntimeError, match="running event loop"):
                bounded_run.run(rejected)
        finally:
            rejected.close()

    bounded_run.run(outer())


def test_the_desktop_entry_uses_the_bounded_runner():
    source = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    start = source.index("        elif args.desktop:\n")
    desktop = source[start : source.index("        elif args.", start + 10)]
    assert "_bounded_run(" in desktop
    assert "asyncio.run(" not in desktop


def test_the_fatal_path_exits_through_the_finalizer_too():
    """2026-09-16, pid 29434: a bare sys.exit(1) after FATAL BOOT ERROR handed
    the exit to the interpreter, whose shutdown joins every executor thread.
    One was hours into an embedding forward pass on a loaded host; the
    process sat in threading._shutdown for forty minutes after the server
    had finished. The finalizer ends in os._exit and waits for no one."""
    source = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    boundary = next(node for node in reversed(main.body) if isinstance(node, ast.Try) and any(
        isinstance(handler.type, ast.Name) and handler.type.id == "_AURA_MAIN_BOUNDARY_ERRORS"
        for handler in node.handlers))
    handler = next(handler for handler in boundary.handlers
                   if isinstance(handler.type, ast.Name)
                   and handler.type.id == "_AURA_MAIN_BOUNDARY_ERRORS")
    assert not any(isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.exit"
                   for node in ast.walk(handler))
    assert any(isinstance(node, ast.Assign) and ast.unparse(node) == "exit_code = 1"
               for node in handler.body)
    finalizer = main.body[main.body.index(boundary) + 1]
    assert isinstance(finalizer, ast.Expr) and isinstance(finalizer.value, ast.Call)
    assert ast.unparse(finalizer.value.func) == "_finalize_root_runtime_process_exit"
    assert any(keyword.arg == "exit_code" and ast.unparse(keyword.value) == "exit_code"
               for keyword in finalizer.value.keywords)
    assert ast.unparse(main.body[-1]) == "if exit_code:\n    sys.exit(exit_code)"
