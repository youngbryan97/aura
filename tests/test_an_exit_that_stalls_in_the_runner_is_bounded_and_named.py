"""asyncio.run's close waits forever for a task that will not end; ours does not.

LIVE, 2026-09-16: the root coroutine had returned and the process sat in
``_cancel_all_tasks`` for fifteen minutes with the port closed and no line
saying why. The bounded runner waits the shutdown budget, names the task
and its frame, and closes the loop underneath it.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.runtime import bounded_run


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
        with pytest.raises(RuntimeError):
            bounded_run.run(asyncio.sleep(0))

    bounded_run.run(outer())


def test_the_desktop_entry_uses_the_bounded_runner():
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "aura_main.py").read_text(encoding="utf-8")
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
    tail = source[source.index("    except _AURA_MAIN_BOUNDARY_ERRORS as e:\n        record_degradation('aura_main', e)\n        logger.critical(\"FATAL BOOT ERROR") :]
    tail = tail[: tail.index("if __name__ ==")]
    assert "sys.exit(1)" not in tail.split("_finalize_root_runtime_process_exit(")[0]
    assert "exit_code = 1" in tail
    assert "exit_code=exit_code," in tail
    assert tail.rstrip().endswith("sys.exit(exit_code)")
