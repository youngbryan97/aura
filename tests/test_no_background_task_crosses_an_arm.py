"""A background task is state, and no arm may leave one running for the next.

There are two ways one could. A free-running loop could survive the wind-down
and keep stepping between arms, or an arm could start a task that is still
alive when the next arm begins. A probe of five arms off one snapshot found
neither: after the wind-down only the state registry's notification dispatcher
was alive, and no arm ended with a task it had started. This keeps it so.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


def _alive() -> set[str]:
    current = asyncio.current_task()
    return {
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not current and not task.done()
    }


@pytest.mark.slow
def test_no_background_task_crosses_from_one_arm_to_the_next() -> None:
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.organism import KEPT_TASKS

    kept = set(KEPT_TASKS)
    by_name = {condition.name: condition for condition in CONDITIONS}

    async def run() -> dict[str, object]:
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=11)
            await start_organism(runtime)
            await quiesce_organism(runtime)
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            runtime.freeze_host()
            seen: dict[str, object] = {
                "after the wind-down": _alive(),
                "reported still running": set(runtime.organism.still_running or ()),
            }
            snapshot = runtime.snapshot()
            for name in ("conversation", "memory", "tool_use"):
                runtime.restore(snapshot)
                # Yield once, so a task scheduled by the restore would be seen.
                await asyncio.sleep(0)
                await runtime.turn_once(by_name[name])
                seen[f"at the end of the {name} arm"] = _alive()
            runtime.restore(snapshot)
            await asyncio.sleep(0)
            seen["after the last restore"] = _alive()
            return seen

    try:
        seen = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()
    extra = {moment: sorted(names - kept) for moment, names in seen.items() if names - kept}
    assert not extra, f"background tasks outside the kept infrastructure: {extra}"
