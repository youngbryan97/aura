"""A boot step that is still completing its parts is working, however slowly.

LIVE 2026-09-16, load 34 on 18 cores, boot offset +286s: the live-mind
activation materialized all eight organs in 17s. Its wall budget was 15s. The
boot raised TimeoutError with the work done and the desktop process exited.

The bound is now on progress. The budget is the longest one part may take to
complete; a host that makes every part slow never trips it, and a part that
never completes still does.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core.runtime.progress_bound import await_while_it_progresses

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_a_step_that_keeps_completing_parts_outlives_the_budget():
    steps: list[int] = []

    async def slow_but_working():
        for i in range(6):
            await asyncio.sleep(0.15)
            steps.append(i)
        return "done"

    # Six parts at 0.15s is 0.9s of wall against a 0.4s budget.
    result = await await_while_it_progresses(
        slow_but_working(), progress=lambda: len(steps), stall_s=0.4, name="six parts"
    )
    assert result == "done"
    assert steps == list(range(6))


@pytest.mark.asyncio
async def test_a_step_whose_parts_stop_completing_is_a_wedge():
    steps: list[int] = []

    async def wedges_after_two():
        for i in range(2):
            await asyncio.sleep(0.05)
            steps.append(i)
        await asyncio.sleep(60)
        return "never"

    with pytest.raises(TimeoutError, match=r"wedge made no progress for .*last progress: 2"):
        await await_while_it_progresses(
            wedges_after_two(), progress=lambda: len(steps), stall_s=0.3, name="wedge"
        )
    assert steps == [0, 1]


def test_the_live_mind_activation_is_bounded_by_its_progress():
    source = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    block = source[source.index("activate_live_mind_runtime") :]
    block = block[: block.index("_mark_runtime_boot_phase(\"live_mind_activation\")")]
    assert "await_while_it_progresses(" in block
    assert "progress=get_live_mind_runtime().materialized" in block
    assert not re.search(r"wait_for\(\s*asyncio\.to_thread\(activate_live_mind_runtime\)", block)

    from core.runtime.live_mind_runtime import LiveMindRuntime

    runtime = LiveMindRuntime()
    assert runtime.materialized() == ""


@pytest.mark.asyncio
async def test_a_stage_working_through_synchronous_steps_outlives_the_budget():
    """The kernel stage: organ after organ, synchronous on the loop thread,
    106s on a loaded host against a 15s wall budget."""
    import time

    from core.runtime.progress_bound import await_while_the_task_moves

    organs: list[str] = []

    async def kernel_like():
        for organ in ("llm", "vision", "memory", "affect", "will", "mesh"):
            time.sleep(0.12)  # synchronous work on the loop thread
            organs.append(organ)
            await asyncio.sleep(0)
        return "kernel ready"

    # 0.72s of blocking work against a 0.3s stall budget.
    result = await await_while_the_task_moves(kernel_like(), stall_s=0.3, name="kernel")
    assert result == "kernel ready"
    assert len(organs) == 6


@pytest.mark.asyncio
async def test_a_stage_awaiting_something_that_never_resolves_is_a_wedge():
    from core.runtime.progress_bound import await_while_the_task_moves

    never = asyncio.get_running_loop().create_future()

    async def wedged():
        await asyncio.sleep(0.05)
        await never
        return "never"

    with pytest.raises(TimeoutError, match=r"boot stage x sat on one await"):
        await await_while_the_task_moves(wedged(), stall_s=0.3, name="boot stage x")


def test_the_boot_stages_are_bounded_by_their_motion():
    source = (ROOT / "core/ops/resilient_boot.py").read_text(encoding="utf-8")
    assert "await_while_the_task_moves(\n                        stage_fn(), stall_s=timeout" in source
    assert "asyncio.wait_for(stage_fn()" not in source
