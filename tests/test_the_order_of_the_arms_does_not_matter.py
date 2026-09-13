"""Running one arm before another does not change what the second arm reads.

Every paired number is an arm measured against another arm run from the same
snapshot, in some order. If an arm left anything the restore missed, the arm
that ran second would read differently from the same arm run first, and the
difference would sit in every contrast as an effect of order. So an arm is run
first, again after itself, and again after a different arm. What order adds has
to be no more than what repeating the same arm adds, which is the floor this
machine gives two identical arms.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

#: The fork test's tolerance: floats accumulated in a different order after a
#: restore, far below the 0.3 standardised effect an edge has to clear.
TOLERANCE = 1e-6


@pytest.mark.slow
def test_an_arm_reads_the_same_after_a_different_arm_as_after_itself(monkeypatch, tmp_path: Path) -> None:
    from core.config import config
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )

    # The state a campaign gives a run: its own root, and a home that resolves
    # through it. The stores the suite moves beside the root stay where the
    # suite put them, because the fork has to hold those too.
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    monkeypatch.setattr(config.paths, "home_dir_override", None)
    by_name = {condition.name: condition for condition in CONDITIONS}

    async def run() -> dict[str, np.ndarray]:
        runtime = build_runtime(tmp_path / "runtime", seed=11)
        await start_organism(runtime)
        await quiesce_organism(runtime)
        await calibrate_clock(runtime, CONDITIONS, turns=1)
        runtime.freeze_host()
        await runtime.turn_once(by_name["conversation"])
        snapshot = runtime.snapshot()

        async def arm(name: str) -> np.ndarray:
            runtime.restore(snapshot)
            frames = await runtime.turn_once(by_name[name])
            return np.vstack([frame.vector() for frame in frames])

        first = await arm("conversation")
        repeated = await arm("conversation")
        await arm("memory")
        after_memory = await arm("conversation")
        memory_after_conversation = await arm("memory")
        await arm("tool_use")
        memory_after_tool_use = await arm("memory")
        return {
            "first": first,
            "repeated": repeated,
            "after_memory": after_memory,
            "memory_after_conversation": memory_after_conversation,
            "memory_after_tool_use": memory_after_tool_use,
        }

    try:
        arms = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()

    floor = float(np.max(np.abs(arms["repeated"] - arms["first"])))
    order = float(np.max(np.abs(arms["after_memory"] - arms["first"])))
    assert order <= floor + TOLERANCE, (
        f"a conversation arm run after a memory arm moved {order:.3g}; "
        f"repeating it moved {floor:.3g}"
    )
    memory_order = float(
        np.max(np.abs(arms["memory_after_tool_use"] - arms["memory_after_conversation"]))
    )
    assert memory_order <= TOLERANCE, (
        f"a memory arm read {memory_order:.3g} differently after a tool-use arm "
        "than after a conversation arm"
    )
