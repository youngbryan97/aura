"""What one arm of an intervention does must not reach the next one.

Every number the battery reports is a displaced arm measured against a sham
run from the same snapshot. That subtraction is only a measurement if the two
arms start from the same place — so after an arm has run and the snapshot has
been put back, the organism has to be where it was, in the state the reading
covers and in the state it does not.

The failures this is written against were not visible in any single number.
`self_prediction` and `comparator` were absent from the hand-kept fork list for
a whole session, so the sham inherited the prediction error the displaced arm
had just made; the reading it corrupted was the self-state, and the criteria
resting on it failed against an organ that was there and working.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from core.subject.driver import (
    CONDITIONS,
    SubjectRuntime,
    build_runtime,
    calibrate_clock,
    start_organism,
)
from core.subject.state import Organs

#: How far a restored reading may sit from the one taken before the arm. Not
#: zero: a few columns are floats accumulated in a different order after a
#: restore. Wide enough to allow that and far below the 0.3 standardised
#: effect an edge has to clear.
TOLERANCE = 1e-6


def test_the_fork_carries_every_organ_the_reading_declares() -> None:
    """The list of what to fork is the list of organs, not a copy of it.

    Two lists of the same organs is two chances to forget one, and forgetting
    one is invisible: the arms still run, the numbers still come out, and the
    floor quietly rises under whichever domain the missed organ writes.
    """
    declared = {field.name for field in dataclasses.fields(Organs)}
    assert set(SubjectRuntime.ORGAN_FIELDS) == declared


def _read_all(runtime: SubjectRuntime) -> dict[str, np.ndarray]:
    return {
        condition.name: runtime.read(condition.name, "probe", {}).vector()
        for condition in CONDITIONS[:3]
    }


def _peripheral(runtime: SubjectRuntime) -> dict[str, float]:
    from core.subject.closure import read_periphery

    return dict(read_periphery(runtime.kernel))


def _world(runtime: SubjectRuntime) -> dict[str, bytes]:
    root = getattr(runtime, "_scratch", None)
    if root is None or not Path(root).exists():
        return {}
    return {
        str(item.relative_to(root)): item.read_bytes()
        for item in sorted(Path(root).rglob("*"))
        if item.is_file()
    }


@pytest.mark.slow
def test_an_arm_that_ran_leaves_the_organism_where_it_found_it() -> None:
    """Take a snapshot, run a turn, put it back, and read again.

    This is the check that says whether the causal experiment is an experiment.
    It reads the ten declared domains and the peripheral state the closure test
    scans, because a variable outside K that survives a restore is a channel
    between two arms exactly as much as one inside it.
    """

    async def run() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, float]]:  # noqa: E501
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=11)
            await start_organism(runtime)
            from core.subject.driver import quiesce_organism

            await quiesce_organism(runtime)
            # The same clock a run installs. Without it the arms are compared
            # on the machine's clock, which is the thing this is here to
            # remove, so a check that ran without it would be checking a
            # configuration the battery never uses.
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            runtime.freeze_host()
            # Move the life on first: a snapshot of a state nothing has
            # touched is a weaker test than a snapshot of one in motion.
            for condition in CONDITIONS[:2]:
                await runtime.turn_once(condition)

            before = _read_all(runtime)
            before_periphery = _peripheral(runtime)
            before_world = _world(runtime)
            snapshot = runtime.snapshot()

            for condition in CONDITIONS[:2]:
                await runtime.turn_once(condition)
            # Act, so the world this arm leaves behind is not empty. A check
            # against a world nothing wrote to cannot fail.
            runtime._act("write the plan down", actor="self")
            runtime.turn += 1
            runtime._act("write the plan down again", actor="self")
            assert _world(runtime) != before_world, "the probe wrote nothing to compare"

            runtime.restore(snapshot)
            after = _read_all(runtime)
            after_periphery = _peripheral(runtime)
            after_world = _world(runtime)
            assert before_world == after_world, (
                "the world an arm acted in survived the restore: "
                + str(sorted(set(after_world) ^ set(before_world))
                      or [k for k in before_world if before_world[k] != after_world.get(k)])
            )
            return before, after, before_periphery, after_periphery

    before, after, before_periphery, after_periphery = asyncio.run(run())

    drifted: list[str] = []
    for name, vector in before.items():
        delta = np.abs(np.asarray(vector, dtype=float) - np.asarray(after[name], dtype=float))
        if float(np.nanmax(delta)) > TOLERANCE:
            worst = int(np.nanargmax(delta))
            drifted.append(f"{name}[{worst}] moved {float(delta[worst]):.6g}")
    assert not drifted, "a restored reading is not the reading taken: " + "; ".join(drifted)

    leaked = [
        f"{key} {before_periphery[key]!r} -> {after_periphery[key]!r}"
        for key in sorted(set(before_periphery) & set(after_periphery))
        if abs(float(before_periphery[key]) - float(after_periphery[key])) > TOLERANCE
    ]
    assert not leaked, "peripheral state survived the restore: " + "; ".join(leaked[:12])
