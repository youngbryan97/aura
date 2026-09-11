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


def test_every_live_cognitive_loop_can_be_advanced_by_a_count() -> None:
    """A loop that only a clock can drive cannot be in a paired measurement.

    Eleven layers ran their own timers. The battery stopped them, because two
    arms cannot be compared while a loop runs at whatever rate the machine
    allows — and stopping them meant that whatever they contribute to the
    coupling between domains was missing from every number the battery
    reported. Each one now has an entry point its own loop already calls, so
    the harness drives the same cognition on a count instead.
    """
    from core.subject.organism import bring_up, quiesce
    from core.subject.steppable import missing_entry_points, step_once

    from core.subject.steppable import LAYERS, layers_of

    #: Long enough for every layer down to one step every two and a half
    #: seconds to take one. A generation of substrate evolution is five
    #: minutes apart and is not expected inside it.
    seconds = 0.05
    frames = int(round(3.0 / seconds))

    async def run() -> tuple[dict[str, str], dict[str, int], dict[str, str], set[str]]:
        organism = await bring_up(quiet=False)
        await quiesce()
        missing = missing_entry_points(organism)
        steps = None
        for frame in range(frames):
            steps = await step_once(organism, frame, steps, seconds=seconds)
        return missing, dict(steps.counts), dict(steps.failures), set(layers_of(organism))

    missing, counts, failures, present = asyncio.run(run())
    assert not missing, f"a live layer has no entry point: {missing}"
    assert not failures, f"a layer raised while being stepped: {failures}"

    # Every layer fast enough to come round inside the window did, and the ones
    # that did not are exactly the ones whose own period is longer than it.
    window = frames * seconds
    due = {layer.name for layer in LAYERS if layer.name in present and 1.0 / layer.hz <= window}
    assert due <= set(counts), f"a layer that was due did not advance: {sorted(due - set(counts))}"
    assert set(counts) <= present, f"a layer advanced that is not here: {sorted(set(counts) - present)}"

    # And each one took about the number of iterations its own rate calls for.
    for layer in LAYERS:
        if layer.name not in counts:
            continue
        expected = layer.hz * window
        assert abs(counts[layer.name] - expected) <= 1.0, (
            f"{layer.name} took {counts[layer.name]} iterations where its "
            f"{layer.hz} Hz calls for about {expected:.1f}"
        )


def test_two_arms_step_the_same_layers_the_same_number_of_times() -> None:
    """Which layer steps on a frame, and how often, is a function of the frame
    count — and the frame count is carried in the snapshot, so a restore puts
    both arms at the same place in the schedule rather than sixty-six frames
    apart."""
    from core.subject.steppable import LAYERS, iterations_at

    seconds = 0.03

    def schedule(start: int, length: int) -> list[list[tuple[str, int]]]:
        return [
            [
                (layer.name, iterations_at(layer, start + step, seconds)[1])
                for layer in LAYERS
            ]
            for step in range(length)
        ]

    assert schedule(0, 40) == schedule(0, 40)
    assert schedule(0, 40) != schedule(7, 40), "the schedule does not depend on the count"


def test_the_fork_carries_the_module_singletons_nothing_holds() -> None:
    """State at module scope that no container and no phase can reach.

    A service is carried under its name; a phase's own attributes go with the
    phase. What is left is an accessor that returns the one object and nobody
    keeping a reference — and the interiority layer publishes every faculty
    through a synaptic cleft got exactly that way. It holds a readiness per
    channel and its receptor bank adapts, so an arm that felt something left the
    medium more excitable for the arm after it, under every domain it touches.
    """
    from core.interiority.cleft import get_cleft
    from core.subject.driver import _restore_singletons, _singleton_state

    before = _singleton_state()
    assert before, "the fork carries no module singletons at all"
    for _ in range(24):
        get_cleft().release("a_fork_probe", 0.7, dt=0.05)
    assert _singleton_state() != before, "the probe moved nothing the fork is watching"
    _restore_singletons(before)
    assert _singleton_state() == before, "a restored singleton is not the one saved"


def test_the_synaptic_medium_rewinds_with_the_process_generator() -> None:
    """Release is probabilistic, so the draw has to be one the fork can rewind.

    The cleft held a `random.Random()` of its own. `random.seed()` did not
    reach it and a snapshot could not put it back, so two arms of a paired
    trial started from one state and drew different quanta — unseeded
    randomness inside the cognitive path, under every number measured through
    the interiority layer.
    """
    import random as _random

    from core.interiority.cleft import SynapticCleft
    from core.interiority.receptors import ReceptorBank

    def run() -> list[float]:
        cleft = SynapticCleft(bank=ReceptorBank())
        return [
            round(cleft.release("rewind_probe", 0.6, dt=0.1).postsynaptic, 9)
            for _ in range(10)
        ]

    saved = _random.getstate()
    first = run()
    _random.setstate(saved)
    assert run() == first, "the medium does not rewind with the process generator"
    assert run() != first, "the medium draws nothing, so nothing was being measured"
