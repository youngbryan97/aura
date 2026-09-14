"""A recorded turn's body read the machine it ran on, not the world it was given.

A condition prepares the body's reading and records it as the environment.
Between phases the proprioceptive loop read the machine through the live
observer and wrote it over the prepared reading. The arms held the host still
and the recorded rounds did not, so on a busy host the body followed a driver
the environment never recorded, and intrinsic persistence fell from 0.52 to
0.0012 on the one seed that ran while other work loaded the machine.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

PREPARED = {"cpu_usage": 12.5, "vram_usage": 34.0, "temperature": 40.0}


@pytest.mark.slow
def test_every_frame_of_a_turn_reads_the_prepared_body(monkeypatch, tmp_path: Path) -> None:
    from core.config import config
    from core.runtime.resource_observation import (
        SimulatedResourceObserver,
        get_resource_observer,
        resource_observer_scope,
    )
    from core.subject.clock import installed_clock
    from core.subject.driver import CONDITIONS, build_runtime, quiesce_organism, start_organism
    from core.subject.state import feature_names

    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.setenv("AURA_STATE_ROOT", str(state_root))
    monkeypatch.setattr(config.paths, "home_dir_override", None)

    def prepared(state, _rng):
        state.soma.hardware.update(PREPARED)
        return {"host_load": 0.125, "host_thermal": 0.0}

    stress = next(condition for condition in CONDITIONS if condition.name == "stress")
    condition = dataclasses.replace(stress, prepare=prepared)
    busy_machine = SimulatedResourceObserver(cpu_percent=97.0, memory_percent=96.0)
    names = list(feature_names("I"))
    cpu = next(index for index, name in enumerate(names) if name.endswith("cpu"))
    memory = next(index for index, name in enumerate(names) if name.endswith("vram"))

    async def run():
        runtime = build_runtime(tmp_path / "runtime", seed=3)
        await start_organism(runtime)
        await quiesce_organism(runtime)
        frames = await runtime.turn_once(condition)
        return runtime, frames

    try:
        with resource_observer_scope(busy_machine):
            runtime, frames = asyncio.run(run())
            observer_after = get_resource_observer()
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()

    readings = [frame.domain("I") for frame in frames]
    assert {round(float(reading[cpu]), 3) for reading in readings} == {PREPARED["cpu_usage"]}
    assert {round(float(reading[memory]), 3) for reading in readings} == {PREPARED["vram_usage"]}
    assert runtime.frozen_host is None and runtime.turn_hold is False
    assert observer_after is busy_machine, "the turn handed back a different observer"
