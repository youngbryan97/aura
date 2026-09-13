"""A restore rewinds an object where it is, so everything holding it sees the rewind.

The fork used to write a copy back over each field. `ConversationalDynamicsPhase`
holds the engine `get_dynamics_engine()` hands every other caller, and after the
first restore the phase had a copy of its own. The phase rewound every arm; the
shared engine, which the response phase updates, counted 18, 19, 20 and 21
messages across four arms from one snapshot.
"""

from __future__ import annotations

import asyncio
import enum
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from core.subject.snapshot import _organ_state, _restore_organ


class Mode(enum.Enum):
    CALM = "calm"
    BUSY = "busy"


@dataclass(frozen=True)
class Frozen:
    n: int


class Topic:
    def __init__(self) -> None:
        self.weight = 0.1


class Engine:
    def __init__(self) -> None:
        self.count = 0
        self.anchors = ["weather"]
        self.by_topic = {"weather": Topic()}
        self.owner = None


class Phase:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.history: deque[str] = deque(["hello"], maxlen=4)
        self.scores = np.zeros(3)
        self.mode = Mode.CALM
        self.frozen = Frozen(1)


def test_an_object_another_caller_holds_keeps_its_identity() -> None:
    engine = Engine()
    phase = Phase(engine)
    topic = engine.by_topic["weather"]
    saved = _organ_state(phase)

    engine.count = 5
    engine.anchors.append("the plan")
    topic.weight = 0.9
    engine.by_topic["the plan"] = Topic()

    _restore_organ(phase, saved)
    assert phase.engine is engine
    assert engine.count == 0
    assert engine.anchors == ["weather"]
    assert engine.by_topic["weather"] is topic and topic.weight == 0.1
    assert "the plan" not in engine.by_topic


def test_every_arm_from_one_snapshot_gets_the_same_start() -> None:
    engine = Engine()
    phase = Phase(engine)
    saved = _organ_state(phase)
    for _ in range(3):
        _restore_organ(phase, saved)
        assert engine.count == 0 and engine.anchors == ["weather"]
        engine.count += 1
        engine.anchors.append("an arm was here")


def test_containers_and_arrays_keep_their_identity() -> None:
    phase = Phase(Engine())
    history, scores = phase.history, phase.scores
    saved = _organ_state(phase)

    history.append("the arm spoke")
    scores += 1.0
    _restore_organ(phase, saved)

    assert phase.history is history and list(history) == ["hello"]
    assert phase.scores is scores and not scores.any()


def test_an_enum_field_is_swapped_and_the_enum_is_untouched() -> None:
    phase = Phase(Engine())
    saved = _organ_state(phase)
    phase.mode = Mode.BUSY
    _restore_organ(phase, saved)
    assert phase.mode is Mode.CALM
    assert Mode.BUSY.value == "busy"


def test_what_cannot_change_in_place_is_replaced() -> None:
    phase = Phase(Engine())
    saved = _organ_state(phase)
    phase.frozen = Frozen(2)
    phase.scores = np.zeros(5)
    _restore_organ(phase, saved)
    assert phase.frozen == Frozen(1)
    assert phase.scores.shape == (3,)


def test_a_cycle_through_the_held_object_restores_and_ends() -> None:
    engine = Engine()
    phase = Phase(engine)
    engine.owner = phase
    saved = _organ_state(phase)
    engine.count = 7
    _restore_organ(phase, saved)
    assert phase.engine is engine and engine.owner is phase and engine.count == 0


@pytest.mark.slow
def test_the_dynamics_phase_and_every_other_caller_share_one_engine_after_restores() -> None:
    from core.conversational.dynamics import get_dynamics_engine
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )

    async def run() -> list[tuple[bool, int]]:
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=11)
            await start_organism(runtime)
            await quiesce_organism(runtime)
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            runtime.freeze_host()
            conversation = CONDITIONS[0]
            await runtime.turn_once(conversation)
            snapshot = runtime.snapshot()
            phase = next(
                p for p in runtime.kernel._phases
                if p.__class__.__name__ == "ConversationalDynamicsPhase"
            )
            seen = []
            for _ in range(3):
                runtime.restore(snapshot)
                shared = get_dynamics_engine()
                seen.append((phase._engine is shared, shared._message_count))
                await runtime.turn_once(conversation)
            return seen

    try:
        seen = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()
    assert all(same for same, _ in seen), f"the phase and the accessor hold different engines: {seen}"
    assert len({count for _, count in seen}) == 1, f"the shared engine kept counting across arms: {seen}"
