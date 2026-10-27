"""The field's input weights learn from a turn whatever its tick count was when the turn began.

The field ticks at 20 Hz and the bridge feeds it at 10 Hz, after the field's
tick in each frame (core/subject/steppable.py), so the field takes input on
every other tick and clears it as it reads it. Its input trace was kept only on
its plasticity ticks, every tenth, and on an odd tick count every one of those
fell on a tick with no input: the trace stayed zero and no turn's worth ever
reached the input weights. A run inherits the parity from the anchor it forks,
so a whole content run either taught the input weights or never did: 12 of 17
saved runs never did, a seed and a commit decided nothing (seed 31 at
24d1a0191 was untaught and its scrambled twin taught), and no run that taught
them had passed the structure ground.

The trace is now kept on every tick, at a tenth of a plasticity step's weight
each, so it holds what the field was driven with over the whole turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.unified_field import UnifiedField

pytestmark = pytest.mark.unit


def _a_turn_on_the_harness_schedule(start_tick: int, frames: int = 40) -> UnifiedField:
    """One field tick a frame, and the bridge's feed after it on every other frame."""
    field = UnifiedField()
    field._tick_count = start_tick
    pattern = np.random.default_rng(5).uniform(-1.0, 1.0, field.cfg.mesh_input_dim).astype(np.float32)
    for frame in range(frames):
        field._tick()
        if frame % 2 == 1:
            field.receive_mesh(pattern)
    return field


@pytest.mark.parametrize("start_tick", [0, 1, 2, 3])
def test_a_good_turn_moves_the_input_weights_from_any_tick_count(start_tick: int) -> None:
    field = _a_turn_on_the_harness_schedule(start_tick)
    before = field.input_shares()
    lesson = field.teach(1.0)
    assert lesson["taught"]
    assert lesson["shares"]["mesh"] > before["mesh"]


def test_the_lesson_does_not_depend_on_the_parity_it_started_at() -> None:
    gains = []
    for start_tick in (0, 1):
        field = _a_turn_on_the_harness_schedule(start_tick)
        before = field.input_shares()["mesh"]
        gains.append(field.teach(1.0)["shares"]["mesh"] - before)
    assert gains[0] > 0.0
    assert gains[1] == pytest.approx(gains[0], rel=0.25)


def test_a_field_fed_on_every_tick_learns_at_the_rate_it_did() -> None:
    """Fed every tick, the trace over ten ticks is one plasticity step's worth, as it was."""
    field = UnifiedField()
    pattern = np.random.default_rng(3).uniform(-1.0, 1.0, field.cfg.mesh_input_dim).astype(np.float32)
    traces = []
    for _ in range(10):
        field.receive_mesh(pattern)
        field._tick()
        traces.append(np.outer(field.F, field._last_input))
    assert np.allclose(field._eligible_input, np.mean(traces, axis=0), atol=1e-6)
