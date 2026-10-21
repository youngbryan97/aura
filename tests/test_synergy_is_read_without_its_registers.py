"""The synergy line is also read with the conjunctive units held flat, and condition by condition.

A unit bound from two domains is a column of the domain it acts in, so a
synergy reading can find the sources in the target by reading the unit. The
diagnostic holds the units flat; what is left is what they did to the rest of
the target. Leaving each condition out shows whether one carries a pass.
"""
from __future__ import annotations

import numpy as np

from core.subject.recording import Recording
from core.subject.synergy import leave_one_condition_out, synergy, without_registers


def _recording(n: int = 400, seed: int = 0, through_register_only: bool = True) -> Recording:
    rng = np.random.default_rng(seed)
    s = np.zeros(n + 1)
    d = np.zeros(n + 1)
    unit = np.zeros(n + 1)
    c = np.zeros(n + 1)
    for t in range(n):
        s[t + 1] = 0.7 * s[t] + rng.normal(0, 0.6)
        d[t + 1] = 0.7 * d[t] + rng.normal(0, 0.6)
        unit[t + 1] = 0.25 * (1 + np.tanh(s[t])) * (1 + np.tanh(d[t]))
        effect = 0.0 if through_register_only else 1.2 * unit[t]
        c[t + 1] = 0.5 * c[t] + effect + rng.normal(0, 0.2)
    x = np.stack([s, d, unit, c], axis=1)
    conditions = tuple(("calm", "busy")[(t // 3) % 2] for t in range(n + 1))
    return Recording(
        x=x, conditions=conditions, tags=tuple("ontogeny" for _ in range(n + 1)), times=np.arange(n + 1),
        env=np.zeros((n + 1, 0)), env_names=(), columns=("S.a", "D.a", "C.conj_recurrent_0", "C.b"),
        slices={"S": slice(0, 1), "D": slice(1, 2), "C": slice(2, 4)}, notes={},
    )


def test_holding_the_registers_flat_leaves_everything_else_as_recorded():
    recording = _recording()
    flat = without_registers(recording)
    assert np.allclose(flat.x[:, 2], 0.0)
    assert np.allclose(flat.x[:, [0, 1, 3]], recording.x[:, [0, 1, 3]])


def test_a_register_that_moved_nothing_tells_the_target_less_once_held_flat():
    recording = _recording(through_register_only=True)
    read = synergy(recording, "S", "D", "C", seed=1, of="change")
    held = synergy(without_registers(recording), "S", "D", "C", seed=1, of="change")
    assert held.joint < read.joint


def test_each_condition_is_left_out_in_turn_and_no_step_spans_the_gap():
    recording = _recording(n=240)
    out = leave_one_condition_out(recording, seed=1, triples=(("S", "D", "C"),))
    assert set(out) == {"calm", "busy"}
    assert all(len(rows) == 1 for rows in out.values())
