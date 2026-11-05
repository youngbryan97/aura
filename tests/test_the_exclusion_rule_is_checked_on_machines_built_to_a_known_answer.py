"""The machines for checking J*'s exclusion rule are built as their names say (core/subject/built_candidates.py).

These hold the construction, never the outcome: the outcome is the preregistered run's. Each machine is wired as its
answer requires, and the batched forks are the toy's own step, so the run measures the machines it says it measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject import built_candidates as bc
from core.subject.state import DOMAINS

FIRST, SECOND = bc.HALVES


def _reads(system, target: str, source: str) -> bool:
    block = system.coupling.get(target, {}).get(source)
    return block is not None and bool(np.any(block != 0))


def test_the_halves_split_the_ten_domains():
    assert set(FIRST) | set(SECOND) == set(DOMAINS)
    assert not set(FIRST) & set(SECOND)


def test_two_apart_has_no_wire_between_its_halves():
    system = bc.build("two_apart", seed=3)
    assert not any(_reads(system, t, s) for t in FIRST for s in SECOND)
    assert not any(_reads(system, t, s) for t in SECOND for s in FIRST)
    assert all(_reads(system, t, s) for half in bc.HALVES for t in half for s in half if s != t)


def test_one_drives_runs_from_the_first_half_to_the_second_and_never_back():
    system = bc.build("one_drives", seed=3)
    assert all(_reads(system, t, s) for t in SECOND for s in FIRST)
    assert not any(_reads(system, t, s) for t in FIRST for s in SECOND)


def test_joined_weakly_joins_both_ways_at_half_strength(monkeypatch):
    # Every drawn block made one, so what is left is the construction: within a half one, across it WEAK_SHARE,
    # both scaled alike to the radius.
    monkeypatch.setattr(bc, "_matrix", lambda rng, rows, cols, strength: np.ones((rows, cols)))
    weak = bc.build("joined_weakly", seed=3)
    within = {float(weak.coupling[t][s][0, 0]) for half in bc.HALVES for t in half for s in half if s != t}
    across = {float(weak.coupling[t][s][0, 0]) for t in FIRST for s in SECOND} | {
        float(weak.coupling[t][s][0, 0]) for t in SECOND for s in FIRST
    }
    assert len(within) == 1 and len(across) == 1
    assert next(iter(across)) == pytest.approx(bc.WEAK_SHARE * next(iter(within)))
    whole = bc.build("one_whole", seed=3)
    assert all(_reads(whole, t, s) for t in DOMAINS for s in DOMAINS if s != t)


def test_every_machine_has_its_answer_written_down():
    assert set(bc.EXPECTED) == set(bc.MACHINES)
    with pytest.raises(KeyError):
        bc.build("three_apart")


def test_the_batched_fork_is_the_toy_s_own_step():
    system = bc.build("one_drives", seed=5, noise=0.0)
    rng = np.random.default_rng(0)
    one = system.start(rng)
    batch = bc._batch_roll(system, {k: v[:, None] for k, v in one.items()}, 6, np.random.default_rng(1))
    state = one
    for step in range(6):
        state = system.step(state, np.random.default_rng(2))
        for key in DOMAINS:
            assert np.allclose(batch[step][key][:, 0], state[key])


def test_a_held_side_does_not_move_and_the_free_side_does():
    system = bc.build("two_apart", seed=5)
    start = {key: np.random.default_rng(9).normal(size=(system.widths[key], 4)) for key in DOMAINS}
    rows = bc._batch_roll(system, start, 5, np.random.default_rng(1), held=SECOND)
    assert all(np.array_equal(rows[-1][key], start[key]) for key in SECOND)
    assert not all(np.array_equal(rows[-1][key], start[key]) for key in FIRST)
