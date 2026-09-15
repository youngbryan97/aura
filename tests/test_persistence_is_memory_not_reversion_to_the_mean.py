"""ISC-v2's persistence line gives the known answers before it reads Aura.

`intrinsic_gain` predicts each domain's change, and a change is predictable from
the present level exactly when the level has no memory: memoryless noise passed
at a gain near one half, and a random walk, which keeps all of its history,
failed. `persistence` predicts the next level beyond the environment and elapsed
time over forward-chaining folds. These are its known answers, measured before
it was run on any of Aura's recordings: noise, a trend and a state driven only
by its input fail; strong autoregression, a random walk and an oscillator pass.
The first test pins the defect in the v1 reading so it stays visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.intrinsic import intrinsic_gain, persistence
from core.subject.recording import Recording
from core.subject.state import DOMAINS

pytestmark = pytest.mark.unit

ROWS = 800
WIDTH = 3


def _recording(state: np.ndarray, env: np.ndarray) -> Recording:
    slices = {key: slice(i * WIDTH, (i + 1) * WIDTH) for i, key in enumerate(DOMAINS)}
    return Recording(
        x=state,
        conditions=tuple("x" for _ in range(len(state))),
        tags=tuple("" for _ in range(len(state))),
        times=np.arange(len(state), dtype=np.float64),
        env=env,
        env_names=tuple(f"e{i}" for i in range(env.shape[1])),
        columns=tuple(f"{key}.{i}" for key in DOMAINS for i in range(WIDTH)),
        slices=slices,
        notes={},
    )


def _autoregressive(coefficient: float, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(ROWS, WIDTH * len(DOMAINS)))
    state = np.zeros_like(noise)
    state[0] = noise[0]
    for row in range(1, ROWS):
        state[row] = coefficient * state[row - 1] + noise[row]
    return _recording(state, rng.normal(size=(ROWS, 2)))


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_the_v1_reading_passes_memoryless_noise(seed: int) -> None:
    assert intrinsic_gain(_autoregressive(0.0, seed), seed=seed).passes


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_memoryless_noise_fails(seed: int) -> None:
    assert not persistence(_autoregressive(0.0, seed), seed=seed).passes


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_strong_autoregression_passes(seed: int) -> None:
    assert persistence(_autoregressive(0.95, seed), seed=seed).passes


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_a_random_walk_passes(seed: int) -> None:
    assert persistence(_autoregressive(1.0, seed), seed=seed).passes


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_a_trend_with_noise_is_not_persistence(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 8.0, ROWS).reshape(-1, 1)
    state = ramp * rng.uniform(0.5, 1.5, size=(1, WIDTH * len(DOMAINS))) + rng.normal(size=(ROWS, WIDTH * len(DOMAINS)))
    assert not persistence(_recording(state, rng.normal(size=(ROWS, 2))), seed=seed).passes


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_a_state_that_is_only_its_input_fails(seed: int) -> None:
    rng = np.random.default_rng(seed)
    drive = rng.normal(size=(ROWS, 3))
    state = np.hstack([np.tanh(drive * scale) for scale in np.linspace(0.3, 2.0, 10)])
    assert not persistence(_recording(state, drive), seed=seed).passes


def test_an_oscillator_passes() -> None:
    rng = np.random.default_rng(0)
    angle = 0.4
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    state = np.zeros((ROWS, WIDTH * len(DOMAINS)))
    phases = rng.normal(size=(len(DOMAINS), 2))
    for row in range(ROWS):
        for index in range(len(DOMAINS)):
            phases[index] = phases[index] @ rotate.T
            state[row, index * WIDTH : index * WIDTH + 2] = phases[index]
        state[row] += rng.normal(scale=0.05, size=WIDTH * len(DOMAINS))
    assert persistence(_recording(state, rng.normal(size=(ROWS, 2)))).passes
