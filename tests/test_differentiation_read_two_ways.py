"""The participation ratio is maximised by destroying integration.

The completion tracker records that the D_eff/D >= 0.40 bar can reward
degenerate controls while a healthy recurrent reference scores much lower, and
that the reason is structural: variables that constrain each other share
variance, so the better integrated a system is the fewer independent directions
its covariance has.

Measured on run_032, six hours and 79,200 frames of the real organism:

    participation ratio   real 0.077   column-shuffled 0.998   degenerate 0.004
    distinguishable       real 0.790   column-shuffled 1.000   degenerate 0.000

A surrogate that destroys every coupling and keeps every marginal scores
almost the maximum on the criterion. These tests hold that contrast on
synthetic data of the same shape, so the finding is checkable without the
six-hour recording.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from core.subject.differentiation import distinguishable_states, effective_dimension
from core.subject.recording import Recording, slices_from_columns
from core.subject.state import DOMAINS, feature_names


def _recording(x: np.ndarray) -> Recording:
    columns = tuple(feature_names())[: x.shape[1]]
    frames = x.shape[0]
    return Recording(
        x=x,
        conditions=tuple("conversation" if i % 2 else "idle" for i in range(frames)),
        tags=tuple("" for _ in range(frames)),
        times=np.arange(frames, dtype=np.float64),
        env=np.zeros((frames, 1)),
        env_names=("host_load",),
        columns=columns,
        slices=slices_from_columns(columns),
        notes={},
    )


def _coupled(frames: int = 400, width: int = 24, seed: int = 7) -> np.ndarray:
    """A system whose columns constrain each other, with its own noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(frames, 3))
    mixing = rng.normal(size=(3, width))
    return latent @ mixing + rng.normal(0, 0.15, size=(frames, width))


def test_shuffling_away_every_coupling_raises_the_participation_ratio() -> None:
    x = _coupled()
    real = _recording(x)
    rng = np.random.default_rng(11)
    shuffled_x = x.copy()
    for column in range(shuffled_x.shape[1]):
        rng.shuffle(shuffled_x[:, column])
    shuffled = dataclasses.replace(real, x=shuffled_x)

    assert effective_dimension(shuffled).normalised > effective_dimension(real).normalised, (
        "the criterion is supposed to be higher for the system with no coupling, "
        "which is the finding this holds"
    )


def test_the_second_reading_does_not_fall_when_integration_rises() -> None:
    x = _coupled()
    real = _recording(x)
    rng = np.random.default_rng(11)
    shuffled_x = x.copy()
    for column in range(shuffled_x.shape[1]):
        rng.shuffle(shuffled_x[:, column])
    shuffled = dataclasses.replace(real, x=shuffled_x)

    coupled_states = distinguishable_states(real).normalised
    assert coupled_states > 0.5, coupled_states
    # It does not have to beat the shuffle — differentiation alone never
    # separates a subject from noise — but it must not collapse.
    assert coupled_states >= distinguishable_states(shuffled).normalised * 0.5


def test_both_readings_reject_a_system_that_is_one_scalar() -> None:
    frames, width = 400, 24
    rng = np.random.default_rng(3)
    one = rng.normal(size=(frames, 1))
    degenerate = _recording(np.repeat(one, width, axis=1) + rng.normal(0, 1e-3, (frames, width)))
    assert effective_dimension(degenerate).normalised < 0.05
    assert distinguishable_states(degenerate).normalised < 0.05


def test_the_resolution_comes_from_her_own_variation() -> None:
    """Rescaling a column cannot change how many states it distinguishes."""
    x = _coupled()
    real = _recording(x)
    scaled = x.copy()
    scaled[:, 0] *= 1_000.0
    rescaled = dataclasses.replace(real, x=scaled)
    assert distinguishable_states(rescaled).states == distinguishable_states(real).states


def test_a_recording_with_nothing_in_it_says_so() -> None:
    empty = _recording(np.zeros((400, 12)))
    reading = distinguishable_states(empty)
    assert reading.normalised == 0.0
    assert "distinguishable" in reading.why or "columns" in reading.why


def test_the_reading_is_reported_beside_the_first_and_not_instead_of_it() -> None:
    from pathlib import Path

    runner = Path("tools/run_subject_core.py").read_text(encoding="utf-8")
    assert 'evidence["differentiation"] = effective_dimension(recording).as_dict()' in runner
    assert 'evidence["differentiation"]["distinguishable"] = distinguishable_states(' in runner


def test_the_domains_have_not_moved_under_the_helper() -> None:
    assert len(DOMAINS) == 10
