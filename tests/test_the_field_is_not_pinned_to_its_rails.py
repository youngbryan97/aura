"""The unified field integrates everything, and it was a constant.

Its integrator is `F + (-decay*F + drive)*dt`, so the field settles where
`decay*F` equals the drive — at `drive / decay`. The leak was a fiftieth and
the drive is a tanh, so that equilibrium sat fifty times outside the range the
field is clipped to. Every run pinned it to the rails within seconds: mean |F|
of 0.906 in the campaign logs, an anti-degeneracy rescue firing over a thousand
times, and the field's own degradation record saying the rescue was not working
and the state carried little information.

A twenty-hertz layer that is a constant contributes nothing to any coupling
measured through it, and reads as high coherence while doing so — a railed
field looks maximally unified.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.substrate_accel import field_integrate
from core.consciousness.unified_field import FieldConfig, UnifiedField


def _drive(field: UnifiedField, rng: np.random.Generator) -> np.ndarray:
    """Input of the size the real projection matrices produce."""
    return np.asarray(
        field._W_input_batched @ (rng.standard_normal(148).astype(np.float32) * 0.5),
        dtype=np.float32,
    )


def _run(field: UnifiedField, steps: int, rng: np.random.Generator, driven: bool) -> np.ndarray:
    cfg = field.cfg
    leak = float(cfg.decay)
    state = np.asarray(field.F, dtype=np.float32)
    drive = _drive(field, rng) if driven else np.zeros(cfg.dim, dtype=np.float32)
    for step in range(steps):
        if driven and step % 200 == 0:
            drive = _drive(field, rng)
        activity = np.tanh(
            cfg.activation_gain * (field._W_field_sparse @ state + drive)
        ).astype(np.float32)
        noise = (rng.standard_normal(cfg.dim) * cfg.noise_sigma).astype(np.float32)
        state = field_integrate(
            state,
            (activity * leak).astype(np.float32),
            (noise * leak).astype(np.float32),
            leak,
            cfg.dt,
        )
    return state


def test_the_equilibrium_lies_inside_the_range_the_field_is_clipped_to() -> None:
    """The arithmetic the defect was, run.

    Hold the drive at the largest value it can take — a tanh, so one — and let
    the integrator settle. It has to come to rest at the drive, inside the
    clip. Scaled the old way it comes to rest at `drive / decay`, which for a
    leak of a fiftieth is fifty, and the clip turns that into a rail.
    """
    cfg = FieldConfig()
    leak = float(cfg.decay)
    full = np.ones(cfg.dim, dtype=np.float32)
    zero = np.zeros(cfg.dim, dtype=np.float32)

    settled = np.zeros(cfg.dim, dtype=np.float32)
    for _ in range(int(round(20.0 / leak / cfg.dt))):
        settled = field_integrate(settled, (full * leak).astype(np.float32), zero, leak, cfg.dt)
    assert float(np.mean(settled)) == pytest.approx(1.0, abs=0.01), (
        "the field does not settle at the drive that holds it"
    )

    as_written = np.zeros(cfg.dim, dtype=np.float32)
    for _ in range(200):
        as_written = field_integrate(as_written, full, zero, leak, cfg.dt)
    assert float(np.mean(np.abs(as_written) > 0.99)) == 1.0, (
        "the unscaled drive no longer rails the field; this test has lost its subject"
    )


def test_a_driven_field_is_not_on_its_rails() -> None:
    field = UnifiedField()
    state = _run(field, 1200, np.random.default_rng(3), driven=True)
    railed = float(np.mean(np.abs(state) > 0.99))
    assert railed == 0.0, f"{railed:.1%} of the field is pinned at the rail"
    assert 0.02 < float(np.mean(np.abs(state))) < 0.90, (
        f"mean |F| is {float(np.mean(np.abs(state))):.4f}, which is silence or a rail"
    )
    assert float(np.std(state)) > 0.05, "the field is the same number in every dimension"


def test_the_loop_gain_is_the_declared_one_with_the_tanh_counted() -> None:
    """The spectral radius alone is not the loop gain.

    A disturbance grows or dies by the radius times the gain it is fed
    through. Scaling the weights and leaving the tanh out of the arithmetic
    gave a field written for the edge of stability a loop gain of 0.40.
    """
    field = UnifiedField()
    radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(field.W_field)))))
    loop = radius * field.cfg.activation_gain
    assert abs(loop - field.cfg.spectral_radius) < 0.01, (
        f"loop gain {loop:.4f} against a declared {field.cfg.spectral_radius}"
    )
    assert loop < 1.0, "a loop gain at or above one is a field that runs away"


def test_the_response_outlasts_the_input_and_then_fades() -> None:
    """What the docstring's second property means once it is measurable."""
    field = UnifiedField()
    rng = np.random.default_rng(5)
    driven = _run(field, 1200, rng, driven=True)
    field.F = driven.copy()
    after_a_second = _run(field, int(round(1.0 / field.cfg.dt)), rng, driven=False)
    field.F = driven.copy()
    long_after = _run(field, int(round(40.0 / field.cfg.dt)), rng, driven=False)

    start = float(np.mean(np.abs(driven)))
    assert float(np.mean(np.abs(after_a_second))) > start * 0.5, (
        "the field goes silent the moment its input does"
    )
    assert float(np.mean(np.abs(long_after))) < start * 0.8, (
        "the field holds its pattern for ever, which is a loop gain above one"
    )
