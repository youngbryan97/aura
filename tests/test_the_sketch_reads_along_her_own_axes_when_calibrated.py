"""With calibrated axes, the sketch's four projections read where an array actually moves."""
from __future__ import annotations

import numpy as np

from core.subject import calibrated_axes
from core.subject.sketch import sketch


def _life(seed: int = 0, frames: int = 60, size: int = 256):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((size, 4)))[0].T
    weights = rng.standard_normal((frames, 4)) * np.array([3.0, 2.0, 1.0, 0.5])
    return basis, [w @ basis + 0.01 * rng.standard_normal(size) for w in weights]


def test_calibrated_projections_recover_how_far_the_array_moved_along_its_axes():
    calibrated_axes.clear()
    basis, readings = _life()
    described = calibrated_axes.calibrate({"organ:substrate.x": readings})
    assert described["organ:substrate.x"]["calibrated"] is True
    axes = calibrated_axes.axes_for("organ:substrate.x")
    moved = readings[0] + 1.0 * basis[0]
    gap = abs(sketch(moved, axes=axes)["p0"] - sketch(readings[0], axes=axes)["p0"])
    assert gap > 0.95
    calibrated_axes.clear()


def test_without_calibration_the_frozen_random_directions_are_read():
    calibrated_axes.clear()
    _basis, readings = _life(seed=1)
    assert sketch(readings[0], axes=None) == sketch(readings[0])
    assert calibrated_axes.axes_for("organ:substrate.x") is None


def test_too_few_readings_leave_the_array_uncalibrated():
    calibrated_axes.clear()
    _basis, readings = _life(seed=2, frames=3)
    assert calibrated_axes.calibrate({"organ:mesh.column_activations": readings})["organ:mesh.column_activations"]["calibrated"] is False
    calibrated_axes.clear()
