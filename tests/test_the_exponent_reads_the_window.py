"""An exponent that moves with the recording was reading the recording.

Her avalanche size exponent was 3.69 against cortex's 1.5, measured on sixty of
her 4,096 units. Sixty was chosen by analogy to Beggs and Plenz's sixty
electrodes, and an LFP electrode integrates thousands of cells, so the analogy
undercounted by three orders of magnitude. The fit that came out of it ran from
18 to 66 — half a decade, ending at the largest cascade sixty units can make.

This holds the sweep that showed it: refit the same recording while reading
more of it, and the exponent has to move if it was the slope of a cutoff.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.connectome.criticality import (
    MINIMUM_DECADES,
    exponent_against_recording_width,
)


def _cascading_raster(ticks: int = 4000, units: int = 512, seed: int = 5) -> np.ndarray:
    """A raster whose cascades have a size distribution wider than any window.

    Each cascade picks a size from a heavy-tailed draw and spends it over
    consecutive bins, so the largest cascade a recording can see is set by how
    many of the units it reads.
    """
    rng = np.random.default_rng(seed)
    raster = np.zeros((ticks, units), dtype=np.float64)
    tick = 0
    while tick < ticks - 1:
        size = min(int((1.0 - rng.random()) ** (-1.0 / 0.5)), units * 4)
        length = max(1, min(int(np.sqrt(size)), ticks - tick - 1))
        for step in range(length):
            spread = max(1, size // length)
            who = rng.choice(units, size=min(spread, units), replace=False)
            raster[tick + step, who] = 1.0
        tick += length + 1 + int(rng.integers(1, 4))
    return raster


def test_the_sweep_refuses_an_empty_recording() -> None:
    report = exponent_against_recording_width(np.zeros((0, 0)))

    assert report["rows"] == []
    assert report["verdict"] == "no recording"


def test_every_row_names_the_bin_it_used() -> None:
    report = exponent_against_recording_width(
        _cascading_raster(), widths=(64, 256, 0)
    )

    assert [row["units_recorded"] for row in report["rows"]] == [64, 256, 512]
    for row in report["rows"]:
        assert row["bin_ticks"] >= 1
        assert 0.0 <= row["active_fraction"] <= 1.0


def test_a_narrow_window_makes_the_exponent_steeper() -> None:
    """The claim the cascade comparison rests on, on a raster built to have it.

    The same cascades read through more units reach further, so the fitted
    exponent falls. If it did not, the number would be a property of the
    dynamics and the old measurement would have stood.
    """
    report = exponent_against_recording_width(
        _cascading_raster(), widths=(32, 128, 0)
    )
    rows = {row["units_recorded"]: row for row in report["rows"]}

    assert rows[32]["largest"] < rows[512]["largest"]
    fitted = [row for row in report["rows"] if row["size_exponent"] > 0.0]
    assert len(fitted) >= 2
    assert report["exponent_range"] > 0.0


@pytest.mark.parametrize("width", [16, 64])
def test_a_window_that_cannot_support_a_fit_says_so(width: int) -> None:
    """Under a decade of tail is reported, not quietly fitted and compared."""
    report = exponent_against_recording_width(
        _cascading_raster(ticks=600, units=128), widths=(width,)
    )
    row = report["rows"][0]

    if row["size_exponent"] > 0.0 and row["size_decades"] < MINIMUM_DECADES:
        assert row["size_decades"] < MINIMUM_DECADES
    assert "size_decades" in row
