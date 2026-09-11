"""An exponent read through a narrow window is steeper than the one underneath.

Her avalanche size exponent was 3.69 against cortex's 1.5, and the whole
distance was being charged to her dynamics. The fit ran from 18 to 66 — half a
decade, ending at the largest cascade sixty recorded units can produce — and a
power law fitted through a finite system's cutoff always comes out steep.

So the published number gets pushed through her own window before the miss is
taken. These tests hold the two halves of that: a fit with no room to scale is
refused rather than compared, and a system that really is cortical measures
2.07 here, not 1.5.
"""

from __future__ import annotations

import random

from core.connectome.criticality import MINIMUM_DECADES
from core.connectome.human_dynamics import expected_through_this_window
from core.connectome.topology import power_law_fit


def _draw(exponent: float, cutoff: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < count:
        value = int((1.0 - rng.random()) ** (-1.0 / (exponent - 1.0)))
        if 1 <= value <= cutoff:
            out.append(value)
    return out


def test_the_fit_reports_how_far_its_tail_runs() -> None:
    fit = power_law_fit(_draw(1.5, 100_000, 4000, seed=7))

    assert fit["decades"] > MINIMUM_DECADES
    assert fit["decades"] == round(
        __import__("math").log10(
            max(_draw(1.5, 100_000, 4000, seed=7)) / fit["xmin"]
        ),
        4,
    )


def test_a_distribution_with_no_room_to_scale_reports_under_a_decade() -> None:
    wide = _draw(1.5, 100_000, 4000, seed=7)
    narrow = [value for value in wide if value <= 3]

    assert power_law_fit(narrow)["decades"] < MINIMUM_DECADES


def test_truncation_alone_steepens_a_true_exponent() -> None:
    """The control the cascade comparison was missing.

    One sample, read through windows of different widths. If the fitted
    exponent were a property of the distribution it would not move.
    """
    sample = _draw(1.5, 100_000, 40_000, seed=7)
    narrow = power_law_fit([v for v in sample if v <= 66])
    wide = power_law_fit(sample)

    assert narrow["alpha"] > wide["alpha"] + 0.4
    assert abs(wide["alpha"] - 1.5) < 0.1
    assert narrow["decades"] < wide["decades"]


def test_a_cortical_system_measures_two_not_one_and_a_half_through_her_window() -> None:
    """Her recording: 2,810 avalanches, the largest of them 66 unit-bins.

    The published 1.5 is fitted over nearly two decades of avalanche size. Read
    through this window it measures 2.07, so 0.57 of her 2.19 miss was the
    instrument and the rest is hers.
    """
    mean, spread = expected_through_this_window(1.5, largest=66, samples=2810)

    assert 1.9 < mean < 2.3
    assert 0.0 < spread < 0.4
    assert mean > 1.5


def test_a_wide_window_gives_the_published_number_back() -> None:
    """The correction has to vanish when the window stops mattering."""
    mean, _spread = expected_through_this_window(1.5, largest=100_000, samples=20_000)

    assert abs(mean - 1.5) < 0.1


def test_an_unusable_window_is_refused_rather_than_guessed() -> None:
    assert expected_through_this_window(1.5, largest=1, samples=2810) == (0.0, 0.0)
    assert expected_through_this_window(1.5, largest=66, samples=4) == (0.0, 0.0)
    assert expected_through_this_window(0.5, largest=66, samples=2810) == (0.0, 0.0)
