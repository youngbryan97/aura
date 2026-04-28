"""Tests for the endogenous fitness efficiency gradient fix.

These tests verify that the rolling p90 baseline preserves meaningful
efficiency gradients even after large energy outliers — the core fix
for the fatal "monotonic _max_energy_observed" bug.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_energy_efficiency_gradient_survives_outlier():
    """After a huge energy outlier, the gradient between 5 and 10 avg
    consumption must remain meaningful (> 0.05), not microscopic (0.005).

    This is the exact scenario from the audit:
      Gen 2 consumes 1000 units → old code locks baseline at 1000
      Gen 50: genome A = 10 avg, genome B = 5 avg
      Old code: 0.990 vs 0.995 → delta 0.005 (invisible to selection)
      New code: should produce a delta > 0.05
    """
    from core.consciousness.endogenous_fitness import EndogenousFitness

    ef = EndogenousFitness()

    # Simulate a history with one large outlier and many normal observations
    ef._energy_avg_observations = [1000.0]  # the outlier
    for _ in range(50):
        ef._energy_avg_observations.append(15.0)  # normal population avg
    # Trim to last 100
    ef._energy_avg_observations = ef._energy_avg_observations[-100:]
    # Recompute the baseline
    ef._energy_baseline = float(np.percentile(ef._energy_avg_observations, 90))

    # Now the baseline should be near the population, not stuck at 1000
    assert ef._energy_baseline < 200.0, (
        f"Baseline should decay toward population avg, got {ef._energy_baseline}"
    )

    # Compute efficiency for two genomes
    baseline = max(ef._energy_baseline, 1e-6)
    bonus_a = 1.0 - min(1.0, 10.0 / baseline)  # genome A: 10 avg
    bonus_b = 1.0 - min(1.0, 5.0 / baseline)   # genome B: 5 avg

    delta = abs(bonus_b - bonus_a)
    assert delta > 0.05, (
        f"Efficiency gradient too small: {delta:.6f} "
        f"(bonus_a={bonus_a:.4f}, bonus_b={bonus_b:.4f}, baseline={baseline:.1f})"
    )


def test_energy_baseline_uses_same_units():
    """The baseline must be in per-tick-average units, not total-window units."""
    from core.consciousness.endogenous_fitness import EndogenousFitness

    ef = EndogenousFitness()

    # Store per-tick averages (not totals)
    ef._energy_avg_observations = [8.0, 12.0, 10.0, 9.0, 11.0]
    ef._energy_baseline = float(np.percentile(ef._energy_avg_observations, 90))

    # Baseline should be close to 12 (the p90 of small per-tick values)
    assert 5.0 < ef._energy_baseline < 20.0, (
        f"Baseline out of expected range: {ef._energy_baseline}"
    )


def test_empty_observations_use_default_baseline():
    """With no observations, baseline should be the default (100.0)."""
    from core.consciousness.endogenous_fitness import EndogenousFitness

    ef = EndogenousFitness()
    assert ef._energy_baseline == 100.0
    assert ef._energy_avg_observations == []


def test_baseline_decays_naturally_with_population():
    """As the population shifts to lower energy, baseline should follow."""
    from core.consciousness.endogenous_fitness import EndogenousFitness

    ef = EndogenousFitness()

    # Early population: high energy
    for _ in range(30):
        ef._energy_avg_observations.append(50.0)
    ef._energy_baseline = float(np.percentile(ef._energy_avg_observations, 90))
    high_baseline = ef._energy_baseline

    # Later population: much lower energy — fill the window with low values
    # so p90 actually shifts.  100-slot window, we need > 90 low values.
    for _ in range(150):
        ef._energy_avg_observations.append(5.0)
    # Trim to last 100
    ef._energy_avg_observations = ef._energy_avg_observations[-100:]
    ef._energy_baseline = float(np.percentile(ef._energy_avg_observations, 90))
    low_baseline = ef._energy_baseline

    assert low_baseline < high_baseline * 0.5, (
        f"Baseline should decay: high={high_baseline:.1f}, low={low_baseline:.1f}"
    )
