"""Three of four triples failed against a null that scored higher than the system.

`A+S->G` read a synergy fraction of 0.079 against a shifted null's 0.255 in
run_019, and the reason was the estimator rather than the organism.

A Gaussian mutual information from log determinants is biased upward, and the
bias grows with the number of columns. A synergy is a joint over eight columns
minus two marginals over four, so the joint carries about twice the bias of
either marginal and the difference is positive whatever the data. Under the
null there is no signal to divide that by, so the fraction — the value over the
joint — goes wherever the noise goes.

Cross-fitting removes it. The covariance is fitted on one block of rows and the
log-likelihood ratio scored on the next, so a cross-block structure that was
noise does not improve the held-out likelihood. Under independence the estimate
falls to zero instead of to the bias, and the null collapses the way a null
should.

Two more things follow. The shift slides both sources together rather than one
against the other, because redundancy between the sources is structure an
integrated system is entitled to and taking it away measures a smaller system.
And the fraction is only taken while the null's joint is large enough to divide
by.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.synergy import _gaussian_mi, _plugin_mi

pytestmark = pytest.mark.unit


def test_the_plugin_estimator_is_biased_upward_on_independent_blocks() -> None:
    """The defect, measured. This is what the null was reading."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(4000, 4)), rng.normal(size=(4000, 4))
    assert _plugin_mi(x, y) > 0.0


def test_the_bias_grows_with_width() -> None:
    """Which is why it does not cancel between a joint and its marginals."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(4000, 4))
    narrow = _plugin_mi(rng.normal(size=(4000, 4)), y)
    wide = _plugin_mi(rng.normal(size=(4000, 8)), y)
    assert wide > narrow


def test_cross_fitting_takes_it_to_zero() -> None:
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(4000, 4)), rng.normal(size=(4000, 4))
    assert _gaussian_mi(x, y) == pytest.approx(0.0, abs=1e-3)


def test_cross_fitting_takes_the_wide_case_to_zero_too(seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    assert _gaussian_mi(rng.normal(size=(4000, 8)), rng.normal(size=(4000, 4))) == pytest.approx(
        0.0, abs=1e-3
    )


def test_a_real_signal_survives_cross_fitting() -> None:
    """An unbiased estimator that cannot see anything is not an improvement."""
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(4000, 1))
    x = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    y = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    crossfit = _gaussian_mi(x, y)
    assert crossfit > 1.0
    assert crossfit == pytest.approx(_plugin_mi(x, y), rel=0.05)


def test_too_few_rows_falls_back_rather_than_returning_nothing() -> None:
    """Both arms of the comparison then get the plug-in value the same way."""
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
    assert _gaussian_mi(x, y) == pytest.approx(_plugin_mi(x, y))


def test_the_null_slides_both_sources_together() -> None:
    """Redundancy between the sources is structure the real system has."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "subject" / "synergy.py"
    ).read_text(encoding="utf-8")
    assert "slid_a = np.roll(a, shift, axis=0)" in source
    assert "slid_b = np.roll(b, shift, axis=0)" in source


def test_a_triple_needs_the_interaction_gain_as_well() -> None:
    """It was computed, reported, and decided nothing."""
    from core.subject.synergy import SynergyReport

    strong = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.5, null_q99=0.1,
        interaction_gain=0.05, rows=500, raw_null_q99=0.1,
    )
    assert strong.passes is True
    flat = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.5, null_q99=0.1,
        interaction_gain=-0.01, rows=500, raw_null_q99=0.1,
    )
    assert flat.passes is False


def test_a_triple_needs_the_raw_value_above_its_own_null() -> None:
    """A ratio whose denominators differ between the arms is not one comparison."""
    from core.subject.synergy import SynergyReport

    thin = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.05, normalised=0.5, null_q99=0.1,
        interaction_gain=0.05, rows=500, raw_null_q99=0.3,
    )
    assert thin.passes is False
