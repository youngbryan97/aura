"""Three of four triples failed against a null that scored higher than the system.

`A+S->G` read a synergy fraction of 0.079 against a shifted null's 0.255 in
run_019, and the reason was the estimator rather than the organism.

A Gaussian mutual information from log determinants is biased upward, and the
bias grows with the number of columns. A synergy is a joint over eight columns
minus two marginals over four, so the joint carries about twice the bias of
either marginal and the difference is positive whatever the data. Under the
null there is no signal to divide that by, so the fraction — the value over the
joint — goes wherever the noise goes.

Cross-fitting removed the bias and broke on a life that drifts. It scored each
contiguous quarter against a covariance fitted on the other three, and in
run_023 the first quarter of the affect domain scored a held-out ratio of -51
nats. The fold average clamped to zero, the self-model's information survived,
and A+S->G read a synergy of -1.88, which no decomposition can produce.

The bias of a Gaussian log determinant is known in closed form, so it is taken
off analytically instead, with every row used. Each column is carried to a
standard normal by rank first, so one column's shape cannot pass for
dependence.

Two more things follow. The shift slides both sources together rather than one
against the other, because redundancy between the sources is structure an
integrated system is entitled to and taking it away measures a smaller system.
And the fraction is only taken while the null's joint is large enough to divide
by.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.synergy import _copula_normal, _gaussian_mi, _plugin_mi

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


def _independent_means(width_x: int, width_y: int, rows: int = 4000, seeds: int = 20):
    corrected, plugin = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        x, y = rng.normal(size=(rows, width_x)), rng.normal(size=(rows, width_y))
        corrected.append(_gaussian_mi(x, y))
        plugin.append(_plugin_mi(x, y))
    return float(np.mean(corrected)), float(np.mean(plugin))


@pytest.mark.parametrize("width_x,width_y", [(4, 4), (8, 4)])
def test_the_corrected_estimate_is_centred_on_zero_under_independence(width_x, width_y) -> None:
    """Within four of its own standard errors.

    Under independence twice rows times the information is chi-squared on
    width_x * width_y degrees of freedom, so one estimate has a standard
    deviation of sqrt(width_x * width_y / 2) / rows, and a mean of twenty has
    that over the square root of twenty. The bound comes from there.
    """
    rows, seeds = 4000, 20
    corrected, _ = _independent_means(width_x, width_y, rows, seeds)
    standard_error = np.sqrt(width_x * width_y / 2.0) / rows / np.sqrt(seeds)
    assert abs(corrected) < 4.0 * standard_error


@pytest.mark.parametrize("width_x,width_y", [(4, 4), (8, 4)])
def test_the_correction_takes_off_the_bias_the_plugin_carries(width_x, width_y) -> None:
    """The plug-in sits at width_x * width_y / (2 * rows), and the corrected
    estimate keeps under a fifth of that."""
    rows = 4000
    corrected, plugin = _independent_means(width_x, width_y, rows)
    expected = width_x * width_y / (2.0 * rows)
    assert plugin == pytest.approx(expected, rel=0.25)
    assert abs(corrected) < 0.2 * expected


def test_a_real_signal_survives_the_correction() -> None:
    """An unbiased estimator that cannot see anything is not an improvement."""
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(4000, 1))
    x = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    y = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    corrected = _gaussian_mi(x, y)
    assert corrected > 1.0
    assert corrected == pytest.approx(_plugin_mi(x, y), rel=0.05)


def test_too_few_rows_says_nothing() -> None:
    """The correction needs more rows than columns, and a covariance on fewer
    is singular. Zero is the honest reading, and both arms get it."""
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    assert _gaussian_mi(x, y) == 0.0


def test_stretching_one_column_changes_nothing_after_ranks() -> None:
    """Information does not change under a monotone transform, and a raw
    Gaussian estimate does. The ranks are what make the estimate respect that."""
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(4000, 1))
    x = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    y = np.hstack([latent + 0.3 * rng.normal(size=(4000, 1)) for _ in range(4)])
    stretched = x.copy()
    stretched[:, 0] = np.exp(3.0 * stretched[:, 0])
    before = _gaussian_mi(_copula_normal(x), _copula_normal(y))
    after = _gaussian_mi(_copula_normal(stretched), _copula_normal(y))
    assert after == pytest.approx(before, abs=1e-9)
    assert abs(_gaussian_mi(stretched, y) - _gaussian_mi(x, y)) > 0.01


def test_a_flat_stretch_stays_flat_under_ranks() -> None:
    """Ties share a rank. Ordering them by row would give a constant stretch
    a trend it does not have."""
    column = np.concatenate([np.full(50, 2.0), np.linspace(0.0, 1.0, 50)])[:, None]
    ranked = _copula_normal(column)
    assert np.ptp(ranked[:50, 0]) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_first_quarter_unlike_the_rest_does_not_read_as_negative_synergy(seed) -> None:
    """run_023, reduced to the part that mattered.

    One source spends its first quarter at a different level, nearly still,
    the way a domain does before a life has settled. The cross-fitted estimate
    scored that quarter against a model of the other three, clamped the
    source's information and the joint to zero, and returned a synergy near
    minus the other source's information. The corrected joint cannot fall
    below the larger marginal by more than noise, and the synergy stays
    positive.
    """
    rng = np.random.default_rng(seed)
    rows, early = 479, 479 // 4
    source_a, source_b = rng.normal(size=(rows, 1)), rng.normal(size=(rows, 1))
    a = source_a + 0.4 * rng.normal(size=(rows, 3))
    b = source_b + 0.4 * rng.normal(size=(rows, 3))
    y = np.hstack([source_a + source_b, source_a - source_b, rng.normal(size=(rows, 1))])
    y = y + 0.4 * rng.normal(size=(rows, 3))
    a[:early] = 4.0 + 0.02 * rng.normal(size=(early, 3))
    a, b, y = _copula_normal(a), _copula_normal(b), _copula_normal(y)
    joint = _gaussian_mi(np.hstack([a, b]), y)
    larger = max(_gaussian_mi(a, y), _gaussian_mi(b, y))
    assert joint >= larger - 0.01
    assert joint - larger > 0.05


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
        null_draws=1000, null_spread=0.02,
    )
    assert strong.passes is True
    flat = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.5, null_q99=0.1,
        interaction_gain=-0.01, rows=500, raw_null_q99=0.1,
        null_draws=1000, null_spread=0.02,
    )
    assert flat.passes is False


def test_a_triple_needs_the_raw_value_above_its_own_null() -> None:
    """A ratio whose denominators differ between the arms is not one comparison."""
    from core.subject.synergy import SynergyReport

    thin = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.05, normalised=0.5, null_q99=0.1,
        interaction_gain=0.05, rows=500, raw_null_q99=0.3,
        null_draws=1000, null_spread=0.02,
    )
    assert thin.passes is False


def test_a_margin_narrower_than_the_bars_own_spread_does_not_pass() -> None:
    """`null_spread` was computed, reported, and decided nothing.

    Its own comment beside the draw loop says what it is for: a synergy a
    hundredth above a null estimated to within two hundredths has not cleared
    it. The predicate read the quantile alone, so it did.
    """
    from core.subject.synergy import SynergyReport

    wide_bar = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.11, null_q99=0.10,
        interaction_gain=0.05, rows=500, raw_null_q99=0.1,
        null_draws=1000, null_spread=0.02,
    )
    assert wide_bar.normalised > wide_bar.null_q99
    assert wide_bar.passes is False, "a hundredth over a two-hundredth bar is not a margin"

    tight_bar = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.11, null_q99=0.10,
        interaction_gain=0.05, rows=500, raw_null_q99=0.1,
        null_draws=1000, null_spread=0.005,
    )
    assert tight_bar.passes is True


def test_a_bar_with_no_draws_behind_it_cannot_be_cleared() -> None:
    """A spread of 0.0 from one draw is absence, not a measurement.

    Comparing a margin against it passes everything, which is the absence of
    a check reported as a passed check.
    """
    from core.subject.synergy import SynergyReport

    unmeasured = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.5, null_q99=0.1,
        interaction_gain=0.05, rows=500, raw_null_q99=0.1,
        null_draws=1, null_spread=0.0,
    )
    assert unmeasured.passes is False

    measured = SynergyReport(
        sources=("A", "S"), target="G", joint=1.0, unique_a=0.1, unique_b=0.1,
        redundancy=0.2, synergy=0.5, normalised=0.5, null_q99=0.1,
        interaction_gain=0.05, rows=500, raw_null_q99=0.1,
        null_draws=1000, null_spread=0.0,
    )
    assert measured.passes is True
