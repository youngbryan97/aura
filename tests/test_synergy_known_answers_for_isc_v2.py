"""The known answers ISC-v2 has to give before it reads Aura.

docs/ISC_V2_PREREGISTRATION.md scores a triple's synergy on the target's change
and names three answers that must hold first: a product under drift passes, the
same drift with the sources acting additively fails, and drift alone fails.
The second did not hold. An additive target passed 10 of 20 seeds, on an
interaction gain a few millionths either side of zero, because a positive
single split was all the bar asked for. The gain now has to clear zero by its
lower bound over the forward-chaining folds irreducibility is scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.recording import Recording
from core.subject.state import DOMAINS
from core.subject.synergy import synergy

pytestmark = pytest.mark.unit

#: The battery's own length: 60 rounds of 8 conditions, one row per turn.
ROWS = 479
SEEDS = (1, 2, 3)


def _recording(kind: str, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=ROWS)
    b = 0.8 * a + 0.6 * rng.normal(size=ROWS)
    drift = 0.5 + 0.3 * rng.normal(size=ROWS)
    noise = 0.3 * rng.normal(size=ROWS)
    change = {"product": (a - b) + a * b + noise, "additive": a + b + noise, "drift": noise}[kind]
    step = np.zeros(ROWS)
    step[1:] = change[:-1] + drift[1:]
    matrix = np.zeros((ROWS, 30))
    matrix[:, 0], matrix[:, 10], matrix[:, 20] = a, b, np.cumsum(step)
    per = matrix.shape[1] // len(DOMAINS)
    slices = {key: slice(index * per, (index + 1) * per) for index, key in enumerate(DOMAINS)}
    return Recording(
        x=matrix,
        conditions=tuple("x" for _ in range(ROWS)),
        tags=tuple("" for _ in range(ROWS)),
        times=np.arange(ROWS, dtype=np.float64),
        env=np.zeros((ROWS, 1)),
        env_names=("clock",),
        columns=tuple(f"c{i}" for i in range(matrix.shape[1])),
        slices=slices,
        notes={},
    )


def _report(kind: str, seed: int, of: str):
    return synergy(_recording(kind, seed), DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=seed, of=of)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_product_under_drift_passes_on_the_change(seed: int) -> None:
    report = _report("product", seed, "change")
    assert report.passes, report.as_dict()


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_drift_with_additive_sources_fails(seed: int) -> None:
    report = _report("additive", seed, "change")
    assert not report.passes, report.as_dict()


@pytest.mark.parametrize("seed", SEEDS)
def test_drift_alone_fails(seed: int) -> None:
    assert not _report("drift", seed, "change").passes


def test_the_product_is_lost_in_the_drift_when_read_on_the_level() -> None:
    """Why v2 reads the change: v1's reading misses a real interaction here."""
    assert not any(_report("product", seed, "level").passes for seed in SEEDS)


def test_an_additive_gain_is_not_established_by_one_positive_split() -> None:
    """The defect: information bars cleared, a single-split gain at zero.

    An additive target gave a positive single-split gain on some seeds, a few
    millionths either side of zero, and the fold lower bound is what refused
    it. Since ed76a8273 the fit takes the most-shrunk candidate within one
    standard error of the best, so an interaction that is not measurably
    better is switched off and the gain reads exactly zero on every seed. The
    lower bound still has to hold whatever the point estimate does.
    """
    reports = [_report("additive", seed, "change") for seed in range(1, 21)]
    assert all(report.interaction_lower_bound <= 0.0 for report in reports)
    assert all(report.interaction_gain <= 0.0 for report in reports)
