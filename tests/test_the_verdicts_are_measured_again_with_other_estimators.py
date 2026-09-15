"""The alternate estimators P54.25 reads the battery's verdicts with give known answers.

A second estimator is only evidence if it is right where the answer is known.
These pin the nearest-neighbour mutual information against the closed form for
Gaussians, the nearest-neighbour synergy against the ISC-v2 known answers at the
length it was shown at (a product passes, an additive target does not), and
irreducibility at the battery's own settings against the battery's own reading.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.subject.alternates import irreducibility_under, ksg_mutual_information, ksg_synergy
from core.subject.recording import Recording
from core.subject.state import DOMAINS

pytestmark = pytest.mark.unit

ROWS = 2400


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.9])
def test_nearest_neighbour_information_matches_the_gaussian_closed_form(rho: float) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    y = rho * x + math.sqrt(1.0 - rho * rho) * rng.normal(size=4000)
    exact = -0.5 * math.log(1.0 - rho * rho)
    assert ksg_mutual_information(x, y) == pytest.approx(exact, abs=0.03)


def _recording(kind: str, seed: int) -> Recording:
    """The ISC-v2 known-answer shape: two sources and a drifting target, one row per turn."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=ROWS)
    b = 0.8 * a + 0.6 * rng.normal(size=ROWS)
    drift = 0.5 + 0.3 * rng.normal(size=ROWS)
    noise = 0.3 * rng.normal(size=ROWS)
    change = {"product": (a - b) + a * b + noise, "additive": a + b + noise}[kind]
    step = np.zeros(ROWS)
    step[1:] = change[:-1] + drift[1:]
    matrix = rng.normal(size=(ROWS, 30)) * 0.05
    matrix[:, 0], matrix[:, 9], matrix[:, 18] = a, b, np.cumsum(step)
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


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_nearest_neighbour_synergy_tells_a_product_from_a_sum(seed: int) -> None:
    product = ksg_synergy(_recording("product", seed), DOMAINS[0], DOMAINS[3], DOMAINS[6], draws=40, seed=seed)
    additive = ksg_synergy(_recording("additive", seed), DOMAINS[0], DOMAINS[3], DOMAINS[6], draws=40, seed=seed)
    assert not product.underpowered
    assert product.clears_its_null, product.as_dict()
    assert not additive.clears_its_null, additive.as_dict()


def test_irreducibility_at_the_battery_s_settings_is_the_battery_s_reading() -> None:
    from core.subject.irreducibility import COMPONENTS, FOLDS, LOWER_BOUND_Z, phi_do

    recording = _recording("product", 2)
    battery = phi_do(recording)
    again = irreducibility_under(recording, components=COMPONENTS, folds=FOLDS)
    assert again.phi == pytest.approx(battery.phi)
    assert again.lower_bound == pytest.approx(battery.phi - LOWER_BOUND_Z * battery.standard_error)
