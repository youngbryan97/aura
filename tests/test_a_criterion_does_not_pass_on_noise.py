"""A point estimate whose own interval contains zero has established nothing.

run_019 read `partition_irreducibility` at 0.0622 against a bar of 0.05 and
recorded it as a pass. Its standard error was 0.0595, its lower bound was
-0.0544, and the five cross-fitted readings it was the mean of ran from -0.118
to +0.249. A fresh estimate on the same saved recording came out at 0.0026 and
picked a different cheapest cut.

So the criterion was deciding on the sign of a number the estimator could not
resolve. What finite measurement can establish is a lower bound above the
instrument floor, and that is now what the criterion asks for.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _verdict(phi: dict) -> dict:
    from core.subject.battery import assemble

    report = assemble({
        "phi": phi,
        "graph": {},
        "nulls": {},
        "differentiation": {},
        "intrinsic": {},
        "closure": {},
        "perturbation": {},
        "synergy": {},
        "metastability": {},
        "global_access": {},
        "agency": {},
        "timescale": {},
        "lesion": {},
        "per_condition": {},
        "recording": {},
    })
    return {item.key: item for item in report.criteria}


def test_a_wide_estimate_above_the_bar_does_not_pass() -> None:
    """run_019's actual numbers."""
    rows = _verdict({
        "phi_do": 0.062173,
        "lower_bound": -0.054406,
        "standard_error": 0.059479,
        "held_out": [-0.117785, 0.249243, 0.021927, 0.054177, 0.103304],
    })
    assert rows["partition_irreducibility"].passed is False


def test_a_tight_estimate_above_the_bar_passes() -> None:
    rows = _verdict({
        "phi_do": 0.09,
        "lower_bound": 0.071,
        "standard_error": 0.01,
        "held_out": [0.08, 0.09, 0.1, 0.09, 0.09],
    })
    assert rows["partition_irreducibility"].passed is True


def test_a_tight_estimate_below_the_bar_still_fails() -> None:
    rows = _verdict({
        "phi_do": 0.02, "lower_bound": 0.018, "standard_error": 0.001, "held_out": [0.02],
    })
    assert rows["partition_irreducibility"].passed is False


def test_a_missing_bound_is_not_a_pass() -> None:
    """Evidence that was never measured is not evidence that cleared a bar."""
    rows = _verdict({"phi_do": 0.9})
    assert rows["partition_irreducibility"].passed is False


def test_the_criterion_says_it_is_asking_about_the_bound() -> None:
    rows = _verdict({"phi_do": 0.09, "lower_bound": 0.071})
    assert "lower bound" in str(rows["partition_irreducibility"].bar)
