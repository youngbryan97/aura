"""A detected fold cannot certify a negative pooled sensitivity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.subject.closure import ClosureReport, _resolution


@pytest.mark.parametrize("resolution", [-0.0013518023886530315, 0.0, math.nan, math.inf, -math.inf])
def test_an_invalid_resolution_cannot_close(resolution):
    report = ClosureReport(
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2, floor_high=.01,
        leak_lower=-.001, leak_upper=.001, smallest_leak_resolved=resolution,
        margin=.02/1.02, folds=({"leak_upper": .001, "open": False},)*3,
    )
    assert report.verdict == "UNRESOLVED"
    assert "finite positive" in report.as_dict()["why_unresolved"]


def test_a_positive_resolution_below_the_margin_can_close():
    report = ClosureReport(
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2, floor_high=.01,
        leak_lower=-.001, leak_upper=.001, smallest_leak_resolved=.005,
        margin=.02/1.02, folds=({"leak_upper": .001, "open": False},)*3,
    )
    assert report.closed


def test_detection_in_one_fold_does_not_make_a_negative_pooled_gain_a_resolution(monkeypatch):
    import core.subject.closure as closure

    def scored(*args, **kwargs):
        return {"folds": [{"open": True}], "leak": -.001, "leak_lower": -.003,
                "leak_upper": .001, "floor_high": .01, "shuffled_leak": 0.0}

    monkeypatch.setattr(closure, "_scored", scored)
    x = np.arange(100, dtype=float).reshape(-1, 1)
    result = _resolution(x, np.zeros((100, 0)), x, x, seed=2, draws=1)
    assert all(rung["detected"] for rung in result["ladder"])
    assert result["smallest_leak_resolved"] is None
    assert not result["detected"]
