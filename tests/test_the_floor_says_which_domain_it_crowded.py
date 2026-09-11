"""Sham against sham is the instrument's own noise, and one number hides it.

A floor that is clean in nine domains and half the effect threshold in the
tenth leaves exactly one domain uninterpretable — a negative result there is
about restoration rather than about coupling — and a pooled mean says none of
that. The floor is reported by target domain, by condition, by source and over
lag, and any domain whose floor reaches a third of the bar is named.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.causal import EDGE_EFFECT, InterventionSet, Trial

pytestmark = pytest.mark.unit

DOMAINS = "PIAGCSMWDN"


def _set(crowd: str | None = None, level: float = 0.15) -> InterventionSet:
    rng = np.random.default_rng(0)
    trials = []
    for source in ("P", "I", "A"):
        for condition in ("idle", "stress"):
            for index in range(3):
                floor = {t: float(abs(rng.normal(scale=0.02))) for t in DOMAINS}
                if crowd:
                    floor[crowd] = level
                trials.append(
                    Trial(
                        source=source, condition=condition, index=index,
                        effect={t: 0.4 for t in DOMAINS}, floor=floor,
                        trace={}, floor_trace={t: [0.01, 0.02, 0.03] for t in DOMAINS},
                        took=True, self_effect=1.0,
                    )
                )
    return InterventionSet(trials=trials)


def test_a_clean_floor_reads_as_clean() -> None:
    report = _set().floor_report()
    assert report["floor_is_clean"] is True
    assert report["crowded_by_restoration_noise"] == []


def test_one_crowded_domain_is_named() -> None:
    report = _set(crowd="G").floor_report()
    assert report["floor_is_clean"] is False
    assert report["crowded_by_restoration_noise"] == ["G"]
    assert report["worst_target"] == "G"


def test_the_pooled_mean_would_have_hidden_it() -> None:
    """Why the breakdown exists: one domain at half the bar barely moves the mean."""
    crowded = _set(crowd="G")
    pooled = float(
        np.mean([v for t in crowded.trials for v in t.floor.values()])
    )
    assert pooled < EDGE_EFFECT / 3.0
    assert crowded.floor_report()["crowded_by_restoration_noise"] == ["G"]


def test_the_floor_is_broken_out_four_ways() -> None:
    report = _set().floor_report()
    assert set(report["by_target"]) == set(DOMAINS)
    assert set(report["by_condition"]) == {"idle", "stress"}
    assert set(report["by_source"]) == {"P", "I", "A"}
    assert report["by_lag"] == [0.01, 0.02, 0.03]


def test_each_breakdown_carries_a_tail_and_not_only_a_mean() -> None:
    """A mean over a skewed floor is the number that hid the problem."""
    for row in _set(crowd="G").floor_report()["by_target"].values():
        assert {"mean", "q95", "max"} == set(row)


def test_the_threshold_it_is_measured_against_is_in_the_report() -> None:
    assert _set().floor_report()["threshold"] == EDGE_EFFECT


def test_no_trials_is_not_a_clean_floor() -> None:
    """Nothing measured is not the same as nothing found."""
    report = InterventionSet(trials=[]).floor_report()
    assert report == {"trials": 0}
    assert "floor_is_clean" not in report
