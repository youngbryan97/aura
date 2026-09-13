"""Two instruments on the same organism, and whether they agree about her.

The connectome watches cells fire during a turn and scores phase stations
against its own rotations. The subject battery perturbs a state domain and
scores the others against its own null suite. They share no code, no recording
and no null, and six of the things they name are the same thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.connectome.corroboration import (
    CORRESPONDENCE,
    FALSIFIED_BELOW,
    Agreement,
    corroborate,
    corroborate_from_disk,
    read_domain_edges,
)

STATIONS = Path("artifacts/connectome/turn_user/all_station_pairs.json")
RECORD = Path("artifacts/connectome/corroboration.json")


def _newest_edges() -> Path | None:
    runs = sorted(Path("artifacts/subject_core").glob("run_*/edges.csv"))
    return runs[-1] if runs else None


def test_the_correspondence_says_what_has_no_counterpart():
    """A domain mapped to nothing is the honest half of the table."""
    without = [domain for domain, station in CORRESPONDENCE.items() if not station]
    assert set(without) == {"P", "M", "W", "N"}
    named = {station for station in CORRESPONDENCE.values() if station}
    assert named == {
        "interoception",
        "affect",
        "workspace",
        "higher_order",
        "self_model",
        "planning",
    }


def test_no_domain_claims_two_stations():
    stations = [station for station in CORRESPONDENCE.values() if station]
    assert len(stations) == len(set(stations))


def test_a_pair_neither_instrument_measured_is_not_scored():
    """Scoring a pair one of them never looked at would score a coincidence."""
    report = corroborate(
        {("affect", "workspace"): (True, 0.5)},
        {("A", "G"): (True, 1.0, 0.001), ("P", "M"): (True, 1.0, 0.001)},
    )
    assert len(report.pairs) == 1
    assert report.pairs[0].domain_link == "A -> G"


def test_a_pair_within_one_station_is_not_scored():
    report = corroborate(
        {("affect", "affect"): (True, 0.5)},
        {("A", "A"): (True, 1.0, 0.001)},
    )
    assert report.pairs == []


def test_agreement_is_about_whether_influence_is_there():
    both = Agreement("a -> b", "A -> B", True, 0.9, True, 1.0, 0.001)
    neither = Agreement("a -> b", "A -> B", False, 0.0, False, 0.0, 1.0)
    split = Agreement("a -> b", "A -> B", True, 0.9, False, 0.0, 1.0)
    assert both.agree and neither.agree and not split.agree


def test_an_empty_comparison_does_not_hold():
    """Nothing compared is not agreement."""
    assert not corroborate({}, {}).holds


def test_the_battery_threshold_is_the_batterys_own():
    edges = read_domain_edges(_newest_edges() or STATIONS, q_ceiling=0.05)
    carrying = [key for key, (carries, _e, _q) in edges.items() if carries]
    assert carrying, "no battery edge survived its own correction"


# ── the recorded comparison ────────────────────────────────────────────────


def test_both_recordings_are_on_disk():
    assert STATIONS.exists(), "the connectome's station pairs were never recorded"
    assert _newest_edges() is not None, "the battery's edges were never recorded"


def test_the_two_instruments_agree_more_than_they_differ():
    report = corroborate_from_disk(STATIONS, _newest_edges())
    assert report.pairs, "the two instruments have no pair in common"
    assert report.rate >= FALSIFIED_BELOW, (
        f"only {report.agreed} of {len(report.pairs)} pairs agree; two instruments "
        "that disagree more than they agree cannot both be measuring what they are "
        "named after"
    )


def test_the_recorded_comparison_says_what_it_compared():
    if not RECORD.exists():
        pytest.skip("the comparison has not been recorded yet")
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    assert payload["compared"] >= 20
    assert payload["holds"] is True
    assert payload["unmatched_stations"] == ["action"]
    assert set(payload["unmatched_domains"]) == {"M", "N", "P", "W"}


def test_the_disagreements_are_the_connectomes_weakest_pairs():
    """Where they differ, it is the sensitivity floor rather than a contradiction.

    Stated as a claim that can fail: if a pair with a large connectome gain
    ever falls out of agreement, that is a contradiction between two
    instruments and not a matter of one being quieter than the other.
    """
    report = corroborate_from_disk(STATIONS, _newest_edges())
    loud = [
        pair
        for pair in report.pairs
        if not pair.agree and pair.connectome_gain >= 0.05
    ]
    assert not loud, [pair.as_json() for pair in loud]
