"""Her activity held against the published statistics of human cortex.

These test the instrument, not the mesh. A change that makes the mesh match
cortex better should move the numbers in `artifacts/connectome/human_dynamics.json`;
a change that makes the comparison easier to pass should fail here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.connectome.criticality import (
    extract_avalanches,
    extract_avalanches_per_unit,
)
from core.connectome.human_dynamics import (
    HUMAN_STATISTICS,
    compare_to_human_cortex,
)


def test_every_statistic_carries_its_source_and_its_falsifier():
    """A target with no way to be refused is a target that always passes."""
    for statistic in HUMAN_STATISTICS:
        assert statistic.source.strip()
        assert statistic.recorded_in.strip()
        assert statistic.falsified_by.strip()
        assert statistic.tolerance > 0.0


def test_the_published_exponents_are_the_ones_that_were_published():
    targets = {statistic.name: statistic.value for statistic in HUMAN_STATISTICS}
    assert targets["avalanche_size_exponent"] == 1.5
    assert targets["avalanche_duration_exponent"] == 2.0
    assert targets["branching_parameter"] == 0.98


def test_the_crackling_relation_is_scored_against_the_system_itself():
    """Cortex's 2.0 falls out of its two exponents; it is not a third measurement."""
    crackling = next(
        statistic for statistic in HUMAN_STATISTICS if statistic.name == "crackling_relation"
    )
    assert crackling.value == 0.0
    assert "own two exponents" in crackling.what_it_is


def test_a_dead_trace_holds_nothing():
    report = compare_to_human_cortex([0.0] * 500)
    assert report["held"] == 0


def test_a_trace_with_no_cascades_says_so_rather_than_scoring():
    report = compare_to_human_cortex(list(np.ones(400)))
    for entry in report["statistics"]:
        assert not entry["holds"]


# ── The two extractors ─────────────────────────────────────────────────────


def test_size_is_measured_in_units_of_the_threshold():
    """A trace at a ten-thousandth gave every cascade a size of exactly 1."""
    small = [0.0001, 0.0009, 0.0008, 0.0001, 0.0001, 0.0007, 0.0001]
    large = [value * 10_000 for value in small]
    assert extract_avalanches(small).sizes == extract_avalanches(large).sizes


def test_a_flat_trace_produces_no_cascades():
    assert extract_avalanches([0.5] * 50).sizes == []


def test_a_per_unit_cascade_is_counted_in_unit_bins():
    """Beggs and Plenz's definition: how many electrodes fired, summed over the run."""
    raster = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    avalanches = extract_avalanches_per_unit(raster, percentile=0.0)
    assert avalanches.sizes == [4, 1]
    assert avalanches.durations == [2, 1]


def test_a_per_unit_raster_with_nothing_in_it():
    assert extract_avalanches_per_unit(np.zeros((10, 4)), percentile=0.0).sizes == []


def test_a_one_dimensional_trace_is_refused_by_the_per_unit_extractor():
    assert extract_avalanches_per_unit(np.zeros(10), percentile=0.0).sizes == []


# ── The recorded result ────────────────────────────────────────────────────


def test_the_recorded_comparison_is_on_disk_and_says_what_it_measured():
    """If this file goes, the claim that the comparison was ever run goes too."""
    path = Path("artifacts/connectome/human_dynamics.json")
    assert path.exists(), "the human-cortex comparison has never been recorded"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["of"] == len(HUMAN_STATISTICS)
    assert report["avalanches"]["count"] > 100, "too few cascades to have measured anything"
    assert report["size_fit"]["tail_n"] >= 32, "the size fit was never usable"
    assert report["size_fit"]["ks"] < 0.2
    for entry in report["statistics"]:
        assert entry["source"].strip()


def test_the_recorded_result_is_not_quietly_a_pass():
    """The mesh does not match cortex's exponents, and the record has to show it."""
    report = json.loads(
        Path("artifacts/connectome/human_dynamics.json").read_text(encoding="utf-8")
    )
    by_name = {entry["name"]: entry for entry in report["statistics"]}
    assert by_name["avalanche_size_exponent"]["hers"] > 2.0, (
        "her size exponent has moved into cortex's range; update this test and say why"
    )


def test_a_target_cannot_be_widened_without_this_failing():
    """Tolerances are part of the claim. Loosening one is a change to the result."""
    tolerances = {statistic.name: statistic.tolerance for statistic in HUMAN_STATISTICS}
    assert tolerances == pytest.approx(
        {
            "avalanche_size_exponent": 0.3,
            "avalanche_duration_exponent": 0.4,
            "crackling_relation": 0.2,
            "branching_parameter": 0.05,
        }
    )
