"""A substitute reported without its limits reads as the thing it replaced.

Five of the eighteen gates need evidence this machine cannot produce. The
instruction was to find the closest possible alternative and clear that too,
and the danger is obvious: "we ran something adjacent" becomes "we ran it".

So every row carries what the external measurement would be, whether it ran,
what was run instead, and what the substitute does NOT establish — and the
last of those is the one the gate checks.
"""
from __future__ import annotations

import pytest

from core.verify.what_was_measured_outside import (
    THE_OUTSIDE_MEASUREMENTS,
    how_it_stands,
    what_cannot_be_run_here,
    what_is_claimed_without_a_limit,
)


def test_nothing_is_claimed_without_a_limit():
    """The gate. A substitute with no stated limit is a claim in a costume."""
    assert what_is_claimed_without_a_limit() == []


@pytest.mark.parametrize("name", sorted(THE_OUTSIDE_MEASUREMENTS))
def test_every_row_says_what_the_external_measurement_would_be(name):
    row = THE_OUTSIDE_MEASUREMENTS[name]
    assert len(row["would_measure"].split()) >= 5, name


@pytest.mark.parametrize("name", sorted(THE_OUTSIDE_MEASUREMENTS))
def test_every_row_says_what_it_fails_to_show(name):
    assert THE_OUTSIDE_MEASUREMENTS[name]["does_not_show"].strip(), name


def test_a_row_that_ran_reports_a_result():
    for name, row in THE_OUTSIDE_MEASUREMENTS.items():
        if row.get("ran"):
            assert row["result"].strip(), name


def test_a_row_that_did_not_run_gives_a_reason():
    for name, row in THE_OUTSIDE_MEASUREMENTS.items():
        if not row.get("ran"):
            assert row["why_not"].strip(), name


def test_arc_is_the_one_that_ran():
    assert how_it_stands()["ran"] == ["ARC-AGI"]
    assert "1 in 87" in THE_OUTSIDE_MEASUREMENTS["ARC-AGI"]["result"]


def test_the_arc_result_carries_its_own_control():
    """0.955 may not be quoted without 0.011."""
    said = THE_OUTSIDE_MEASUREMENTS["ARC-AGI"]
    assert "19 in 20" in said["result"]
    assert "0.011" in said["does_not_show"]


def test_the_human_row_offers_no_substitute():
    """Inventing one would be the worst row in the table."""
    said = THE_OUTSIDE_MEASUREMENTS["human judges"]
    assert said["instead"] == ""
    assert said["does_not_show"].startswith("anything")


def test_the_repository_row_says_it_is_not_sealed():
    """The defects were found by the same agent that fixed them."""
    said = THE_OUTSIDE_MEASUREMENTS["SWE-bench after the cutoff"]
    assert "sealed" in said["does_not_show"]
    assert "same agent" in said["does_not_show"]


def test_every_named_external_benchmark_has_a_row():
    """The list the review named, so none can quietly drop out."""
    named = {"ARC-AGI", "GAIA", "OSWorld", "hours-long autonomy", "human judges"}
    assert named <= set(THE_OUTSIDE_MEASUREMENTS)
    assert any("SWE" in one for one in THE_OUTSIDE_MEASUREMENTS)


def test_a_row_missing_its_limit_is_caught():
    """The gate has to be able to fail."""
    THE_OUTSIDE_MEASUREMENTS["a row somebody added"] = {
        "would_measure": "something nobody has measured yet at all",
        "ran": False,
        "why_not": "it was never tried",
        "instead": "something adjacent",
        "does_not_show": "",
    }
    try:
        assert any(
            "a row somebody added" in one for one in what_is_claimed_without_a_limit()
        )
    finally:
        del THE_OUTSIDE_MEASUREMENTS["a row somebody added"]


def test_the_reasons_read_as_reasons():
    for one in what_cannot_be_run_here():
        assert ": " in one
        assert len(one.split(": ", 1)[1].split()) >= 5, one
