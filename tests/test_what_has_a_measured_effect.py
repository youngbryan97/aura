"""How many named faculties have a measured downstream effect. Today: none.

An external review wrote the distinction this file exists for: a channel wired
to a consumer is not a measured downstream effect. Aura has the framework —
treatment against null, a lesion registry, an influence ledger with three
verdicts — and does not yet have coverage.

Publishing that number is the difference between having a standard and meeting
one, and the number is what the ratchet moves.
"""
from __future__ import annotations

import json

import pytest

import core.brain.cognitive_engine  # noqa: F401 — importing registers its lesions
from core.verify.what_has_a_measured_effect import (
    BASELINE,
    how_much_is_measured,
    the_baseline,
    what_is_still_unmeasured,
)


@pytest.fixture(scope="module")
def measured():
    return how_much_is_measured()


# ------------------------------------------------------------ what it says


def test_the_count_covers_every_declared_service(measured):
    assert measured["declared_services"] > 50
    assert (
        measured["lesionable"] + measured["not_lesionable"]
        == measured["declared_services"]
    )


def test_a_faculty_that_cannot_be_lesioned_is_not_counted_as_unmeasured(measured):
    """It cannot be measured at all, and that is a bigger number."""
    assert measured["not_lesionable"] > measured["unmeasured"]


def test_the_declared_count_is_the_same_however_much_was_imported():
    """Read from the source, so two processes agree."""
    from core.verify.what_has_a_measured_effect import the_declared_lesions

    assert the_declared_lesions() == the_declared_lesions()
    assert "channel" not in the_declared_lesions(), (
        "a wrapper passing its own argument through is not a channel"
    )


def test_measured_is_influential_plus_inert_and_nothing_else(measured):
    """Inert is a measurement. The faculty ran and did not change the output."""
    assert measured["measured"] == len(measured["influential"]) + len(
        measured["inert"]
    )


def test_lesionable_is_measured_plus_unmeasured(measured):
    assert measured["lesionable"] == measured["measured"] + measured["unmeasured"]


def test_the_result_carries_the_sentence_the_counts_are_for(measured):
    assert "not a measured downstream effect" in measured["what_this_means"]


# --------------------------------------------------------------- the truth


def test_nothing_is_measured_yet(measured):
    """The honest state today, pinned so improving it is visible."""
    assert measured["measured"] == 0
    assert measured["unmeasured"] == measured["lesionable"]


def test_the_live_mind_controls_are_lesionable_and_unmeasured(measured):
    """The mechanism the review called demonstrated internal-state actuation.

    Lesionable is what makes the claim falsifiable. Unmeasured is what it is.
    """
    from core.verify.what_has_a_measured_effect import the_declared_lesions

    assert "LIVE_MIND_GENERATION_CONTROLS" in the_declared_lesions()
    assert measured["measured"] == 0


# -------------------------------------------------------------- the ratchet


def test_the_baseline_names_the_channels_and_not_only_the_count():
    """A count with no names cannot say which one stopped being measured."""
    held = the_baseline()
    assert isinstance(held["still_unmeasured"], list)
    assert held["measured"] == len(held["measured_channels"])
    assert BASELINE.exists()


def test_the_number_of_declared_lesions_only_goes_up(measured):
    """Counted from the source, not the live registry.

    The first version counted registered channels, which said 7 in one process
    and 1 in another depending on what had been imported. A ratchet on an
    import-order-dependent number is worse than no ratchet.
    """
    held = the_baseline()
    assert measured["declared_lesions"] >= held["declared_lesions"], (
        "a faculty stopped being lesionable, which makes it unfalsifiable"
    )


def test_the_number_of_measured_faculties_only_goes_up(measured):
    assert measured["measured"] >= the_baseline()["measured"]


def test_no_faculty_that_was_measured_becomes_unmeasured(measured):
    """A newly lesionable channel starts unmeasured, and that is progress.

    The first version of this test compared the unmeasured SETS and failed
    the moment a new channel became lesionable — which is the improvement it
    was supposed to protect. What regresses is a channel that had a verdict
    losing it.
    """
    had_one = set(the_baseline().get("measured_channels") or ())
    still_has = set(measured["influential"]) | set(measured["inert"])
    assert not (had_one - still_has), (
        f"lost the verdict for: {sorted(had_one - still_has)}"
    )


def test_a_channel_can_only_leave_the_unmeasured_list_by_being_measured():
    """The other direction: unregistering it would also empty the list."""
    was = set(the_baseline()["still_unmeasured"])
    now = set(what_is_still_unmeasured())
    from core.verify.lesion_registry import get_lesion_registry

    lesionable = set(get_lesion_registry().channels())
    left = was - now
    assert left <= lesionable, (
        f"stopped being lesionable rather than being measured: "
        f"{sorted(left - lesionable)}"
    )


def test_the_baseline_is_json_a_person_can_read():
    held = json.loads(BASELINE.read_text("utf-8"))
    assert "only go up" in held["note"]


def test_the_count_is_in_the_health_report():
    """From the baseline, not worked out: the scan takes eight seconds."""
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["what_has_a_measured_effect"]
    assert set(block) >= {"declared_lesions", "measured", "unmeasured"}
    assert "not a measured downstream effect" in block["what_this_means"]
