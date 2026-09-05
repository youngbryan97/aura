"""Independent reproducibility has two halves and only one is code.

The half that is code: the freeze, the environments, the receipts, and enough
of the protocol that a run elsewhere is the same run. The half that is not: a
team that builds its own task families after this freeze, never sees the ones
here, and reports what it found.

A bundle that carried the first and implied the second would be the more
comfortable object and the less honest one.
"""

from __future__ import annotations

import json

from tools.agi_gauntlet.bundle import (
    WHAT_A_HUMAN_WOULD_SCORE,
    the_bundle,
    write_the_bundle,
)


def test_it_carries_the_freeze_the_run_would_have_to_match():
    bundle = the_bundle()
    assert bundle["freeze"]["commit"]
    assert bundle["freeze"]["source_digest"]
    assert "seed" in bundle["freeze"]


def test_no_human_baseline_is_filled_in_with_a_guess():
    """A gate whose pass condition is "roughly competent-human" and whose
    harness has never seen a human is comparing a number to an assumption."""

    for gate, slot in WHAT_A_HUMAN_WOULD_SCORE.items():
        assert slot["measured"] is None, f"{gate} has a baseline nobody measured"
        assert slot["needs"], f"{gate} does not say what would fill it"


def test_every_gate_that_mentions_a_person_has_a_slot():
    from tools.agi_gauntlet.gates import THE_GATES

    for gate in THE_GATES:
        said = f"{gate.passes_when} {gate.control}".lower()
        if "human" in said or "person" in said or "people" in said:
            assert gate.name in WHAT_A_HUMAN_WOULD_SCORE, (
                f"{gate.name} is judged against a person and has no baseline slot"
            )


def test_it_says_what_does_not_reproduce():
    """A bundle that claimed everything reproduces would be wrong about the
    one search bounded by a clock."""

    bundle = the_bundle()
    assert bundle["what_does_not"]
    assert any("clock" in one for one in bundle["what_does_not"])


def test_it_says_what_is_still_needed_rather_than_implying_it_is_done():
    bundle = the_bundle()
    said = " ".join(bundle["still_needed"]).lower()
    assert "after this freeze" in said
    assert "plain" in said and "scaffold" in said
    assert "second group" in said


def test_every_lesion_is_listed_with_whether_it_runs():
    bundle = the_bundle()
    assert bundle["lesions"]
    assert any(not one["runs_here"] for one in bundle["lesions"]), (
        "the comparison against the same weights in a plain scaffold is the "
        "one that cannot run here, and it must be listed as such"
    )
    for one in bundle["lesions"]:
        assert one["runs_here"] or one["needs"]


def test_it_writes(tmp_path):
    where = write_the_bundle(tmp_path)
    assert where.exists()
    assert json.loads(where.read_text())["gates"]
