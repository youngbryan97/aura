"""A weak cut named two blocks and said nothing about which one was weak.

`phi_do` reports the cheapest bipartition and the loss on either side of it
summed together. That tells you the system can be split there. It does not tell
you which half is the one predicting itself too independently, and that is the
half an engineering change has to reach.

So the cheapest cut is now reported one side at a time: what that side's own
model loses, what the intact model loses on the same targets, and the gain
between them. A side whose gain is near zero sees nothing it needs from the
other half, and strengthening the channel into it is what raises the cut.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report():
    from core.subject.irreducibility import phi_do
    from core.subject.nulls import architecture, toy_recording

    return phi_do(toy_recording(architecture("recurrent", seed=7), steps=700, seed=7))


def test_both_sides_of_the_cheapest_cut_are_reported(report) -> None:
    sides = report.as_dict()["sides"]
    assert len(sides) == 2
    left, right = report.best_cut
    assert set(sides) == {"".join(left), "".join(right)}


def test_each_side_carries_what_it_lost_and_what_the_intact_model_lost(report) -> None:
    for row in report.as_dict()["sides"].values():
        assert {"own_loss", "intact_loss", "gain", "width"} <= set(row)
        assert row["own_loss"] >= 0.0
        assert row["intact_loss"] >= 0.0


def test_the_gain_is_what_the_intact_model_recovers_on_that_side(report) -> None:
    for row in report.as_dict()["sides"].values():
        if row["own_loss"] <= 0.0:
            continue
        expected = (row["own_loss"] - row["intact_loss"]) / row["own_loss"]
        # The report rounds to six places, and the inputs it rounds are what
        # this recomputes from, so the tolerance is the rounding.
        assert row["gain"] == pytest.approx(expected, abs=1e-5)


def test_a_side_that_needs_the_other_half_shows_a_gain(report) -> None:
    """On the recurrent reference at least one side is genuinely dependent."""
    gains = [row["gain"] for row in report.as_dict()["sides"].values()]
    assert max(gains) > 0.05, (
        "neither side of the cheapest cut gained anything from the other, "
        "which would mean the reference architecture is two systems"
    )


def test_an_independent_system_has_nothing_to_gain_on_either_side() -> None:
    """Ten domains with no coupling: both sides predict themselves perfectly well."""
    from core.subject.irreducibility import phi_do
    from core.subject.nulls import architecture, toy_recording

    loose = phi_do(toy_recording(architecture("independent", seed=7), steps=700, seed=7))
    gains = [row["gain"] for row in loose.as_dict()["sides"].values()]
    assert max(gains) < 0.05, f"an uncoupled system reported a gain: {gains}"
