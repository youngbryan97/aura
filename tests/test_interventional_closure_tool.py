"""The interventional closure tool's own rules: what a swap carries, and what the verdict says."""
from __future__ import annotations

import pytest

from tools.run_interventional_closure import aligned, donor_offsets, periods_in_phase, verdict_of

pytestmark = pytest.mark.unit


def test_a_swap_carries_state_and_leaves_the_receivers_instants():
    donor = {"level": 0.9, "last_update": 1_790_000_100.0, "inner": (("nested",), {"seen_at": 1_790_000_200.0, "n": 4})}
    receiver = {"level": 0.1, "last_update": 1_790_000_000.0, "inner": (("nested",), {"seen_at": 1_790_000_050.0, "n": 2})}
    swapped = aligned(donor, receiver)
    assert swapped["level"] == 0.9
    assert swapped["last_update"] == 1_790_000_000.0
    assert swapped["inner"][1]["n"] == 4
    assert swapped["inner"][1]["seen_at"] == 1_790_000_050.0


def test_a_monotonic_reading_is_the_receivers_too():
    assert aligned({"last_update_monotonic": 900.0}, {"last_update_monotonic": 100.0}) == {"last_update_monotonic": 100.0}


def test_the_verdict_names_residue_as_unresolved_never_closed():
    assert verdict_of([], [], decided=True) == "CLOSED"
    assert verdict_of(["belief_authority"], [], decided=True) == "OPEN"
    assert verdict_of([], [{"after": "x", "anchor": 0, "drift": 0.1}], decided=True) == "UNRESOLVED"
    assert verdict_of([], [], decided=False) == "UNRESOLVED"


def test_donors_one_anchor_and_half_the_bank_away():
    assert donor_offsets("1,half", 8) == [1, 4]
    assert donor_offsets("half", 6) == [3]
    assert donor_offsets("0,8,1", 8) == [1]


def test_no_cycle_shorter_than_the_spacing_hides_behind_the_donors():
    """Seed 23's one donor sat 48 turns away, in phase with every cycle of 2, 3, 4, 6, 8 or 12 turns.

    Six messages are 48 turns when a message comes every eighth turn, which is how
    the discourse tracker's cycle hid there. Anchors 13 turns apart with donors one
    anchor and half the bank away leave no cycle up to 12 turns in phase.
    """
    assert periods_in_phase([3], 16, 12) == [2, 3, 4, 6, 8, 12]
    assert periods_in_phase(donor_offsets("1,half", 8), 13, 12) == []
