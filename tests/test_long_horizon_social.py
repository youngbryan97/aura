"""Trust that was tested, and beliefs about beliefs with evidence.

A bond that has never been under strain and a bond that broke and was repaired
can sit at the same number and are not the same thing. Any model reporting
only a level loses that, and it is the distinction that matters — it is why a
repaired relationship can end up stronger than one never strained, which a
monotone trust score cannot represent at all.

And the recursive theory-of-mind module here was gutted for fabricating three
nested minds out of one caller-supplied trust value. What replaced it says it
"does not claim recursive beliefs without recursive evidence". Nothing then
produced any. These tests are what that evidence looks like.
"""

from __future__ import annotations

import time

import pytest

from core.social.long_horizon import (
    BREAK_COST,
    IDLE_HALF_LIFE_S,
    KEEP_GAIN,
    Act,
    Discriminates,
    Episode,
    Event,
    second_order,
    standing,
)

DAY = 86400.0
NOW = time.time()
K, B, R = Episode.KEPT, Episode.BROKE, Episode.REPAIRED


def history(spec, spacing=3 * DAY, end=NOW):
    moment = end - len(spec) * spacing
    events = []
    for kind in spec:
        events.append(Event(kind, at=moment))
        moment += spacing
    return events


# ── the distinction a level cannot carry ─────────────────────────────────


def test_a_tested_bond_and_an_untested_one_can_sit_at_the_same_level():
    untested = standing(history([K] * 20), now=NOW)
    tested = standing(history([K] * 10 + [B, R] + [K] * 6), now=NOW)
    assert tested.trust == pytest.approx(untested.trust, abs=0.02)
    assert untested.proved == 0.0
    assert tested.proved > 0.0
    assert tested.stronger_for_it and not untested.stronger_for_it


def test_an_untested_bond_can_never_be_proved_however_long_it_lasts():
    assert standing(history([K] * 200), now=NOW).proved == 0.0


def test_a_repair_that_has_not_been_re_tested_is_not_yet_proved():
    repaired = standing(history([K] * 10 + [B, R]), now=NOW)
    assert repaired.repairs == 1
    assert repaired.proved == 0.0
    assert repaired.stronger_for_it is False


def test_an_unrepaired_break_is_not_stronger_for_it():
    broken = standing(history([K] * 10 + [B] + [K] * 6), now=NOW)
    assert broken.unrepaired == 1
    assert broken.stronger_for_it is False


# ── the asymmetry ────────────────────────────────────────────────────────


def test_a_break_costs_more_than_a_keep_gains():
    """Trust that recovered as fast as it fell would not be worth having."""
    assert BREAK_COST > KEEP_GAIN
    kept = standing(history([K]), now=NOW).trust
    broke = standing(history([B]), now=NOW).trust
    assert (0.5 - broke) > (kept - 0.5)


def test_a_repair_is_not_an_undo():
    broke_then_repaired = standing(history([K] * 5 + [B, R]), now=NOW)
    never_broke = standing(history([K] * 5), now=NOW)
    assert broke_then_repaired.trust < never_broke.trust


def test_repeated_breaking_drives_trust_down():
    assert standing(history([K, B] * 8), now=NOW).trust < 0.1


def test_what_was_at_stake_scales_the_effect():
    trivial = standing([Event(B, at=NOW - DAY, weight=0.1)], now=NOW)
    serious = standing([Event(B, at=NOW - DAY, weight=1.0)], now=NOW)
    assert trivial.trust > serious.trust


# ── time ─────────────────────────────────────────────────────────────────


def test_a_relationship_nothing_happens_in_drifts_back():
    events = history([K] * 20)
    fresh = standing(events, now=NOW)
    later = standing(events, now=NOW + 400 * DAY)
    assert fresh.trust > 0.95
    assert 0.4 < later.trust < 0.7


def test_the_drift_is_slow_enough_to_survive_a_normal_gap():
    events = history([K] * 20)
    assert standing(events, now=NOW + IDLE_HALF_LIFE_S / 10).trust > 0.9


def test_events_are_read_in_the_order_they_happened_not_the_order_given():
    forward = history([K] * 5 + [B])
    shuffled = list(reversed(forward))
    assert standing(forward, now=NOW).trust == pytest.approx(
        standing(shuffled, now=NOW).trust
    )


def test_a_repair_with_nothing_to_repair_does_nothing():
    assert standing(history([K, R]), now=NOW).repairs == 0


def test_no_history_is_neutral():
    assert standing([], now=NOW).trust == pytest.approx(0.5)
    assert standing([], now=NOW).tested is False


# ── beliefs about beliefs ────────────────────────────────────────────────


ACTS = (
    Act("explained the deadline again", Discriminates.THINKS_SHE_DOES_NOT_KNOW, "deadline"),
    Act("said hello", Discriminates.NEITHER, "deadline"),
    Act("skipped the setup step", Discriminates.THINKS_SHE_KNOWS, "setup"),
    Act("skipped it again", Discriminates.THINKS_SHE_KNOWS, "setup"),
)


def test_an_act_that_only_makes_sense_one_way_licenses_a_belief():
    belief = second_order("setup", ACTS)
    assert belief.held and belief.believes_she_knows is True
    assert len(belief.evidence) == 2


def test_explaining_something_again_says_he_thinks_she_missed_it():
    belief = second_order("deadline", ACTS)
    assert belief.held and belief.believes_she_knows is False


def test_a_subject_with_no_discriminating_act_licenses_nothing():
    """Which is most subjects, and saying so is the whole difference."""
    belief = second_order("something_never_mentioned", ACTS)
    assert belief.held is False
    assert belief.confidence == 0.0


def test_an_act_consistent_with_either_belief_contributes_nothing():
    only_neutral = (Act("said hello", Discriminates.NEITHER, "x"),)
    assert second_order("x", only_neutral).held is False


def test_acting_both_ways_supports_neither():
    both = (
        Act("explained it", Discriminates.THINKS_SHE_DOES_NOT_KNOW, "x"),
        Act("skipped it", Discriminates.THINKS_SHE_KNOWS, "x"),
    )
    belief = second_order("x", both)
    assert belief.held is False
    assert belief.evidence, "the conflicting acts are a real observation and are kept"


def test_more_agreeing_acts_raise_confidence_without_reaching_certainty():
    one = second_order("setup", ACTS[2:3])
    two = second_order("setup", ACTS[2:4])
    assert two.confidence > one.confidence < 1.0
    assert two.confidence < 1.0
