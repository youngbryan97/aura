"""Clicks at nothing in particular go first to the middle, then about, then more to where they paid.

LIVE 2026-10-08 games played with the mouse ("click to launch", "use the mouse
to aim") were clicked in one place or not at all.
"""
from __future__ import annotations

import pytest

from core.agency.where_clicks_pay import (
    PLACES,
    WhereClicksPay,
    _click_attribution_invariant,
    _click_attribution_respects_stretch_boundaries,
)


@pytest.mark.unit
def test_the_middle_first_then_everywhere_once():
    pay = WhereClicksPay()
    first = [pay.where(0, 0) for _ in range(4)]
    assert all(0.25 <= x <= 0.75 and 0.25 <= y <= 0.75 for x, y in first)
    rest = {pay.where(0, 0) for _ in range(PLACES * PLACES - 4)}
    assert len(rest) == PLACES * PLACES - 4


@pytest.mark.unit
def test_a_place_that_paid_is_clicked_more():
    pay = WhereClicksPay()
    gains = 0
    paying = None
    for _ in range(PLACES * PLACES):
        at = pay.where(gains, 0)
        if paying is None and at[0] > 0.75:
            paying = at
            gains += 1  # what came of this click
    later = [pay.where(gains, 0) for _ in range(40)]
    assert later.count(paying) > 40 / (PLACES * PLACES)


@pytest.mark.unit
def test_clicks_that_cost_are_given_up():
    pay = WhereClicksPay()
    losses = 0
    for _ in range(4):
        pay.where(0, losses)
        losses += 1
    pay.credit(0, losses)
    assert pay.costing()


@pytest.mark.unit
@pytest.mark.parametrize(("old_counts", "new_counts"), [
    ((9, 2), (0, 0)),
    ((2, 9), (0, 0)),
    ((9, 2), (5, 1)),
    ((2, 9), (1, 5)),
])
def test_a_counter_reset_cannot_reward_or_punish_a_pending_click(old_counts, new_counts):
    pay = WhereClicksPay()
    pay.where(0, 0)
    pay.credit(4, 1)
    pay.where(*old_counts)
    pending = pay.last
    learned = {place: list(values) for place, values in pay.places.items()}

    pay.begin_stretch(*new_counts)
    pay.credit(*new_counts)
    assert pay.places == learned
    assert not pay.costing()

    pay.where(*new_counts)
    current = pay.last
    pay.credit(new_counts[0] + 2, new_counts[1])
    assert pay.places[pending] == learned[pending]
    assert pay.places[current] == [1.0, 2.0]


@pytest.mark.unit
def test_a_new_stretch_keeps_what_places_paid_and_how_often_they_were_tried():
    pay = WhereClicksPay()
    gains = 0
    for _ in range(PLACES * PLACES):
        position = pay.where(gains, 0)
        if position == (0.375, 0.375):
            gains += 4
    pay.credit(gains, 0)
    learned = {place: list(values) for place, values in pay.places.items()}

    pay.begin_stretch()
    assert pay.places == learned
    assert pay.where(0, 0) == (0.375, 0.375)
    assert pay.places[(1, 1)] == [2.0, 4.0]


@pytest.mark.unit
def test_click_attribution_boundary_invariant_measures_the_reset():
    assert _click_attribution_respects_stretch_boundaries()
    assert _click_attribution_invariant() == ()
