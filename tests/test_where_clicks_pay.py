"""Clicks at nothing in particular go first to the middle, then about, then more to where they paid.

LIVE 2026-10-08 games played with the mouse ("click to launch", "use the mouse
to aim") were clicked in one place or not at all.
"""
from __future__ import annotations

import pytest

from core.agency.where_clicks_pay import PLACES, WhereClicksPay


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
