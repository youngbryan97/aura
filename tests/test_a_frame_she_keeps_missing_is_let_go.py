"""A frame she keeps judging not in front of her is let go, even when each look still fits a piece of it.

LIVE 2026-10-08 she held a two-place frame from a game's menu; every look
after put something into it (so its misses were counted back to nought), and
every look was judged not the thing (so nothing was pressed). She pressed
nothing for four minutes.
"""
from __future__ import annotations

import pytest

from core.perception.the_lattice_she_holds import TheLatticeSheHolds


@pytest.mark.unit
def test_misses_by_looking_are_counted_apart_from_misses_by_fitting():
    lattice = TheLatticeSheHolds(down_at=(0.5,), across_at=(0.3, 0.7))
    for _look in range(TheLatticeSheHolds.CHANGED_AFTER):
        lattice.would_not_fit = 0  # each look put something into the frame
        lattice.not_here_for += 1
        assert not lattice.has_changed()
    lattice.not_here_for += 1
    assert lattice.has_changed()
