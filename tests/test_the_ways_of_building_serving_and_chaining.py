"""Building up, giving each what it asks for, and putting parts on in a chain: three ways a place may ask to be played.

The design map's games ask for all three (stack blocks of a colour, build steps from boxes; put tubes from where a
thing drops to where it goes; give each what it asks for). Nothing here is any of them: pieces and a pile, a
customer and what she asks for, parts and the places they go.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _thing(x, y, *, w=20.0, h=10.0, vx=0.0, vy=0.0):
    return SimpleNamespace(x=x, y=y, w=w, h=h, vx=vx, vy=vy)


def test_a_piece_is_dropped_square_over_the_top_of_what_is_built_and_not_elsewhere():
    from core.agency.building_up import drop_now, the_top

    base, second = _thing(100, 200, w=60), _thing(104, 190, w=30)
    piece = _thing(60, 40, w=30, vx=80.0)
    assert the_top([base, second, piece], piece) is second
    assert not drop_now([base, second, piece])                         # far to the left of the top
    piece.x = 100.0
    assert drop_now([base, second, piece])                             # square over it
    piece.x = 90.0
    assert drop_now([base, second, piece], ahead_s=0.15)               # it will be, when the press lands
    assert not drop_now([base, second])                                # nothing swinging: nothing to drop


def test_where_a_place_is_of_pieces_dropped_to_build_up_play_knows_it():
    from core.agency.building_up import builds_up

    assert builds_up("Stack the blocks as high as you can; don't let the tower fall.")
    assert not builds_up("Use the arrow keys to move and space to jump.")


def test_parts_are_put_on_from_where_the_last_one_that_worked_went_nearest_first_and_down_where_things_fall():
    from core.agency.putting_things_in_place import a_carry_of, in_chain_order

    where = {"chute": (0.2, 0.1), "near": (0.25, 0.3), "far": (0.8, 0.9), "beside": (0.3, 0.12)}
    offered = [a_carry_of("tube", place) for place in ("far", "near", "beside")]
    assert in_chain_order(offered, where, None) == offered                              # nothing built yet: as it was
    ordered = in_chain_order(offered, where, where["chute"])
    assert ordered[0] == a_carry_of("tube", "beside") and ordered[-1] == a_carry_of("tube", "far")
    falling = in_chain_order(offered, where, where["chute"], falls=True)
    assert falling.index(a_carry_of("tube", "near")) < falling.index(a_carry_of("tube", "far"))


def test_a_carry_that_worked_is_where_the_chain_goes_on_from():
    from core.agency.putting_things_in_place import a_carry_of
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

    here = WhatWorksHere()
    moves = (a_click_on("tube"), a_click_on("the shape at 20% across, 10% down"), a_click_on("the shape at 80% across, 90% down"),
             a_click_on("the shape at 25% across, 30% down"))
    where = {moves[1]: (0.2, 0.1), moves[2]: (0.8, 0.9), moves[3]: (0.25, 0.3)}
    here.asked_for_by("Drag each tube so the gumball rolls from the chute to the bucket: a chain reaction.", moves)
    here.looked_at(moves, where=where)
    here.looked_at(moves, where=where)
    assert "chains" in here.mechanics_said
    here.carried(a_carry_of("tube", "the shape at 20% across, 10% down"), changed=True)
    carries = here.carries(here.on_screen)
    assert carries and carries[0] == a_carry_of("tube", "the shape at 25% across, 30% down")


def test_each_is_given_what_it_asks_for_first():
    from core.agency.taking_and_using import a_use_of
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

    here = WhatWorksHere()
    red, blue = (0.9, 0.1, 0.1) * 16, (0.1, 0.1, 0.9) * 16
    moves = (a_click_on("burger"), a_click_on("customer one"), a_click_on("customer two"))
    here.asked_for_by("Serve each customer the order they ask for before they leave.", moves)
    here.looked_at(moves, looks={moves[0]: red, moves[1]: blue, moves[2]: red})
    here.looked_at(moves, looks={moves[0]: red, moves[1]: blue, moves[2]: red})
    here.taken["burger"] = "burger"
    assert "serving" in here.mechanics_said
    uses = here.uses(here.on_screen)
    assert uses[0] == a_use_of("burger", "customer two")


def test_writing_there_to_be_read_is_never_taken_or_used():
    """LIVE 2026-10-10 "LEVEL" was taken off a game's bar and used on "HITS LEFT"."""
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

    here = WhatWorksHere()
    moves = (a_click_on("LEVEL"), a_click_on("HITS LEFT"), a_click_on("key"))
    here.taken["LEVEL"] = "LEVEL"
    assert not here.uses(moves)
