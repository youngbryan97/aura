"""A lattice held, and then immediately thrown away by the crop.

The comparison that teaches her how a world moves needs both readings in the
same frame of reference. A lattice is what holds one: worked out once from
where things have been seen to happen, and every reading placed into it, so
the squares are there when they are empty.

The crop is what she has INSTEAD of a lattice — the largest regular block in a
reading, for when she is not holding a frame yet. Run on a reading that was
just placed into one, it cuts the frame down to whichever block the tiles
happened to make this glance, and it slices the frame's own lines along with
the cells. LIVE 2026-08-31: a correct four-by-four lattice held across forty
acts of the real game, and one comparison out of forty reached the rule.
"""

from __future__ import annotations

from core.perception.the_lattice_she_holds import TheLatticeSheHolds
from core.perception.where_it_responds import what_is_there
from core.skills.screen_pursuit import _in_the_same_grid, _the_thing_she_is_acting_in


def _four_by_four() -> TheLatticeSheHolds:
    """A lattice over the sixteen places of a board."""
    lattice = TheLatticeSheHolds()
    places = [(int(x * 100), int(y * 100)) for y in (0.2, 0.4, 0.6, 0.8) for x in (0.2, 0.4, 0.6, 0.8)]
    assert lattice.built_from(places, acts=8)
    assert (lattice.rows, lattice.columns) == (4, 4)
    return lattice


def _placed(lattice: TheLatticeSheHolds, rows: list[list[str]]):
    """One glance of that board, placed into the lattice she is holding."""
    said = [
        (0.2 + 0.2 * row, 0.2 + 0.2 * column, value)
        for row, line in enumerate(rows)
        for column, value in enumerate(line)
        if value
    ]
    seen = lattice.fit(said)
    assert seen is not None
    return seen


def test_a_reading_placed_into_the_lattice_keeps_its_shape():
    lattice = _four_by_four()
    # A glance where the top row happens to be empty, which happens
    # constantly and is not a smaller board.
    seen = _placed(lattice, [["", "", "", ""], ["2", "4", "", ""], ["", "8", "", ""], ["", "", "", "16"]])
    thing = _the_thing_she_is_acting_in(seen, lattice)
    assert (thing.rows, thing.columns) == (4, 4)


def test_two_sparse_glances_stay_comparable():
    """The whole point: this is the pair that teaches her how the world moves."""
    lattice = _four_by_four()
    before = _the_thing_she_is_acting_in(
        _placed(lattice, [["2", "", "", ""], ["", "", "", ""], ["", "4", "", ""], ["", "", "", ""]]),
        lattice,
    )
    after = _the_thing_she_is_acting_in(
        _placed(lattice, [["", "", "", ""], ["", "", "", ""], ["", "", "", ""], ["2", "4", "", ""]]),
        lattice,
    )
    assert _in_the_same_grid(lattice, before, after), (
        "two glances of one board were not in one frame, so the move between them taught her nothing"
    )


def test_without_a_lattice_the_crop_still_does_its_work():
    """The crop is what she has before she is holding anything."""
    page = what_is_there(
        {
            "ok": True,
            "text": "Score 1234 Best 9999\n2 4 8 2\n16 32 64 4\n2 8 4 2\n4 2 8 16",
            "layout": [],
            "bounds": [],
        },
        None,
    )
    thing = _the_thing_she_is_acting_in(page, TheLatticeSheHolds())
    assert thing.rows <= page.rows and thing.occupied() > 0


def test_a_reading_of_a_different_shape_is_not_forced_into_the_frame():
    """Evidence the thing has changed shape must still be able to arrive."""
    lattice = _four_by_four()
    other = what_is_there({"ok": True, "text": "2 4\n8 16", "layout": [], "bounds": []}, None)
    thing = _the_thing_she_is_acting_in(other, lattice)
    assert (thing.rows, thing.columns) != (lattice.rows, lattice.columns)
