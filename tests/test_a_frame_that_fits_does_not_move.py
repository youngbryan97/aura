"""The lattice moving under her on the turn she learns something.

The places a lattice is built from are gathered over acts, so the set grows as
she watches. Rebuilt from the larger set, every line moves — a line is the
mean of the places on it, and a tile centre read to the nearest hundredth of
the window lands a hundredth either way between one glance and the next. The
frame then changes on the turn she learns something, and the two readings on
either side of that turn are in different frames however alike they look.

Which is the whole point of holding one. A lattice is not permanent, but what
replaces it is evidence that the thing has changed shape — not another place
belonging to the shape it already has.
"""

from __future__ import annotations

from core.perception.the_lattice_she_holds import TheLatticeSheHolds


def _four_by_four() -> TheLatticeSheHolds:
    lattice = TheLatticeSheHolds()
    # Offered until it settles. A set still growing is not evidence yet, so
    # the first offer only records it and the second builds from it.
    places = [(x, y) for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)]
    lattice.built_from(places, acts=8)
    lattice.built_from(places, acts=8)
    assert lattice.held
    return lattice


def test_a_place_that_already_fits_leaves_the_lines_where_they_are():
    lattice = _four_by_four()
    was = (lattice.down_at, lattice.across_at)
    # The same board seen again, read a hundredth to one side.
    assert not lattice.built_from(
        [(x, y) for y in (19, 41, 60, 81) for x in (21, 40, 59, 80)], acts=9
    )
    assert (lattice.down_at, lattice.across_at) == was


def test_more_of_the_same_board_does_not_move_the_frame():
    """She keeps finding places belonging to a shape she already holds."""
    lattice = TheLatticeSheHolds()
    lattice.built_from([(20, 20), (40, 20), (20, 40), (40, 40)], acts=2)
    was = (lattice.down_at, lattice.across_at)
    lattice.built_from([(20, 20), (40, 20), (20, 40), (40, 40), (21, 19)], acts=3)
    assert (lattice.down_at, lattice.across_at) == was


def test_places_that_do_not_fit_are_still_evidence_of_a_new_shape():
    """A frame that can never change is not a frame, it is an assumption."""
    lattice = _four_by_four()
    was = (lattice.down_at, lattice.across_at)
    # A window resized, or a different thing on the screen: the places are
    # nowhere near the lines she is holding.
    moved = [(x, y) for y in (5, 15, 25) for x in (5, 15, 25)]
    lattice.built_from(moved, acts=20)
    changed = lattice.built_from(moved, acts=20)
    assert changed
    assert (lattice.down_at, lattice.across_at) != was
    assert (lattice.rows, lattice.columns) == (3, 3)


def test_more_lines_than_she_holds_is_a_bigger_view_of_the_same_thing():
    """A frame that could only stay still could never grow into the thing.

    Early on she has seen two of a board's places. The gap between them is the
    whole board, so every later place lands inside it and fits — and a frame
    that declined to move on that would be two by two for the rest of the run.
    """
    lattice = TheLatticeSheHolds()
    corners = [(20, 20), (80, 20), (20, 80), (80, 80)]
    lattice.built_from(corners, acts=2)
    lattice.built_from(corners, acts=2)
    assert (lattice.rows, lattice.columns) == (2, 2)
    wider = [(x, y) for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)]
    lattice.built_from(wider, acts=6)
    grew = lattice.built_from(wider, acts=6)
    assert grew
    assert (lattice.rows, lattice.columns) == (4, 4)


def test_the_places_are_still_written_down_when_the_frame_does_not_move():
    """Given the same places again this must do nothing, twice as well."""
    lattice = _four_by_four()
    more = [(x, y) for y in (19, 41, 60, 81) for x in (21, 40, 59, 80)]
    assert not lattice.built_from(more, acts=9)
    assert lattice.from_acts == 9
    assert not lattice.built_from(more, acts=10)


def test_a_lattice_with_nothing_held_still_builds():
    lattice = TheLatticeSheHolds()
    lattice.built_from([(20, 20), (40, 20), (20, 40), (40, 40)], acts=2)
    lattice.built_from([(20, 20), (40, 20), (20, 40), (40, 40)], acts=2)
    assert lattice.held
    assert lattice.held
