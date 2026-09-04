"""A number split by the reader is still one number.

A reader splits a two-digit value now and then — "16" comes back as "1" and
"6", both centred on the same place of the board. Keeping the nearer of them
puts a 6 where a 16 is, and from there the wrong value sits in the frame the
next move is compared against, so it costs two comparisons rather than one.

LIVE 2026-09-04: three readings in twenty-one, and the true rule sat at 58% of
64 because of them — under the bar to be trusted, so nothing looked ahead.

Joined only where every run in the place is a number. That is what says they
are pieces of one value rather than something lying across the board: a line
of prose over a tile is not digits, and there the nearer thing still wins.
"""

from __future__ import annotations

from core.perception.the_lattice_she_holds import TheLatticeSheHolds

DOWN = (0.34, 0.48, 0.62, 0.76)
ACROSS = (0.25, 0.42, 0.58, 0.74)


def _held() -> TheLatticeSheHolds:
    lattice = TheLatticeSheHolds()
    lattice.down_at, lattice.across_at = DOWN, ACROSS
    return lattice


def _says(lattice, said):
    placed = lattice.fit(said)
    return {(cell.row, cell.column): cell.says for cell in placed.cells}


def test_a_split_number_is_put_back_together():
    said = [(DOWN[1], ACROSS[2] - 0.012, "1"), (DOWN[1], ACROSS[2] + 0.012, "6")]
    assert _says(_held(), said) == {(1, 2): "16"}


def test_it_is_joined_in_reading_order_not_nearness():
    """The nearer run is the second digit here, and it still goes second."""
    said = [(DOWN[0], ACROSS[0] + 0.004, "2"), (DOWN[0], ACROSS[0] - 0.02, "5")]
    assert _says(_held(), said) == {(0, 0): "52"}


def test_prose_lying_over_a_tile_still_loses_to_the_tile():
    said = [
        (DOWN[2], ACROSS[1], "64"),
        (DOWN[2], ACROSS[1] + 0.03, "Are you sure you want to quit?"),
    ]
    assert _says(_held(), said) == {(2, 1): "64"}


def test_one_run_in_a_place_is_left_alone():
    said = [(DOWN[3], ACROSS[3], "128")]
    assert _says(_held(), said) == {(3, 3): "128"}


def test_crowding_is_still_noticed():
    lattice = _held()
    lattice.fit([(DOWN[0], ACROSS[0] - 0.01, "1"), (DOWN[0], ACROSS[0] + 0.01, "6")])
    assert lattice.crowded_for == 1


def test_numbers_with_separators_count_as_numbers():
    said = [(DOWN[0], ACROSS[1] - 0.01, "1"), (DOWN[0], ACROSS[1] + 0.01, "024")]
    assert _says(_held(), said) == {(0, 1): "1024"}
