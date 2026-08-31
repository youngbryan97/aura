"""One grid, held across time, that readings are placed into.

Worked out from each glance instead, a four by four board whose top row is
empty has three rows in it, so a tile that was in the second row is now in the
first and the move between two readings is unreadable — not because either
reading was wrong but because they were never in the same frame of reference.

The end of this file runs the whole chain: a drawn screen with furniture and
OCR drift, through the real readers, into the rule learner.
"""

from __future__ import annotations

import random

from core.perception.the_lattice_she_holds import TheLatticeSheHolds
from core.perception.what_moves_within_itself import MovesWithinItself
from core.perception.where_it_responds import (
    Responsive,
    noticed,
    places_and_text,
    what_is_there,
)

PLACES = frozenset(
    (int(round((0.20 + col * 0.15) * 100)), int(round((0.35 + row * 0.12) * 100)))
    for row in range(4)
    for col in range(4)
)


def _held() -> TheLatticeSheHolds:
    lattice = TheLatticeSheHolds()
    assert lattice.built_from(PLACES, acts=6)
    return lattice


def test_an_empty_row_does_not_shrink_the_grid() -> None:
    lattice = _held()
    assert (lattice.rows, lattice.columns) == (4, 4)
    # Only the bottom two rows have anything in them this turn.
    said = [
        (0.35 + 2 * 0.12, 0.20 + 1 * 0.15, "4"),
        (0.35 + 3 * 0.12, 0.20 + 3 * 0.15, "2"),
    ]
    placed = lattice.fit(said)
    assert placed is not None
    assert (placed.rows, placed.columns) == (4, 4)
    at = {(cell.row, cell.column): cell.says for cell in placed.cells}
    assert at == {(2, 1): "4", (3, 3): "2"}


def test_furniture_outside_it_is_left_out_rather_than_making_a_row() -> None:
    lattice = _held()
    placed = lattice.fit(
        [
            (0.35, 0.20, "2"),
            (0.10, 0.62, "3184"),  # the score, well above the board
            (0.92, 0.50, "Install"),  # an advert, well below it
        ]
    )
    assert placed is not None
    assert (placed.rows, placed.columns) == (4, 4)
    assert [cell.says for cell in placed.cells] == ["2"]


def test_the_nearer_thing_wins_a_place_and_crowding_is_noticed() -> None:
    """A real capture had a system dialog over the board, its lines of prose
    landing on the board's places. Refusing the whole reading threw away the
    tiles that were plainly visible beside it, every turn."""
    lattice = _held()
    placed = lattice.fit([(0.352, 0.203, "2"), (0.35, 0.20, "some prose")])
    assert placed is not None
    at = {(cell.row, cell.column): cell.says for cell in placed.cells}
    assert at == {(0, 0): "some prose"}, "the thing centred on the place wins"
    assert not lattice.looks_covered(), "one crowded reading is a stray region"
    lattice.fit([(0.352, 0.203, "2"), (0.35, 0.20, "some prose")])
    assert lattice.looks_covered(), "several in a row is something over it"
    # And it clears when the view does.
    lattice.fit([(0.35, 0.20, "2")])
    assert not lattice.looks_covered()


def test_it_gives_way_when_the_thing_has_changed_and_not_before() -> None:
    lattice = _held()
    assert lattice.fit([(0.35, 0.20, "2")]) is not None
    assert not lattice.has_changed()
    assert lattice.fit([(0.0, 0.0, "somewhere else")]) is None
    assert not lattice.has_changed(), "one bad glance is a bad glance"
    assert lattice.fit([(0.0, 0.0, "somewhere else")]) is None
    assert lattice.has_changed()


def test_it_survives_the_process() -> None:
    lattice = _held()
    again = TheLatticeSheHolds.from_memory(lattice.as_memory())
    assert again.held
    assert (again.rows, again.columns) == (4, 4)
    assert again.fit([(0.35, 0.20, "2")]) is not None
    assert not again.built_from(PLACES), "the same places move nothing"


def _slid(row: list[int]) -> list[int]:
    kept = [one for one in row if one]
    out, at = [], 0
    while at < len(kept):
        if at + 1 < len(kept) and kept[at] == kept[at + 1]:
            out.append(kept[at] * 2)
            at += 2
        else:
            out.append(kept[at])
            at += 1
    return out + [0] * (len(row) - len(out))


def _move(grid, act):
    rows = [list(one) for one in grid]
    if act == "left":
        return [_slid(one) for one in rows]
    if act == "right":
        return [list(reversed(_slid(list(reversed(one))))) for one in rows]
    if act == "up":
        return [list(one) for one in zip(*[_slid(list(one)) for one in zip(*rows)], strict=False)]
    return [
        list(one)
        for one in zip(
            *[
                list(reversed(_slid(list(reversed(list(one))))))
                for one in zip(*rows, strict=False)
            ],
            strict=False,
        )
    ]


def test_the_whole_chain_learns_the_rule_from_a_drawn_screen() -> None:
    """Not the reasoning and not the reader: the two of them together.

    A screen with a board, a title, a score, a best score, a move counter, a
    New Game button and an advert on it, drawn with the OCR drift a real
    reading has, played for forty moves. What comes out is the rule.
    """
    from core.perception.how_it_moves import HowItMoves

    roll = random.Random(7)
    drift = 0.004

    def screen(grid, score, moves):
        out = [
            {
                "text": str(grid[row][col]),
                "center_x": 0.20 + col * 0.15 + roll.uniform(-drift, drift),
                "center_y": 0.35 + row * 0.12 + roll.uniform(-drift, drift),
            }
            for row in range(4)
            for col in range(4)
            if grid[row][col]
        ]
        for text, x, y in (
            ("2048", 0.20, 0.10),
            (str(score), 0.62, 0.10),
            ("20000", 0.80, 0.10),
            (str(moves), 0.62, 0.16),
            ("New Game", 0.50, 0.24),
            ("Install", 0.50, 0.92),
        ):
            out.append({"text": text, "center_x": x, "center_y": y})
        return {"layout": out, "text": " ".join(one["text"] for one in out)}

    acts = ["left", "right", "up", "down"]
    grid = [[0] * 4 for _ in range(4)]
    for _ in range(2):
        grid[roll.randrange(4)][roll.randrange(4)] = 2
    score = moves = 0
    band, moving, rules = Responsive(), MovesWithinItself(), HowItMoves()
    lattice = TheLatticeSheHolds()
    before, was = screen(grid, score, moves), None
    once_held: list[tuple[int, int]] = []
    for _ in range(40):
        act = roll.choice(acts)
        went = _move(grid, act)
        worked = went != grid
        grid = went
        if worked:
            empty = [
                (r, c) for r in range(4) for c in range(4) if not grid[r][c]
            ]
            if empty:
                row, col = roll.choice(empty)
                grid[row][col] = roll.choice([2, 2, 2, 4])
            score += roll.randrange(4, 40)
        moves += 1
        after = screen(grid, score, moves)
        noticed(band, before, after, worked=worked)
        moving.saw(places_and_text(before), places_and_text(after))
        if moving.settled():
            lattice.built_from(moving.the_thing_itself(), moving.acts)
        laid = what_is_there(after, band.band(), None, None, lattice)
        if lattice.held:
            once_held.append((laid.rows, laid.columns))
        if was is not None:
            rules.watched(was, act, laid)
        was, before = laid, after

    assert moving.the_thing_itself() == PLACES, "the board is the sixteen places"
    assert (lattice.rows, lattice.columns) == (4, 4)
    # It settles and then stays. Early on it is built from the three or four
    # places that have moved twice so far, which is a grid of the wrong size,
    # and readings fall back to being worked out. As more places earn their
    # way in it is rebuilt, and from then on every reading is in it — no shape
    # drift at all, which is the thing that was making the moves unreadable.
    assert len(once_held) > 20
    assert set(once_held[-20:]) == {(4, 4)}
    # Not all of them, and it should not be. The first turns happen before she
    # has acted enough for anything to have moved twice, so there is no
    # lattice to place them in and they are read the old way. What matters is
    # that she gets there and stays there.
    assert rules.confidence() > 0.9, rules.says()
    assert "slides and combines" in rules.says()


# From a reading actually captured off the screen on 2026-08-29, of a browser
# with the game open in it. The pitch is what matters and it is tighter than a
# made-up one: rows 0.113 of the window apart, columns 0.105, against a
# quantisation of a hundredth. Only the geometry is taken; what the tabs said
# is his and stays on his machine.
BOARD_TOP, ROW_PITCH = 0.3070, 0.1132
BOARD_LEFT, COLUMN_PITCH = 0.3946, 0.1050
#: Where the furniture was: browser chrome along the top, a score and a best
#: score and a New Game above the board, a footer under it.
FURNITURE = (
    (0.0198, 0.1294, "a tab"),
    (0.0209, 0.7035, "another tab"),
    (0.0605, 0.1381, "the address"),
    (0.1186, 0.5305, "BEST"),
    (0.1198, 0.4695, "SCORE"),
    (0.1303, 0.7522, "New Game"),
    (0.1314, 0.2638, "the title"),
    (0.9012, 0.4993, "a footer"),
)


def test_the_rule_comes_out_on_the_geometry_of_a_real_reading() -> None:
    """The same chain, at the pitch a real capture had rather than a roomy one.

    This is where the quantisation bites: at 0.105 between columns, one tile
    drifting by a few thousandths of the window rounds into two different
    hundredths, so a single place is counted as several. The lattice is built
    by clustering what has been seen rather than by matching keys, which is
    what makes that survivable.
    """
    from core.perception.how_it_moves import HowItMoves

    roll = random.Random(11)
    drift = 0.003

    def screen(grid, score):
        out = [
            {"text": text, "center_x": x, "center_y": y}
            for y, x, text in FURNITURE
        ]
        out.append({"text": str(score), "center_x": 0.4688, "center_y": 0.1372})
        out += [
            {
                "text": str(grid[row][col]),
                "center_x": BOARD_LEFT + col * COLUMN_PITCH + roll.uniform(-drift, drift),
                "center_y": BOARD_TOP + row * ROW_PITCH + roll.uniform(-drift, drift),
            }
            for row in range(4)
            for col in range(4)
            if grid[row][col]
        ]
        return {"layout": out, "text": " ".join(one["text"] for one in out)}

    acts = ["left", "right", "up", "down"]
    grid = [[0] * 4 for _ in range(4)]
    for _ in range(2):
        grid[roll.randrange(4)][roll.randrange(4)] = 2
    score = 112
    band, moving, rules = Responsive(), MovesWithinItself(), HowItMoves()
    lattice = TheLatticeSheHolds()
    before, was, exact, seen = screen(grid, score), None, 0, 0
    for _ in range(60):
        act = roll.choice(acts)
        went = _move(grid, act)
        worked = went != grid
        grid = went
        if worked:
            empty = [(r, c) for r in range(4) for c in range(4) if not grid[r][c]]
            if empty:
                row, col = roll.choice(empty)
                grid[row][col] = roll.choice([2, 2, 2, 4])
            score += roll.randrange(4, 40)
        after = screen(grid, score)
        noticed(band, before, after, worked=worked)
        moving.saw(places_and_text(before), places_and_text(after))
        if moving.settled():
            lattice.built_from(moving.the_thing_itself(), moving.acts)
        laid = what_is_there(after, band.band(), None, None, lattice)
        truth = {
            (r, c): str(grid[r][c])
            for r in range(4)
            for c in range(4)
            if grid[r][c]
        }
        seen += 1
        exact += {(one.row, one.column): one.says for one in laid.cells} == truth
        if was is not None:
            rules.watched(was, act, laid)
        was, before = laid, after

    assert (lattice.rows, lattice.columns) == (4, 4)
    assert not lattice.looks_covered()
    assert exact >= seen * 0.9, f"{exact}/{seen} readings exactly right"
    # The misses are the first turns, before anything has moved twice and
    # there is any lattice to place them in.
    assert rules.confidence() > 0.9, rules.says()
    assert "slides and combines" in rules.says()
