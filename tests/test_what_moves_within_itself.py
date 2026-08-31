"""Separating the board from the score, which one frame cannot do.

The screen is a board, a score, a best-score, a move counter and an
advertisement. Every one of them changes when she presses a key, so how often
a place changes says nothing. What the board does that none of the others do
is show her values that were already in it.
"""

from __future__ import annotations

import random

from core.perception.what_moves_within_itself import (
    MovesWithinItself,
    what_moved_within,
)

BOARD = [(col, row) for row in range(4) for col in range(4)]
SCORE, BEST, MOVES, ADVERT = (9, 0), (10, 0), (11, 0), (9, 5)


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


def _screen(grid, score: int, moves: int) -> dict:
    seen = {
        (col, row): (str(grid[row][col]) if grid[row][col] else "")
        for col, row in BOARD
    }
    seen[SCORE] = str(score)
    seen[BEST] = "20000"
    seen[MOVES] = str(moves)
    seen[ADVERT] = "Install"
    return seen


def _a_game(turns: int = 8, seed: int = 1):
    """A run of ordinary left-slides, with the score going up each time."""
    roll = random.Random(seed)
    grid = [[roll.choice([0, 2, 2, 4, 4, 8]) for _ in range(4)] for _ in range(4)]
    score, moves = 1000, 0
    frames = [_screen(grid, score, moves)]
    for _ in range(turns):
        grid = [_slid(list(row)) for row in grid]
        empty = [(r, c) for r in range(4) for c in range(4) if not grid[r][c]]
        if empty:
            row, col = roll.choice(empty)
            grid[row][col] = 2
        score += roll.randrange(4, 60)
        moves += 1
        frames.append(_screen(grid, score, moves))
    return frames


def test_the_board_is_the_part_that_rearranges() -> None:
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    assert watching.settled()
    thing = watching.the_thing_itself()
    assert thing, "nothing looked like a board"
    assert thing <= set(BOARD), sorted(thing - set(BOARD))


def test_the_score_and_the_move_counter_are_not_part_of_it() -> None:
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    reports = watching.the_things_that_report()
    assert SCORE in reports
    assert MOVES in reports
    assert SCORE not in watching.the_thing_itself()
    assert MOVES not in watching.the_thing_itself()


def test_furniture_that_never_changes_is_in_neither() -> None:
    """A best score that stands still, and an advert. Not the thing, and not
    a report on it — she has seen no evidence about them either way."""
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    assert BEST not in watching.the_thing_itself()
    assert BEST not in watching.the_things_that_report()
    assert ADVERT not in watching.the_thing_itself()


def test_one_act_settles_nothing_and_says_so() -> None:
    frames = _a_game(turns=1)
    watching = MovesWithinItself()
    watching.saw(frames[0], frames[1])
    assert not watching.settled()
    # It still answers, for a caller with only one act to go on.
    moved, came = what_moved_within(frames[0], frames[1])
    assert isinstance(moved, frozenset) and isinstance(came, frozenset)


def test_it_is_not_about_boards() -> None:
    """Files being moved between folders, and a count of them underneath."""
    before = {(0, 0): "notes.md", (0, 1): "todo.md", (1, 0): "", (5, 5): "2 files"}
    after = {(0, 0): "notes.md", (0, 1): "", (1, 0): "todo.md", (5, 5): "1 file"}
    moved, came = what_moved_within(before, after)
    assert (1, 0) in moved
    assert (5, 5) in came


def test_what_it_found_survives_the_process() -> None:
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    again = MovesWithinItself.from_memory(watching.as_memory())
    assert again.the_thing_itself() == watching.the_thing_itself()
    thinner = MovesWithinItself.from_memory(watching.as_memory(), trust=0.0)
    assert not thinner.the_thing_itself()


def test_the_pursuit_narrows_to_the_thing_with_it() -> None:
    """The wiring: the same places the responsiveness band uses, so the two
    sets can be intersected at all. Two callers rounding differently would
    look like a board with nothing in it rather than like a bug."""
    from core.perception.where_it_responds import Responsive, noticed, places_and_text

    def as_reading(seen: dict) -> dict:
        return {
            "layout": [
                {"text": text, "center_x": col / 100, "center_y": row / 100}
                for (col, row), text in seen.items()
                if text
            ]
        }

    frames = [as_reading(one) for one in _a_game()]
    band, moving = Responsive(), MovesWithinItself()
    for before, after in zip(frames, frames[1:], strict=False):
        noticed(band, before, after)
        moving.saw(places_and_text(before), places_and_text(after))

    assert moving.settled()
    itself = moving.the_thing_itself()
    assert itself, "no places looked like the thing"
    # The score answers to her as reliably as the board does, which is why the
    # band alone cannot be the answer.
    at_score = (round(SCORE[0] / 100 * 100), round(SCORE[1] / 100 * 100))
    assert at_score in band.answered
    assert at_score not in itself
