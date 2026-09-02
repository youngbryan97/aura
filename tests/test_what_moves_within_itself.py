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
    watching = MovesWithinItself()
    steps = [
        {(0, 0): "notes.md", (0, 1): "todo.md", (1, 0): "plan.md", (5, 5): "3 files"},
        {(0, 0): "notes.md", (0, 1): "todo.md", (1, 0): "", (5, 5): "2 files"},
        {(0, 0): "notes.md", (0, 1): "", (1, 0): "todo.md", (5, 5): "2 files"},
        {(0, 0): "", (0, 1): "notes.md", (1, 0): "todo.md", (5, 5): "2 files"},
    ]
    for before, after in zip(steps, steps[1:], strict=False):
        watching.saw(before, after)
    assert (0, 1) in watching.the_thing_itself()
    assert (5, 5) in watching.the_things_that_report()


def test_what_it_found_survives_the_process_as_evidence_not_as_fact() -> None:
    """What comes back is a head start, not an answer.

    It used to come back as an answer, and that is what put a second sitting's
    places on top of the first: the counts arrive already made, so nothing
    that tells places apart within a sitting can touch them. What survives now
    is the evidence, and a place has to be seen again to be counted.
    """
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    knew = watching.the_thing_itself()
    assert knew

    again = MovesWithinItself.from_memory(watching.as_memory())
    assert again.rearranged, "the evidence is still there"
    assert again.the_thing_itself() == frozenset(), "and none of it is claimed yet"
    for before, after in zip(frames, frames[1:], strict=False):
        again.saw(before, after)
    assert again.the_thing_itself(), "seeing them again earns them back"

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


def test_a_score_reading_the_same_as_a_tile_is_still_a_score() -> None:
    """Early in a game the score passes through 4, 8 and 16, which are also
    tile values. Asking only whether the value had been on the screen called
    the score a tile, and every reporting place was swallowed the same way."""
    before = {(0, 0): "8", (0, 1): "2", (9, 0): "8"}
    # The score goes 8 -> 16 while a tile reading 8 is still sitting there.
    after = {(0, 0): "8", (0, 1): "2", (9, 0): "16"}
    moved, came = what_moved_within(before, after)
    assert (9, 0) in came, "the score arrived at a value it did not move to"
    assert (9, 0) not in moved


def test_a_tally_going_back_to_the_start_is_a_new_game_not_a_fall() -> None:
    """A score goes to nought when a new game begins. Counting that as falling
    makes every tally look like it goes both ways, so she finds no measure of
    progress at all from the second game onward — which is when she most needs
    one."""
    watching = MovesWithinItself()

    def a_game(from_: int, to: int) -> None:
        for score in range(from_, to, 4):
            watching.saw(
                {(0, 0): "2", (9, 0): str(score)},
                {(0, 1): "2", (9, 0): str(score + 4)},
            )

    a_game(0, 40)
    assert watching.what_measures_doing_well() == frozenset({(9, 0)})
    # A new game: the score drops to nought and climbs again.
    watching.saw({(0, 0): "2", (9, 0): "40"}, {(0, 1): "2", (9, 0): "0"})
    a_game(0, 40)
    assert watching.what_measures_doing_well() == frozenset({(9, 0)})
    # A genuine fall, to somewhere above where it began, is still a fall.
    watching.saw({(0, 0): "2", (9, 0): "40"}, {(0, 1): "2", (9, 0): "20"})
    assert watching.what_measures_doing_well() == frozenset()


def test_a_place_she_was_told_about_has_to_be_seen_again() -> None:
    """Where a thing is is only worth remembering while the thing is there.

    The window gets moved, the page reflows, the game restarts a little to the
    left — and what comes back from the last sitting is then a second set of
    places laid over this one. Measured live 2026-09-02: a four by four board
    came back as four by EIGHT, both sittings counted at once, and every
    reading placed into it was wrong.
    """
    watching = MovesWithinItself()
    frames = _a_game()
    for before, after in zip(frames, frames[1:], strict=False):
        watching.saw(before, after)
    knew = watching.the_thing_itself()
    assert knew

    # A new sitting, with the thing in the same place: it earns its places
    # back as it sees them, and ends up where it was.
    same = MovesWithinItself.from_memory(watching.as_memory())
    assert same.the_thing_itself() == frozenset(), "nothing seen yet this sitting"
    for before, after in zip(frames, frames[1:], strict=False):
        same.saw(before, after)
    found = same.the_thing_itself()
    assert found, "seeing them again should earn them back"
    assert found <= set(BOARD), sorted(found - set(BOARD))

    # And a new sitting with the thing somewhere else: what it knew does not
    # get laid over what it can see.
    def shifted(seen: dict) -> dict:
        return {(col + 40, row): text for (col, row), text in seen.items()}

    moved = MovesWithinItself.from_memory(watching.as_memory())
    for before, after in zip(frames, frames[1:], strict=False):
        moved.saw(shifted(before), shifted(after))
    found = moved.the_thing_itself()
    assert found
    assert not (found & knew), "the old places must not come back too"
    assert found == {(col + 40, row) for col, row in knew}
