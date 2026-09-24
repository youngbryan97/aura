"""Waiting for the move to land before judging whether it landed.

There was no wait at all after a keystroke. The key returned, the loop came
round, and the next reading was of a board mid-slide — or of one the game had
not begun moving yet. Compared with the reading before it, that says nothing
happened.

Of what she had kept about playing this game on 2026-08-30, ninety-eight of a
hundred acts were written down as having changed nothing, and the only rule
she ever confirmed was "this does not move", ninety-eight times.
"""

from __future__ import annotations

from screen_pursuit_support import patch_pursuit, pursuit_function_source

import asyncio

import pytest

from core.skills import screen_pursuit


def _reading(text: str) -> dict:
    return {"ok": True, "text": text, "layout": [], "scoped_to": "a thing"}


@pytest.mark.asyncio
async def test_it_waits_for_the_change_and_then_for_the_stillness(monkeypatch) -> None:
    """A board mid-slide has half moved, and half a move is not a state any
    rule describes."""
    frames = iter(
        [
            _reading("before"),  # the key has landed, nothing has moved yet
            _reading("before"),
            _reading("mid slide"),  # moving
            _reading("after"),  # arrived
            _reading("after"),  # and still
            _reading("after"),
        ]
    )

    async def read(app, over=None):
        return next(frames, _reading("after"))

    patch_pursuit(monkeypatch, "read_screen", read)
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.0
    seen, moved = await screen_pursuit._settled_after(_reading("before"), "a thing")
    assert moved
    assert seen["text"] == "after", "it should not stop on a board mid-slide"


@pytest.mark.asyncio
async def test_a_move_that_really_changes_nothing_says_so(monkeypatch) -> None:
    async def read(app, over=None):
        return _reading("before")

    patch_pursuit(monkeypatch, "read_screen", read)
    from core.skills import screen_pursuit_looking as looking

    looking._ANSWERS.clear()
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.4
    seen, moved = await screen_pursuit._settled_after(_reading("before"), "a thing")
    assert not moved
    assert seen["text"] == "before"


def test_how_long_to_wait_comes_from_how_long_it_has_taken() -> None:
    """Nothing is chosen. Before she has seen a change there is no
    measurement, so the old default stands; after that it is a little more
    than the longest she has seen lately."""
    from core.skills import screen_pursuit_looking as looking

    looking._ANSWERS.clear()
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.0
    assert screen_pursuit._how_long_to_wait() == 4.0
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.9
    assert screen_pursuit._how_long_to_wait() == pytest.approx(1.8)
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.1
    assert screen_pursuit._how_long_to_wait() == 1.0, "never less than a second"


def test_the_move_path_waits_at_all() -> None:
    """The mechanism existed and was wired only to the restart path."""
    # The act was a closure inside the decision until the module went back
    # under the size gate's ceiling. Asked for by name rather than sliced.
    act = pursuit_function_source("carry_out_the_move")
    assert "_settled_after(" in act, "a keystroke must be given time to land"


def test_the_pause_between_looks_is_probed_not_timed_from_before_it():
    """Timing an answer from before her own pause counted the pause as the world's.

    The next pause was the middle of those, so each move waited longer than
    the one before: two seconds a move on a board that answers in a third of
    one (live, 2026-09-18). A look that finds the world already still halves
    the pause; one that finds it still moving adds what it was short by.
    """
    from core.skills import screen_pursuit_looking as looking

    was, floor = dict(looking._WAIT), dict(looking._STILL_FLOOR)
    try:
        looking._WAIT["seconds"] = 0.0
        looking._STILL_FLOOR["seconds"] = float("inf")
        # Nothing measured yet: she looks at once, because the look measures it.
        assert looking._before_looking_again() == 0.0
        # She looked at once and the world was still moving for 0.3s more
        # than a still picture costs.
        looking._it_was_ready(0.0, True, {"seconds_to_still": 0.2})
        looking._it_was_ready(0.0, True, {"seconds_to_still": 0.5})
        assert looking._before_looking_again() == 0.3
        # Waiting that long found it already still: the wait was enough, so
        # less is tried.
        looking._it_was_ready(0.3, True, {"seconds_to_still": 0.2})
        assert looking._before_looking_again() == 0.15
        # And her own waiting is never counted as the world's.
        for _ in range(10):
            looking._it_was_ready(looking._before_looking_again(), True, {"seconds_to_still": 0.2})
        assert looking._before_looking_again() < 0.01
    finally:
        looking._WAIT.update(was)
        looking._STILL_FLOOR.update(floor)
