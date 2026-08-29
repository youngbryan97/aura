"""Being above her window is not being in her way.

A notification banner sits in a corner. The board she is playing is in the
middle. Nothing about the banner stops her, and treating it as an obstacle
cost a whole run.

LIVE 2026-08-29, on play2048.co with the board found and read — "0 screenfuls
down: 22x19 with 41 things in it" — she reported "UserNotificationCenter was
in front of this. Closed it." four times over and then ended the run with
"nothing on screen offered a move (after 0 moves)". The board was untouched
and entirely visible the whole time.

Two things were wrong and they hid each other. Anything above her window
counted as in front of her, wherever it was. And the check for whether she had
closed it asked what was on top while EXCLUDING the thing it had just tried to
close, so it reported success whenever the overlay was the only thing above
her — every claim it made was unfalsifiable.
"""

from __future__ import annotations

import pytest

from core.skills.screen_pursuit import _covers


class Screen:
    """A main display of a given size, shaped the way Quartz reports one."""

    class Size:
        def __init__(self, w, h):
            self.width, self.height = w, h

    def __init__(self, w=1600.0, h=1000.0):
        self.size = Screen.Size(w, h)


def window(x, y, w, h):
    return {"kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}


#: The middle of the screen, where a board sits: left, top, right, bottom.
BOARD = (0.30, 0.30, 0.70, 0.80)


# ── what is really in her way ────────────────────────────────────────────

def test_a_banner_in_the_corner_is_not_over_the_board():
    """The exact case that ended a live run."""
    assert _covers(window(1300, 20, 280, 90), BOARD, Screen()) is False


def test_a_dialog_over_the_middle_is():
    assert _covers(window(500, 400, 600, 200), BOARD, Screen()) is True


def test_a_sheet_that_covers_everything_is():
    assert _covers(window(0, 0, 1600, 1000), BOARD, Screen()) is True


def test_something_clipping_one_edge_of_it_is():
    assert _covers(window(0, 0, 500, 400), BOARD, Screen()) is True


@pytest.mark.parametrize(
    ("x", "y", "w", "h"),
    [
        (0, 0, 400, 200),      # top-left, clear of the board
        (1200, 850, 400, 150),  # bottom-right, clear of it
        (0, 900, 1600, 100),    # a strip along the bottom
    ],
)
def test_anything_beside_it_is_not_in_the_way(x, y, w, h):
    assert _covers(window(x, y, w, h), BOARD, Screen()) is False


def test_touching_the_edge_is_not_covering_it():
    left = int(0.30 * 1600)
    assert _covers(window(0, 0, left, 1000), BOARD, Screen()) is False


# ── and when she cannot tell, she assumes the worst ──────────────────────

def test_a_window_whose_bounds_cannot_be_read_is_in_the_way():
    assert _covers({}, BOARD, Screen()) is True
    assert _covers({"kCGWindowBounds": "nonsense"}, BOARD, Screen()) is True


def test_a_screen_with_no_size_is_no_help_either():
    assert _covers(window(0, 0, 10, 10), BOARD, Screen(0.0, 0.0)) is True


# ── the check that could never fail ──────────────────────────────────────

def test_the_close_check_asks_about_the_thing_it_closed():
    """It passed the overlay as the window to EXCLUDE, so success was certain."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.clear_what_is_in_front)
    # The code, not the comment explaining what it used to do.
    doing = [line for line in source.splitlines() if not line.strip().startswith("#")]
    assert not any("_whats_on_top(on_top)" in line for line in doing)
    assert any('_everything_on_top("")' in line for line in doing)
