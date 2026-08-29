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


#: Her window on the desktop: x, y, width, height in pixels. A band is a share
#: of THIS, which is the space read_screen measures in.
HER_WINDOW = (0, 100, 1600, 900)


def window(x, y, w, h):
    return {"kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}


#: The middle of her window, where a board sits: left, top, right, bottom.
BOARD = (0.30, 0.30, 0.70, 0.80)


# ── what is really in her way ────────────────────────────────────────────

def test_a_banner_in_the_corner_is_not_over_the_board():
    """The exact case that ended a live run."""
    assert _covers(window(1350, 120, 240, 90), BOARD, HER_WINDOW) is False


def test_a_dialog_over_the_middle_is():
    assert _covers(window(600, 500, 500, 200), BOARD, HER_WINDOW) is True


def test_a_sheet_that_covers_everything_is():
    assert _covers(window(0, 0, 1600, 1200), BOARD, HER_WINDOW) is True


def test_something_clipping_one_edge_of_it_is():
    assert _covers(window(0, 0, 600, 500), BOARD, HER_WINDOW) is True


@pytest.mark.parametrize(
    ("x", "y", "w", "h"),
    [
        (0, 100, 400, 150),      # top-left of her window, clear of the board
        (1300, 900, 300, 100),   # bottom-right, clear of it
        (0, 950, 1600, 50),      # a strip along the bottom
    ],
)
def test_anything_beside_it_is_not_in_the_way(x, y, w, h):
    assert _covers(window(x, y, w, h), BOARD, HER_WINDOW) is False


def test_a_band_is_a_share_of_her_window_not_of_the_display():
    """Measured against the screen, a banner halfway down the display reads as
    sitting on a board halfway down a window that starts lower."""
    lower = (0, 400, 1600, 600)
    banner = window(700, 300, 200, 80)
    assert _covers(banner, BOARD, HER_WINDOW) is True
    assert _covers(banner, BOARD, lower) is False


# ── and when she cannot tell, she assumes the worst ──────────────────────

def test_a_window_whose_bounds_cannot_be_read_is_in_the_way():
    assert _covers({}, BOARD, HER_WINDOW) is True
    assert _covers({"kCGWindowBounds": "nonsense"}, BOARD, HER_WINDOW) is True


def test_a_window_of_no_size_is_no_help_either():
    assert _covers(window(0, 0, 10, 10), BOARD, (0, 0, 0, 0)) is True


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
