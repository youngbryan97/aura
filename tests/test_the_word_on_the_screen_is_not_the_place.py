"""Being in a thing is not the same as its name being visible somewhere.

A screen reading is of the screen, not of her window, so anything else on the
display counts as evidence that she has arrived. Measured live 2026-08-26: the
word she was looking for was in a terminal on the same display, the arrival
test passed, and she played thirty-five moves into that instead of into the
game — which was open, one window back, and never brought forward.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.skills.screen_pursuit import am_i_there

SOURCE = Path("core/skills/screen_pursuit.py").read_text()


# ── identity beats what happens to be visible ────────────────────────────

def test_the_window_she_is_in_is_what_says_where_she_is():
    assert am_i_there("2048", "anything at all", "https://play2048.co", "Google Chrome")


def test_the_name_in_somebody_elses_window_is_not_arrival():
    assert not am_i_there("2048", "... 2048 moves=35 cycles=27 ...", "", "Claude")


def test_a_wrong_page_in_the_right_browser_is_not_arrival():
    assert not am_i_there("2048", "2048 is mentioned here", "https://reddit.com", "Google Chrome")


def test_with_no_identity_at_all_the_reading_is_all_there_is():
    assert am_i_there("2048", "2048 SCORE 368 BEST 6068", "", "")
    assert not am_i_there("2048", "New Artifacts Routines", "", "")


def test_nothing_asked_for_is_nowhere_to_fail_to_be():
    assert am_i_there("", "anything", "", "Claude")
    assert am_i_there("  ", "anything", "", "Claude")


@pytest.mark.parametrize("wanted", ["a", "an", "of"])
def test_a_name_too_short_to_identify_anything_is_not_a_test(wanted):
    assert am_i_there(wanted, "", "", "Claude")


def test_several_words_any_of_which_identifies_it():
    assert am_i_there("lichess chess", "", "https://lichess.org/abc", "Safari")


# ── and the window is brought forward, named or learned ──────────────────

def test_a_run_fronts_the_window_it_belongs_to_even_unnamed():
    body = SOURCE[SOURCE.index("async def observe") :]
    assert 'mine = target_app or anchor["app"]' in body
    assert "await _ensure_frontmost(mine)" in body
    assert "if target_app:\n            try:\n                await _ensure_frontmost(target_app)" not in body
