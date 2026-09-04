"""The picture she watched come to rest is the picture of what the act did.

Waiting for a move to land means watching until the screen changes and then
stops changing, which is two readings at least. The cycle after it took a
third of the same still surface, and a reading is a screenshot and an OCR —
about a third of the whole cost of a move on the real board.

It is only the same picture under conditions the loop can check: nothing was
moved to bring the window forward, the reading was of the whole window rather
than a part of it, and a reading was actually taken.
"""

from __future__ import annotations

import inspect

from core.skills import screen_pursuit

SOURCE = inspect.getsource(screen_pursuit.pursue_on_screen)


def test_the_settled_reading_is_kept():
    assert "at_rest[\"reading\"] = (" in SOURCE
    assert "came_to_rest, _ = await _settled_after(" in SOURCE


def test_it_is_used_instead_of_a_third_picture():
    at = SOURCE.index('ready = at_rest["reading"]')
    nearby = SOURCE[at : at + 500]
    assert "return ready" in nearby


def test_it_is_refused_when_a_window_had_to_be_brought_forward():
    at = SOURCE.index('ready = at_rest["reading"]')
    assert "undisturbed" in SOURCE[at : at + 300]
    assert "undisturbed = await _ensure_frontmost(mine)" in SOURCE


def test_it_is_refused_when_the_reading_was_of_part_of_the_window():
    at = SOURCE.index('ready = at_rest["reading"]')
    assert 'drawn["where"] is None' in SOURCE[at : at + 300]


def test_a_settle_that_read_nothing_keeps_nothing():
    """It hands back what it was given, which is the board before the act."""
    at = SOURCE.index("came_to_rest, _ = await _settled_after(")
    nearby = SOURCE[at : at + 800]
    assert 'came_to_rest is not pending["watched"]' in nearby


def test_it_is_used_once_and_not_again():
    at = SOURCE.index('ready = at_rest["reading"]')
    assert 'at_rest["reading"] = None' in SOURCE[at : at + 120]
