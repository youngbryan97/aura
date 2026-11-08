"""A look that saw nothing is a cut, not what an event showed, where a later look in it saw something.

LIVE 8 November 2026, rehearsal 12, "Kiwi!": an event from 1:16 to 2:25 began on a black frame and was handed to her
thoughts after as "Entirely black", though from 1:50 her eyes saw the kiwi flying in its goggles. She told the person
she must have invented what she had said of it while it played.
"""
from __future__ import annotations

import pytest

from core.together.what_moves_her import WhatMovesHer, shows_nothing

BLACK = "Entirely black — there are no visible objects, people, or actions depicted."
KIWI = "A cartoon kiwi bird wearing aviator goggles, flying against a bright, cloudy sky."


def _watched_through_a_cut() -> WhatMovesHer:
    moves = WhatMovesHer("Kiwi!")
    for look in range(10):
        moves.saw(5.0 * look, 0.1 + 0.001 * look, "a kiwi dragging a stick across a field")
    moves.saw(55.0, 8.0, BLACK)
    for look in range(12, 16):
        moves.saw(5.0 * look, 0.12, "")
    moves.saw(80.0, 0.15, KIWI)
    return moves


def test_the_event_a_black_frame_opened_shows_what_was_seen_after_it():
    moves = _watched_through_a_cut()
    assert moves.events[-1].seen == KIWI
    assert "goggles" in moves.events_text() and "Entirely black" not in moves.events_text()


def test_an_event_that_was_only_black_says_so():
    moves = WhatMovesHer("x")
    for look in range(10):
        moves.saw(5.0 * look, 0.1 + 0.001 * look, "a field")
    moves.saw(55.0, 8.0, BLACK)
    assert moves.events[-1].seen == BLACK


@pytest.mark.parametrize(("seen", "nothing"), [
    (BLACK, True),
    ("A blank or empty image with no content.", True),
    ("The screen is completely dark.", True),
    ("A black cat on a white sofa.", False),
    (KIWI, False),
])
def test_what_shows_nothing(seen, nothing):
    assert shows_nothing(seen) is nothing
