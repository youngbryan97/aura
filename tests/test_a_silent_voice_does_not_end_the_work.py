"""A voice that does not answer is not a reason to stop.

Live, 2026-09-18: at an impasse she was offered the two ways out — start over,
or see it through — and both needed a reason in words. The voice timed out, so
nothing was speakable, and the run ended "everything available needs a reason
I cannot put into words right now" after eighty-eight moves. She is the one
deciding; the voice advises.

Needing words protects live work from being thrown away on a ranking, which is
what starting over does. Seeing it through throws nothing away, so it is hers
to take without words.
"""

from __future__ import annotations

from core.agency.deliberate_action import choose_without_language
from core.skills.screen_pursuit import SEE_IT_THROUGH, START_OVER, ways_out

BOARD = {
    "layout": [
        {"text": "New Game", "center_x": 0.8, "center_y": 0.1},
        {"text": "2 4 8", "center_x": 0.5, "center_y": 0.5},
    ]
}


def test_throwing_the_work_away_still_needs_a_reason():
    by_name = {option.name: option for option in ways_out(BOARD)}
    assert by_name[START_OVER].needs_words


def test_keeping_the_work_does_not():
    by_name = {option.name: option for option in ways_out(BOARD)}
    assert not by_name[SEE_IT_THROUGH].needs_words


def test_without_words_she_keeps_going_rather_than_stopping():
    chosen, why = choose_without_language(ways_out(BOARD))
    assert chosen is not None, why
    assert chosen.name == SEE_IT_THROUGH
