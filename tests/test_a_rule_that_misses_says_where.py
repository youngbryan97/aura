"""Where the rule she holds was wrong, said in full the first few times.

LIVE 2026-09-24 the held rule was right 39% of the time on the real board and
100% in her own model of it, and nothing said which places disagreed.
"""

from __future__ import annotations

import logging

from core.perception import how_it_moves
from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell


def _board(*tiles: tuple[int, int, str]) -> Arrangement:
    return Arrangement(4, 4, tuple(Cell(r, c, s, (0.0, 0.0)) for r, c, s in tiles))


def test_a_miss_names_the_places_that_disagreed(caplog):
    knows = HowItMoves()
    with caplog.at_level(logging.INFO, logger="Aura.HowItMoves"):
        knows._say_what_it_missed(
            _board((0, 1, "2"), (0, 2, "2")), "left",
            _board((0, 0, "4")), _board((0, 0, "8"), (3, 3, "2")),
        )
    said = caplog.text
    assert "her rule missed on left" in said
    assert "(0,0) said '4' saw '8'" in said


def test_only_the_first_few_are_said(caplog):
    knows = HowItMoves()
    with caplog.at_level(logging.INFO, logger="Aura.HowItMoves"):
        for _ in range(how_it_moves._MISSES_WORTH_SAYING + 5):
            knows._say_what_it_missed(_board((0, 0, "2")), "up", _board((0, 0, "4")), _board())
    assert caplog.text.count("her rule missed") == how_it_moves._MISSES_WORTH_SAYING
