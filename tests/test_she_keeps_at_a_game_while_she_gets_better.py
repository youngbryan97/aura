"""She keeps at a game while she is getting better at it, and says so and leaves it when she is not.

LIVE 2026-10-07 she played one game for twelve minutes and another for thirteen, each round lost as fast as the last.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser_drawing import not_getting_better

pytestmark = pytest.mark.unit


def _r(gains, took, ended="lost"):
    return {"gains": gains, "took_s": took, "ended": ended}


@pytest.mark.parametrize(("runs", "leave"), [
    ([_r(0, 20), _r(0, 21)], False),                        # too few rounds to say
    ([_r(0, 20), _r(0, 19), _r(0, 21)], True),              # three rounds, the last two no better
    ([_r(0, 20), _r(0, 19), _r(0, 30)], False),             # lasting much longer is better
    ([_r(1, 20), _r(0, 40), _r(2, 15)], False),             # gaining more is better
    ([_r(0, 20), _r(0, 19), _r(0, 18, "won")], False),      # a round won is better than any
    ([_r(3, 60), _r(0, 30), _r(1, 50), _r(2, 55)], True),   # the best was early, and not reached again
])
def test_whether_she_is_still_getting_better(runs, leave):
    assert not_getting_better(runs) is leave
