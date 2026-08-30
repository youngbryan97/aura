"""A way to start again, where there was none, is the thing saying it finished.

The other test asks whether what she was acting on is gone. On a board that
keeps its tiles visible under a "Play Again" overlay it never fires, so she
went on pressing keys into a game that was over — thirty-nine moves after Game
Over, each one costing a language pass because the situation was unusual.

A control that appears only at the end is better evidence than the absence of
one, and it is general: a finished form, an expired session and a lost game
all put one up.
"""

from __future__ import annotations

import pytest

from core.skills.screen_pursuit import RESTART_LABELS, restart_control, ways_out


def _screen(*texts: str) -> dict:
    return {
        "layout": [
            {"text": text, "center_x": 0.5, "center_y": 0.9 - n * 0.1}
            for n, text in enumerate(texts)
        ]
    }


@pytest.mark.parametrize("label", RESTART_LABELS)
def test_every_way_of_labelling_a_restart_is_found(label):
    assert restart_control(_screen(label.title(), "2", "4")) is not None


def test_a_board_with_no_way_out_offers_none():
    assert restart_control(_screen("2", "4", "8")) is None


def test_a_finished_thing_offers_the_way_out_and_not_the_moves():
    """Pressing a move key into something finished is not a thing she can do."""
    out = ways_out(_screen("Play Again", "2", "4"), ended=True)
    assert out
    assert any("start over" in option.name for option in out)


def test_the_pursuit_treats_an_appearing_restart_as_the_end():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    at = source.index("ended = responds[\"state\"].nothing_answers()")
    nearby = source[at : at + 1200]
    assert "restart_control(observation) is not None" in nearby
    assert "ended = True" in nearby


def test_it_is_said_once_rather_than_every_cycle():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit)
    assert 'offered_a_restart["value"]' in source
