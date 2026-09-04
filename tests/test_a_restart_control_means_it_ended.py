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

from core.skills.screen_pursuit import (
    RESTART_LABELS,
    a_way_back_that_was_not_there,
    restart_control,
    restart_controls,
    ways_out,
)


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


def test_a_control_that_was_always_there_is_not_an_ending():
    """The 2048 desktop app keeps "New Game" above the board the whole game.

    LIVE 2026-09-04: presence alone was the test, so the first cycle of every
    run on that app declared the task finished. Twelve cycles, eleven clicks
    on New Game, and not one arrow key pressed.
    """
    playing = restart_controls(_screen("New Game", "2", "4"))
    assert not a_way_back_that_was_not_there(playing, playing)


def test_a_control_that_turns_up_is_an_ending():
    began = restart_controls(_screen("New Game", "2", "4"))
    over = restart_controls(_screen("New Game", "Try again", "Game over!"))
    assert a_way_back_that_was_not_there(over, began)


def test_nothing_has_appeared_before_there_is_anything_to_compare_with():
    over = restart_controls(_screen("New Game", "Try again"))
    assert not a_way_back_that_was_not_there(over, None)


def test_a_way_back_where_there_was_none_at_all_is_an_ending():
    assert a_way_back_that_was_not_there(
        restart_controls(_screen("Play Again", "2")), restart_controls(_screen("2", "4"))
    )


def test_a_way_back_that_goes_away_is_not_an_ending():
    began = restart_controls(_screen("Try again", "2"))
    playing = restart_controls(_screen("2", "4"))
    assert not a_way_back_that_was_not_there(playing, began)


def test_the_pursuit_asks_whether_it_appeared_rather_than_whether_it_is_there():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('ended = responds["state"].nothing_answers()')
    nearby = source[at : at + 1800]
    assert "a_way_back_that_was_not_there(" in nearby
    assert "ended = True" in nearby


def test_it_is_said_once_rather_than_every_cycle():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert 'offered_a_restart["said"]' in source
