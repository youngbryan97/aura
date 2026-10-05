"""An end screen says who won in a sentence; a request says whether a win is wanted."""
from __future__ import annotations

import pytest

from core.language.how_a_game_ended import asks_to_win, how_it_ended

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("words", "ended"),
    [
        ("You win! You 5 Computer 3 Press SPACE to play again", "won"),
        ("The computer wins. You 2 Computer 5 Press SPACE to play again", "lost"),
        ("GAME OVER Score 7 PLAY AGAIN", "lost"),
        ("Congratulations! Level complete", "won"),
        ("Johnny wins!", "lost"),
        ("YOU WIN", "won"),
        ("You 3 Computer 5", "lost"),
        ("PONG Press SPACE to play", ""),
        ("Winner: Johnny", ""),
    ],
)
def test_who_won_is_the_subject_of_winning(words, ended):
    assert how_it_ended(words) == ended


@pytest.mark.parametrize(
    ("request_words", "wanted"),
    [
        ("Fix it, then play it against the computer until you win.", True),
        ("play three of the games and beat each one", True),
        ("Go play Food Bash and win", True),
        ("play the game", False),
    ],
)
def test_a_request_for_a_win(request_words, wanted):
    assert asks_to_win(request_words) is wanted


def test_a_screen_read_in_pieces_is_read_line_by_line():
    """A large "You win!" read as two pieces among the smaller line under it is still a win."""
    from core.language.how_a_game_ended import how_it_ended_in
    from core.skills.screen_pursuit_as_it_happens import _lines_of

    def region(text, x, y, h):
        return {"text": text, "center_x": x, "center_y": y, "x": x - 0.05, "y": y - h / 2, "height": h}

    layout = [region("You", 0.42, 0.30, 0.08), region("You 5", 0.40, 0.48, 0.03), region("win!", 0.58, 0.31, 0.08),
              region("Computer o", 0.6, 0.48, 0.03), region("Press SPACE to play again", 0.5, 0.78, 0.04)]
    assert _lines_of(layout) == ["You win!", "You 5 Computer o", "Press SPACE to play again"]
    assert how_it_ended_in(_lines_of(layout)) == "won"


def test_who_won_outweighs_a_screen_that_only_says_it_is_over():
    from core.language.how_a_game_ended import how_it_ended_in

    assert how_it_ended_in(["The computer wins.", "Press SPACE to play again"]) == "lost"
    assert how_it_ended_in(["Press SPACE to play again", "You 5 Computer 2"]) == "won"
