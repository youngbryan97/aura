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
