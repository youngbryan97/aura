"""What a screen asks to have typed is typed where it points, and sent the way it says; nothing is typed where she has
nothing to give. Nothing here is a game: an oracle that answers questions, a high-score table, a riddle, a search box."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.agency.typing_what_is_asked import places_to_type, type_what_is_asked, what_to_type
from core.language.words_asked_for import ANSWER, CODE, NAME, QUESTION, words_asked_for

pytestmark = pytest.mark.unit


def test_what_each_screen_asks_for_is_read_off_its_words():
    oracle = words_asked_for('1. Try to type only "Yes" or "No" questions. 2. Don\'t forget to click "Ask." TYPE YOUR QUESTION ABOVE')
    assert oracle.kind == QUESTION and oracle.where == "above" and oracle.sent_by == "Ask"
    assert words_asked_for("GAME OVER Enter your name: SUBMIT").kind == NAME
    assert words_asked_for("Riddle me this: What has keys but no locks? Answer?").kind == ANSWER
    assert words_asked_for("Type your answer and press Enter.").sent_by_key == "return"
    assert words_asked_for("Enter the secret code").kind == CODE
    assert words_asked_for("Use the arrow keys to move. Collect the stars.") is None


def test_she_types_her_name_a_question_or_an_answer_she_has_and_nothing_she_does_not():
    async def knows(question):
        return "a piano" if "keys" in question else None

    assert asyncio.run(what_to_type(words_asked_for("Enter your name"))) == "Aura"
    assert asyncio.run(what_to_type(words_asked_for("Type your question above"), goal="play until you win")) == "Will I win?"
    assert asyncio.run(what_to_type(words_asked_for("Riddle me this: What has keys but no locks? Answer?"), answer=knows)) == "a piano"
    assert asyncio.run(what_to_type(words_asked_for("Enter the secret code"))) is None


def test_the_place_the_words_point_to_comes_first():
    asked = words_asked_for("TYPE YOUR QUESTION ABOVE")
    regions = [{"text": "TYPE YOUR QUESTION ABOVE", "center_x": 0.5, "center_y": 0.6, "height": 0.04, "width": 0.4}]
    (x, y), *_rest = places_to_type(asked, regions)
    assert x == pytest.approx(0.5) and y < 0.6


class _Oracle:
    """A screen with a field above its words; what is typed shows only where the field was clicked; Ask answers."""

    def __init__(self):
        self.focused = False
        self.typed = ""
        self.answered = False

    async def look(self):
        return np.zeros((10, 10, 3), np.uint8), 0.0

    def words(self, _picture):
        regions = [{"text": "TYPE YOUR QUESTION ABOVE", "center_x": 0.5, "center_y": 0.6, "height": 0.04, "width": 0.4},
                   {"text": "ASK", "center_x": 0.8, "center_y": 0.7, "height": 0.04, "width": 0.08}]
        if self.typed:
            regions.append({"text": self.typed, "center_x": 0.5, "center_y": 0.5, "height": 0.04, "width": 0.4})
        if self.answered:
            regions.append({"text": "Like, wow! Most definitely!", "center_x": 0.5, "center_y": 0.3})
        return regions

    async def click(self, x, y):
        if abs(x - 0.8) < 0.05 and abs(y - 0.7) < 0.05:
            self.answered = bool(self.typed)
            return
        self.focused = abs(x - 0.5) < 0.1 and 0.45 < y < 0.55

    async def type_text(self, text):
        if self.focused:
            self.typed += text

    async def tap(self, key):
        if key == "backspace":
            self.typed = self.typed[:-1]


def test_she_types_the_question_into_the_field_above_and_asks():
    oracle = _Oracle()
    asked = words_asked_for('Try to type only "Yes" or "No" questions. Don\'t forget to click "Ask." TYPE YOUR QUESTION ABOVE')
    done = asyncio.run(type_what_is_asked(asked, oracle.look, oracle.words, oracle, goal="play it until you win"))
    assert oracle.typed == "Will I win?" and oracle.answered, done


def test_a_waiting_screen_that_asks_for_words_has_them_typed_on_her_page(monkeypatch):
    import time

    import core.skills.screen_pursuit_as_it_happens as reflexes

    oracle = _Oracle()

    class _OnHerPage(reflexes.PlayingAsItHappens):
        async def look(self):
            return await oracle.look()

        async def click(self, x, y):
            await oracle.click(x, y)

        async def type_text(self, text):
            await oracle.type_text(text)

        async def tap(self, key):
            await oracle.tap(key)

    monkeypatch.setattr(reflexes, "_said_while_playing", lambda line: None)
    monkeypatch.setattr("core.perception.what_the_pixels_show.recognize_text", oracle.words)
    playing = _OnHerPage(page=None, band=(0, 0, 1, 1), goal="play it until you win", ends_at=time.monotonic() + 30)
    playing.words = ['Try to type only "Yes" or "No" questions. Click "Ask." TYPE YOUR QUESTION ABOVE']
    assert asyncio.run(playing._typed_what_it_asks())
    assert oracle.typed == "Will I win?" and oracle.answered
    assert asyncio.run(playing._typed_what_it_asks())
    assert not asyncio.run(playing._typed_what_it_asks()), "the same ask is typed for twice at most"


def test_the_box_drawn_under_the_words_that_ask_is_clicked_first_in_its_middle():
    """LIVE 2026-10-09 a golf game's name box sat half a line under "Enter your name."; clicked a line and more below
    the words, and to their right, the name went nowhere."""
    import numpy as np

    from core.agency.typing_what_is_asked import places_to_type
    from core.language.words_asked_for import words_asked_for

    picture = np.full((400, 600, 3), (250, 245, 190), np.uint8)
    picture[140:160, 190:300] = (200, 60, 40)                   # the words that ask, drawn
    picture[168:198, 200:350] = (215, 205, 150)                 # the box to type in
    regions = [{"text": "Enter your name.", "x": 0.32, "y": 0.35, "width": 0.18, "height": 0.05,
                "center_x": 0.41, "center_y": 0.375}]
    first = places_to_type(words_asked_for("Enter your name."), regions, picture)[0]
    assert abs(first[0] - 275 / 600) < 0.03 and abs(first[1] - 183 / 400) < 0.02, first
