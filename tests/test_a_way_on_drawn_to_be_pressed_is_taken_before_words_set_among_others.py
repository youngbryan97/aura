"""Looking for the way on, she weighs what is drawn to be pressed before words set on the page among other words.

LIVE 23 November 2026 at a finished game in En Croissant, the heading over the players' names, "BLACK Stockfish", was
nearer "play a game of chess against Stockfish" (0.77) than the "New Game" button (0.60). She pressed the heading,
and it laid "Player not found" over the board. A button's words sit in a box drawn around them and nothing else; a
heading's sit in a card with the other player's name, the date and the event. The encoder here is a stand-in that
counts shared words, so the test is of the choosing, not of a model.
"""
from __future__ import annotations

import asyncio
import re

import pytest

import core.skills.screen_pursuit as pursuit
import core.skills.screen_pursuit_a_way_on as way_on
import core.skills.screen_pursuit_surface as surface

#: The card that names the players, and the box of each button, as shares of the window.
CARD = [0.53, 0.10, 0.98, 0.28]
MOVES = [0.53, 0.53, 0.98, 0.93]


def _button(left: float) -> list[float]:
    return [left, 0.44, left + 0.13, 0.51]


#: The finished game as her reading gave it, 23 November 2026, with the box each run of words is drawn in.
FINISHED = [
    ("WHITE", 0.57, 0.15, CARD), ("Aura", 0.56, 0.19, CARD), ("Unknown ELO", 0.58, 0.23, CARD),
    ("Player vs Engine", 0.76, 0.15, CARD), ("En Croissant", 0.70, 0.19, CARD), ("2026.11.23", 0.78, 0.19, CARD),
    ("Stockfish", 0.94, 0.19, CARD),
    ("New Game", 0.60, 0.47, _button(0.54)), ("Analyze", 0.75, 0.47, _button(0.69)),
    ("Engine Log", 0.90, 0.47, _button(0.84)),
    ("Starting Position", 0.63, 0.55, MOVES), ("Black wins", 0.75, 0.86, MOVES),
]
ASKED = "Open En Croissant and play a game of chess against Stockfish. Tell me each move as you go."


def _words_in_common(texts):
    vocabulary = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
    return [[float(w in re.findall(r"[a-z]+", t.lower())) + 1e-3 for w in vocabulary] for t in texts]


def _layout(boxed: bool) -> list[dict]:
    return [{"text": t, "center_x": x, "center_y": y, **({"within": box} if boxed else {})}
            for t, x, y, box in FINISHED]


@pytest.fixture
def screen(monkeypatch):
    layout: list[dict] = []
    pressed: list[str] = []

    async def read_screen(app, over=None):
        return {"layout": list(layout), "bounds": [0, 0, 1000, 800]}

    async def click(x, y, **_kwargs):
        pressed.append(next(r["text"] for r in layout if (r["center_x"], r["center_y"]) == (x, y)))
        layout[:] = [{"text": "FIDE Player Info", "center_x": 0.37, "center_y": 0.14}] if pressed[-1] == "Stockfish" \
            else [{"text": "Starting Position", "center_x": 0.63, "center_y": 0.55}]
        return True

    monkeypatch.setattr(pursuit, "read_screen", read_screen)
    monkeypatch.setattr(pursuit, "_tell", lambda line: None)
    monkeypatch.setattr(surface, "click_normalized", click)
    monkeypatch.setattr("core.language.learned_matcher.embed_sentences", _words_in_common)
    monkeypatch.setattr(way_on, "SHOWS_IN_S", 0.0)

    async def no_window(_app):  # the real desktop is never read in a test
        return None

    monkeypatch.setattr(way_on, "_its_own_pixels_read", no_window)
    return layout, pressed


def test_the_button_is_pressed_before_a_heading_that_names_the_opponent(screen) -> None:
    layout, pressed = screen
    layout[:] = _layout(boxed=True)
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert pressed == ["New Game"]


def test_where_nothing_is_seen_drawn_around_the_words_the_words_alone_decide(screen) -> None:
    layout, pressed = screen
    layout[:] = _layout(boxed=False)
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, {}, nothing_answered=True))
    assert pressed == ["Stockfish"]


def test_a_box_holding_other_words_is_not_a_box_of_its_own() -> None:
    controls = way_on._controls(way_on._regions({"layout": _layout(boxed=True)}))
    own = {" ".join(r["text"] for r in c): way_on._a_box_of_its_own(c, controls) for c in controls}
    assert own["New Game"] and own["Analyze"] and own["Engine Log"]
    assert not any(own[name] for name in own if "Stockfish" in name or "Aura" in name)


def test_a_way_on_that_changed_the_screen_without_what_was_asked_coming_of_it_leaves_the_choice_to_her(
        screen, monkeypatch) -> None:
    """LIVE 26 November 2026 the heading was pressed a second time, each press followed by twenty seconds at a board
    that took no move, before "New Game" was put to her. A press that changed the screen reached the control, and with
    what was asked still undone it has answered: which control does the next step is put to her at the next look."""
    layout, pressed = screen
    offered: list[list[str]] = []

    async def her_understanding(_app, _act, texts):
        offered.append(list(texts))
        return texts.index("New Game")

    monkeypatch.setattr(way_on, "_her_understanding", her_understanding)
    keep: dict = {}
    layout[:] = _layout(boxed=False)
    keep["way_on_looked_at"] = -1e9
    asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=True))
    assert pressed == ["Stockfish"]
    layout[:] = _layout(boxed=False)
    keep["way_on_looked_at"] = -1e9
    assert asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=True))
    assert pressed == ["Stockfish", "New Game"]
    assert offered, "the choice was hers"


def test_a_way_on_pressed_as_often_as_it_is_given_without_the_screen_changing_leaves_the_choice_to_her(
        screen, monkeypatch) -> None:
    """A press that changed nothing may not have reached the control: it is tried as often as a way on is given, then
    which other control does the next step is put to her, with the controls not yet pressed that often as the answers."""
    layout, pressed = screen
    offered: list[list[str]] = []

    async def her_understanding(_app, _act, texts):
        offered.append(list(texts))
        return texts.index("New Game")

    async def a_press_that_changes_nothing(x, y, **_kwargs):
        pressed.append(next(r["text"] for r in layout if (r["center_x"], r["center_y"]) == (x, y)))
        return True

    monkeypatch.setattr(way_on, "_her_understanding", her_understanding)
    monkeypatch.setattr(surface, "click_normalized", a_press_that_changes_nothing)
    keep: dict = {}
    for _ in range(way_on.PRESSES_OF_A_WAY_ON):
        layout[:] = _layout(boxed=False)
        keep["way_on_looked_at"] = -1e9
        asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=True))
    assert pressed == ["Stockfish"] * way_on.PRESSES_OF_A_WAY_ON
    layout[:] = _layout(boxed=False)
    keep["way_on_looked_at"] = -1e9
    asyncio.run(way_on.took_a_way_on("en-croissant", ASKED, keep, nothing_answered=True))
    assert pressed[-1] == "New Game"
    assert offered and "Stockfish" not in offered[-1]
