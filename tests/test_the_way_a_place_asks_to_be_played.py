"""The way a place says it is played is taken up and played, again and again, while it pays; not looked round from.

LIVE 2026-10-09 a game said "click and hold the mouse button to aim, then release to shoot". She threw for a round
and scored; in the next, its player's idle bobbing read as a world moving on its own, she steered him and never threw,
and pressed arrows and clicked shapes "to see what it does" for three minutes.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency.playing_by_shots import sends_by_letting_go
from core.agency.the_way_it_is_played import AS_IT_HAPPENS, FOLLOW, SEND, HeldTo, ways_asked

pytestmark = pytest.mark.unit

THROW_BY_HOLDING = "Use the mouse to control Wilt, click and hold the mouse button to aim, then release to shoot."
THROW_BY_CLICKING = "Take aim with your mouse and click to throw. Dodge food by pressing the space bar."


def test_the_words_say_which_way_first():
    assert ways_asked(THROW_BY_HOLDING)[0] == SEND
    assert ways_asked(THROW_BY_CLICKING) == [AS_IT_HAPPENS]          # a click that throws is a shot, made at once
    assert ways_asked("Follow the arrow keys in the book to cast the spell.")[0] == FOLLOW
    assert ways_asked("Control the girls with the arrow keys. SPACE press to fire.") == [AS_IT_HAPPENS]
    assert ways_asked("Thanks for playing!") == []


@pytest.mark.parametrize(("words", "sent"), [
    ("click to throw the ball and crack open the eggs", False),
    ("Use the mouse to aim and toss Bloo", True),
    ("Move your mouse to aim, then click to release your top", True),
    ("Pull back and let go to launch the hamster", True),
])
def test_a_throw_a_click_makes_is_not_a_throw_held_and_let_go(words, sent):
    assert sends_by_letting_go(words) is sent


def test_a_way_is_held_to_while_it_is_learned_and_while_it_pays_and_let_go_when_it_gets_nowhere():
    held = HeldTo()
    asked = ways_asked(THROW_BY_HOLDING)
    assert held.choose(asked, THROW_BY_HOLDING) == SEND
    for _ in range(2):
        held.took(SEND, {"by_shots": {"gains": 0}}, THROW_BY_HOLDING)
        assert held.holding()                                        # still being learned
    held.took(SEND, {"by_shots": {"gains": 2}}, THROW_BY_HOLDING)
    for _ in range(2):
        held.took(SEND, {"by_shots": {"gains": 0}}, THROW_BY_HOLDING)
        assert held.holding()                                        # it paid in its latest stretches
    held.took(SEND, {"by_shots": {"gains": 0}}, THROW_BY_HOLDING)
    assert not held.holding()                                        # three running, nothing: let go
    assert held.choose(asked, THROW_BY_HOLDING) == AS_IT_HAPPENS     # the next way the words allow
    held.took(AS_IT_HAPPENS, {}, THROW_BY_HOLDING)
    held.took(AS_IT_HAPPENS, {}, THROW_BY_HOLDING)
    held.took(AS_IT_HAPPENS, {}, THROW_BY_HOLDING)
    assert held.choose(asked, THROW_BY_HOLDING) == ""                # both had their turn: the pursuit looks round
    assert held.choose(asked, THROW_BY_HOLDING + " Level 2: the basket moves!") == SEND   # the place says more


def test_holding_her_own_steering_what_is_hers_pays_where_nothing_is_counted():
    held = HeldTo()
    held.choose([AS_IT_HAPPENS], "arrows")
    for _ in range(4):
        held.took(AS_IT_HAPPENS, {"hers": "red thing", "seconds": 30.0, "gains": 0, "losses": 0}, "arrows")
    assert held.holding()


class _Reflexes:
    """The reflexes' part the look drives: it plays each time it is asked, and the run ends on the fourth look."""

    def __init__(self, *, paying: bool):
        from core.agency.the_way_it_is_played import HeldTo

        self.played_at, self.plays, self.looks = 0.0, 0, 0
        self.held = HeldTo(way=SEND, stretches=[] if paying else [{}, {}, {}])
        self.ends_at = float("inf")
        self.way_on_shown = False

    async def while_it_moves(self):
        import time

        self.plays += 1
        self.played_at = time.monotonic()

    def read(self, _seen):
        self.looks += 1

    async def read_a_legend(self):
        return None

    def run_is_over(self, _seen):
        return self.looks >= 4

    def held_to(self):
        return self.held

    def goes_on_playing(self, observation, played_before):
        from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

        return PlayingAsItHappens.goes_on_playing(self, observation, played_before)


@pytest.mark.parametrize("paying", [True, False])
def test_while_a_way_pays_she_plays_on_through_the_looks_and_the_pursuit_gets_the_screen_when_the_run_is_over(paying):
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS as REFLEXES
    from core.skills.screen_pursuit_as_it_happens import looked_at_as_it_happens

    reflexes = _Reflexes(paying=paying)

    async def look():
        return {"ok": True}

    async def go():
        token = REFLEXES.set(reflexes)
        try:
            return await looked_at_as_it_happens(look)
        finally:
            REFLEXES.reset(token)

    asyncio.run(go())
    assert reflexes.plays == (4 if paying else 1), reflexes.plays


@pytest.mark.parametrize(("said", "rules"), [
    ("HOW TO USE THE PUTTER: 1. Click on your ball and hold down the mouse button.", True),
    ("INSTRUCTIONS CANDY Gives you invincibility HEART Restores your health", True),
    ("SCORE 1200 LIVES 3", False), ("Hole 3 Par 2 Strokes 1", False),
])
def test_a_screen_that_teaches_how_to_play_is_gone_on_from_not_played_on(said, rules):
    """LIVE 2026-10-09 a putter lesson drew its "next" in letters she could not read, and she sent shots at it."""
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens, reads_as_rules

    assert reads_as_rules(said) is rules
    reflexes = PlayingAsItHappens(page=None, band=(0.0, 0.0, 1.0, 1.0), goal="play", ends_at=0.0)
    reflexes.words = [said]
    assert reflexes._a_menu_first() is rules


@pytest.mark.parametrize(("said", "no_shots"), [
    ("HOW TO USE THE PUTTER: 1. Click on your ball and hold down the mouse button.", True),
    ("Please Select Number Of Players", True), ("Choose your character", True),
    ("LEVEL 1. Paint 502 of the ground to continue", False), ("Hole 1 Par 3 Strokes 0", False),
])
def test_nothing_is_sent_into_a_screen_that_teaches_or_asks_her_to_choose(said, no_shots):
    """LIVE 2026-10-09 a shot on a putter lesson pressed its "back", and she went round title, welcome and lesson for a
    round without reaching the course."""
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

    reflexes = PlayingAsItHappens(page=None, band=(0.0, 0.0, 1.0, 1.0), goal="play", ends_at=0.0)
    reflexes.words = [said]
    assert reflexes._no_place_for_shots() is no_shots


def test_a_way_on_read_before_is_looked_for_where_it_was_on_a_lesson_whose_own_cannot_be_read():
    """LIVE 2026-10-09 a welcome had "next" she could read; the putter lesson after it drew the same "back" and "next" in
    letters of which she read only "back", and she went back and forth between the two for a round."""
    from core.agency.what_i_can_do_here import a_click_on
    from core.skills.screen_pursuit_bearings import where_to_click
    from core.skills.screen_pursuit_looking import _where_the_way_on_was

    paced: dict = {}
    welcome = {"layout": [{"text": "next", "center_x": 0.86, "center_y": 0.9}, {"text": "back", "center_x": 0.72, "center_y": 0.9}]}
    _where_the_way_on_was(welcome, (a_click_on("next"), a_click_on("back")), "Welcome to the island", paced)
    lesson = {"layout": [{"text": "back", "center_x": 0.72, "center_y": 0.9}]}
    offered = _where_the_way_on_was(lesson, (a_click_on("back"),), "HOW TO USE THE PUTTER: 1. Click on your ball", paced)
    assert a_click_on("next") in offered and where_to_click(lesson, "next") == (0.86, 0.9)
    play = {"layout": [{"text": "SCORE 120", "center_x": 0.1, "center_y": 0.05}, {"text": "MENU", "center_x": 0.9, "center_y": 0.05}]}
    assert a_click_on("next") not in _where_the_way_on_was(play, (a_click_on("MENU"),), "SCORE 120 MENU", paced)
