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


def test_a_key_is_pressed_by_the_browsers_name_for_it_and_one_it_does_not_know_does_not_end_the_game():
    """LIVE 2026-10-09 a name typed into a game's scorecard was cleared with "backspace", which the browser calls
    "Backspace", and the error ended the whole game."""
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

    pressed = []

    class _Keyboard:
        async def press(self, key):
            if key not in {"Backspace", "ArrowUp", "x", "F2"}:
                raise RuntimeError(f'Keyboard.press: Unknown key: "{key}"')
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

    reflexes = PlayingAsItHappens(page=_Page(), band=(0.0, 0.0, 1.0, 1.0), goal="play", ends_at=0.0)
    reflexes._focused = True
    for key in ("backspace", "up", "x", "f2", "no such key"):
        asyncio.run(reflexes.tap(key))
    assert pressed == ["Backspace", "ArrowUp", "x", "F2"]


def test_inside_a_thing_handed_over_its_own_dialog_is_gone_on_through_not_the_end():
    """LIVE 2026-10-09 a golf game's name box, typed into, was taken for an overlay that would not go, and ended it."""
    from types import SimpleNamespace

    from core.skills.screen_pursuit import MAX_BLOCKER_ATTEMPTS
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS as REFLEXES
    from core.skills.screen_pursuit_decision_branches import (
        _FALL_THROUGH,
        _decide_the_next_move_blocker,
    )

    async def blocker(_observation):
        return SimpleNamespace(name="dismiss the dialog")

    def decide():
        attempts = {"count": MAX_BLOCKER_ATTEMPTS, "last": "", "dismissed": 0}
        no_move = {"because": ""}
        made = asyncio.run(_decide_the_next_move_blocker(attempts, blocker, {"reason": ""}, no_move, {"ok": True}))
        return made, no_move["because"]

    assert decide() == (None, "something is in front of it that will not move")    # a page's own overlay: stopped
    token = REFLEXES.set(SimpleNamespace())
    try:
        assert decide() == (_FALL_THROUGH, "")                                       # a game's own dialog: gone on
    finally:
        REFLEXES.reset(token)


def test_a_way_on_is_looked_for_where_it_was_on_a_screen_that_asks_her_to_choose():
    """LIVE 2026-10-09 a choice of players drew its "next" faded until a choice was made, and she never read it."""
    from core.agency.what_i_can_do_here import a_click_on
    from core.skills.screen_pursuit_looking import _where_the_way_on_was

    paced: dict = {}
    welcome = {"layout": [{"text": "next", "center_x": 0.86, "center_y": 0.72}]}
    _where_the_way_on_was(welcome, (a_click_on("next"),), "Welcome to the island", paced)
    choose = {"layout": [{"text": "1 Player", "center_x": 0.4, "center_y": 0.55}, {"text": "back", "center_x": 0.7, "center_y": 0.72}]}
    offered = _where_the_way_on_was(choose, (a_click_on("1 Player"), a_click_on("back")), "Please Select Number Of Players", paced)
    assert a_click_on("next") in offered


def test_a_heading_read_as_a_label_still_says_the_screen_asks_her_to_choose():
    """LIVE 2026-10-09 "Please Select Number Of Players" was read as a thing to click, not as the screen's words."""
    from core.agency.what_i_can_do_here import a_click_on
    from core.skills.screen_pursuit_looking import _where_the_way_on_was

    paced: dict = {}
    _where_the_way_on_was({"layout": [{"text": "next", "center_x": 0.86, "center_y": 0.72}]}, (a_click_on("next"),),
                          "Welcome to the island", paced)
    choose = {"layout": [{"text": "Please Select Number Of Players", "center_x": 0.4, "center_y": 0.25}]}
    moves = (a_click_on("Please Select Number Of Players"), a_click_on("1 Player"), a_click_on("back"))
    assert a_click_on("next") in _where_the_way_on_was(choose, moves, "", paced)


def test_once_a_choice_is_made_on_a_screen_that_asks_for_one_what_goes_on_is_wanted_and_the_other_choices_are_not():
    """LIVE 2026-10-09 a choice of players drew its "next" faded until a player was chosen; she clicked "next" first,
    it did nothing, and after choosing she went on clicking the choices and the shapes for three minutes."""
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
    from core.skills.screen_pursuit_decision import _once_chosen_go_on

    here = WhatWorksHere()
    moves = (a_click_on("1 Player"), a_click_on("2 Players"), a_click_on("next"))
    here.asked_for_by("Please Select Number Of Players 1 Player 2 Players next", moves)
    here.looked_at(moves)
    here.looked_at(moves)
    exploring = {a_click_on("2 Players"): 0.09, a_click_on("the shape at 40% across, 45% down"): 0.06}
    assert _once_chosen_go_on(here, exploring) == exploring             # nothing chosen yet: as it was
    here.tried(a_click_on("next"), changed=False)
    here.tried(a_click_on("1 Player"), changed=True)
    after = _once_chosen_go_on(here, exploring)
    assert after[a_click_on("next")] == 1.0 and max(v for k, v in after.items() if k != a_click_on("next")) < 0.01
    here.asked_for_by("Choose A Ball Pick your favorite color", moves)  # the next screen: chosen afresh
    assert not here.chose_here


@pytest.mark.parametrize("said", ["USE WASD TO MOVE", "Use WASD to move, mouse to aim", "use the WASD keys to move"])
def test_a_cluster_of_letters_named_as_one_word_is_the_keys_it_names(said):
    """LIVE 2026-10-09 "USE WASD TO MOVE" was read as no keys at all, and she played a game steered by keys with the
    pointer for three minutes."""
    from core.agency.playing_as_it_happens import controls_named_in

    assert controls_named_in(said, keys_without_words=(), during_play=True)[0][:4] == ["w", "a", "s", "d"]


def test_a_pause_is_let_go_by_what_made_it_and_what_made_it_is_not_taken_again_in_play():
    """LIVE 2026-10-09 trying a game's corner button paused it; on "PAUSED" she clicked the word, the middle of the
    picture and every shape but that button, and the game stood paused until its time ran out."""
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
    from core.language.a_way_on import how_much_it_leads_on, says_it_is_paused
    from core.skills.screen_pursuit_decision import _go_on_from_a_pause

    corner, word = a_click_on("the shape at 5% across, 5% down"), a_click_on("PAUSED")
    here = WhatWorksHere()
    here.asked_for_by("SCORE: 0 USE WASD TO MOVE CHARGING", (corner,))
    here.asked_for_by("SCORE: 0 PAUSED LEVEL SELECT", (corner, word))
    here.tried(corner, changed=True)
    assert here.paused_here and here.paused_by == corner
    paused = _go_on_from_a_pause(here, {word: 0.8, corner: 0.1, a_click_on("LEVEL SELECT"): 0.3})
    assert max(paused, key=paused.get) == corner
    here.asked_for_by("SCORE: 0 USE WASD TO MOVE CHARGING", (corner,))
    here.tried(corner, changed=True)
    playing = _go_on_from_a_pause(here, {corner: 0.8, a_click_on("the middle of the picture"): 0.5})
    assert playing[corner] < 0.1
    assert says_it_is_paused("GAME PAUSED") and not says_it_is_paused("Pause the music with P")
    assert how_much_it_leads_on("Resume") > 1.0


def test_a_pause_she_did_not_make_is_let_go_by_the_keys_a_pause_is_usually_let_go_by():
    from core.agency.what_i_can_do_here import WhatWorksHere

    here = WhatWorksHere()
    here.asked_for_by("PAUSED click to continue", ())
    assert {"p", "escape"} <= set(here.asked_for)
