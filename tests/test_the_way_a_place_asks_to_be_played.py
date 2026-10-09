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
