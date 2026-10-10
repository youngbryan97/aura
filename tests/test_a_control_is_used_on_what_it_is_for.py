"""A control is used on the thing it is for when she is beside it, and what came of it is kept.

"Press the SPACE BAR to open doors": beside a door her eyes have seen, she presses space; a door she is far from is
not pressed at. What a press did is judged: the door gone, and it did something; nothing changed twice, and that key
is no longer used on that kind. Nothing here is any one game.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.using_a_control_on_a_thing import (
    JUDGED_AFTER_S,
    UsingOnThings,
    a_name_for,
    a_place_to_go,
    use_controls_on_things,
)
from core.cognition.a_guide_to_a_place import THE_GUIDE, TOLD, Guide

pytestmark = pytest.mark.unit


def _thing(number: int, kind: int, x: float, y: float, w: float = 20.0, h: float = 30.0) -> SimpleNamespace:
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, w=w, h=h)


class _Hands:
    def __init__(self) -> None:
        self.done: list[tuple[str, object]] = []

    async def tap(self, key: str) -> None:
        self.done.append(("tap", key))

    async def down(self, key: str) -> None:
        self.done.append(("down", key))

    async def click(self, x: float, y: float) -> None:
        self.done.append(("click", (round(x, 2), round(y, 2))))


def _world(things: list[SimpleNamespace]) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    moves = SimpleNamespace(things={t.number: t for t in things}, share=lambda x, y: (x / 400, y / 300), shape=(300, 400))
    hers = SimpleNamespace(thing=lambda m: m.things.get(1))
    run = SimpleNamespace(using=UsingOnThings(), held="", letting_go={}, input_key_downs=Counter(), last_click=0.0)
    return moves, hers, run


def _guide(*said: str) -> Guide:
    guide = Guide(place="a place")
    guide.take_in(TOLD, list(said))
    guide.seen.saw(1, "boy")
    guide.seen.saw(5, "wooden door")
    guide.seen.saw(6, "ladder")
    return guide


@pytest.mark.asyncio
async def test_beside_a_door_she_presses_the_key_that_opens_doors_and_sees_it_did_something():
    guide = _guide("Press the SPACE BAR to open doors, revealing secret passageways.")
    token = THE_GUIDE.set(guide)
    try:
        hands, said = _Hands(), []
        moves, hers, run = _world([_thing(1, 1, 100.0, 200.0), _thing(2, 5, 128.0, 200.0), _thing(3, 5, 380.0, 200.0)])
        assert await use_controls_on_things(hands, run, moves, hers, [], 1.0, lambda line, once: said.append(line))
        assert hands.done == [("down", "space")] and "space" in run.letting_go     # held a moment, as a person's is
        assert said and "wooden door next to me" in said[0] and "space is to open doors" in said[0].lower()
        del moves.things[2]                                       # the door opened and is gone
        await use_controls_on_things(hands, run, moves, hers, [], 1.0 + JUDGED_AFTER_S, lambda line, once: said.append(line))
        assert ("space", "wooden door") in run.using.worked and "did something" in said[-1]
        # The far door is not pressed at from where she stands.
        assert hands.done == [("down", "space")]
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_a_key_that_does_nothing_twice_at_a_kind_is_not_used_on_it_again():
    guide = _guide("Press RETURN to open doors.")
    token = THE_GUIDE.set(guide)
    try:
        hands = _Hands()
        moves, hers, run = _world([_thing(1, 1, 100.0, 200.0), _thing(2, 5, 112.0, 200.0)])
        at = 0.0
        for _ in range(6):
            await use_controls_on_things(hands, run, moves, hers, [], at)
            run.letting_go.clear()                                            # let go, as play lets go
            at += 3.0
        assert run.using.nothing[("return", "wooden door")] == 2
        assert hands.done.count(("down", "return")) == 2
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_a_climb_is_held_and_a_click_named_for_things_goes_to_the_thing():
    guide = _guide("Press the up arrow to climb ladders.", "Click on the doors to open them.")
    token = THE_GUIDE.set(guide)
    try:
        hands = _Hands()
        moves, hers, run = _world([_thing(1, 1, 100.0, 200.0), _thing(4, 6, 90.0, 190.0, 16.0, 80.0)])
        assert await use_controls_on_things(hands, run, moves, hers, [], 1.0)
        assert hands.done == [("down", "up")] and "up" in run.letting_go
        moves2, hers2, run2 = _world([_thing(2, 5, 200.0, 150.0)])
        assert await use_controls_on_things(hands, run2, moves2, hers2, [], 1.0)
        assert hands.done[-1] == ("click", (0.5, 0.5))
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize(("on", "seen", "bears", "usually", "act", "someone", "used"), [
    ("doors", "red door", "", "", "open", False, True),
    ("boxes", "wooden box", "", "", "push", False, True),
    ("people", "old man", "", "", "talk", True, True),
    ("items", "golden key", "to get", "", "pick up", False, True),
    ("enemies", "zombie", "danger", "", "punch", False, True),
    ("enemies", "coin", "to get", "", "punch", False, False),
    ("", "rope ladder", "to use", "a ladder is climbed to reach somewhere higher", "climb", False, True),
    ("", "rock", "scenery", "a rock is scenery", "climb", False, False),
])
def test_what_a_control_is_used_on_is_known_by_name_or_by_what_such_things_are(on, seen, bears, usually, act, someone, used):
    assert a_name_for(on, seen, bears, usually, act, someone) is used


def test_her_eyes_answer_is_read_as_the_things_standing_in_the_place():
    from core.perception.what_is_in_a_place import landmarks_from

    said = landmarks_from('```json\n{"things": [{"what": "wooden door", "bbox_2d": [250, 600, 310, 880]}, '
                          '{"what": "Score", "bbox_2d": [10, 10, 1000, 1000]}, {"what": "", "bbox_2d": [1, 1, 5, 5]}, '
                          '{"what": "ladder", "bbox_2d": [700, 300, 740, 900]}]}\n```')
    assert [s.what for s in said] == ["wooden door", "ladder"]          # the whole picture and the unnamed, left out
    assert said[0].box == pytest.approx((0.25, 0.6, 0.31, 0.88))
    assert [s.what for s in landmarks_from('[{"label": "chest", "bbox_2d": [100, 100, 200, 200]}]')] == ["chest"]


@pytest.mark.asyncio
async def test_a_door_standing_in_the_place_is_gone_to_used_and_seen_to_have_opened():
    """A door is part of the backdrop to play; her eyes saw it, she goes to it, presses space, and the picture there
    changed: it did something."""
    from core.perception.what_is_in_a_place import Landmark

    guide = _guide("Press the SPACE BAR to open doors.")
    token = THE_GUIDE.set(guide)
    try:
        hands, said = _Hands(), []
        moves, hers, run = _world([_thing(1, 1, 60.0, 250.0)])
        run.using.landmarks, run.using.surveyed = [Landmark("door", (0.5, 0.7, 0.56, 0.9))], 0   # this screen's look, had
        where = a_place_to_go(run, moves, hers, 1.0, lambda line, once: said.append(line))
        assert where == pytest.approx((212.0, 240.0)) and said == ["I'll go over to the door and press space."]
        picture = np.full((300, 400, 3), 200, np.uint8)
        picture[210:270, 200:224] = (139, 90, 43)                           # the door
        assert not await use_controls_on_things(hands, run, moves, hers, [], 2.0, picture=picture)   # not beside it yet
        moves.things[1] = _thing(1, 1, 190.0, 250.0)
        assert await use_controls_on_things(hands, run, moves, hers, [], 3.0, picture=picture)
        assert hands.done == [("down", "space")]
        picture[210:270, 200:224] = 200                                     # opened: gone from the picture
        await use_controls_on_things(hands, run, moves, hers, [], 3.0 + JUDGED_AFTER_S, picture=picture)
        assert ("space", "door") in run.using.worked and run.using.landmarks == []
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_after_a_press_that_did_nothing_she_comes_right_up_to_the_thing_before_pressing_again():
    guide = _guide("Press the SPACE BAR to open doors.")
    token = THE_GUIDE.set(guide)
    try:
        hands = _Hands()
        moves, hers, run = _world([_thing(1, 1, 100.0, 200.0), _thing(2, 5, 128.0, 200.0)])
        assert await use_controls_on_things(hands, run, moves, hers, [], 0.0)
        run.letting_go.clear()
        await use_controls_on_things(hands, run, moves, hers, [], JUDGED_AFTER_S)
        assert run.using.nothing[("space", "wooden door")] == 1
        assert not await use_controls_on_things(hands, run, moves, hers, [], 5.0)     # beside, but not up to it
        moves.things[1] = _thing(1, 1, 120.0, 200.0)
        assert await use_controls_on_things(hands, run, moves, hers, [], 6.0)
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_controls_for_the_game_itself_or_for_moving_are_not_used_on_things():
    """"P to pause", "Space to jump", "Space to fire": none is done to the thing beside her."""
    guide = _guide("Press P to pause.", "Press space to jump.", "Press Z to fire at the doors.")
    token = THE_GUIDE.set(guide)
    try:
        hands = _Hands()
        moves, hers, run = _world([_thing(1, 1, 100.0, 200.0), _thing(2, 5, 112.0, 200.0)])
        assert not await use_controls_on_things(hands, run, moves, hers, [], 1.0)
        assert hands.done == []
    finally:
        THE_GUIDE.reset(token)
