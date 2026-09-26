"""Find the thing, face it, walk while it grows, and do what the screen says.

A small world stands in for a game: things at bearings and distances, a
camera with a field of view, a mouse that turns it and a key that walks.
What is on screen reaches her the way her eyes report it, as text with a box.
She is told none of the numbers: the turn is measured from one test move.
"""

from __future__ import annotations

import math

from core.agency.going_to_what_she_sees import GoingTo, a_cue_to_press, seen_named
from core.agency.what_hands_do import Chunk

FIELD = 90.0      # degrees across the view
TURN = 0.2        # degrees of turn per point of mouse travel
PACE = 2.0        # distance walked per second held
REACH = 1.5


class World:
    def __init__(self, things, *, walls: bool = False) -> None:
        self.things = things  # name -> (x, y)
        self.x, self.y, self.facing = 0.0, 0.0, 0.0  # facing 0 is along +y
        self.walls = walls
        self.pressed: list[str] = []

    def _bearing(self, x, y) -> float:
        return math.degrees(math.atan2(x - self.x, y - self.y)) - self.facing

    def layout(self) -> list[dict]:
        seen = []
        for name, (x, y) in self.things.items():
            off = self._bearing(x, y)
            far = math.hypot(x - self.x, y - self.y)
            if abs(off) < FIELD / 2 and far > 0.01:
                high = min(0.9, 0.5 / far)
                seen.append({"text": name, "center_x": 0.5 + off / FIELD, "center_y": 0.5,
                             "width": high * 0.6, "height": high})
                if far < REACH and abs(off) < 10:
                    seen.append({"text": "Press E to open", "center_x": 0.5, "center_y": 0.8,
                                 "width": 0.2, "height": 0.03})
        return seen

    def play(self, chunk: Chunk) -> None:
        for slot in chunk.slots:
            self.facing += TURN * slot.moved[0]
            if "w" in slot.held and not self.walls:
                step = PACE * chunk.slot_s
                self.x += step * math.sin(math.radians(self.facing))
                self.y += step * math.cos(math.radians(self.facing))
            self.pressed.extend(sorted(slot.held - {"w"}))


def _a_turn_measured_in(world: World, named: str):
    """One test move of the mouse, and how far the named thing slid for it."""
    before = seen_named(world.layout(), named)[0].across
    world.facing += TURN * 10
    after = seen_named(world.layout(), named)[0].across
    world.facing -= TURN * 10
    per_point = (after - before) / 10.0
    return lambda share: int(round(share / per_point))


def _go(world: World, named: str, most: int = 60) -> GoingTo:
    trip = GoingTo(named, turn_for=_a_turn_measured_in(world, named), walks="w")
    for _ in range(most):
        chunk = trip.next_chunk(world.layout(), slot_s=0.1, slots=5)
        world.play(chunk)
        if chunk.done or chunk.think:
            break
    return trip


def test_she_turns_to_it_walks_to_it_and_does_what_the_screen_says():
    world = World({"door": (6.0, 8.0)})
    trip = _go(world, "door")
    assert trip.ended == "the screen said to press e"
    assert world.pressed == ["e"]
    assert math.hypot(6.0 - world.x, 8.0 - world.y) < REACH


def test_two_things_that_answer_to_the_name_are_the_persons_to_choose_between():
    world = World({"red door": (-3.0, 8.0), "blue door": (3.0, 8.0)})
    trip = _go(world, "door")
    assert "2 things answer to 'door'" in trip.ended
    assert "blue door" in trip.ended and "red door" in trip.ended


def test_what_is_not_on_screen_is_not_walked_toward():
    trip = GoingTo("chest", turn_for=lambda share: 0, walks="w")
    chunk = trip.next_chunk([{"text": "door", "center_x": 0.5, "width": 0.1, "height": 0.1}],
                            slot_s=0.1, slots=5)
    assert chunk.think and not chunk.slots and "nothing on screen" in trip.ended


def test_a_thing_that_stops_getting_nearer_ends_the_trip():
    world = World({"door": (0.0, 8.0)}, walls=True)
    trip = _go(world, "door")
    assert trip.ended == "'door' stopped getting nearer"


def test_a_cue_to_hold_is_held():
    layout = [{"text": "Hold [F] to absorb the goo", "center_x": 0.5}]
    assert a_cue_to_press(layout) == ("f", True)
    chunk = GoingTo("goo", turn_for=lambda share: 0, walks="w").next_chunk(layout, slot_s=0.1, slots=4)
    assert chunk.done and [slot.held for slot in chunk.slots] == [frozenset({"f"})] * 4


def test_words_that_only_look_like_a_cue_are_not_one():
    assert a_cue_to_press([{"text": "Impress your friends"}]) is None
    assert a_cue_to_press([{"text": "Press Cmd to quit"}]) is None


def test_a_prompt_that_belongs_to_something_else_is_not_answered():
    """Measured in generated worlds: another thing stood between her and the
    chest, its prompt came up as she passed, and she used the altar."""
    layout = [
        {"text": "Altar", "center_x": 0.5, "width": 0.3, "height": 0.4},
        {"text": "Chest", "center_x": 0.45, "width": 0.1, "height": 0.1},
        {"text": "Press F to use the altar", "center_x": 0.5, "width": 0.3, "height": 0.03},
    ]
    chunk = GoingTo("chest", turn_for=lambda share: 0, walks="w").next_chunk(layout, slot_s=0.1, slots=3)
    assert not chunk.done and all("f" not in slot.held for slot in chunk.slots)


def test_a_prompt_that_names_nothing_is_hers_only_when_her_thing_is_in_front():
    ahead = [{"text": "Chest", "center_x": 0.5, "width": 0.2, "height": 0.2},
             {"text": "Press E", "center_x": 0.5, "width": 0.1, "height": 0.03}]
    aside = [{"text": "Chest", "center_x": 0.8, "width": 0.1, "height": 0.1},
             {"text": "Press E", "center_x": 0.5, "width": 0.1, "height": 0.03}]
    walker = GoingTo("chest", turn_for=lambda share: 7, walks="w")
    assert walker.next_chunk(ahead, slot_s=0.1, slots=1).done
    turned = GoingTo("chest", turn_for=lambda share: 7, walks="w").next_chunk(aside, slot_s=0.1, slots=1)
    assert not turned.done and turned.slots[0].moved == (7, 0)

