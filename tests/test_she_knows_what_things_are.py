"""She sees things as what they are, and knows what such things usually mean: a zombie is probably trouble, a coin is
probably worth getting, a bat is probably swung; and she wonders at what looks unlike its kind.

Her eyes and her model are stood in for here; what is tested is what she makes of what they say. Nothing here is any
one game.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.cognition.a_guide_to_a_place import THE_GUIDE, TOLD, Guide

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _kept_apart(tmp_path, monkeypatch):
    import core.runtime.what_she_learned as learned

    monkeypatch.setattr(learned, "_KEPT_IN", tmp_path)


def _thing(number, kind, x, y, w=20.0, h=20.0, moved=True):
    def box():
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, w=w, h=h, moved=moved, vx=0.0, vy=0.0, box=box)


def _moves(things, colours):
    return SimpleNamespace(things={t.number: t for t in things}, kinds=[SimpleNamespace(colour=c) for c in colours],
                           shape=(200, 300), scale=1.0, share=lambda x, y: (x / 300, y / 200))


def test_what_her_eyes_say_is_read_plainly_and_nothing_odd_is_nothing():
    from core.perception.what_things_look_like import sighting_from

    seen = sighting_from('Sure: {"is": "A Zombie", "odd": "No"}')
    assert seen is not None and seen.what == "zombie" and seen.odd == ""
    assert sighting_from('{"is": "unclear", "odd": ""}') is None
    assert sighting_from('{"is": "orange", "odd": "it has a face and eyes"}').odd == "has a face and eyes"


def test_her_eyes_are_asked_of_each_kind_once_hers_first_and_never_of_a_speck_or_the_backdrop():
    from core.perception.what_things_look_like import LookingAtThings

    asked: list[str] = []

    async def see(prompt, image):
        asked.append(image)
        return {1: '{"is": "red car"}', 2: '{"is": "zombie", "odd": "it is green and glowing"}'}[len(asked)] \
            if len(asked) <= 2 else '{"is": "unclear"}'

    async def play():
        looking = LookingAtThings(see=see)
        moves = _moves([_thing(5, 0, 50, 50), _thing(6, 1, 150, 80, w=30, h=30), _thing(7, 2, 10, 10, w=2, h=2),
                        _thing(8, 3, 150, 100, w=290, h=190, moved=False)], [(200, 30, 30), (60, 180, 70),
                                                                            (0, 0, 0), (40, 40, 40)])
        picture = np.zeros((200, 300, 3), dtype=np.uint8)
        assert not looking.look(picture, moves, 0.0, mine=5)          # play has only begun
        assert looking.look(picture, moves, 4.0, mine=5)
        await looking._asking
        assert not looking.look(picture, moves, 5.0, mine=5)          # not again so soon, and nothing new to ask of
        return looking

    looking = asyncio.run(play())
    assert [k for k, _s in looking.newly()] == [0, 1]                 # hers first, then the other; no speck, no backdrop
    assert looking.by_kind[0].what == "red car" and looking.by_kind[1].odd == "is green and glowing"
    assert len(asked) == 2


def test_what_a_kind_of_thing_usually_means_is_asked_once_and_kept_for_every_place():
    from pydantic import BaseModel

    from core.cognition.what_things_are import WhatThingsAre, first_guess, what_things_are

    assert first_guess("spikes").bears == "danger"
    asked: list[str] = []

    async def ask(prompt, schema, most):
        asked.append(prompt)
        assert issubclass(schema, BaseModel)
        return schema(things=[{"name": "zombie", "usually": "a walking corpse that attacks people", "bears": "danger",
                               "thought": "Zombies are usually hostile; best kept clear of, or fought."},
                              {"name": "baseball bat", "usually": "swung to hit a ball, or anything", "bears": "to use",
                               "thought": "Maybe something to swing, to defend myself."}])

    async def asking():
        store = what_things_are()
        assert store.ask_about(["zombie", "baseball bat"], ask)
        assert not store.ask_about(["zombie"], ask)                   # already being asked
        for _ in range(5):
            await asyncio.sleep(0)
        return store

    store = asyncio.run(asking())
    assert store.of("zombie").bears == "danger" and store.of("a green zombie").bears == "danger"
    assert store.of("old baseball bat").bears == "to use"            # known by its last word too
    assert len(asked) == 1
    from core.runtime.what_she_learned import named, recall

    again = WhatThingsAre()
    again.take_in(recall(named("what holds everywhere", "what things are")))
    assert again.of("zombie").thought.startswith("Zombies are usually hostile")


def test_the_names_a_place_gives_say_who_she_plays_what_to_flee_and_whom_to_save():
    guide = Guide(place="Buddy's Big Adventure")
    guide.take_in(TOLD, ["LEVEL 1 Park the red car.", "Escape the zombies!", "Rescue the princess before dawn."])
    assert guide.names["me"] == ["red car", "buddy"]                  # what the words say she drives, then the title's
    assert guide.names["avoid"] == ["zombies"] and guide.names["friend"] == ["princess"]


def test_the_place_s_characters_as_her_model_knows_them_are_who_she_sees():
    guide = Guide(place="a place")
    filled = guide.take_in_what_i_know({"who": [{"name": "Cow", "what": "a big-hearted cow, Chicken's sister",
                                                  "looks": "a purple cow", "side": "with you"},
                                                 {"name": "Red Guy", "what": "a red devil who schemes against them",
                                                  "looks": "a red man with no trousers", "side": "against you"}]})
    assert "who" in filled
    assert guide.seen.saw(3, "cow in a tutu") == "Cow"
    assert guide.seen.as_seen(3) == "Cow"
    assert "Who's who, from what I know: Cow" in guide.for_thinking()


def test_what_she_sees_a_thing_as_names_it_sets_what_she_does_about_it_and_what_looks_odd_she_wonders_at():
    from core.agency.naming_what_she_sees import SeeingInPlay, describe, seen_in_play
    from core.agency.what_meeting_things_does import AVOID, MEET, WhatMeetingDoes
    from core.perception.what_things_look_like import Sighting

    async def ask(prompt, schema, most):
        if "things" in schema.model_fields:
            return schema(things=[{"name": "zombie", "usually": "a walking corpse that attacks people",
                                   "bears": "danger", "thought": "zombies are usually hostile; best kept clear of"},
                                  {"name": "coin", "usually": "money", "bears": "to get", "thought": ""}])
        return schema(maybe="maybe it has been made into one of them")

    class Eyes:
        def __init__(self):
            self.given = [(1, Sighting("zombie", "it wears a crown")), (2, Sighting("coin"))]

        def look(self, *args, **kwargs):
            return True

        def newly(self):
            given, self.given = self.given, []
            return given

    guide = Guide(place="a place")
    token = THE_GUIDE.set(guide)
    try:
        said: list[str] = []
        meeting = WhatMeetingDoes()
        run = SimpleNamespace(seeing=SeeingInPlay(looking=Eyes()))
        hers = SimpleNamespace(number=5, kind=0)
        moves = _moves([_thing(5, 0, 50, 50), _thing(6, 1, 150, 80), _thing(7, 2, 200, 120)],
                       [(200, 30, 30), (60, 180, 70), (240, 210, 50)])
        picture = np.zeros((200, 300, 3), dtype=np.uint8)

        async def play():
            for step in range(4):
                seen_in_play(run, moves, hers, meeting, picture, 10.0 + step * 7, lambda line, once: said.append(line),
                             ask=ask)
                for _ in range(5):
                    await asyncio.sleep(0)

        asyncio.run(play())
        assert meeting.supposed == {1: AVOID, 2: MEET}
        assert meeting.stance(1) == AVOID                              # until meeting one says otherwise
        assert said and said[0].startswith("That looks like a zombie. Zombies are usually hostile")
        assert describe(moves, 1) == "green zombie" and describe(moves, 2) == "yellow coin"
        wondered = [n.said for n in guide.notes.notes.values() if n.kind == "wondered"]
        assert any("wears a crown" in w and "made into one of them" in w for w in wondered)
        assert "What I see here: zombie (danger: a walking corpse" in guide.for_thinking()
    finally:
        THE_GUIDE.reset(token)


def test_what_meeting_a_thing_shows_outranks_what_she_supposed_of_it():
    from core.agency.what_meeting_things_does import AVOID, MEET, WhatMeetingDoes

    meeting = WhatMeetingDoes()
    meeting.supposed[4] = AVOID
    assert meeting.stance(4) == AVOID
    kept = meeting.evidence[4]
    kept.touch_sum, kept.touches_settled, kept.touched = 2.0, 2, 2       # met twice, and it paid both times
    assert meeting.stance(4) == MEET


def test_what_goes_where_the_mouse_goes_is_her_pointer_where_her_eyes_see_a_cursor_and_herself_otherwise():
    """LIVE 2026-10-10 a game's own cursor followed the mouse and she said "That's me: the white thing"."""
    from core.agency.naming_what_she_sees import i_go_where_the_mouse_goes

    guide = Guide(place="a place")
    token = THE_GUIDE.set(guide)
    try:
        moves = _moves([_thing(5, 0, 150, 100), _thing(6, 1, 40, 40)], [(240, 240, 240), (240, 140, 30)])
        guide.seen.saw(0, "white arrow cursor")
        assert i_go_where_the_mouse_goes(moves, 0, moves.things[5]).startswith(
            "The white arrow cursor at the middle is my pointer")
        guide.seen.saw(1, "orange with a face")
        assert i_go_where_the_mouse_goes(moves, 1, moves.things[6]).startswith("That's me: the orange with a face")
    finally:
        THE_GUIDE.reset(token)


def test_a_busy_model_is_asked_again_after_a_wait_rather_than_left_unasked(monkeypatch):
    """LIVE 2026-10-10 her model was busy with her thinking at each move, and what things are came back empty."""
    import core.cognition.what_things_are as things

    monkeypatch.setattr(things, "WAITS_S", (0.0, 0.0))
    tries: list[int] = []

    async def ask(prompt, schema, most):
        tries.append(1)
        if len(tries) < 3:
            raise RuntimeError("an empty answer")
        return "answered"

    assert asyncio.run(things.asked_patiently(ask, "?", None, 10, matters=0)) == "answered" and len(tries) == 3


def test_a_name_that_says_what_anything_is_and_an_odd_look_that_is_only_how_it_is_drawn_are_not_kept():
    """LIVE 2026-10-10 her eyes called things "object" and "character", and said a bare "yes" or "cartoonish" for odd."""
    from core.perception.what_things_look_like import sighting_from

    assert sighting_from('{"is": "object", "odd": ""}') is None
    assert sighting_from('{"is": "emoji", "odd": "yes"}').odd == ""
    assert sighting_from('{"is": "barrel", "odd": "a cartoonish, stylized object in a game setting"}').odd == ""
    assert sighting_from('{"is": "apple", "odd": "has a cartoon face with wide eyes"}').odd.startswith("has a cartoon face")


def test_who_she_is_is_said_once_play_has_held_it_and_not_for_a_cursor():
    """LIVE 2026-10-10 play's guess went back and forth, and each change was said."""
    from core.agency.naming_what_she_sees import HERS_HELD_S, SeeingInPlay, _read_the_place

    guide = Guide(place="a place")
    said: list[str] = []
    calls: list[int] = []
    import core.cognition.what_this_place_is as place

    original = place.played_shows_hers
    place.played_shows_hers = lambda g, kind: calls.append(kind) or f"I'm the {kind}."
    original_ask = place.ask_for_a_reading
    place.ask_for_a_reading = lambda *a, **k: None
    try:
        seeing = SeeingInPlay()
        for at, kind in ((0.0, 1), (3.0, 2), (5.0, 1), (5.0 + HERS_HELD_S, 1), (30.0, 2), (30.0 + HERS_HELD_S, 2)):
            _read_the_place(guide, seeing, SimpleNamespace(number=9, kind=kind, follows_pointer=False), at, None,
                            lambda line, once: said.append(line))
        assert said == ["I'm the 1."]                                   # once held, and once a game
        guide.seen.saw(4, "cursor")
        seeing = SeeingInPlay()
        for at in (0.0, HERS_HELD_S + 1):
            _read_the_place(guide, seeing, SimpleNamespace(number=9, kind=4, follows_pointer=True), at, None,
                            lambda line, once: said.append(line))
        assert said == ["I'm the 1."]                                   # a cursor is not who she is
    finally:
        place.played_shows_hers = original
        place.ask_for_a_reading = original_ask


def test_what_things_are_worth_is_said_a_few_times_a_game_paced_and_never_both_ways_of_one_name():
    """LIVE 2026-10-10 sixteen such lines in a minute, and "the red things cost me" then "are worth getting to"."""
    from core.agency.naming_what_she_sees import (
        MOST_STANCES_SAID,
        STANCE_EVERY_S,
        SeeingInPlay,
        a_stance_said,
    )
    from core.agency.what_meeting_things_does import AVOID, MEET

    moves = _moves([_thing(1, 0, 10, 10), _thing(2, 1, 50, 50), _thing(3, 2, 90, 90)] +
                   [_thing(10 + k, 3 + k, 20 * k, 30) for k in range(8)],
                   [(210, 40, 40), (210, 40, 40), (60, 180, 70), (240, 210, 50), (50, 110, 230), (60, 210, 230),
                    (150, 60, 200), (240, 110, 200), (140, 90, 45), (240, 140, 30), (240, 240, 240)])
    run = SimpleNamespace(seeing=SeeingInPlay())
    assert a_stance_said(run, moves, 0, AVOID, 0.0) == "The red things cost me. Keeping clear of them."
    assert a_stance_said(run, moves, 2, MEET, 1.0) == ""                      # too soon after the last
    assert a_stance_said(run, moves, 1, MEET, STANCE_EVERY_S + 1) == ""       # red things were said to cost her
    said = [a_stance_said(run, moves, kind, MEET, 100.0 * kind) for kind in range(2, 11)]
    assert len([line for line in said if line]) == MOST_STANCES_SAID - 1


def test_a_thing_her_eyes_know_by_name_is_called_by_it():
    from core.agency.naming_what_she_sees import describe
    from core.perception.what_things_look_like import sighting_from

    seen = sighting_from('{"is": "hedgehog", "unusual": false, "how": "", "who": "Amy Rose"}')
    assert seen.what == "hedgehog" and seen.who == "Amy Rose"
    assert sighting_from('{"is": "car", "who": "car"}').who == ""
    guide = Guide(place="a place")
    token = THE_GUIDE.set(guide)
    try:
        moves = _moves([_thing(5, 0, 50, 50)], [(140, 90, 45)])
        guide.seen.saw(0, seen.what, seen.odd, seen.who)
        assert describe(moves, 0) == "Amy Rose"
    finally:
        THE_GUIDE.reset(token)


def test_her_own_kind_is_looked_at_where_she_is_not_at_the_biggest_thing_like_her():
    from core.perception.what_things_look_like import LookingAtThings

    looking = LookingAtThings()
    moves = _moves([_thing(5, 0, 50, 50, w=8, h=10), _thing(6, 0, 150, 80, w=40, h=30)], [(240, 240, 240)])
    chosen = looking._to_look_at(moves, 5, set())
    assert chosen and chosen[0][1] == moves.things[5].box()


def test_her_kind_looked_at_before_she_was_known_is_looked_at_again_where_she_is():
    from core.perception.what_things_look_like import LookingAtThings

    looking = LookingAtThings()
    moves = _moves([_thing(5, 0, 50, 50, w=8, h=10), _thing(6, 0, 150, 80, w=40, h=30)], [(240, 240, 240)])
    first = looking._to_look_at(moves, None, set())
    looking.asked |= {kind for kind, _box in first}
    assert first[0][1] == moves.things[6].box()                       # before she is known: the biggest of the look
    again = looking._to_look_at(moves, 5, set())
    assert again and again[0][1] == moves.things[5].box()             # once she is: where she is
    looking.asked |= {kind for kind, _box in again}
    assert looking._to_look_at(moves, 5, set()) == []                 # and only once
