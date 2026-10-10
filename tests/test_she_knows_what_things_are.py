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
    assert sighting_from('{"is": "orange", "odd": "it has a face and eyes"}').odd == "it has a face and eyes"


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
        assert looking.look(picture, moves, 2.0, mine=5)
        await looking._asking
        assert not looking.look(picture, moves, 3.0, mine=5)          # not again so soon, and nothing new to ask of
        return looking

    looking = asyncio.run(play())
    assert [k for k, _s in looking.newly()] == [0, 1]                 # hers first, then the other; no speck, no backdrop
    assert looking.by_kind[0].what == "red car" and looking.by_kind[1].odd == "it is green and glowing"
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
