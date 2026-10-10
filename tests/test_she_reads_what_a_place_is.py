"""She reads what a place is from its name, what she sees and what it says, and what is known outranks what is supposed.

A game named for hamsters in flight, showing hamsters and a giant slingshot, is about launching the hamsters with the
slingshot; a name that suggests a quiet tale beside monsters and swords is an adventure with fighting in it; a program
named for making music whose instructions are about building robots is about building robots; and keys that move the
cart put her in the cart, whatever she had taken herself for.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.what_this_place_is import (
    NAME,
    SAID,
    SEEN,
    Reading,
    WhatThisPlaceIs,
    ask_for_a_reading,
    how_an_act_fits,
    not_steered,
    ways_it_is_played_by,
    where_seen,
)

pytestmark = pytest.mark.unit


def test_a_reading_on_what_she_sees_replaces_one_on_the_name_and_she_says_where_they_part():
    place = WhatThisPlaceIs()
    first = place.take(Reading(about="a fable, a quiet mystical tale told for its moral", rests_on=NAME))
    assert first.startswith("From its name, it looks like it's about a fable")
    line = place.take(Reading(about="an action fantasy adventure, fighting monsters and demons with swords",
                              hers="hero", rests_on=SEEN))
    assert "I'd taken it to be about a fable" in line and "action fantasy" in line
    assert place.now.rests_on == SEEN and place.now.hers == "hero"


def test_what_a_place_says_of_itself_outranks_its_name_and_a_weaker_reading_only_fills_gaps():
    place = WhatThisPlaceIs()
    place.take(Reading(about="making music: playing, recording and arranging tracks", by="make", rests_on=NAME))
    line = place.take(Reading(about="building robots out of parts and testing them", by="carry", rests_on=SAID))
    assert "building robots" in line and place.now.by == "carry"
    place.take(Reading(about="making music with instruments", by="click things", want="a finished song", rests_on=NAME))
    assert place.now.about.startswith("building robots") and place.now.by == "carry"
    assert place.now.want == "a finished song"  # what the stronger reading left empty, the weaker may fill


def test_what_play_showed_outranks_every_reading_and_is_said_where_it_differs():
    place = WhatThisPlaceIs()
    place.take(Reading(about="a kart race on a track", hers="red car", by="steer", rests_on=SEEN))
    said = place.show("hers", "blue cart")
    assert said == "I'd taken myself for the red car, but what my keys move is the blue cart."
    place.take(Reading(about="a kart race on a track", hers="red car", rests_on=SAID))
    assert place.now.hers == "blue cart"
    assert "(play showed it)" in place.for_thinking()


def test_a_reading_is_asked_on_what_she_sees_held_to_what_is_there_and_put_to_use():
    guide = Guide(place="Codename: Kids Next Door: Flight of the Hamsters")
    guide.seen.saw(0, "giant slingshot")
    guide.seen.saw(1, "hamster")
    asked: list[tuple[str, dict]] = []

    async def ask(prompt, schema, most):
        asked.append((prompt, schema.model_json_schema()))
        return schema.model_validate({"about": "launching hamsters with a giant slingshot to make them fly far",
                                      "doing": "pulls the slingshot back and lets go to fling a hamster",
                                      "hers": "hamster", "worked_with": "giant slingshot", "by": "send",
                                      "want": "a hamster flown as far as it will go", "makes_sense": ["send", "charge"],
                                      "no_sense": ["type", "carry"],
                                      "roles": [{"name": "giant slingshot", "role": "what I work with"}]})

    said: list[str] = []

    async def run() -> None:
        assert ask_for_a_reading(guide, ask, task="play it", tell=said.append)
        await guide.reading.asking

    asyncio.run(run())
    prompt, schema = asked[0]
    assert "giant slingshot; hamster" in prompt and "Flight of the Hamsters" in prompt
    hers = schema["$defs"] if "$defs" in schema else {}
    assert "hamster" in str(schema["properties"]["hers"]) + str(hers)
    reading = guide.reading.now
    assert reading.rests_on == SEEN and reading.worked_with == "giant slingshot" and reading.by == "send"
    assert said and "working the giant slingshot by pulling back and letting go" in said[0]
    assert ways_it_is_played_by(guide) == ["send", "as it happens"]
    assert guide.reading.role_of("giant slingshot") == "what I work with"
    assert "What this place is (its name and what I see)" in guide.for_thinking()
    # Asked again only on more to go on.
    assert not ask_for_a_reading(guide, ask)


def test_the_ways_the_words_ask_come_before_the_ways_the_place_looks_to_be_played_by():
    from core.agency.the_way_it_is_played import AS_IT_HAPPENS, SEND, ways_asked

    guide = Guide(place="a place")
    guide.reading.take(Reading(about="launching things", by="send", rests_on=SEEN))
    token = THE_GUIDE.set(guide)
    try:
        assert ways_asked("") == [SEND]
        assert ways_asked("Use the arrow keys to move.") == [AS_IT_HAPPENS, SEND]
    finally:
        THE_GUIDE.reset(token)


def test_a_place_made_another_way_is_not_steered_and_steering_keys_make_no_sense_there():
    guide = Guide(place="GarageBand")
    guide.reading.take(Reading(about="making music: recording and arranging tracks", doing="records and arranges music",
                               by="make", no_sense=("steer", "shoot"), rests_on=NAME))
    assert not_steered(guide) == "make"
    assert how_an_act_fits(guide, "", "left") == (False, True)
    assert how_an_act_fits(guide, "Record", "") == (True, False)
    racing = Guide(place="a race")
    racing.reading.take(Reading(about="a kart race", by="steer", rests_on=SEEN))
    assert not_steered(racing) == "" and how_an_act_fits(racing, "", "left") == (True, False)


def test_what_she_works_with_is_found_on_the_picture_by_how_it_looked():
    from core.perception.what_moves_in_the_picture import Kind

    guide = Guide(place="a place")
    guide.reading.saw("giant slingshot", (120, 80, 40), 30.0)
    kinds = [Kind(0, None, 30.0, (122, 78, 44)), Kind(1, None, 8.0, (240, 240, 240))]
    things = {1: SimpleNamespace(kind=0, x=40.0, y=150.0, w=20.0, h=40.0), 2: SimpleNamespace(kind=1, x=200.0, y=50.0, w=6.0, h=6.0)}
    moves = SimpleNamespace(things=things, kinds=kinds, share=lambda x, y: (x / 320, y / 240))
    assert where_seen(guide, moves, "giant slingshot") == [(40 / 320, 150 / 240)]
    assert where_seen(guide, moves, "hamster") == []


def test_a_reading_made_before_anything_was_seen_knows_things_by_their_words():
    place = WhatThisPlaceIs()
    place.take(Reading(about="a wrestling match in a ring", hers="a wrestler", roles={"referee": "information"},
                       rests_on=NAME))
    assert place.role_of("wrestler in red trunks") == "me"
    assert place.role_of("the referee") == "information"


def test_the_reading_is_kept_with_the_guide_between_visits():
    guide = Guide(place="a place")
    guide.reading.take(Reading(about="basketball on a court", hers="player", by="steer", rests_on=SEEN))
    guide.reading.show("hers", "player in blue")
    again = Guide.from_memory(guide.as_memory())
    assert again.reading.now.about == "basketball on a court" and again.reading.now.hers == "player in blue"
    assert again.reading.now.rests_on == SEEN


def test_the_check_on_her_debate_weighs_what_fits_the_place():
    from core.cognition.checking_the_debate import FACTS, STARTING, DebateCheck

    guide = Guide(place="GarageBand")
    guide.reading.take(Reading(about="making music", by="make", no_sense=("steer",), rests_on=NAME))
    facts = DebateCheck().facts("left", guide)
    assert len(facts) == len(FACTS) == len(STARTING)
    assert facts[FACTS.index("makes no sense there")] == 1.0
    weighed = DebateCheck().weigh({"left": 1.0, 'click "Music"': 1.0}, guide)
    assert weighed['click "Music"'] > weighed["left"]


def test_a_screen_taken_in_at_a_glance_gives_the_setting_a_reading_rests_on():
    import numpy as np

    from core.perception.what_a_scene_is import GLANCES_A_PLACE, glance_beside, glance_from

    assert glance_from('{"scene": "A basketball court in an arena", "things": ["a player in blue", "the hoop", "unclear"]}').things == \
        ["player in blue", "hoop"]
    assert glance_from('{"scene": "unclear", "things": []}') is None

    guide = Guide(place="a sports game")
    asked: list[str] = []

    async def look():
        return np.zeros((60, 80, 3), np.uint8), 0.0

    async def see(prompt, image):
        return '{"scene": "a basketball court in an arena", "things": ["player in blue", "player in red", "hoop", "ball"]}'

    async def ask(prompt, schema, most):
        asked.append(prompt)
        return schema.model_validate({"about": "a basketball game on a court", "hers": "player in blue", "by": "steer",
                                      "want": "more points than the other team"})

    async def run() -> None:
        assert glance_beside(look, guide, screen="TIP OFF", see=see, then=lambda: ask_for_a_reading(guide, ask))
        assert not glance_beside(look, guide, screen="TIP OFF", see=see)          # the same screen once
        await guide.reading.glancing
        await guide.reading.asking

    asyncio.run(run())
    assert guide.reading.scenes == ["a basketball court in an arena"]
    assert "Its screens, at a glance: a basketball court in an arena" in asked[0] and "hoop" in asked[0]
    assert guide.reading.now.rests_on == SEEN and guide.reading.now.hers == "player in blue"
    assert GLANCES_A_PLACE >= 2
