"""A thing whose own words set a player to make something has no winner, and asked to win it she says so.

LIVE 2026-10-07 a character maker ("Create your own KND by selecting your
favorite body parts ... Then click on the GENERATE button to see your
character in action!") was played for four minutes to be won, and reported
"Played, and not won: I could not get further with it."
"""
from __future__ import annotations

import time

import pytest

from core.language.how_a_game_ended import what_it_asks_of_a_player
from core.skills.screen_pursuit_as_it_happens import TRYING_OUT_S, PlayingAsItHappens
from core.skills.sovereign_browser_drawing import (
    _PLAY_OVER,
    NOTHING_TO_WIN,
    _address,
    _what_it_is_for,
    played_first,
)

pytestmark = pytest.mark.unit

MAKER = ("Create your own KND by selecting your favorite body parts, accessories and equipment. "
         "Then click on the GENERATE button to see your character in action!")


def test_its_own_words_say_whether_it_is_won_scored_or_made():
    assert what_it_asks_of_a_player(MAKER) == "make"
    assert what_it_asks_of_a_player("Dress up the princess for the ball.") == "make"
    assert what_it_asks_of_a_player("Create your own fighter, then battle to win the tournament.") == "win"
    assert what_it_asks_of_a_player("You have two minutes to collect as many points as possible.") == "score"
    assert what_it_asks_of_a_player("Use the arrow keys to move. Catch the fruit.") == ""


def test_a_thing_for_making_is_over_once_she_has_tried_it_out_and_made_something_in_it():
    """Offline 2026-10-11 a scrapbook's photo was taken for made after ninety seconds of arrows and a menu clicked."""
    from core.skills.screen_pursuit_as_it_happens import TRYING_OUT_MOST_S

    reflexes = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="win it", ends_at=time.monotonic() + 600)
    reflexes.words = ["KND NUMBUH GENERATOR", MAKER]
    assert not reflexes.run_is_over({"text": MAKER})
    reflexes.for_making_since = time.monotonic() - TRYING_OUT_S - 1
    assert not reflexes.run_is_over({"text": MAKER})                  # its time had, but nothing made in it yet
    reflexes.acts_that_changed = reflexes.made_before + 500
    assert not reflexes.run_is_over({"text": MAKER})                 # menus changed; no artifact witness
    stuck = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="win it", ends_at=time.monotonic() + 600)
    stuck.words = ["KND NUMBUH GENERATOR", MAKER]
    stuck.run_is_over({"text": MAKER})
    stuck.for_making_since = time.monotonic() - TRYING_OUT_MOST_S - 1
    assert stuck.run_is_over({"text": MAKER})                         # one that will not change is left in time
    from core.skills.screen_pursuit_as_it_happens import MAKING_STALLED

    assert stuck.over_because == MAKING_STALLED


def test_she_says_what_it_is_for_in_its_own_words_and_ends_it_as_played():
    assert _what_it_is_for(["KND NUMBUH GENERATOR", MAKER]).startswith("Create your own KND")
    url = "https://archive.org/details/a-maker"
    _PLAY_OVER[_address(url)] = NOTHING_TO_WIN
    done = played_first({"url": url}, "play it and win it")
    assert done["done"] and done["why"] == f"Played; {NOTHING_TO_WIN}."


def test_an_invitation_to_imagine_and_make_is_read_as_made():
    # Offline 2026-10-10 a photo game said "Use your imagination to make the craziest, creepiest, or goofiest scenes".
    said = ("As the gang's photographer, set up and take photos for the scrap book. Use your imagination to make the "
            "craziest, creepiest, or goofiest scenes you can think of.")
    assert what_it_asks_of_a_player(said) == "make"
    assert what_it_asks_of_a_player("Make it to the end of the level to win.") == "win"
    assert what_it_asks_of_a_player("Make the highest score you can before time runs out.") == "score"


def test_a_goal_said_with_who_it_is_for_or_where_a_thing_is_brought_is_an_end_and_points_are_a_score():
    """LIVE 2026-10-11 "help the Eds get their jawbreakers" with a score on the screen was played for its score."""
    from core.language.what_counts_as_done import END, MEASURE, shape_in

    assert shape_in("Help the Eds get their jawbreakers by using crazy contraptions.", a_measure_seen=True) == END
    assert shape_in("Create elaborate tubes to get a jawbreaker into a cup.", a_measure_seen=True) == END
    assert shape_in("Drop the coins into the slots for points.", a_measure_seen=True) == MEASURE


def test_what_she_made_is_said_as_her_eyes_see_it_and_why_in_its_own_words(monkeypatch):
    """Offline 2026-10-11 she said of a scrapbook she had put nothing into "I made it: at the end it shows 'PUBLIC
    LIBRARY'", and why she made it was not said."""
    import asyncio

    import core.perception.her_eyes as eyes
    import core.skills.sovereign_browser_drawing as drawing

    class _Page:
        viewport_size = {"width": 200, "height": 100}

        async def screenshot(self, clip):
            import io

            from PIL import Image

            out = io.BytesIO()
            Image.new("RGB", (int(clip["width"]), int(clip["height"])), (90, 40, 160)).save(out, format="PNG")
            return out.getvalue()

    async def seen(prompt, image, **_kw):
        return '{"shows": "Scooby and Shaggy hiding behind a bookshelf in a haunted library."}'

    monkeypatch.setattr(eyes, "look_with_her_eyes", seen)
    words = ["INSTRUCTIONS", "Use your imagination to make the craziest, creepiest, or goofiest scenes you can think of."]
    said = asyncio.run(drawing._what_was_made_as_seen(_Page(), (0.0, 0.0, 1.0, 1.0), {"words": "NEW PHOTO"}, [], words))
    assert "shows scooby and shaggy hiding behind a bookshelf" in said.lower()
    assert "because it asks to" in said and "craziest" in said


@pytest.mark.parametrize(("words", "asks"), [
    ('Instructions 1. Try to type only "Yes" or "No" questions. 2. Don\'t forget to click "Ask."', "make"),
    ("Ask the Magic 8 ball any question", "make"),
    ("Answer the questions to win the quiz", "win"),
    ("Type your name and press enter to begin the race", ""),
])
def test_a_thing_that_answers_what_is_asked_of_it_is_played_once_as_one_for_making_is(words, asks):
    """Offline 2026-10-11 a fortune-teller asked "Will I win?" was played as it happened until its time ran out, and
    what it answered was never said."""
    assert what_it_asks_of_a_player(words) == asks


def test_what_a_thing_that_answers_said_is_told_with_what_she_asked_it(monkeypatch):
    import asyncio

    import core.perception.her_eyes as eyes
    import core.skills.sovereign_browser_drawing as drawing

    class _Page:
        viewport_size = {"width": 100, "height": 100}

        async def screenshot(self, clip):
            import io

            from PIL import Image

            out = io.BytesIO()
            Image.new("RGB", (100, 100), (20, 20, 20)).save(out, format="PNG")
            return out.getvalue()

    async def seen(prompt, image, **_kw):
        assert "answer" in prompt                                         # asked for its answer, not what it shows
        return '{"answered": "Swami Shaggy saying Like, no way, man!"}'

    monkeypatch.setattr(eyes, "look_with_her_eyes", seen)
    said = asyncio.run(drawing._what_was_made_as_seen(_Page(), (0.0, 0.0, 1.0, 1.0), {"words": ""}, [], [],
                                                      {"typed": "Will I win?"}))
    assert "I asked it 'Will I win?'" in said and "it answered: swami shaggy saying like, no way" in said.lower()


def test_a_place_understood_as_for_making_is_known_when_her_time_runs_out():
    """Offline 2026-10-12 a scrapbook of photographs was worked for five minutes and, its time up, left with nothing
    said of what was in it: where she understands a place is for making, what she made is said when her time is up."""
    from types import SimpleNamespace

    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.skills.sovereign_browser_drawing import _a_place_for_making

    for mechanics, expected in (({"making": object()}, False), ({"avoiding": object()}, False)):
        guide = SimpleNamespace(controls={}, mechanics=mechanics, how=None, how_seen={}, how_said="", goals=[], things={},
                                rules=None, readouts={}, said="", program="")
        token = THE_GUIDE.set(guide)
        try:
            from core.agency import how_a_game_works

            original = how_a_game_works.understood
            how_a_game_works.understood = lambda _guide, also="": how_a_game_works.HowItWorks()
            try:
                assert _a_place_for_making() is expected
            finally:
                how_a_game_works.understood = original
        finally:
            THE_GUIDE.reset(token)
    assert _a_place_for_making() is False


def test_a_place_for_making_is_known_after_its_runs_from_the_guide_kept_for_it():
    """Offline 2026-10-13 an alien-maker was worked for five minutes and, its time up, left with "I've given this one
    its share of the time": no run had the guide in hand any more, and the place was taken for one with no guide.
    Invented places: a robot shop and a rock run, each known only by the guide kept for it between its runs."""
    from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide
    from core.skills.sovereign_browser_drawing import _a_place_for_making, _the_guide_in_hand

    shop, run = Guide(place="The Robot Shop"), Guide(place="Rock Run")
    shop.take_in(SCREEN, "ROBOT SHOP. Create and customize your own robot, then give it a name!")
    run.take_in(SCREEN, "Jump over the rocks with the up arrow. Reach the flag to win!")
    assert THE_GUIDE.get() is None
    assert _a_place_for_making({"guide": shop}) is True
    assert _a_place_for_making({"guide": run}) is False
    assert _a_place_for_making({}) is False
    # The guide a run has in hand comes first, and the kept one is what is left when none has.
    assert _the_guide_in_hand({"guide": shop}) is shop
    token = THE_GUIDE.set(run)
    try:
        assert _the_guide_in_hand({"guide": shop}) is run
    finally:
        THE_GUIDE.reset(token)


def test_what_a_machine_answered_is_what_is_new_on_it_not_the_label_over_it(monkeypatch):
    """Offline 2026-10-12 a machine that answered a name with a made-up one, under a lit label saying what it was, was
    said to have answered the label, twice. Nothing here is that machine: a box that answers a word with another."""
    import asyncio

    import core.perception.her_eyes as eyes
    import core.skills.sovereign_browser_drawing as drawing

    class _Page:
        viewport_size = {"width": 100, "height": 100}

        async def screenshot(self, clip):
            import io

            from PIL import Image

            out = io.BytesIO()
            Image.new("RGB", (100, 100), (20, 20, 20)).save(out, format="PNG")
            return out.getvalue()

    async def seen(prompt, image, **_kw):
        return '{"answered": "YOUR LUCKY COLOUR"}'                    # the lit label, not the answer

    monkeypatch.setattr(eyes, "look_with_her_eyes", seen)
    keep = {"typed": "Maple", "typed_kind": "a name", "typed_on": "TYPE A WORD HERE GO YOUR LUCKY COLOUR",
            "answered": "TYPE A WORD HERE TRY AGAIN DEEP ORANGE YOUR LUCKY COLOUR"}
    said = asyncio.run(drawing._what_was_made_as_seen(_Page(), (0.0, 0.0, 1.0, 1.0), {"words": ""}, [], [], keep))
    assert said.endswith("it answered: DEEP ORANGE.")


def test_a_name_given_to_a_thing_made_is_said_as_part_of_what_was_made(monkeypatch):
    """Offline 2026-10-13 an alien made, named and saved was said to have been given her name and to have answered
    "ERROR Could not connect". Invented: a robot shop that asks for a name; and a box that answers a name, which is
    not for making, still said to answer."""
    import asyncio

    import core.perception.her_eyes as eyes
    import core.skills.sovereign_browser_drawing as drawing
    from core.cognition.a_guide_to_a_place import SCREEN, Guide

    class _Page:
        viewport_size = {"width": 100, "height": 100}

        async def screenshot(self, clip):
            import io

            from PIL import Image

            out = io.BytesIO()
            Image.new("RGB", (100, 100), (20, 20, 20)).save(out, format="PNG")
            return out.getvalue()

    asked: list[str] = []

    async def seen(prompt, image, **_kw):
        asked.append(prompt)
        return '{"shows": "A blue robot with claw hands and a red antenna"}' if "made" in prompt else '{"answered": "ZAP"}'

    monkeypatch.setattr(eyes, "look_with_her_eyes", seen)
    shop = Guide(place="The Robot Shop")
    shop.take_in(SCREEN, "ROBOT SHOP. Create and customize your own robot, then give it a name!")
    keep = {"guide": shop, "typed": "Aura", "typed_kind": "a name", "typed_on": "NAME YOUR ROBOT"}
    said = asyncio.run(drawing._what_was_made_as_seen(_Page(), (0.0, 0.0, 1.0, 1.0), {"words": ""}, [], [], keep))
    assert "answer" not in asked[-1]
    assert "What I made shows a blue robot with claw hands and a red antenna." in said
    assert "It asked for a name, and I gave it 'Aura'." in said and "answered" not in said
    box = Guide(place="The Name Box")
    box.take_in(SCREEN, "Type your name and the box will tell you your secret word.")
    keep = {"guide": box, "typed": "Aura", "typed_kind": "a name", "typed_on": "TYPE YOUR NAME", "answered": "TYPE YOUR NAME ZAP"}
    said = asyncio.run(drawing._what_was_made_as_seen(_Page(), (0.0, 0.0, 1.0, 1.0), {"words": ""}, [], [], keep))
    assert "I gave it my name, 'Aura', and it answered: ZAP." in said
