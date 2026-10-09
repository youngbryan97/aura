"""When she keeps failing or gets nowhere, she stops and asks what she knows, weighs what she hears, and remembers
whether it helped.

LIVE 2026-10-09 she lost a game over and over that wanted two keys pressed in
turn, fast, when she was caught. Nothing on its screens said so in words; a
page about the game did. Nothing here is about games: the situation, the
questions, the sources and the weighing are the same for a form or a program.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency.playing_as_it_happens import controls_named_in
from core.cognition.taking_stock import (
    FAILING,
    HELPED_BEFORE,
    STUCK,
    Counsel,
    Heard,
    Situation,
    WhatHelped,
    questions_for,
    take_stock,
    what_to_take,
)
from core.skills.looking_it_up import _yahoo
from core.skills.sovereign_browser_taking_stock import Stocktaking, _better, the_thing

GAME = "Regular Show: All-Nighter"
SITUATION = Situation(GAME, "Play this game and win it.",
                      ("Steer the boat. Aim with your MOUSE and click to sling Rigby after the fireflies!",),
                      "OUCH! You got captured. TRY AGAIN", FAILING)


def _source(*heard: Heard, slow_s: float = 0.0, fails: bool = False):
    async def ask(_questions, _seconds):
        if fails:
            raise RuntimeError("no answer")
        await asyncio.sleep(slow_s)
        return list(heard)
    return ask


@pytest.mark.unit
def test_the_questions_come_from_the_situation():
    asked = questions_for(SITUATION)
    assert asked[0] == f"how to win {GAME}"
    assert any("captured" in q and "ouch" not in q.lower() for q in asked)
    assert any("controls" in q for q in asked)


@pytest.mark.unit
def test_what_several_sources_say_to_do_is_taken_and_what_says_nothing_to_do_is_not():
    heard = [
        Heard("the web", "“All Nighter” on a fan wiki", "If Rigby gets captured by a croc, press the left and right arrow keys "
              "rapidly to break free. Catch fireflies to fill the jar."),
        Heard("my model", "", "Press left and right quickly when captured to escape."),
        Heard("my own copy of Wikipedia", "“Regular Show”", "Regular Show is an animated sitcom created for Cartoon Network."),
        Heard("the web", "a video page", "About Press Copyright Contact us Creators Advertise Developers Terms Privacy Policy."),
    ]
    kept = what_to_take(heard, SITUATION, questions_for(SITUATION))
    assert [h.source for h in kept] == ["the web", "my model"]
    assert "rapidly" in kept[0].text and controls_named_in(Counsel((), tuple(kept)).told)[0][:2] == ["left", "right"]


@pytest.mark.unit
def test_what_helped_before_comes_first():
    heard = [Heard("the web", "", "Use the arrow keys to steer the boat and catch every firefly before dawn."),
             Heard(HELPED_BEFORE, "", "Press left and right in turn, fast, when Rigby is caught by a croc.")]
    assert what_to_take(heard, SITUATION, questions_for(SITUATION))[0].source == HELPED_BEFORE


@pytest.mark.unit
def test_every_source_is_asked_at_once_and_one_slow_or_failing_does_not_hold_up_the_rest():
    async def go():
        return await take_stock(SITUATION, {
            "the web": _source(Heard("the web", "", "If you get captured, press left and right rapidly to break free.")),
            "my model": _source(Heard("my model", "", "Press space to win at once."), slow_s=5.0),
            "my memory": _source(fails=True),
        }, seconds=0.5)

    counsel = asyncio.run(go())
    assert counsel and counsel.kept[0].source == "the web" and counsel.took_s < 2.0
    assert "From the web" in counsel.said()


@pytest.mark.unit
def test_nothing_found_is_said_as_nothing_found():
    counsel = asyncio.run(take_stock(SITUATION, {"the web": _source()}, seconds=0.5))
    assert not counsel and "nothing said more than this place itself" in counsel.said()


@pytest.mark.unit
def test_what_helped_is_kept_and_what_did_not_is_not_offered_again():
    good = Counsel((), (Heard("the web", "", "Press left and right rapidly to break free when captured."),))
    bad = Counsel((), (Heard("my model", "", "Press space when captured to win the round at once."),))
    WhatHelped.of(GAME).came_of(GAME, good, True)
    WhatHelped.of(GAME).came_of(GAME, bad, False)
    again = WhatHelped.of(GAME)
    assert again.helped == [good.kept[0].text]
    assert not again.without_what_did_not(bad)
    assert asyncio.run(again.source()([], 1.0))[0].source == HELPED_BEFORE


@pytest.mark.unit
def test_stock_is_taken_after_two_losses_running_and_its_counsel_judged_by_the_runs_after():
    stock = Stocktaking(the_thing(f"Play this game and win it. (It is “{GAME}”, the first of the 3 picked.)"))
    assert stock.thing == GAME
    assert stock.due([{"ended": "lost"}]) == ""
    assert stock.due([{"ended": "lost"}, {"ended": "lost"}]) == FAILING
    assert stock.due([{"ended": "won"}, {"ended": "lost"}]) == ""
    before = [{"ended": "lost", "gains": 1, "took_s": 20.0}]
    assert _better([{"ended": "lost", "gains": 3, "took_s": 20.0}], before)
    assert _better([{"ended": "won"}], before)
    assert not _better([{"ended": "lost", "gains": 1, "took_s": 21.0}], before)
    assert STUCK != FAILING


@pytest.mark.unit
def test_a_letter_named_as_a_key_is_a_key_and_a_word_is_not():
    assert controls_named_in("Grenades (activated with the X key) clear tight spots.", keys_without_words=(), during_play=True)[0] == ["x"]
    assert controls_named_in("Press Z to shoot and X to throw.", keys_without_words=(), during_play=True)[0] == ["z", "x"]
    assert controls_named_in("Press a button to begin.", keys_without_words=(), during_play=True)[0] == []
    assert controls_named_in("Press S to start.", keys_without_words=(), during_play=True)[0] == []


@pytest.mark.unit
def test_a_search_results_heading_is_its_title_not_the_address_above_it():
    page = ('<a href="https://r.search.yahoo.com/_ylt=x/RU=https%3a%2f%2fwww.example.com%2fgame%2fall-nighter%2f/RK=2/RS=y">'
            '<span>Example</span>https://www.example.com › game › all-nighter'
            '<h3 class="title"><span class="fw-500">Regular Show: All Nighter | Example</span></h3></a>')
    assert _yahoo(page) == [("Regular Show: All Nighter | Example", "https://www.example.com/game/all-nighter/")]
