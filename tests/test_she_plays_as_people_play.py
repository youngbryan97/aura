"""She plays as people play: the crafts people show, playing the games of the design map, said for any place, with
the reason under each; found by what a place says, and given to what she says, plans and reasons with.

Nothing here is any one game.
"""
from __future__ import annotations

import pytest

from core.agency.the_craft_of_play import CRAFTS, PRINCIPLES, crafts_for
from core.cognition.a_guide_to_a_place import TOLD, Guide

pytestmark = pytest.mark.unit


def test_every_craft_says_how_and_why_and_names_no_game():
    for craft in CRAFTS:
        assert craft.first and craft.moment and craft.strategy and craft.mistakes and craft.why, craft.name
    assert len(PRINCIPLES) >= 8


@pytest.mark.parametrize(("said", "craft"), [
    ("Use the mouse to move him back and forth to catch the eggs before they crash. Avoid the bombs!",
     "getting under what to catch"),
    ("Press the SPACE BAR to open doors, revealing secret passageways, power-ups, or monsters!", "opening every door"),
    ("Run 1 of 3. You need 5000 points to advance.", "reaching a bar to go on"),
    ("Throw food at your opponents, avoid their attacks, and be the last one left standing!", "lining up to throw or shoot"),
    ("Choose your moves: select 3 more, then start wrestling.", "planning moves before they play out"),
    ("Give the tambourine to him and he will give you the pineapple in exchange.", "fetching and trading"),
    ("Complete each level to gather the items shown on the blueprint: find them in the room.", "finding what is shown"),
])
def test_what_a_place_says_finds_the_craft_it_asks_for_first(said, craft):
    found = [c for c in crafts_for(said) if not c.foundational]
    assert found and found[0].name == craft, [c.name for c in found]


def test_the_guide_gives_the_craft_and_its_reason_to_what_she_says_and_reasons_with():
    guide = Guide(place="a place")
    guide.take_in(TOLD, ["Use the mouse to move him back and forth to catch the eggs before they crash.", "Avoid the bombs!"])
    said = guide.says()
    assert "As people play it: go to where the next good one will land" in said and "because what lands first" in said
    assert "What loses: Use the mouse" not in said                         # how it is worked is no loss
    thinking = guide.for_thinking()
    assert "How people play it (getting under what to catch)" in thinking and "Why: what lands first decides" in thinking
    assert "What all good play rests on" in thinking
    assert "As people play it:" in guide.in_brief()


@pytest.mark.parametrize(("said", "key", "act", "on"), [
    ("Press the SPACE BAR to open doors, revealing secret passageways.", "space", "open", "doors"),
    ("Press space to pick up items.", "space", "pick up", "items"),
    ("Press the up arrow to climb ladders.", "up", "climb", "ladders"),
    ("Use the mouse to move Frankie back and forth.", "the pointer", "move", "frankie"),
    ("Click on the doors to open them.", "the pointer", "open", "doors"),
    ("Walk up to people and press space to talk to them.", "space", "talk", "people"),
    ("Go to the door and press up to enter it.", "up", "enter", "door"),
])
def test_a_control_is_known_with_what_it_is_used_on(said, key, act, on):
    """What a key does and what it is done to: what she needs to use it on the thing when she is beside it."""
    guide = Guide(place="a place")
    guide.take_in(TOLD, [said])
    control = guide.controls[key]
    assert (control.act, control.on) == (act, on)


@pytest.mark.parametrize(("said", "bar"), [
    ("Run 1 of 3. You need 5000 points to advance.", 5000),
    ("Score at least 1,200 points to pass the level.", 1200),
    ("Target: 300", 300),
    ("Catch the fish before time runs out.", None),
])
def test_a_bar_to_go_on_is_read_apart_from_winning(said, bar):
    from core.agency.how_the_contest_stands import what_wins

    wins = what_wins(said)
    assert wins.to_pass == bar and wins.to_score is None


def test_a_count_out_of_a_whole_is_read_as_what_she_has_of_it_and_said_with_what_is_left():
    """"Fish 2/40" is two fish of forty, not forty: what she catches counts, and what is left is known."""
    from core.agency.how_the_contest_stands import ContestStands
    from core.agency.what_meeting_things_does import Readouts, readouts_in

    read = readouts_in([{"text": "Fish 2/40", "center_x": 0.1, "center_y": 0.05},
                        {"text": "Run 1 of 3", "center_x": 0.5, "center_y": 0.05}])
    assert [(r.label, r.value, r.out_of) for r in read] == [("Fish", 2, 40), ("Run", 1, 3)]
    readouts = Readouts()
    for at, fish in ((0.0, 2), (0.5, 3), (1.0, 3)):
        readouts.read([{"text": f"Fish {fish}/40", "center_x": 0.1, "center_y": 0.05},
                       {"text": "Score 3200", "center_x": 0.8, "center_y": 0.05},
                       {"text": "Run 1 of 3", "center_x": 0.5, "center_y": 0.05}], at, None)
    contest = ContestStands()
    contest.heard("You need 5000 points to advance.")
    contest.counted(readouts.current, readouts.where, None, 1.0, readouts.out_of)
    assert contest.says() == "I have 3200; 1800 more to go on; 3 of 40, 37 to go."


def test_going_again_she_says_what_cost_her_and_that_she_keeps_clear_of_it():
    from types import SimpleNamespace

    from core.skills.sovereign_browser_drawing import _what_i_do_differently

    played = SimpleNamespace(stretches=[{"came_to_avoid": ["red car"]}, {"came_to_avoid": ["red car", "other", "bomb"]}])
    assert _what_i_do_differently(played) == " The red car and the bomb cost me; this time I keep clear of them."
    assert _what_i_do_differently(SimpleNamespace(stretches=[{"came_to_avoid": []}])) == ""


@pytest.mark.parametrize(("said", "errand"), [
    ("Give the tambourine to him and he will give you the pineapple in exchange.",
     "So he wants the tambourine, and I'll get the pineapple for it."),
    ("Bring me my glasses and I'll open the gate.", "So someone here wants the glasses, and they will open the gate for them."),
    ("Grandma: I lost my knitting needles somewhere in the garden.", "So Grandma wants the knitting needles."),
    ("Give the banana to Monkey Bob.", "So Monkey Bob wants the banana."),
])
def test_what_the_people_of_a_place_ask_is_kept_as_an_errand_and_said(said, errand):
    guide = Guide(place="a place")
    news = guide.take_in(TOLD, [said])
    assert news and news[0] == errand
    want = guide.errands[0].want
    assert guide.things[want] == "get" and "Asked of me: " + errand in guide.for_thinking()


@pytest.mark.parametrize("said", ["Press space to talk to people.", "Collect coins and avoid bombs.", "Find your way out!"])
def test_an_instruction_is_no_errand(said):
    guide = Guide(place="a place")
    guide.take_in(TOLD, [said])
    assert guide.errands == []


def test_catching_someone_is_no_catching_of_what_falls():
    """LIVE 2026-10-10 "PICK A ROOM AND TRY TO CATCH JERRY!" (a trap to build) was taken for catching what falls."""
    said = ("PICK A ROOM AND TRY TO CATCH JERRY! DRAG THE DEVICE AT THE END OF ANOTHER DEVICE'S ARROW TO MAKE A "
            "CONNECTION. TEST TRAP")
    found = [c.name for c in crafts_for(said) if not c.foundational]
    assert "getting under what to catch" not in found and found[0] == "building a way through", found


def test_a_round_far_worse_than_her_best_goes_back_to_how_she_played_her_best(monkeypatch):
    """As the agents that improve most over attempts do: a revised way of playing that drops below half the best is
    rolled back to the best."""
    import core.skills.sovereign_browser_drawing as drawing

    said = []
    monkeypatch.setattr(drawing, "_tell", said.append)
    keep = {"counsel": "Stay low and jump late."}
    runs = [{"score": 400, "counsel": "Jump early."}, {"score": 900, "counsel": "Stay low."},
            {"score": 300, "counsel": "Stay low and jump late."}]
    drawing._back_to_what_went_best(runs, keep)
    assert keep["counsel"] == "Stay low." and "going back to how I played my best one" in said[0]
    keep = {"counsel": "Stay low."}
    drawing._back_to_what_went_best([{"score": 900, "counsel": "Stay low."}, {"score": 600, "counsel": "Stay low."}], keep)
    assert keep["counsel"] == "Stay low." and len(said) == 1          # not much worse: nothing rolled back


def test_counsel_that_did_not_help_is_no_longer_read_by_play(monkeypatch):
    import core.skills.sovereign_browser_taking_stock as stocktaking

    monkeypatch.setattr(stocktaking, "_tell", lambda line: None)
    monkeypatch.setattr("core.cognition.taking_stock.WhatHelped.of", lambda thing: type("H", (), {
        "came_of": lambda self, *a: None})())
    stock = stocktaking.Stocktaking(thing="a game")
    stock.counsel = type("C", (), {"told": "Hold left."})()
    stock.at_run, stock.before, stock.counsel_before = 1, [{"ended": "lost", "gains": 5, "took_s": 30.0}], "Watch the clock."
    keep = {"counsel": "Watch the clock. Hold left."}
    stock.judge([{}, {"ended": "lost", "gains": 1, "took_s": 20.0}, {"ended": "lost", "gains": 2, "took_s": 25.0}], keep)
    assert keep["counsel"] == "Watch the clock."
