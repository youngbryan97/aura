"""She calls things by what the place calls them, and notices what follows what, most of it let go, some of it kept.

The machinery is general; what she says and thinks is of the place in front of her: "the lunar lander", not "the
orange thing"; "after I press space, I tend to gain", not a count of pictures. Nothing here is any one game.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import TOLD, Guide
from core.cognition.what_she_notices import Notebook

pytestmark = pytest.mark.unit


def test_things_are_called_by_what_the_place_calls_them_for_the_part_they_play():
    guide = Guide()
    guide.take_in(TOLD, ["Use the arrow keys to guide Tommy's Lunar Lander safely onto the landing platform!",
                         "Avoid the asteroids.", "Collect the red gems and shoot the robot dogs.",
                         "Help Bloo escape from the monster."])
    assert guide.names["me"][:2] == ["lunar lander", "bloo"]
    assert guide.name_for("me", "orange") == "lunar lander"
    assert guide.name_for("goal", "white") == "landing platform"
    assert guide.name_for("get", "red") == "red gems"
    assert guide.name_for("avoid", "grey") == "asteroids" and guide.name_for("shoot", "black") == "robot dogs"
    assert guide.name_for("avoid", "grey") == "asteroids"                 # the same thing keeps its name
    assert guide.name_for("avoid", "blue") == "monster"                   # what Bloo escapes from is kept clear of
    assert guide.name_for("avoid", "purple") == ""                        # and two things are not given one name


def test_a_noticing_seen_often_and_holding_is_a_theory_and_one_the_evidence_turns_against_is_let_go():
    book = Notebook()
    for at in range(3):
        became = book.notice("after press space gain", "after I press space, I tend to gain", float(at),
                             about="space", pays=1)
    assert became is not None and became.theory and book.paying("space") == 1
    for at in range(3, 10):
        book.notice("after press space gain", "after I press space, I tend to gain", float(at), held=False)
    assert not book.notes["after press space gain"].theory and book.paying("space") == 0


def test_a_passing_remark_is_forgotten_and_a_theory_is_kept():
    book = Notebook()
    book.wonder("odd flicker", "that's odd: the corner flickers", 0.0)
    for at in range(3):
        book.notice("score by itself", "my score goes up with nothing I did beside it", float(at), about="score", pays=1)
    book.notice("later", "something later", 2000.0)
    assert "odd flicker" not in book.notes and "score by itself" in book.notes
    again = Notebook.from_memory(book.as_memory())
    assert again.theories() and again.theories()[0].said.startswith("my score goes up")


def _run(**kw):
    defaults = {"held": "", "input_key_downs": {}, "last_click": float("-inf")}
    return SimpleNamespace(**{**defaults, **kw})


def test_what_follows_an_act_more_often_than_chance_is_noticed_and_becomes_a_theory():
    from core.agency.noticing_in_play import NoticingInPlay

    seeing, book, said = NoticingInPlay(), Notebook(), []
    meeting = SimpleNamespace(verdicts=[], _met={})
    moves = SimpleNamespace(things={})
    presses = 0
    for second in range(30):
        at = float(second)
        if second % 3 == 0:
            presses += 1
        run = _run(input_key_downs={"space": presses})
        if second % 3 == 1:
            meeting.verdicts.append({"what": "gain", "at": at + 0.1, "counter": "score"})   # 0.6 s after the press
        seeing.saw(run, moves, meeting, at + 0.5, book, None, said.append)
    assert book.paying("space") == 1
    assert said and "after I press space, I tend to gain" in said[0]


def test_a_thing_that_stops_after_she_meets_it_is_noticed_by_the_place_s_name_for_it():
    from core.agency.noticing_in_play import NoticingInPlay

    seeing, book = NoticingInPlay(), Notebook()
    for time in range(4):
        robot = SimpleNamespace(kind=2, vx=60.0, vy=0.0)
        met = {7 + time: 10.0 * time}
        seeing.saw(_run(), SimpleNamespace(things={7 + time: robot}), SimpleNamespace(verdicts=[], _met=met),
                   10.0 * time, book, lambda kind, thing, part: "robot dog", None)
        robot.vx = 2.0
        seeing.saw(_run(), SimpleNamespace(things={7 + time: robot}), SimpleNamespace(verdicts=[], _met=met),
                   10.0 * time + 0.5, book, lambda kind, thing, part: "robot dog", None)
    theory = [n for n in book.theories() if "stops for a moment" in n.said]
    assert theory and "robot dog" in theory[0].said


def test_what_she_has_come_to_think_is_in_what_she_reasons_with_and_weighs_what_she_does():
    from core.cognition.checking_the_debate import FACTS, DebateCheck

    guide = Guide(place="a place")
    guide.take_in(TOLD, "Use the arrow keys to move. Press space to jump.")
    for at in range(3):
        guide.notes.notice("after press space gain", "after I press space, I tend to gain", float(at), about="space",
                           pays=1)
    assert "I've come to think: after I press space" in guide.for_thinking()
    assert "I've come to think" in guide.in_brief()
    facts = DebateCheck().facts("space", guide)
    assert facts[FACTS.index("what I've noticed of it")] == 1.0


def test_where_every_act_is_followed_by_gains_no_act_is_noticed_for_them():
    """LIVE 2026-10-09 a flickering bar gained and lost all the time, and within seconds every key was a theory."""
    from core.agency.noticing_in_play import NoticingInPlay

    seeing, book = NoticingInPlay(), Notebook()
    meeting = SimpleNamespace(verdicts=[], _met={})
    downs = {"w": 0, "a": 0, "s": 0, "d": 0}
    for tick in range(80):
        at = tick * 0.5
        key = "wasd"[tick % 4]
        downs[key] += 1
        meeting.verdicts.append({"what": "gain", "at": at + 0.2, "counter": "the bar"})
        seeing.saw(_run(input_key_downs=dict(downs)), SimpleNamespace(things={}), meeting, at + 0.3, book, None, None)
    assert not book.theories(), [n.said for n in book.theories()]
