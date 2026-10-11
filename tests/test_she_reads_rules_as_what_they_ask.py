"""She reads a place's rules for what they ask, and follows its lesson as a procedure, a step at a time.

LIVE 2026-10-10 a game's lesson said to open the library, choose a type, drag a device to the end of another's arrow
and test the trap; she matched "drag" against a word list and carried the library's tab names about thirty-five times.
Read by her own model, held to her own ways and to each sentence's words, the lesson is a procedure; the same reading
serves a program's lesson as a game's.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide
from core.cognition.reading_the_rules import Frame, Rules, read_the_rules_beside
from core.runtime.skill_contract import PredicateOperator, SemanticPredicate

pytestmark = pytest.mark.unit

LESSON = ["Click here to open the device library.", "Choose the type of device you wish to use.",
          "Click and drag a device from the library.",
          "Place the device at the end of another device's arrow to make a connection.",
          "Use the controls to rotate or delete the device.", "Don't press start over until the trap is tested.",
          "Tom needs your help to build a trap to the cage that will catch Jerry!"]


def _reader(answers):
    asked = []

    async def ask(prompt, schema, most):
        asked.append(prompt)
        return schema.model_validate({"frames": answers})

    return ask, asked


def test_a_lesson_is_read_into_steps_held_to_her_ways_and_to_its_own_words():
    guide = Guide(place="a place of devices")
    guide.take_in(SCREEN, LESSON)
    ask, asked = _reader([
        {"number": 1, "act": "click things", "thing": "the device library", "using": "here"},
        {"number": 2, "act": "click things", "thing": "the type of device"},
        {"number": 3, "act": "carry", "thing": "a device", "where": "a magic portal"},          # not its words: dropped
        {"number": 4, "act": "carry", "thing": "the device", "where": "the end of another device's arrow"},
        {"number": 5, "act": "click things", "using": "the controls", "thing": "the device"},
        {"number": 6, "act": "click things", "thing": "start over", "never": True},
        {"number": 7, "act": "a chain", "thing": "a trap", "where": "the cage", "is_what_it_is_for": True},
    ])

    async def run():
        assert read_the_rules_beside(guide, ask)
        await guide.rules.asking

    asyncio.run(run())
    rules = guide.rules
    assert "a place of devices" in asked[0] and "1. Click here to open the device library." in asked[0]
    assert rules.read_as(LESSON[2])[0].where == ""                              # what is not in a sentence is not read
    assert rules.carried_to() == "the end of another device's arrow"
    assert rules.next_step().sentence == LESSON[0]
    assert rules.step_of('click "DEVICE LIBRARY"').sentence == LESSON[0]
    assert rules.forbids('click "START OVER"')
    rules.tried('click "DEVICE LIBRARY"', changed=True)
    assert rules.next_step().sentence == LESSON[1]
    assert guide.goals[0] == LESSON[6]                                         # what it says it is for leads
    assert "The rules, as I read them, in order: ✓ clicking things" in guide.for_thinking()


def test_a_step_that_never_answers_stays_required_for_local_repair():
    rules = Rules()
    rules.hear(["Click here to open the device library.", "Drag a device to the room."])
    rules.took([Frame(LESSON[0], act="click things", thing="the device library"),
                Frame("Drag a device to the room.", act="carry", thing="a device", where="the room")])
    for _ in range(2):
        rules.tried('click "DEVICE LIBRARY"', changed=False)
    failed = rules.next_step()
    assert failed.sentence == LESSON[0]
    assert failed.key in rules.failed_steps
    assert not rules.passed(failed)
    assert len(rules.steps()) == 2
    assert rules.the_step_to_do(['drag "a device" to "the room"']) is None


def test_the_lessons_next_step_leads_the_move_choice_and_what_it_forbids_falls():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    guide = Guide(place="a photo program")
    lesson = ["Click New to start a project.", "Drag a photo onto the canvas.", "Never press Delete All."]
    guide.rules.hear(lesson)
    guide.rules.took([Frame(lesson[0], act="click things", thing="New"),
                      Frame(lesson[1], act="carry", thing="a photo", where="the canvas"),
                      Frame(lesson[2], act="click things", thing="Delete All", never=True)])
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "New"': 0.4, 'click "Help"': 0.9, 'click "Delete All"': 0.8})
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "New"'
    assert 'click "Delete All"' not in valued


def test_a_sentence_read_once_is_understood_again_at_once_anywhere():
    first = Rules()
    first.hear(["Drag a photo onto the canvas."])
    first.took([Frame("Drag a photo onto the canvas.", act="carry", thing="a photo", where="the canvas")])
    again = Rules()
    assert again.hear(["Drag a photo onto the canvas."]) == []                 # nothing left to read
    assert again.carried_to() == "the canvas"


def test_once_something_is_built_an_option_that_keeps_it_outweighs_one_that_throws_it_away(monkeypatch):
    # The player whose trap failed chose "Edit the trap", not "Start over"; a person keeps what they built.
    from types import SimpleNamespace

    import core.cognition.reading_the_rules as rules_module
    from core.cognition.checking_the_debate import DebateCheck

    monkeypatch.setattr(rules_module, "keeps_the_work",
                        lambda label: {"EDIT THE TRAP": True, "START OVER": False}.get(label))
    guide = Guide(place="a place of parts")
    built = SimpleNamespace(carried_to={'drag "the shape at 20% across, 40% down" to "the end of the arrow"': True},
                            quiet_since=set())
    weighed = DebateCheck().weigh({'click "EDIT THE TRAP"': 1.0, 'click "START OVER"': 1.0}, guide, built)
    assert weighed['click "EDIT THE TRAP"'] > weighed['click "START OVER"']
    nothing = SimpleNamespace(carried_to={}, quiet_since=set())
    weighed = DebateCheck().weigh({'click "EDIT THE TRAP"': 1.0, 'click "START OVER"': 1.0}, guide, nothing)
    assert weighed['click "EDIT THE TRAP"'] == weighed['click "START OVER"']


def test_a_repeated_step_holds_until_its_own_readiness_effect_is_confirmed():
    # The player in the video added device after device to the arrow's end, and tested the trap only once it reached.
    rules = Rules()
    lesson = ["Drag a device to the end of another device's arrow.",
              "Continue adding devices until you are ready to test the trap.", "Click Test Trap to see if it works."]
    rules.hear(lesson)
    rules.took([Frame(lesson[0], act="carry", thing="a device", where="the end of another device's arrow"),
                Frame(lesson[1], act="carry", thing="devices", again=True, when="you are ready to test the trap"),
                Frame(lesson[2], act="click things", thing="Test Trap")])
    carry = 'drag "the shape at 20% across, 40% down" to "the end of another device\'s arrow"'
    rules.tried(carry, changed=True)
    assert rules.next_step().sentence == lesson[1]
    adding = rules.next_step()
    repeat = 'drag "devices" to "the end of another device\'s arrow"'
    ready = SemanticPredicate("ready to test", "after.ready", PredicateOperator.EQUALS, True)
    rules.effects[adding.key] = (ready,)
    for _ in range(2):
        rules.tried(repeat, changed=True, before={"ready": False}, after={"ready": False})
        assert rules.next_step().sentence == lesson[1]                          # its termination is not confirmed
    rules.tried(repeat, changed=True, before={"ready": False}, after={"ready": True})
    assert rules.passed(adding)
    assert rules.next_step().sentence == lesson[2]
    assert rules.repeat_opportunities([repeat]) == [adding]
    rules.tried('click "TEST TRAP"', changed=True)
    assert rules.next_step() is None and "again and again" in rules.for_thinking()


def test_on_a_first_visit_she_reads_what_a_place_teaches_rather_than_skip_it():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    guide = Guide(place="a place of devices")
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "SKIP INSTRUCTIONS"': 0.709, 'click "the shape at 75% across, 5% down"': 0.644})
        assert max(valued, key=valued.get) == 'click "the shape at 75% across, 5% down"'
        guide.rules.hear(["Drag a device to the room."])
        guide.rules.took([Frame("Drag a device to the room.", act="carry", thing="a device", where="the room")])
        valued = _the_lesson_first({'click "SKIP INSTRUCTIONS"': 0.709, 'click "the shape at 75% across, 5% down"': 0.644})
        assert valued['click "SKIP INSTRUCTIONS"'] < 0.1                        # read this visit: still new to her
    finally:
        THE_GUIDE.reset(token)
    again = Guide(place="a place of devices")                                    # a later visit: seen before
    again.rules.hear(["Drag a device to the room."])
    token = THE_GUIDE.set(again)
    try:
        valued = _the_lesson_first({'click "SKIP INSTRUCTIONS"': 0.709, 'click "the shape at 75% across, 5% down"': 0.644})
        assert valued['click "SKIP INSTRUCTIONS"'] == 0.709
    finally:
        THE_GUIDE.reset(token)


def test_while_a_new_lesson_plays_she_reads_it_and_once_it_stops_giving_words_she_acts():
    rules = Rules()
    rules.hear(["Click here to open the device library."])
    shown = ["SKIP INSTRUCTIONS", "DEVICE LIBRARY"]
    began = rules.heard_times[-1]
    assert rules.a_lesson_plays(shown, now=began + 1.0)
    assert not rules.a_lesson_plays(["DEVICE LIBRARY"], now=began + 1.0)          # no lesson offered to be skipped
    assert not rules.a_lesson_plays(shown, now=began + rules.beat() + 0.1)      # it waits for her: she acts


def test_a_screen_whose_lesson_is_laid_out_around_its_controls_is_read_whole_into_its_steps():
    """LIVE 2026-10-10 a lesson's callouts read off the screen as one run of words with button names between them:
    "PLaCE THE DEViCE aT + TURN → THE END OF ANOTHER DELETE X DEVICE'S aRROW TO MAKE A CONNECTION". Its labels, and her
    own names for the shapes drawn on it, had been read as rules one by one."""
    guide = Guide(place="a place of devices")
    screen = "PLaCE THE DEViCE aT + TURN → THE END OF ANOTHER DELETE X DEVICE'S aRROW TO MAKE A CONNECTION"
    guide.take_in(SCREEN, [screen, "TURN", "DELETE X", "the shape at 50% across, 80% down"], passages=[screen])
    assert guide.rules.heard == [screen]
    ask, _asked = _reader([
        {"number": 1, "act": "carry", "thing": "THE DEViCE", "where": "THE END OF ANOTHER DEVICE'S aRROW"},
        {"number": 1, "act": "click things", "thing": "THE DEViCE", "using": "TURN"},
        {"number": 1, "act": "click things", "thing": "THE DEViCE", "using": "the rotate dial"},   # not its words
    ])

    async def run():
        assert read_the_rules_beside(guide, ask)
        await guide.rules.asking

    asyncio.run(run())
    steps = guide.rules.read_as(screen)
    assert [f.act for f in steps] == ["carry", "click things", "click things"]
    assert guide.rules.carried_to() == "THE END OF ANOTHER DEVICE'S aRROW"
    assert steps[2].using == ""
    guide.rules.hear(["DEVICE'S aRROW TO MAKE A CONNECTION"])                   # a piece of what was heard
    assert guide.rules.heard == [screen]


def test_a_place_whose_lesson_asks_only_to_build_is_not_played_as_it_happens():
    from core.cognition.what_this_place_is import done_by_its_lesson

    guide = Guide(place="a place of devices")
    lesson = ["Click here to open the device library.", "Drag a device to the end of another device's arrow."]
    guide.rules.hear(lesson)
    guide.rules.took([Frame(lesson[0], act="click things", thing="the device library"),
                      Frame(lesson[1], act="carry", thing="a device", where="the end of another device's arrow")])
    assert done_by_its_lesson(guide) == "carry"
    racer = Guide(place="a race")
    racer.rules.hear(["Steer the car around the track with the arrows.", "Drag the sticker onto the car."])
    racer.rules.took([Frame("Steer the car around the track with the arrows.", act="steer", thing="the car"),
                      Frame("Drag the sticker onto the car.", act="carry", thing="the sticker", where="the car")])
    assert done_by_its_lesson(racer) == ""                                       # it is steered, too
    assert done_by_its_lesson(Guide(place="unread")) == ""


def test_a_screen_read_again_a_little_differently_is_understood_at_once():
    first = Rules()
    seen = "PiCK a ROOM aND TRY TO CaTCH JERRY! THE KITCHEN THE LIVING ROOM"
    first.hear([seen])
    first.took([Frame(seen, act="click things", thing="ROOM", is_what_it_is_for=True)])
    again = Rules()
    assert again.hear(["PICK a ROOM AND TRY TO CATCH JERRY! THE THE LIVING THE KITCHEN ROOM"]) == []
    assert again.in_order()[0].act == "click things" and again.in_order()[0].thing == "ROOM"


def test_where_things_are_carried_follows_the_lesson_from_its_first_part_to_the_rest():
    rules = Rules()
    lesson = ["Place the device on any active square.", "Place the device at the end of another device's arrow."]
    rules.hear(lesson)
    rules.took([Frame(lesson[0], act="carry", thing="the device", where="any active square"),
                Frame(lesson[1], act="carry", thing="the device", where="the end of another device's arrow")])
    assert rules.carried_to() == "any active square"
    rules.tried('drag "the shape at 20% across, 40% down" to "any active square"', changed=True)
    assert rules.carried_to() == "the end of another device's arrow"
    rules.tried("drag \"the shape at 30% across, 40% down\" to \"the end of another device's arrow\"", changed=True)
    assert rules.carried_to() == "the end of another device's arrow"            # the last that says, from then on


def test_what_she_built_before_this_run_still_makes_keeping_it_worth_more_than_starting_over(monkeypatch):
    from types import SimpleNamespace

    import core.cognition.reading_the_rules as rules_module
    from core.cognition.checking_the_debate import DebateCheck

    monkeypatch.setattr(rules_module, "keeps_the_work",
                        lambda label: {"EDIT THE TRAP": True, "start over": False}.get(label))
    guide = Guide(place="a place of parts")
    guide.built = "2 parts put in place, the last at 40% across and 50% down"
    fresh = SimpleNamespace(carried_to={}, quiet_since=set())                 # a new run: nothing carried in it yet
    weighed = DebateCheck().weigh({'click "EDIT THE TRAP"': 1.0, "start over": 1.0}, guide, fresh)
    assert weighed['click "EDIT THE TRAP"'] > weighed["start over"]
