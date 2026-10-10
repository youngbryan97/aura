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
    assert rules.frames[LESSON[2]].where == ""                                  # what is not in a sentence is not read
    assert rules.carried_to() == "the end of another device's arrow"
    assert rules.next_step().sentence == LESSON[0]
    assert rules.step_of('click "DEVICE LIBRARY"').sentence == LESSON[0]
    assert rules.forbids('click "START OVER"')
    rules.tried('click "DEVICE LIBRARY"', changed=True)
    assert rules.next_step().sentence == LESSON[1]
    assert guide.goals[0] == LESSON[6]                                         # what it says it is for leads
    assert "The rules, as I read them, in order: ✓ clicking things" in guide.for_thinking()


def test_a_step_that_never_answers_is_passed_over_and_holds_her_nowhere():
    rules = Rules()
    rules.hear(["Click here to open the device library.", "Drag a device to the room."])
    rules.took([Frame(LESSON[0], act="click things", thing="the device library"),
                Frame("Drag a device to the room.", act="carry", thing="a device", where="the room")])
    for _ in range(2):
        rules.tried('click "DEVICE LIBRARY"', changed=False)
    assert rules.next_step().sentence == "Drag a device to the room."


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
    assert valued['click "Delete All"'] < 0.1


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
