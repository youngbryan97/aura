"""A round in which steps of the lesson or of her plan were done is getting somewhere, though nothing was scored.

LIVE 2026-10-10 a trap-building game, where what is gained is the build, was begun again from the top every four
minutes for gaining nothing.
"""
from __future__ import annotations

import pytest

from core.cognition.a_guide_to_a_place import Guide
from core.cognition.a_plan_to_an_end import Plan
from core.cognition.reading_the_rules import Frame
from core.skills.sovereign_browser_drawing import _steps_done

pytestmark = pytest.mark.unit


def test_steps_done_of_the_lesson_and_the_plan_are_counted():
    guide = Guide(place="a place of devices")
    keep = {"guide": guide}
    assert _steps_done(keep) == 0 and _steps_done({}) == 0
    guide.rules.hear(["Click here to open the device library."])
    guide.rules.took([Frame("Click here to open the device library.", act="click things", thing="the device library")])
    guide.rules.tried('click "DEVICE LIBRARY"', changed=True)
    plan = Plan()
    plan.took([Frame("Open the device library", act="click things", thing="DEVICE LIBRARY")])
    plan.tried('click "DEVICE LIBRARY"', changed=True)
    guide.plan = plan
    assert _steps_done(keep) == 2
