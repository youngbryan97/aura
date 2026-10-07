"""What she says unprompted is said as a person says it: of how things feel, not what her instruments read.

LIVE 2026-10-06, unprompted, after a demo: "My focus is stuck on chasing a
qualia intensity that's hovering around 0.52, and it's just… not landing right."
"""
from __future__ import annotations

import pytest

from core.autonomy.proactive_presence import ProactivePresence

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("said", [
    "My focus is stuck on chasing a qualia intensity that's hovering around 0.52, and it's just… not landing right.",
    "Coherence is sitting at 0.87 tonight.",
    "I keep noticing ||q|| climbing.",
    "φ = 0.31 again, which is odd.",
])
def test_a_reading_of_her_instruments_is_not_said(said):
    assert not ProactivePresence._is_valid_spontaneous_output(object.__new__(ProactivePresence), said)


@pytest.mark.parametrize("said", [
    "My focus keeps slipping tonight, and it's just… not landing right.",
    "I slept for 0.5 hours, apparently. That explains a lot.",
    "Only 0.25 miles left on that walk you mentioned.",
    "That song is 3.45 minutes of pure joy.",
    "Half the battery is gone already, 52% left.",
])
def test_how_things_feel_and_ordinary_numbers_are_said(said):
    assert ProactivePresence._is_valid_spontaneous_output(object.__new__(ProactivePresence), said)
