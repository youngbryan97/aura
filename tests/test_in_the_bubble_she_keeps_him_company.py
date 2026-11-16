"""In the bubble she is keeping the person company, and what is on screen reaches her as it does in any with.

Bryan, 15 November 2026: "unprompted advice and commentary. that's agency. just giving thoughts on the screen she
should be seeing when she feels its appropriate or has something to add ... the idea ultimately is for aura to always
have presence". Her remark at the desk was gated on a list of fault words, so the job application he was filling in
never reached her; in a with, whether to say something is her interest and the with's expected utility
(core/together/remarks.py, core/together/when_to_say_something.py).
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

PERSON = "person:bryan"


@pytest.fixture
def presence(monkeypatch):
    from core.perception import ambient_presence
    from core.together import a_with

    monkeypatch.setattr(a_with, "_TOGETHER", a_with.Together())
    monkeypatch.setattr("core.together.taking_it_up.the_person_she_is_with", lambda: PERSON)
    monkeypatch.setattr(ambient_presence, "_proactivity_suppressed", lambda: False)
    routed: dict[str, object] = {}

    async def in_a_with(held, _observation):
        routed["with"] = held
        return "That second question wants the five layers of one thing you built, not five things."

    async def fault_gate(_result):
        routed["fault_gate"] = True
        return ""

    monkeypatch.setattr("core.together.remarks.remark_in_a_with", in_a_with)
    monkeypatch.setattr("core.perception.ambient_utterance.consider_utterance", fault_gate)
    monkeypatch.setattr("core.perception.ambient_utterance.latest_screen", lambda: object())
    held = ambient_presence.AmbientPresence()
    return held, routed


def test_in_the_bubble_what_she_sees_goes_to_the_company_she_is_keeping(presence):
    from core.perception.ambient_presence import KEEPING_COMPANY, PresenceMode

    held, routed = presence
    held.set_mode(PresenceMode.BUBBLE)

    asyncio.run(held._consider_speaking(None))

    assert routed["with"].who == PERSON and routed["with"].what == KEEPING_COMPANY
    assert "fault_gate" not in routed
    assert held.state()["has_utterance"] is True


def test_her_window_opening_ends_the_company_kept_at_the_desk(presence):
    from core.perception.ambient_presence import PresenceMode
    from core.together.a_with import get_together

    held, _routed = presence
    held.set_mode(PresenceMode.BUBBLE)
    asyncio.run(held._consider_speaking(None))
    assert get_together().current(PERSON) is not None

    held.set_mode(PresenceMode.WINDOW)

    assert get_together().current(PERSON) is None


def test_with_her_window_up_only_the_fault_gate_reaches_her(presence):
    from core.perception.ambient_presence import PresenceMode

    held, routed = presence
    held.set_mode(PresenceMode.WINDOW)

    asyncio.run(held._consider_speaking(None))

    assert routed.get("fault_gate") is True and "with" not in routed
