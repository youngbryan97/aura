"""Asked to think out loud while something plays, a remark is what was asked for.

LIVE 20 October 2026, rehearsal 8: "think out loud while we watch", and six minutes in nothing said: "welcome 0.250 x
worth 0.954 x (1 + 1.00 lost by waiting) = 0.477 against (1 - welcome) x attention held 1.000 = 0.750".
"""
from __future__ import annotations

import pytest

from core.intent.asks_for_remarks import asks_for_remarks
from core.together.a_with import THEM, With
from core.together.when_to_say_something import worth_saying_now

ASKED = ("Let's watch something together. Go to YouTube and find a random video from Mark Rober, under 15 minutes. "
         "Before it starts, tell me what you think it'll be about, think out loud while we watch, and tell me what you "
         "thought after.")


@pytest.mark.parametrize("said", [
    ASKED,
    "Watch this with me and react as it goes: https://www.youtube.com/watch?v=ujkD4SxPKOI",
    "Put it on and tell me what you think as we watch.",
    "Talk me through it while it plays.",
])
def test_asked_to_say_what_she_thinks_as_it_plays(said):
    assert asks_for_remarks(said)


@pytest.mark.parametrize("said", [
    "Watch this with me: https://www.youtube.com/watch?v=ujkD4SxPKOI",
    "What do you think about volcanoes?",
    "Tell me what you thought after.",
])
def test_not_asked_to_talk_while_it_plays(said):
    assert not asks_for_remarks(said)


def _watching(invited: int) -> With:
    held = With(who="person:bryan", what="watching the gingerbread contest", proposed_by=THEM, began=0.0)
    held.invited = invited
    for at in range(0, 60, 6):  # a contest: something new every look
        held.shown(f"the contest at {at} seconds", float(at), change=1.0)
    return held


def test_invited_a_remark_of_interest_is_said_while_it_plays():
    said = worth_saying_now(_watching(3), 60.0, interest=0.95, thing="the collapse", lost_by_waiting=1.0)
    assert said.speak, said.why


def test_invited_she_still_keeps_quiet_through_what_does_not_interest_her():
    assert not worth_saying_now(_watching(3), 60.0, interest=0.05, thing="a logo", lost_by_waiting=1.0).speak


def test_uninvited_and_with_nothing_yet_to_go_on_she_waits():
    assert not worth_saying_now(_watching(0), 60.0, interest=0.95, thing="the collapse", lost_by_waiting=1.0).speak


def test_her_record_in_the_with_comes_to_outweigh_the_invitation():
    held = _watching(3)
    assert held.welcome() == pytest.approx(0.8)
    held.remarks, held.taken_up = 10, 0  # ten remarks of hers, none taken up
    assert held.welcome() < 0.4
