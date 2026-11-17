"""What she says she cannot do is read for what it names, traced through her turn, and mended where the route dropped
it (Master Todo F16, part 4).

LIVE 16 November 2026, 00:33: "No. I don't have a tool that can highlight or interact with UI elements on your screen
right now." She had one, and its step had run.
"""
from __future__ import annotations

import pytest

from core.agency import gaps_she_names as gaps


class _Meanings:
    """A stand-in for her reading by meaning: fixed answers, and lessons that change them."""

    def __init__(self, which: dict[str, str], nearest: dict[str, str], decisive_once_taught: bool = True):
        self._which, self._nearest, self._decisive = dict(which), dict(nearest), decisive_once_taught
        self.taught: list[tuple[str, str]] = []

    def which(self, text, skills):
        return self._which.get(text)

    def nearest(self, text, skills):
        top = self._nearest.get(text)
        return [(top, 0.6), ("other", 0.4), ("third", 0.3)] if top else []

    def learned(self, text, name):
        self.taught.append((text, name))
        if self._decisive:
            self._which[text] = name

    def forget(self, text, name):
        self.taught.remove((text, name))
        return True


SAID = "I don't have a tool that can highlight or interact with UI elements on your screen right now."
REPLY = f"No. {SAID}"
ASKED = "Can you highlight the back button?"
SKILLS = {"desktop_task": object(), "web_search": object(), "notify_user": object()}


def test_her_sentences_that_say_she_cannot_are_found_and_quoted_speech_is_not():
    assert gaps.denials_in(f"{REPLY} But I can describe it.") == [SAID]
    assert gaps.denials_in('He wrote "I can\'t do that" in his note.') == []


def test_a_capability_she_lacks_is_kept_as_named():
    gap = gaps.trace(SAID, ASKED, [], SKILLS, _Meanings({}, {}))
    assert gap.finding == gaps.NOTHING_FOR_IT and gap.capability == "" and gap.mended == ""


def test_a_capability_that_ran_says_what_it_came_to():
    receipts = [{"tool": "desktop_task", "ok": False, "error": "overlay declined"}]
    gap = gaps.trace(SAID, ASKED, receipts, SKILLS, _Meanings({SAID: "desktop_task"}, {}))
    assert gap.finding == gaps.IT_RAN and gap.detail == "overlay declined"


def test_a_route_that_dropped_it_is_taught_and_kept_only_when_the_request_then_means_it():
    meanings = _Meanings({SAID: "desktop_task"}, {ASKED: "desktop_task"})
    gap = gaps.trace(SAID, ASKED, [], SKILLS, meanings)
    assert gap.finding == gaps.NEVER_REACHED and gap.mended.startswith("taught")
    assert meanings.taught == [(ASKED, "desktop_task")]

    stubborn = _Meanings({SAID: "desktop_task"}, {ASKED: "desktop_task"}, decisive_once_taught=False)
    gap = gaps.trace(SAID, ASKED, [], SKILLS, stubborn)
    assert gap.mended.startswith("taught and taken back") and stubborn.taught == []


def test_a_request_nearest_something_else_is_not_taught():
    meanings = _Meanings({SAID: "desktop_task"}, {ASKED: "web_search"})
    gap = gaps.trace(SAID, ASKED, [], SKILLS, meanings)
    assert gap.finding == gaps.NEVER_REACHED and gap.mended == "" and meanings.taught == []


def test_what_she_said_is_heard_after_her_reply_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(gaps, "_store", lambda: tmp_path / "gaps.json")
    found = gaps.hear_what_she_said_she_cannot(ASKED, REPLY, [], skills=SKILLS,
                                               meanings=_Meanings({SAID: "desktop_task"}, {ASKED: "desktop_task"}))
    assert [gap.finding for gap in found] == [gaps.NEVER_REACHED]
    kept = gaps.kept_gaps()
    assert kept[-1]["asked"] == ASKED and kept[-1]["mended"].startswith("taught")
    assert gaps.hear_what_she_said_she_cannot(ASKED, "Sure, here it is.", [], skills=SKILLS,
                                              meanings=_Meanings({}, {})) == []
