"""In a stretch of play where nothing has been said for a while, how it is going is said: what she has noticed and what
she is doing, each only where it is true and new, in turn.

LIVE 2026-10-13 a point-and-click game was played for 106 seconds with nothing said, before an audience.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from core.agency.keeping_a_watcher_company import KEEP_COMPANY_S, keeping_company
from core.cognition.what_she_notices import Notebook


def _a_stretch(**kept):
    return SimpleNamespace(**{"began": 0.0, "said_at": 0.0, "gains": 0, "losses": 0, "clicks": 0, "taps": 0,
                              "input_key_downs": Counter(), "company": {}, "lines": [], **kept})


def test_nothing_is_said_while_something_was_said_lately():
    run = _a_stretch(clicks=9)
    assert keeping_company(run, KEEP_COMPANY_S - 1, Notebook()) == ""
    run.said_at = 20.0
    assert keeping_company(run, 30.0, Notebook()) == ""


def test_after_a_silence_the_kinds_of_line_take_turns_and_nothing_is_made_up():
    notes = Notebook()
    notes.notice("new lamp", "that's new: a lamp showed up", 5.0, kind="new")
    run = _a_stretch(gains=159, losses=9, clicks=30)
    said = []
    for at in (20.0, 40.0, 60.0):
        said.append(keeping_company(run, at, notes))
        run.said_at = at
    assert said[0] == "That's new: a lamp showed up."
    assert said[1] == "Still at it: 30 clicks in the last 40 seconds."
    assert said[2] == ""                                        # nothing new noticed, no moves since
    run.taps = 1
    run.input_key_downs["space"] = 6
    assert keeping_company(run, 80.0, notes) == "Still at it: 6 presses in the last 40 seconds."
    assert "for me" not in " ".join(said)                       # what she counts is not the game's own count


def test_a_noticing_is_said_as_something_she_noticed():
    notes = Notebook()
    notes.notice("after click gain", "after I click, I tend to gain", 5.0, kind="after")
    assert keeping_company(_a_stretch(), 20.0, notes) == "I've noticed after I click, I tend to gain."


def test_a_note_she_has_already_said_is_not_said_again():
    notes = Notebook()
    notes.notice("pot", "the white one looks like a pot", 5.0)
    run = _a_stretch(lines=["Clicking near the pot: the white one looks like a pot."])
    assert keeping_company(run, 20.0, notes) == ""


def test_the_play_loop_keeps_company(monkeypatch):
    """A pass of her play loop with nothing else to say says what keeping a watcher company finds."""
    from core.agency import playing_as_it_happens as play

    heard: list[str] = []
    monkeypatch.setattr(play, "keeping_company", lambda run, at: "I've noticed the white one looks like a pot.")
    monkeypatch.setattr(play, "_say", lambda run, say, line, at, **_k: heard.append(line) if line else None)
    run = SimpleNamespace(keys=[], began=0.0, contest=SimpleNamespace(says=lambda: ""), contest_first=None,
                          counters_first=None, contest_said="", counted={}, said_at=0.0, stance_said_at=0.0, said=set())
    hers = SimpleNamespace(thing=lambda moves: None, keys_that_move_her=lambda keys: [], kind=None, makes=[])
    meeting = SimpleNamespace(evidence={}, readouts=SimpleNamespace(values={}))
    play._what_she_says(run, None, SimpleNamespace(things={}), hers, meeting, 20.0)
    assert heard == ["I've noticed the white one looks like a pot."]
