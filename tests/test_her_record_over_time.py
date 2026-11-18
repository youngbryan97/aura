"""The record of what she changed in herself says what landed, what came back out, and what stopped the rest."""
from __future__ import annotations

from types import SimpleNamespace

from tools.her_record_over_time import events, summary

_DAY = 86400.0


def _record(goal, offer=None, before=(), said=()):
    return SimpleNamespace(goal=goal, offer=offer or {}, offers_before=list(before), said=list(said))


def test_a_landing_a_taking_back_and_each_refusal_are_read():
    commits = [("abcdef123", _DAY, "Mend a: it raised KeyError"), ("feed00000", 2 * _DAY, 'Revert "Mend a: it raised KeyError"')]
    records = [
        _record("Mend a", {"commit": "abcdef1234", "outside_bar_at_landing": "nothing against it"}),
        _record("Mend b", said=[{"words": "My own yes asked the gates first, and writing failed:\nthe tail", "at": 3 * _DAY},
                                {"words": "Not mended: its test did not raise what she raised.", "at": 4 * _DAY},
                                {"words": "Something else she said.", "at": 5 * _DAY}]),
    ]
    assert [(e["what"], e["why"]) for e in events(records, commits)] == [
        ("landed", "nothing against it"), ("taken back", ""), ("gate refused", "writing"),
        ("not mended", "its test did not raise what she raised.")]


def test_a_landing_from_before_the_outside_bar_says_so():
    found = events([], [("abcdef123", _DAY, "Mend c")])
    assert found[0]["why"] == "landed before the outside bar was read"


def test_the_summary_counts_what_stopped_her_and_leaves_out_what_came_before():
    found = events([_record("Mend e", said=[{"words": "Not mended: nothing proposed.", "at": _DAY},
                                            {"words": "Not mended: nothing proposed.", "at": 3 * _DAY},
                                            {"words": "Not mended: nothing proposed.", "at": 4 * _DAY}])])
    report = summary(found, since=2 * _DAY)
    assert report["stopped_by"] == [("nothing proposed.", 2)]
    assert sum(sum(day.values()) for day in report["days"].values()) == 2
