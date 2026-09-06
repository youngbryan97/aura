"""One action history, read off the log rather than kept beside it.

AutoGPT keeps one action-history component; the closure asked for a canonical
ActionEpisode projection sourced from the event spine, with adapters rather
than a second store.

Aura has several places that remember what was done — the record of her own
work, the delivery journal, the influence ledger, the degradation register —
each written for its own question, and none able to answer "what did she do,
in order, and what came of it".
"""
from __future__ import annotations

import pytest

from core.runtime.event_spine import EventLog
from core.runtime.what_she_did_and_what_happened import (
    THE_ENDS,
    THE_STARTS,
    how_the_history_stands,
    the_action_history,
    what_never_finished,
)


@pytest.fixture
def log():
    return EventLog()


def test_an_action_that_began_and_ended_is_one_episode(log):
    log.append("action_started", {"action_id": "a1", "what": "read a file"})
    log.append("action_finished", {"action_id": "a1", "outcome": "ok"})

    history = the_action_history(log)
    assert len(history) == 1
    assert history[0].what == "read a file"
    assert history[0].finished
    assert history[0].outcome == "ok"


def test_an_action_that_never_finished_is_kept(log):
    """The question anybody actually has after a crash."""
    log.append("tool_called", {"call_id": "c9", "what": "search"})
    assert [one["what"] for one in what_never_finished(log)] == ["search"]


def test_they_come_back_in_the_order_they_happened(log):
    for at in range(3):
        log.append("action_started", {"action_id": at, "what": f"thing {at}"})
        log.append("action_finished", {"action_id": at, "outcome": "ok"})
    assert [one.what for one in the_action_history(log)] == [
        "thing 0", "thing 1", "thing 2"
    ]


def test_an_end_with_no_start_is_still_an_episode(log):
    """An action whose beginning is off the front of the log still happened."""
    log.append("action_finished", {"action_id": "gone", "outcome": "ok"})
    history = the_action_history(log)
    assert len(history) == 1
    assert history[0].began_at_event == 0
    assert history[0].finished


def test_two_actions_are_joined_by_their_own_ids(log):
    log.append("action_started", {"action_id": "a", "what": "first"})
    log.append("action_started", {"action_id": "b", "what": "second"})
    log.append("action_finished", {"action_id": "b", "outcome": "b done"})
    log.append("action_finished", {"action_id": "a", "outcome": "a done"})

    by_what = {one.what: one for one in the_action_history(log)}
    assert by_what["first"].outcome == "a done"
    assert by_what["second"].outcome == "b done"


def test_an_action_with_no_id_is_joined_by_its_actor(log):
    log.append("action_started", {"what": "a thing"}, actor="tools")
    log.append("action_finished", {"outcome": "ok"}, actor="tools")
    assert the_action_history(log)[0].finished


def test_the_starts_and_ends_are_declared_not_guessed():
    """An event kind that merely contains the word "action" is not one."""
    assert "action_started" in THE_STARTS
    assert "action_finished" in THE_ENDS
    assert not THE_STARTS & THE_ENDS


def test_an_unrelated_event_is_not_an_action(log):
    log.append("something_else_entirely", {"what": "not an action"})
    assert the_action_history(log) == []


def test_the_episode_carries_what_both_events_said(log):
    log.append("action_started", {"action_id": "a", "what": "x", "asked": 1})
    log.append("action_finished", {"action_id": "a", "outcome": "ok", "cost": 2})
    said = the_action_history(log)[0].said
    assert said["asked"] == 1
    assert said["cost"] == 2


def test_it_says_how_long_an_action_took(log):
    log.append("action_started", {"action_id": "a"}, clock=lambda: 100.0)
    log.append("action_finished", {"action_id": "a"}, clock=lambda: 102.5)
    assert the_action_history(log)[0].took_s == 2.5


def test_an_unfinished_action_took_no_measurable_time(log):
    log.append("action_started", {"action_id": "a"}, clock=lambda: 100.0)
    assert the_action_history(log)[0].took_s == 0.0


def test_it_is_a_projection_and_not_a_store(log):
    """If it is wrong it is thrown away and read again."""
    log.append("action_started", {"action_id": "a", "what": "x"})
    assert the_action_history(log)[0].what == "x"
    assert the_action_history(log)[0].what == "x"
    assert "not a fifth store" in how_the_history_stands(log)["what_this_is"]


def test_no_spine_is_an_empty_history_and_not_an_error(monkeypatch):
    import core.runtime.what_she_did_and_what_happened as module

    monkeypatch.setattr(module, "the_action_history", module.the_action_history)
    assert isinstance(the_action_history(EventLog()), list)


def test_the_report_counts_what_it_found(log):
    log.append("action_started", {"action_id": "a", "what": "unfinished"})
    log.append("action_started", {"action_id": "b", "what": "done"})
    log.append("action_finished", {"action_id": "b"})
    said = how_the_history_stands(log)
    assert said["episodes"] == 2
    assert said["unfinished"] == 1
    assert said["what_never_finished"] == ["unfinished"]
