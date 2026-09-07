"""An hourly job with an owner, a schedule, and no verdicts."""
from __future__ import annotations

import pytest

from core.verify import why_the_campaign_did_not_run as record
from core.verify.why_the_campaign_did_not_run import (
    forget_everything,
    how_the_campaign_has_gone,
    note_a_consideration,
    note_a_verdict,
    the_bar_has_never_been_met,
)


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "where_it_is_kept", lambda: tmp_path / "c.json")
    forget_everything()
    yield
    forget_everything()


def test_never_admitted_and_no_verdicts_yet_stop_looking_identical() -> None:
    """A deferral logged a reason and left no count, so both read the same."""
    assert not the_bar_has_never_been_met(), "nothing has come up yet"
    for _ in range(5):
        note_a_consideration("deferred", because="idle_for_only_120s")
    assert the_bar_has_never_been_met()
    said = how_the_campaign_has_gone()
    assert said["considered"] == 5
    assert said["ran"] == 0
    assert said["share_admitted"] == 0.0


def test_the_reason_that_refuses_most_often_is_named() -> None:
    """Which condition to change first, rather than which to guess at."""
    for _ in range(7):
        note_a_consideration("deferred", because="idle_for_only_120s")
    for _ in range(2):
        note_a_consideration("deferred", because="memory_at_91_percent")
    said = how_the_campaign_has_gone()
    assert said["refuses_most_often"] == "idle_for_only_120s"
    assert said["refused_because"]["memory_at_91_percent"] == 2


def test_a_run_that_reached_a_verdict_is_told_from_one_that_added_a_sample() -> None:
    note_a_consideration("ran", channel="affect")
    said = how_the_campaign_has_gone()
    assert said["ran"] == 1
    assert said["seconds_since_a_verdict"] is None, "a sample is not a verdict"

    note_a_verdict("affect")
    said = how_the_campaign_has_gone()
    assert said["seconds_since_a_verdict"] is not None
    assert said["ran"] == 2


def test_the_bar_being_met_once_settles_the_question_about_the_bar() -> None:
    note_a_consideration("deferred", because="idle_for_only_120s")
    assert the_bar_has_never_been_met()
    note_a_consideration("ran", channel="affect")
    assert not the_bar_has_never_been_met()


def test_the_record_survives_the_process(tmp_path) -> None:
    note_a_consideration("deferred", because="idle_for_only_120s")
    assert record.where_it_is_kept().exists()

    # A fresh process reads it back rather than starting from zero.
    forget_everything()
    assert how_the_campaign_has_gone()["considered"] >= 1


def test_recording_never_raises_when_the_path_itself_raises(monkeypatch) -> None:
    """A record that cannot be written is not a reason to stop the job."""
    monkeypatch.setattr(
        record, "where_it_is_kept", lambda: (_ for _ in ()).throw(OSError("no"))
    )
    note_a_consideration("deferred", because="something")
    assert how_the_campaign_has_gone()["considered"] >= 1


def test_the_conductor_records_through_this() -> None:
    """Wired, not beside it: the job's own path notes what happened."""
    import inspect

    from core.runtime import autonomy_conductor

    source = inspect.getsource(
        autonomy_conductor.AutonomyConductor._job_influence_campaign
    )
    assert "note_a_consideration" in source
    # Every way out of the job, not only the one that ran. A job that records
    # its runs and not its deferrals answers the wrong question: the reason it
    # produced nothing is in the deferrals.
    flat = " ".join(source.split())
    for ending in ("deferred", "unavailable", "idle", "ran"):
        assert f'note_a_consideration( "{ending}"' in flat or (
            f'note_a_consideration("{ending}"' in flat
        ), ending
