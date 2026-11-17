"""What she changed in herself is kept and shown, day by day (Master Todo F16, part 5).

Each change she landed keeps its receipt: who took it, the gates it passed, what was loaded into her running process
and what waits for her next start, whether it was undone there because it faulted, and the commit that took it back
out. An undo was held only in memory, so a restart between it and her taking the mend back out left the mend in her
source.
"""
from __future__ import annotations

import time

import pytest

from core.agency import her_goal_at_work as goals


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(goals, "records_dir", lambda: tmp_path)
    return tmp_path


def _landed(slug: str, *, at: float, **offer) -> goals.GoalRecord:
    record = goals.GoalRecord(goal=f"Mend {slug}", slug=slug, status=goals.MERGED)
    record.offer = {"summary": f"Mend {slug}", "commit": "a" * 40, "at": at, "merged_at": at, "taken_by": "herself",
                    "origin": "template:guard_none_value", "files": ["core/x.py"],
                    "gates": {"tests": {"passed": True}, "smoke": {"passed": True}}, **offer}
    goals._write_now(record)
    return record


def test_a_load_and_an_undo_are_kept_on_her_record_and_an_undo_outlives_a_restart(records):
    _landed("mend-x", at=time.time())
    assert goals.note_on_offer("mend-x", loaded={"functions": ["label"], "waits": [], "at": time.time()})
    assert goals.load("mend-x").offer["loaded"]["functions"] == ["label"]
    assert goals.mends_undone_on_record() == []
    goals.note_on_offer("mend-x", undone={"at": time.time(), "because": "core.x.label raised KeyError after it was "
                                                                       "loaded"})
    assert goals.mends_undone_on_record() == ["mend-x"], "read from her record, not from memory"
    goals.note_on_offer("mend-x", reverted="b" * 40)
    assert goals.mends_undone_on_record() == [], "taken back out"
    assert not goals.note_on_offer("no-such-goal", loaded={})


def test_an_undo_in_her_running_process_is_written_on_her_record_when_it_happens(records, monkeypatch):
    from core.self_modification import faults_she_can_reproduce as faults
    from core.self_modification import into_the_running_process as running

    _landed("mend-y", at=time.time())

    class _Undone:
        goal = "mend-y"

    monkeypatch.setattr(running, "undo_if_it_faulted", lambda *args: _Undone())

    class _Record:
        raised_in = ("core.x", "label")
        error_type = "builtins.KeyError"
        failed_call = None

    faults._undo_a_mend_that_did_not_hold(_Record())
    assert goals.load("mend-y").offer["undone"]["because"] == "core.x.label raised KeyError after it was loaded"
    assert faults.mends_undone() == ["mend-y"]


def test_her_changes_by_day_say_what_became_of_each(records):
    from core.agency.what_she_changed import changes_by_day

    day = time.mktime((2026, 11, 16, 15, 0, 0, 0, 0, -1))
    _landed("loaded", at=day, loaded={"functions": ["nth"], "waits": []})
    _landed("waiting", at=day + 60, loaded={"functions": [], "waits": ["a class changed shape"]})
    _landed("undone", at=day + 120, undone={"at": day + 130, "because": "core.x.label raised KeyError after it "
                                                                         "was loaded"})
    taken_back = _landed("back", at=day - 86400, reverted="c" * 40, reverted_at=day - 86000)
    goals._write_now(taken_back)
    unfinished = goals.GoalRecord(goal="Find something productive", slug="no-commit")
    goals._write_now(unfinished)

    days = changes_by_day()
    assert [d["day"] for d in days] == ["2026-11-16", "2026-11-15"]
    states = [c["state"] for c in days[0]["changes"]]
    assert states == ["undone, not yet taken back", "landed, waits for her next start", "landed and loaded"]
    assert days[0]["changes"][2]["loaded"] == ["nth"] and days[0]["changes"][2]["gates"] == {"tests": True,
                                                                                             "smoke": True}
    assert days[1]["changes"][0]["state"] == "taken back out" and days[1]["changes"][0]["taken_back_by"] == "c" * 40
    assert all(c["goal"] != "Find something productive" for d in days for c in d["changes"])


def test_the_page_and_its_route_are_served():
    from pathlib import Path

    from interface.routes import inner_state

    assert any(getattr(route, "path", "") == "/api/self/changes" for route in inner_state.router.routes)
    page = Path("interface/static/what_she_changed.html").read_text(encoding="utf-8")
    assert 'fetch("/api/self/changes"' in page
