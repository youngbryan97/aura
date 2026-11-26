"""A way her work keeps failing, on goal after goal, is a goal of hers (Master Todo F16, 26 November 2026).

Five of her goals stalled on "its test did not raise what she raised", one fault in how her own code made faults again,
and nothing of hers noticed. Her records are read for why her work stopped, and a reason that stopped three of her goals
or more becomes a goal to mend what in her own code makes it happen, with what each showed said in it.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import her_goal_at_work as goals
from core.agency import what_keeps_failing_her as failing

pytestmark = pytest.mark.unit

_NOT_MADE_AGAIN = "its test did not raise what she raised"


def _stalled(slug: str, why: str = _NOT_MADE_AGAIN, *, by_design: bool = False, at: float = 10.0) -> goals.GoalRecord:
    return goals.GoalRecord(goal=f"Mend {slug}", slug=slug, status=goals.STALLED,
                            last_check={"passed": False, "why": why, "evidence": f"FAILED {slug}: 1 passed", "at": at,
                                        "by_design": by_design})


def test_a_reason_that_stopped_three_of_her_goals_is_a_way_she_fails():
    records = [_stalled("a"), _stalled("b"), _stalled("c", at=30.0), _stalled("d", why="something else")]
    (way,) = failing.ways_she_fails(records)
    assert (way.how, way.goals, way.last_at) == (_NOT_MADE_AGAIN, ("a", "b", "c"), 30.0)
    assert way.shown[0] == "a: FAILED a: 1 passed"


def test_two_are_not_a_pattern_and_a_stop_by_design_is_no_failure_of_hers():
    assert failing.ways_she_fails([_stalled("a"), _stalled("b")]) == []
    designed = [_stalled(slug, "it was raised in what governs her", by_design=True) for slug in "abc"]
    assert failing.ways_she_fails(designed) == []


def test_a_gate_that_refused_three_of_her_drafts_is_a_way_she_fails():
    def refused(slug: str) -> goals.GoalRecord:
        return goals.GoalRecord(goal=f"Mend {slug}", slug=slug, status=goals.WORKING,
                                offer={"commit": "x", "asked_at": 5.0,
                                       "gates": {"outside-check": {"passed": False, "tail": "did not hold"},
                                                 "lint": {"passed": True}}})
    stopped = goals.GoalRecord(goal="g", slug="z", offer={"gates": {"outside-check": {"passed": False, "finished": False}}})
    (way,) = failing.ways_she_fails([refused("a"), refused("b"), refused("c"), stopped])
    assert way.how == "her outside-check gate refused a draft of hers" and way.goals == ("a", "b", "c")


def test_a_way_she_fails_is_set_once_as_a_goal_with_what_it_showed(monkeypatch):
    kept: dict[str, goals.GoalRecord] = {}

    async def save(record):
        kept[record.slug] = record
    records = [_stalled("a"), _stalled("b"), _stalled("c")]
    monkeypatch.setattr(goals, "all_records", lambda: [*records, *kept.values()])
    monkeypatch.setattr(goals, "load", lambda slug: kept.get(slug))
    monkeypatch.setattr(goals, "save", save)
    (slug,) = asyncio.run(failing.set_what_keeps_failing_her())
    record = kept[slug]
    assert record.goal == f"Mend what in my own code keeps failing this way: {_NOT_MADE_AGAIN}"
    assert record.status == goals.WORKING and "This stopped 3 of my goals: a, b, c" in record.said[0]["words"]
    assert "FAILED b: 1 passed" in record.said[0]["words"]
    assert asyncio.run(failing.set_what_keeps_failing_her()) == [], "a goal under way is not set again"
    record.status, record.offer = goals.MERGED, {"merged_at": 20.0}
    assert asyncio.run(failing.set_what_keeps_failing_her()) == [], "mended since the way last stopped her"
    records.append(_stalled("d", at=40.0))
    assert asyncio.run(failing.set_what_keeps_failing_her()) == [slug], "it stopped her again after the mend"
    assert kept[slug].status == goals.WORKING and kept[slug].offers_before[-1]["merged_at"] == 20.0
