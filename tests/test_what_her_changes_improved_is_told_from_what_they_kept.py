"""Whether a change of hers made something better is told from the evidence her records keep (F16, outside review).

An outside review of 19 November (F16-09) asked for a record that tells improvement from preservation, keeping the
source before each change, the change, what else was measured with it and how it is undone, and that never calls a
change an improvement because its tests pass. These records stand for her own: a site her loop made cheaper and kept
on its measurement (LIVE 18 November, a collection that held her 203 ms at the median held her 1 ms after), mends of
faults her running raised, a drill, a revert, and a change whose every gate passed with nothing measured.
"""
from __future__ import annotations

import pytest

from core.agency.her_goal_at_work import GoalRecord
from core.agency.mending_a_fault_of_hers import goal_for
from core.agency.what_her_changes_improved import KINDS, improvements, what_her_changes_improved
from core.runtime.a_call_that_failed import FailedCall
from core.self_modification.faults_she_can_reproduce import HeardFault

pytestmark = pytest.mark.unit

LANDED = 1_000_000.0
NOW = LANDED + 3 * 86400.0


def _fault(module: str, qualname: str, *, last_at: float) -> HeardFault:
    call = FailedCall(module=module, qualname=qualname, file=module.replace(".", "/") + ".py", line=12,
                      error_type="builtins.AttributeError", error_message="'NoneType' object has no attribute 'name'")
    return HeardFault(call=call, subsystem="test", first_at=LANDED - 3600.0, last_at=last_at)


def _landed(goal: str, **offer) -> GoalRecord:
    return GoalRecord(goal=goal, slug=goal[:20].lower().replace(" ", "-"), status="merged", base="b" * 40,
                      offer={"commit": "c" * 40, "merged_at": LANDED, "taken_by": "her own yes", **offer})


HELD_FAULT = _fault("core.memory.recall", "Recall.best", last_at=LANDED - 60.0)
AGAIN_FAULT = _fault("core.memory.recall", "Recall.worst", last_at=LANDED + 7200.0)
DRILL_FAULT = _fault("core.self_modification.loop_drill", "label", last_at=LANDED - 60.0)


@pytest.mark.parametrize(("record", "kind"), [
    (_landed("Lower what the full collections asked for at core/ops/lymphatic_reaper.py:360 cost her",
             measured_before={"median_s": 0.2027}, judged={"kept": True, "measured_after": {"median_s": 0.0009},
             "why": "each collection held her 1 ms at the median against 203 ms before"}), "a cost lowered"),
    (_landed(goal_for(HELD_FAULT)), "a failure repaired"),
    (_landed(goal_for(AGAIN_FAULT)), "a repair that did not hold"),
    (_landed(goal_for(DRILL_FAULT)), "a drill"),
    (_landed("Mend core.voice.ears.listen, which raised KeyError while she ran: 'frame'"),
     "a mend whose fault is no longer on record"),
    (_landed("Keep a small note store for user-specific context", reverted="d" * 40), "taken back"),
    (_landed("Keep a small note store for user-specific context",
             gates={name: {"passed": True} for name in ("tests", "smoke", "lint", "outside-check")}),
     "kept, not shown to improve anything"),
])
def test_each_change_is_told_by_what_its_records_show(record, kind):
    found = improvements([record], [HELD_FAULT, AGAIN_FAULT, DRILL_FAULT], now=NOW)
    assert [entry["kind"] for entry in found] == [kind]


def test_a_repair_says_how_long_it_has_held_and_how_it_is_undone():
    entry, = improvements([_landed(goal_for(HELD_FAULT))], [HELD_FAULT], now=NOW)
    assert entry["evidence"] == "not heard in the 3.0 days since it landed"
    assert entry["before"] == "b" * 40 and entry["undo"] == "git revert ccccccccc"


def test_a_change_that_never_landed_is_not_read():
    drafted = GoalRecord(goal="Keep a small note store", slug="keep-a-small-note", status="offered",
                         offer={"commit": "c" * 40})
    assert improvements([drafted], [], now=NOW) == []


def test_every_kind_is_counted():
    counted = what_her_changes_improved([_landed(goal_for(HELD_FAULT))], [HELD_FAULT], now=NOW)["counts"]
    assert set(counted) == set(KINDS) and counted["a failure repaired"] == 1 and sum(counted.values()) == 1
