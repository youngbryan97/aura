"""Her growth is read from the commit that declared what wrote it.

Bryan decided on 23 October that the record of her growth begins at e19047ec7:
before it, 33 steps of her live window changed how far she had grown under a
contract that never named the write, so each read as a defect. Those steps are
before the record and reported beside it; after it, every measured step must
develop, as before.
"""

from __future__ import annotations

from core.subject.bridge import lineage_term
from core.subject.lineage import HER_RECORD_BEGINS_AT, Lineage


def _person(**changes) -> dict:
    person = {"values": ["honesty"], "preferences": {"depth": 0.8}}
    person.update(changes)
    return person


def _wrote(at: float) -> list[dict]:
    return [{"transform": "PreferenceLearning", "honoured": True, "tick": "t", "at": at}]


def _line(record_begins_at: float = 0.0) -> Lineage:
    line = Lineage(record_begins_at=record_begins_at)
    line.record("t1", _person(), at=1.0)
    line.record("t2", _person(preferences={"depth": 0.6}), at=3.0)
    line.record("t3", _person(preferences={"depth": 0.5}), at=5.0)
    # Measured, but the change it made has no recorded writer: a defect.
    line.descend("t1", "t2", writes={"values": _wrote(2.0)})
    line.descend("t2", "t3", writes={"preferences:depth": _wrote(4.0)})
    return line


def test_read_from_the_first_measured_step_the_early_defect_counts():
    line = _line()
    assert lineage_term(line).status == "DOES_NOT_DEVELOP"


def test_read_from_the_fix_the_early_defect_is_before_the_record_and_reported():
    line = _line(record_begins_at=4.0)
    reading = line.development()
    assert reading["steps_since_the_record"] == 1 and reading["developing"] == 1
    assert reading["failed_before_the_record"] == 1
    assert lineage_term(line).status == "RECORDED"


def test_her_record_begins_at_the_fix_commit():
    assert HER_RECORD_BEGINS_AT == 1792766937.0
