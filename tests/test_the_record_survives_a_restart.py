"""Her metacognitive history has to belong to the continuing system.

The developmental policy chooses the next self-change from H_t, the
accumulated evidence about her own performance:

    Δ_t = π_dev(H_t)

Both halves of the persistence existed — ``keep_the_record`` wrote through
the governed file-write gateway, ``recall_the_record`` read it back — and
nothing in the tree called either of them. ``note_an_episode`` updated the
record in memory and stopped there. So every process restart was

    H_t → ∅

for the one input the policy is a function of. Other learned artefacts persist
by other mechanisms, so it did not reset all learning; it reset the history
that decides what to change next, which is worse than it sounds, because the
system then re-derives the same conclusion from the same afternoon forever.

Three things are checked here: it comes back without anybody asking, what
comes back is what went in, and the write never happens on the thread that is
answering.
"""

from __future__ import annotations

import json

import pytest

import core.cognition.the_record_of_her_own_work as record


@pytest.fixture
def a_fresh_state(tmp_path, monkeypatch):
    """Her record, in a file of its own, with nothing remembered yet."""

    monkeypatch.setattr(record, "_KEPT_AT", tmp_path / "the_record.json")
    record.forget_the_record()
    monkeypatch.setattr(record, "_RESTORED", [False])
    monkeypatch.setattr(record, "_UNWRITTEN", [0])
    yield tmp_path / "the_record.json"
    record.forget_the_record()


def _a_life(how_many: int) -> None:
    for turn in range(how_many):
        record.note_an_episode(
            "a shape that recurs",
            route="a meaning she induced" if turn % 2 else None,
            walked=100 + turn,
            used=("mirror",) if turn % 3 == 0 else (),
            tried="an operator she invented",
            about=[((0, 1), (1, 0))],
        )


def _restart(monkeypatch) -> None:
    """What a process restart does to the module: memory gone, file kept."""

    record.forget_the_record()
    monkeypatch.setattr(record, "_RESTORED", [False])
    monkeypatch.setattr(record, "_UNWRITTEN", [0])


def test_a_restart_used_to_empty_the_history(a_fresh_state, monkeypatch):
    """The defect, stated as the measurement it broke."""

    _a_life(40)
    assert record.how_often("a shape that recurs") == 40
    record.keep_the_record()

    _restart(monkeypatch)
    assert record.how_often("a shape that recurs") == 40, (
        "the recurrence estimate the policy reads did not survive"
    )


def test_nothing_has_to_remember_to_restore_it(a_fresh_state, monkeypatch):
    """No boot caller. The first question restores it."""

    _a_life(20)
    record.keep_the_record()
    _restart(monkeypatch)

    # Not `remember_what_she_had` — an ordinary read, the way the policy asks.
    assert record.what_it_has_cost("a meaning she induced") is not None
    assert len(record.episodes()) > 0


def test_what_comes_back_is_what_went_in(a_fresh_state, monkeypatch):
    """Including `tried`, which was dropped in all three places it lived."""

    record.note_an_episode(
        "dear",
        route=None,
        walked=9000,
        tried="an operator she invented",
        about=[((0, 1, 2), (2, 1, 0))],
    )
    before = record.episodes()[-1]
    assert before.tried == "an operator she invented"
    record.keep_the_record()
    _restart(monkeypatch)

    after = record.episodes()[-1]
    assert after.family == before.family
    assert after.walked == before.walked
    assert after.about == before.about
    assert after.tried == before.tried, (
        "everything tried and nothing held came back as nothing ever tried"
    )


def test_compaction_keeps_what_was_tried(a_fresh_state):
    """The oldest episodes lose their cases. They must not lose their attempt.

    A compacted episode rebuilt without `tried` turns "she tried everything
    she has and none of it held" into "she never tried", and those two states
    call for opposite actions: the first wants a new operator, the second
    wants the one she already has.
    """

    for turn in range(record.HOW_MANY_CASES_ARE_KEPT + 10):
        record.note_an_episode(
            f"family {turn}",
            route=None,
            walked=10,
            tried="an operator she invented",
            about=[((0, 1), (1, 0))],
        )
    oldest = record.episodes()[0]
    assert oldest.about == (), "this one should have been compacted"
    assert oldest.tried == "an operator she invented"


def test_it_writes_itself_back_on_a_cadence(a_fresh_state, monkeypatch):
    """Bounded: a hard kill loses an afternoon, not a history."""

    written: list[int] = []
    monkeypatch.setattr(record, "_ASKED_TO_WRITE", _Counting(written))

    _a_life(record.HOW_OFTEN_IT_IS_WRITTEN - 1)
    assert not written, "it asked to write before it had anything to write"
    _a_life(1)
    assert written == [1]
    _a_life(record.HOW_OFTEN_IT_IS_WRITTEN)
    assert written == [1, 1]


def test_the_write_does_not_happen_on_the_answering_thread(a_fresh_state, monkeypatch):
    """An fsync taken here once froze the live event loop for twenty minutes."""

    import threading

    wrote_on: list[str] = []
    real = record.keep_the_record

    def watched() -> bool:
        wrote_on.append(threading.current_thread().name)
        return real()

    monkeypatch.setattr(record, "keep_the_record", watched)
    here = threading.current_thread().name
    _a_life(record.HOW_OFTEN_IT_IS_WRITTEN * 2)
    record._ASKED_TO_WRITE.wait(timeout=5.0)
    assert here not in wrote_on


def test_a_corrupt_record_is_a_first_run_not_a_crash(a_fresh_state, monkeypatch):
    a_fresh_state.write_text("{not json at all")
    _restart(monkeypatch)
    assert record.how_often("anything") == 0
    record.note_an_episode("f", route=None, walked=1)
    assert record.how_often("f") == 1


def test_the_file_it_writes_is_readable_json(a_fresh_state):
    _a_life(4)
    assert record.keep_the_record() is True
    held = json.loads(a_fresh_state.read_text())
    assert held["seen"] == 4
    assert held["families"]["a shape that recurs"] == 4
    assert any(row.get("tried") for row in held["kept"])


class _Counting:
    """Stands in for the writer's event, counting the asks."""

    def __init__(self, into: list[int]) -> None:
        self._into = into

    def set(self) -> None:
        self._into.append(1)

    def clear(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> bool:
        return True
