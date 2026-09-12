"""An intention one arm opened is not open for the arm after it.

The fork rolled the intention database back and left the loop's memory where
the arm had put it. On an isolated organism, an arm of four turns and two acts
opened nine intentions; after the restore the database held the ten rows it
started with and the loop still held all nineteen open, with its persist count
at 57 rather than 30. The count decides when the table is next pruned.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.agency.intention_loop import IntentionLoop
from core.subject.snapshot import _intentions_state, _restore_intentions


def _loop(tmp_path: Path) -> IntentionLoop:
    return IntentionLoop(db_path=str(tmp_path / "intentions.db"))


def _memory(loop: IntentionLoop) -> dict[str, Any]:
    return {
        "open": {
            key: (record.status.value, record.observation, len(record.actions_taken))
            for key, record in loop._active_intentions.items()
        },
        "finished": [(record.id, record.status.value) for record in loop._completed_intentions],
        "persists": loop._persist_count,
        "rows": sorted(map(repr, loop._conn.execute("SELECT * FROM intentions"))),
    }


def test_a_restore_closes_what_the_arm_opened(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    first = loop.intend("tidy the notes", drive="curiosity")
    loop.intend("finish the drawing", drive="competence")
    saved = _intentions_state(loop)
    before = _memory(loop)

    loop.intend("paint the panel", drive="competence")
    loop.abandon(first, "the arm changed its mind")
    assert _memory(loop) != before, "the arm changed nothing, so the check cannot fail"

    _restore_intentions(loop, saved)
    assert _memory(loop) == before


def test_every_arm_from_one_snapshot_gets_its_own_copy(tmp_path: Path) -> None:
    """One snapshot starts many arms, so an arm that edits a record it was
    handed must not edit the record the next arm is handed."""
    loop = _loop(tmp_path)
    kept = loop.intend("tidy the notes", drive="curiosity")
    saved = _intentions_state(loop)
    before = _memory(loop)

    _restore_intentions(loop, saved)
    loop._active_intentions[kept].observation = "the first arm wrote this"
    loop.abandon(kept, "the first arm gave up")

    _restore_intentions(loop, saved)
    assert _memory(loop) == before
    assert saved["active"][kept].observation is None


def test_a_caller_holding_the_open_set_sees_the_restore(tmp_path: Path) -> None:
    """Filled in place: something that took the dict before the arm reads
    what the restore put back, not the arm's leftovers."""
    loop = _loop(tmp_path)
    loop.intend("tidy the notes", drive="curiosity")
    held_open = loop._active_intentions
    held_finished = loop._completed_intentions
    saved = _intentions_state(loop)
    expected = set(held_open)

    added = loop.intend("paint the panel", drive="competence")
    loop.abandon(added, "the arm gave up")
    _restore_intentions(loop, saved)

    assert held_open is loop._active_intentions
    assert held_finished is loop._completed_intentions
    assert set(held_open) == expected
    assert list(held_finished) == []


def test_a_loop_without_a_database_is_left_alone() -> None:
    loop = SimpleNamespace(_conn=None, _active_intentions={"x": object()})
    assert _intentions_state(loop) is None
    _restore_intentions(loop, None)
    assert set(loop._active_intentions) == {"x"}
