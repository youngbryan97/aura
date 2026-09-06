"""The durable event log, attacked the way OpenHands' tests attack theirs.

A blind comparison called the OpenHands event log a stronger canonical
durable-event story than Aura's, and was specific about why: duplicate event-ID
rejection, parent validity, bounded traversal, stale-index recovery, gap
detection — and tests that attack duplicate IDs, missing files, corrupted JSON,
stale mappings, index gaps, concurrent threads and multiple instances writing
to one backing store.

These are those tests.
"""
from __future__ import annotations

import json
import threading

import pytest

from core.runtime.event_spine import EventLog, Lane, LineageBroken


def _a_line(seq: int, *, kind: str = "thing", parent: int = 0) -> str:
    return json.dumps(
        {
            "seq": seq,
            "kind": kind,
            "lane": Lane.SYSTEM.value,
            "payload": {},
            "at": 1.0,
            "actor": "",
            "causal_parent": parent,
        },
        separators=(",", ":"),
    )


# --------------------------------------------------------- parent validity


def test_an_event_may_name_a_parent_that_happened():
    log = EventLog()
    first = log.append("one", {})
    second = log.append("two", {}, causal_parent=first.seq)
    assert second.causal_parent == first.seq


def test_an_event_may_not_name_a_parent_that_has_not_happened():
    """Clamping to zero would make the log agree with itself and not reality."""
    log = EventLog()
    log.append("one", {})
    with pytest.raises(LineageBroken, match="has not happened yet"):
        log.append("two", {}, causal_parent=99)


def test_an_event_may_not_name_a_negative_parent():
    log = EventLog()
    with pytest.raises(LineageBroken):
        log.append("two", {}, causal_parent=-1)


def test_no_parent_is_allowed_and_counted():
    """Legacy events and roots both have no parent. That is not a defect."""
    log = EventLog()
    log.append("root", {})
    assert log.integrity()["whole"] is True


# ------------------------------------------------------ bounded traversal


def test_the_causal_chain_reads_back_in_order():
    log = EventLog()
    seqs = []
    parent = 0
    for name in ("a", "b", "c", "d"):
        parent = log.append(name, {}, causal_parent=parent).seq
        seqs.append(parent)
    assert log.ancestry(seqs[-1]) == list(reversed(seqs))


def test_a_traversal_is_bounded_by_how_far_it_was_asked_to_go():
    log = EventLog()
    parent = 0
    for _ in range(40):
        parent = log.append("x", {}, causal_parent=parent).seq
    assert len(log.ancestry(parent, most=5)) == 5


def test_a_cycle_in_an_edited_file_does_not_hang_the_reader(tmp_path):
    """A durable log is a file, and a file can be edited into a cycle."""
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        _a_line(1, parent=2) + "\n" + _a_line(2, parent=1) + "\n", encoding="utf-8"
    )
    log = EventLog(kept_at=kept)
    chain = log.ancestry(2)
    assert chain == [2, 1]


def test_an_ancestry_of_something_that_is_not_there_is_empty_not_a_raise():
    assert EventLog().ancestry(0) == []


# ------------------------------------------------------ duplicate rejection


def test_a_repeated_sequence_number_is_dropped_not_replayed(tmp_path):
    """Two writers on one file. Applying one event twice counts it twice."""
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        "\n".join([_a_line(1), _a_line(2), _a_line(2), _a_line(3)]) + "\n",
        encoding="utf-8",
    )
    log = EventLog(kept_at=kept)

    assert [one.seq for one in log.events()] == [1, 2, 3]
    integrity = log.integrity()
    assert integrity["duplicate_sequences"] == [2]
    assert integrity["whole"] is False


def test_the_next_sequence_number_clears_everything_in_the_file(tmp_path):
    kept = tmp_path / "spine.jsonl"
    kept.write_text("\n".join([_a_line(1), _a_line(7)]) + "\n", encoding="utf-8")
    log = EventLog(kept_at=kept)
    assert log.append("new", {}).seq == 8


# ---------------------------------------------------------- gap detection


def test_a_hole_in_the_sequence_is_named_by_its_range(tmp_path):
    """"Three gaps" does not say which experience is missing."""
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        "\n".join([_a_line(1), _a_line(2), _a_line(9), _a_line(10)]) + "\n",
        encoding="utf-8",
    )
    log = EventLog(kept_at=kept)
    assert log.integrity()["gaps"] == [[3, 8]]


def test_a_log_with_no_holes_says_it_is_whole(tmp_path):
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        "\n".join(_a_line(n) for n in (1, 2, 3)) + "\n", encoding="utf-8"
    )
    assert EventLog(kept_at=kept).integrity()["whole"] is True


# ------------------------------------------------- corruption and recovery


def test_a_missing_file_is_an_empty_log_and_not_an_error(tmp_path):
    log = EventLog(kept_at=tmp_path / "not-there.jsonl")
    assert log.events() == []
    assert log.append("first", {}).seq == 1


def test_a_half_written_last_line_keeps_everything_before_it(tmp_path):
    """The ordinary case: the process died mid-append."""
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        _a_line(1) + "\n" + _a_line(2) + "\n" + '{"seq": 3, "kind": "th',
        encoding="utf-8",
    )
    log = EventLog(kept_at=kept)

    assert [one.seq for one in log.events()] == [1, 2]
    assert log.integrity()["unreadable_after"] == 2
    assert log.integrity()["whole"] is False


def test_the_sequence_counter_survives_a_corrupt_tail(tmp_path):
    """The stale-index case: the next append must not reuse a kept number."""
    kept = tmp_path / "spine.jsonl"
    kept.write_text(_a_line(1) + "\n" + _a_line(2) + "\n" + "{ not json",
                    encoding="utf-8")
    log = EventLog(kept_at=kept)
    assert log.append("after", {}).seq == 3


def test_blank_lines_are_not_corruption(tmp_path):
    kept = tmp_path / "spine.jsonl"
    kept.write_text("\n" + _a_line(1) + "\n\n" + _a_line(2) + "\n\n",
                    encoding="utf-8")
    log = EventLog(kept_at=kept)
    assert [one.seq for one in log.events()] == [1, 2]
    assert log.integrity()["whole"] is True


# ------------------------------------------------------ concurrent writers


def test_many_threads_appending_produce_one_unbroken_sequence():
    log = EventLog()
    ready = threading.Barrier(8)

    def push():
        ready.wait()
        for _ in range(50):
            log.append("x", {})

    threads = [threading.Thread(target=push) for _ in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    seqs = [one.seq for one in log.events()]
    assert seqs == list(range(1, 401))
    assert len(set(seqs)) == len(seqs)


def test_a_file_that_already_holds_duplicates_is_still_detected(tmp_path):
    """Two writers no longer produce them, and a file may still contain them.

    Written by an older build, or by two processes that raced before the
    sequence lock existed. The reader drops the second copy rather than
    replaying an event that happened once, and says it did.
    """
    kept = tmp_path / "spine.jsonl"
    kept.write_text(
        "\n".join([_a_line(1, kind="from one"), _a_line(1, kind="from other")]) + "\n",
        encoding="utf-8",
    )
    log = EventLog(kept_at=kept)

    assert log.integrity()["duplicate_sequences"] == [1]
    assert [event.kind for event in log.events()] == ["from one"]


def test_the_integrity_report_is_in_the_spine_report():
    from core.runtime.event_spine import EventLog

    report = EventLog().integrity()
    assert set(report) >= {
        "kept_through", "next", "duplicate_sequences", "gaps",
        "unreadable_after", "whole",
    }


# ------------------------------------------------- safe across processes


def test_two_logs_on_one_file_no_longer_mint_the_same_number(tmp_path):
    """They did. The file held each number twice and half were dropped."""
    kept = tmp_path / "spine.jsonl"
    one = EventLog(kept_at=kept)
    other = EventLog(kept_at=kept)

    first = one.append("from one", {})
    second = other.append("from other", {})

    assert {first.seq, second.seq} == {1, 2}
    third = EventLog(kept_at=kept)
    assert [event.kind for event in third.events()] == ["from one", "from other"]
    assert third.integrity()["whole"] is True


def test_the_sequence_lock_is_a_sibling_and_not_the_log(tmp_path):
    """flock is per open-file description.

    Locking the log itself deadlocked the process against its own writer on
    the first append, because atomic_append_text flocks it through a
    different descriptor.
    """
    kept = tmp_path / "spine.jsonl"
    EventLog(kept_at=kept).append("one", {})
    assert kept.exists()
    assert (tmp_path / "spine.seq.lock").exists()


def test_the_write_does_not_happen_under_the_thread_lock(tmp_path):
    """Lockdep refuses an fsync under a lock, and it is right."""
    from core.runtime.lockdep import lockdep_report

    log = EventLog(kept_at=tmp_path / "spine.jsonl")
    for _ in range(5):
        log.append("x", {})

    splats = [
        one for one in (lockdep_report().get("splats") or [])
        if "event_spine" in str(one)
    ]
    assert splats == [], splats


def test_many_threads_on_a_durable_log_keep_one_unbroken_sequence(tmp_path):
    log = EventLog(kept_at=tmp_path / "spine.jsonl")
    ready = threading.Barrier(6)

    def push():
        ready.wait()
        for _ in range(20):
            log.append("x", {})

    threads = [threading.Thread(target=push) for _ in range(6)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    seqs = [one.seq for one in log.events()]
    assert seqs == list(range(1, 121))


# ----------------------------------------------------- deterministic rebuild


def test_a_rebuild_reads_the_file_again_and_says_what_came_back(tmp_path):
    log = EventLog(kept_at=tmp_path / "spine.jsonl")
    for name in ("a", "b", "c"):
        log.append(name, {})

    said = log.rebuild()
    assert said["read_back"] == 3
    assert said["lost"] == 0
    assert said["whole"] is True
    assert [one.kind for one in log.events()] == ["a", "b", "c"]


def test_a_rebuild_is_the_same_twice(tmp_path):
    """Deterministic: the same file gives the same events in the same order."""
    log = EventLog(kept_at=tmp_path / "spine.jsonl")
    for name in ("a", "b"):
        log.append(name, {})

    first = [one.to_dict() for one in (log.rebuild(), log.events())[1]]
    second = [one.to_dict() for one in (log.rebuild(), log.events())[1]]
    assert first == second


def test_a_rebuild_says_what_it_lost(tmp_path):
    """A recovery that cannot be checked is one nobody can rely on."""
    kept = tmp_path / "spine.jsonl"
    log = EventLog(kept_at=kept)
    for name in ("a", "b", "c"):
        log.append(name, {})
    kept.write_text(_a_line(1) + "\n", encoding="utf-8")

    said = log.rebuild()
    assert said["in_memory_before"] == 3
    assert said["read_back"] == 1
    assert said["lost"] == 2


# ------------------------------------------------------------ at scale


def test_ancestry_is_bounded_at_scale(tmp_path):
    """A causal query must not cost the length of the log."""
    import time

    log = EventLog()
    parent = 0
    for _ in range(20_000):
        parent = log.append("x", {}, causal_parent=parent).seq

    began = time.monotonic()
    chain = log.ancestry(parent, most=64)
    bounded = time.monotonic() - began

    assert len(chain) == 64
    assert bounded < 1.0, f"a bounded query took {bounded:.3f}s"
