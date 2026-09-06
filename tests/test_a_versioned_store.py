"""The store, attacked at the properties Home Assistant scored highest for.

Versioned records, a forward-readable minor version, serialised and bounded
loads, deep copies of pending writes, migration hooks, atomic writes, a
read-only mode, a final write at shutdown, corruption detection, and corrupt
state put aside rather than overwritten.

The minor-version rule is the one Aura had nowhere, and it is the one a system
that keeps changing shape needs: a newer writer may only add fields, so an
older reader can still use what it understands, which is what lets a rollback
keep the data written since.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from core.persistence.a_versioned_store import (
    TOO_MANY_AT_ONCE,
    AVersionedStore,
    CannotRead,
)


@pytest.fixture
def where(tmp_path):
    return tmp_path / "kept.json"


# --------------------------------------------------------- reading, writing


def test_nothing_there_is_none_and_not_an_error(where):
    assert AVersionedStore(where, major=1).load() is None


def test_what_was_saved_comes_back(where):
    store = AVersionedStore(where, major=3, minor=2)
    assert store.save({"name": "Aura", "turns": 4})
    kept = store.load()
    assert kept.data == {"name": "Aura", "turns": 4}
    assert (kept.major, kept.minor) == (3, 2)
    assert kept.written_at > 0


def test_the_version_is_written_into_the_record(where):
    AVersionedStore(where, major=7, minor=1).save({"a": 1})
    raw = json.loads(where.read_text("utf-8"))
    assert raw["major"] == 7
    assert raw["minor"] == 1
    assert raw["data"] == {"a": 1}


# ----------------------------------------------------------- the two rules


def test_a_later_minor_version_is_still_readable(where):
    """A newer writer may only add fields."""
    AVersionedStore(where, major=2, minor=9).save({"kept": True, "new_field": 1})
    kept = AVersionedStore(where, major=2, minor=3).load()
    assert kept.data["kept"] is True
    assert kept.minor == 9


def test_a_later_major_version_is_refused_rather_than_guessed(where):
    """Acting on a guess is how a rollback loses data it could have kept."""
    AVersionedStore(where, major=5).save({"a": 1})
    with pytest.raises(CannotRead, match="this reader knows"):
        AVersionedStore(where, major=4).load()


def test_a_refused_forward_version_is_not_put_aside(where):
    """It is not corrupt. A later reader will want it exactly as it is."""
    AVersionedStore(where, major=5).save({"a": 1})
    with pytest.raises(CannotRead):
        AVersionedStore(where, major=4).load()
    assert where.exists()
    assert json.loads(where.read_text("utf-8"))["major"] == 5


# ------------------------------------------------------------- migration


def test_an_earlier_version_is_migrated(where):
    AVersionedStore(where, major=1).save({"old_name": "aura"})

    def rename(data, major, minor):
        return {"name": data.pop("old_name", "")} if major < 2 else data

    kept = AVersionedStore(where, major=2, migrate=rename).load()
    assert kept.data == {"name": "aura"}
    assert kept.major == 2


def test_an_earlier_version_with_no_migration_is_refused(where):
    AVersionedStore(where, major=1).save({"a": 1})
    with pytest.raises(CannotRead, match="nothing was given to migrate it"):
        AVersionedStore(where, major=2).load()


def test_the_migration_sees_the_version_it_is_migrating_from(where):
    AVersionedStore(where, major=1, minor=4).save({"a": 1})
    seen = {}

    def note(data, major, minor):
        seen["from"] = (major, minor)
        return data

    AVersionedStore(where, major=2, migrate=note).load()
    assert seen["from"] == (1, 4)


# ------------------------------------------------------------- corruption


def test_a_file_that_will_not_parse_is_put_aside_not_overwritten(where):
    """Overwriting corruption destroys the only evidence of what went wrong."""
    where.write_text("{ this is not json", encoding="utf-8")
    store = AVersionedStore(where, major=1)

    with pytest.raises(CannotRead) as caught:
        store.load()

    aside = caught.value.put_aside_at
    assert aside is not None and aside.exists()
    assert "this is not json" in aside.read_text("utf-8")
    assert not where.exists()


def test_a_file_that_is_not_a_record_is_put_aside(where):
    where.write_text(json.dumps({"nothing": "familiar"}), encoding="utf-8")
    with pytest.raises(CannotRead, match="not a versioned record"):
        AVersionedStore(where, major=1).load()


def test_a_record_whose_data_is_not_an_object_is_put_aside(where):
    where.write_text(json.dumps({"major": 1, "data": [1, 2, 3]}), encoding="utf-8")
    with pytest.raises(CannotRead, match="where an object belongs"):
        AVersionedStore(where, major=1).load()


def test_a_store_can_be_written_again_after_the_bad_copy_is_put_aside(where):
    where.write_text("{ broken", encoding="utf-8")
    store = AVersionedStore(where, major=1)
    with pytest.raises(CannotRead):
        store.load()
    assert store.save({"fresh": True})
    assert store.load().data == {"fresh": True}


# -------------------------------------------------------------- read-only


def test_a_read_only_store_refuses_to_hold_anything(where):
    AVersionedStore(where, major=1).save({"a": 1})
    store = AVersionedStore(where, major=1, read_only=True)
    assert store.load().data == {"a": 1}
    with pytest.raises(PermissionError):
        store.hold({"b": 2})


def test_a_read_only_store_does_not_put_a_corrupt_file_aside(where):
    """Moving a file is a write, and read-only means read-only."""
    where.write_text("{ broken", encoding="utf-8")
    with pytest.raises(CannotRead) as caught:
        AVersionedStore(where, major=1, read_only=True).load()
    assert caught.value.put_aside_at is None
    assert where.exists()


# -------------------------------------------------------- pending writes


def test_what_is_held_is_a_copy_of_what_the_caller_had(where):
    """A caller that keeps mutating its dict saved something else."""
    live = {"turns": [1, 2]}
    store = AVersionedStore(where, major=1)
    store.hold(live)
    live["turns"].append(3)
    live["added_later"] = True
    store.flush()

    assert store.load().data == {"turns": [1, 2]}


def test_flushing_with_nothing_held_writes_nothing(where):
    assert AVersionedStore(where, major=1).flush() is False
    assert not where.exists()


def test_a_flush_clears_what_was_held(where):
    store = AVersionedStore(where, major=1)
    store.hold({"a": 1})
    assert store.flush() is True
    assert store.flush() is False


# ---------------------------------------------------------------- shutdown


def test_the_final_write_saves_what_was_outstanding(where):
    store = AVersionedStore(where, major=1)
    store.hold({"unsaved": True})
    assert store.final_write() is True
    assert store.load().data == {"unsaved": True}


def test_the_final_write_says_so_when_nothing_was_outstanding(where):
    store = AVersionedStore(where, major=1)
    store.save({"already": True})
    assert store.final_write() is False


# ------------------------------------------------------ bounded concurrency


def test_loads_are_bounded_rather_than_serialised(tmp_path):
    """One at a time makes boot the sum of every load.

    The bound is what stops forty stores parsing together; it is not one.
    """
    assert TOO_MANY_AT_ONCE > 1


def test_many_threads_loading_one_store_all_get_the_same_answer(where):
    AVersionedStore(where, major=1).save({"shared": 1})
    store = AVersionedStore(where, major=1)
    seen: list[Any] = []
    ready = threading.Barrier(8)

    def read():
        ready.wait()
        seen.append(store.load().data)

    threads = [threading.Thread(target=read) for _ in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    assert seen == [{"shared": 1}] * 8


def test_many_threads_saving_leave_one_whole_record(where):
    store = AVersionedStore(where, major=1)
    ready = threading.Barrier(6)

    def write(n: int):
        ready.wait()
        store.save({"who": n})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    kept = AVersionedStore(where, major=1).load()
    assert set(kept.data) == {"who"}


def test_the_report_says_what_it_is_holding(where):
    store = AVersionedStore(where, major=2, minor=1)
    report = store.report()
    assert report["version"] == "2.1"
    assert report["holding"] is False
    store.hold({"a": 1})
    assert store.report()["holding"] is True


# ------------------------------------------------------- files from before


def test_a_file_written_before_this_store_existed_is_version_zero(where):
    """Adopting an existing store must not throw its file away."""
    where.write_text(json.dumps({"measures": [{"at": "gap"}]}), encoding="utf-8")

    def adopt(data, major, minor):
        return {"rows": data.get("measures", [])} if major < 1 else data

    kept = AVersionedStore(where, major=1, migrate=adopt).load()
    assert kept.data == {"rows": [{"at": "gap"}]}
    assert kept.major == 1


def test_a_file_from_before_is_refused_when_nothing_can_migrate_it(where):
    """Reading it as version zero would be a guess wearing a rule's clothes."""
    where.write_text(json.dumps({"measures": []}), encoding="utf-8")
    with pytest.raises(CannotRead, match="not a versioned record"):
        AVersionedStore(where, major=1).load()


# ------------------------------------------------------------- live wiring


def test_what_she_invented_reads_and_writes_through_the_store(tmp_path, monkeypatch):
    """A real store, adopted. Its old files still open."""
    from core.agency import what_she_invented

    kept = tmp_path / "properties.json"
    monkeypatch.setattr(what_she_invented, "_KEPT_AT", kept)

    store = what_she_invented._the_store()
    assert store.path == kept
    assert store.save({"measures": []})
    assert store.load().data == {"measures": []}
    assert json.loads(kept.read_text("utf-8"))["major"] == 1


def test_a_corrupt_file_is_reported_rather_than_read_as_a_fresh_install(
    tmp_path, monkeypatch
):
    """It used to return an empty dict, and the next keep wrote over it."""
    from core.agency import what_she_invented

    kept = tmp_path / "properties.json"
    kept.write_text("{ half a fi", encoding="utf-8")
    monkeypatch.setattr(what_she_invented, "_KEPT_AT", kept)

    seen = {}

    def note(subsystem, exc, **kwargs):
        seen["subsystem"] = subsystem
        seen["action"] = kwargs.get("action", "")

    monkeypatch.setattr(what_she_invented, "record_degradation", note)
    assert what_she_invented.recall() == {"measures": 0}
    assert seen["subsystem"] == "what_she_invented"
    assert "put the unreadable file aside" in seen["action"]
    assert not kept.exists()
    assert list(tmp_path.glob("properties.unreadable.*.json"))


def test_a_file_written_by_the_old_code_still_opens(tmp_path, monkeypatch):
    from core.agency import what_she_invented

    kept = tmp_path / "properties.json"
    kept.write_text(json.dumps({"measures": [], "at": 1.0}), encoding="utf-8")
    monkeypatch.setattr(what_she_invented, "_KEPT_AT", kept)

    assert what_she_invented.recall() == {"measures": 0}
    assert kept.exists(), "an old file must not be put aside as corrupt"
