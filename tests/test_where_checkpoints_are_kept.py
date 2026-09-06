"""Two providers, one suite, and a checkpoint that says what it holds.

CrewAI came out above Aura on engineering maturity and its checkpoint layer
was named: typed trigger events across task, crew, agent, flow, LLM, tool and
memory; JSON and SQLite providers; instances restored from a checkpoint.

Two providers because one provider is an implementation pretending to be an
interface. A trigger taxonomy because "why was this checkpoint made" is a
question a resume has to answer and a timestamp cannot. A digest because
CrewAI's restore trusts the row, and a resume point that silently differs from
what was saved is worse than none — the run continues.
"""
from __future__ import annotations

import threading

import pytest

from core.state.where_checkpoints_are_kept import (
    THE_PROMISES,
    AKeptCheckpoint,
    ATrigger,
    InJson,
    InSqlite,
    TheCheckpointsAreKept,
    WhatCameBackIsNotWhatWentIn,
    what_a_checkpoint_store_promises,
)


@pytest.fixture
def providers(tmp_path):
    counter = {"n": 0}

    def a_json():
        counter["n"] += 1
        return InJson(tmp_path / f"checkpoints-{counter['n']}.json")

    def a_db():
        counter["n"] += 1
        return InSqlite(tmp_path / f"checkpoints-{counter['n']}.db")

    return (("json", a_json), ("sqlite", a_db))


def test_both_providers_keep_every_promise(providers):
    """The same suite, unchanged, against both."""
    for name, make in providers:
        kept = what_a_checkpoint_store_promises(make, called=name)
        broken = {promise: why for promise, why in kept.items() if why != "kept"}
        assert not broken, f"{name} broke: {broken}"
        assert set(kept) == set(THE_PROMISES)


def test_both_providers_are_the_same_kind_of_thing(providers, tmp_path):
    assert isinstance(InJson(tmp_path / "a.json"), TheCheckpointsAreKept)
    assert isinstance(InSqlite(tmp_path / "a.db"), TheCheckpointsAreKept)


def test_the_same_checkpoint_reads_the_same_from_either(tmp_path):
    one = AKeptCheckpoint.of(
        "after the tool", {"plan": ["a", "b"]}, trigger=ATrigger.TOOL_RAN
    )
    from_json = InJson(tmp_path / "a.json")
    from_db = InSqlite(tmp_path / "a.db")
    from_json.put(one)
    from_db.put(one)

    assert from_json.get("after the tool").to_dict() == (
        from_db.get("after the tool").to_dict()
    )


# ---------------------------------------------------------------- triggers


def test_every_trigger_is_one_of_the_named_kinds():
    """A timestamp cannot tell a tool call from the end of a turn."""
    assert {str(one) for one in ATrigger} >= {
        "turn ended", "task finished", "agent decided", "flow branched",
        "model answered", "tool ran", "memory wrote", "asked for",
    }


@pytest.mark.parametrize("trigger", list(ATrigger))
def test_a_trigger_survives_the_round_trip(tmp_path, trigger):
    store = InJson(tmp_path / "a.json")
    store.put(AKeptCheckpoint.of("one", {}, trigger=trigger))
    assert store.get("one").trigger is trigger


# ----------------------------------------------------------------- digests


def test_a_checkpoint_carries_a_digest_of_what_it_holds():
    one = AKeptCheckpoint.of("one", {"a": 1}, trigger=ATrigger.TURN_ENDED)
    other = AKeptCheckpoint.of("one", {"a": 2}, trigger=ATrigger.TURN_ENDED)
    assert one.digest and one.digest != other.digest


def test_the_same_state_written_two_ways_has_one_digest():
    assert AKeptCheckpoint.of(
        "one", {"a": 1, "b": 2}, trigger=ATrigger.TURN_ENDED
    ).digest == AKeptCheckpoint.of(
        "one", {"b": 2, "a": 1}, trigger=ATrigger.TURN_ENDED
    ).digest


def test_a_state_changed_underneath_is_refused(tmp_path):
    """A resume point that silently differs is worse than none: the run goes on."""
    store = InJson(tmp_path / "a.json")
    honest = AKeptCheckpoint.of("one", {"a": 1}, trigger=ATrigger.TURN_ENDED)
    store.put(honest)
    store.put(
        AKeptCheckpoint(
            name="one", trigger=ATrigger.TURN_ENDED, state={"a": 99},
            digest=honest.digest,
        )
    )
    with pytest.raises(WhatCameBackIsNotWhatWentIn, match="came back as"):
        store.get("one")


def test_a_checkpoint_with_no_digest_is_not_refused(tmp_path):
    """Rows written before digests existed still open."""
    store = InJson(tmp_path / "a.json")
    store.put(
        AKeptCheckpoint(name="old", trigger=ATrigger.TURN_ENDED, state={"a": 1})
    )
    assert store.get("old").state == {"a": 1}


# ----------------------------------------------------------- under threads


def test_many_threads_putting_leave_every_checkpoint_readable(tmp_path):
    store = InJson(tmp_path / "a.json")
    ready = threading.Barrier(8)

    def push(n: int):
        ready.wait()
        store.put(AKeptCheckpoint.of(f"c{n}", {"n": n}, trigger=ATrigger.TOOL_RAN))

    threads = [threading.Thread(target=push, args=(n,)) for n in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    assert store.names() == sorted(f"c{n}" for n in range(8))


def test_the_json_store_does_not_hold_its_lock_across_the_write(tmp_path):
    """Lockdep refuses an fsync under a lock, and it is right.

    A blocking call under a lock stalls everyone waiting on it. The document
    is written whole, so a write a newer one has overtaken may be skipped.
    """
    from core.runtime.lockdep import lockdep_report

    store = InJson(tmp_path / "a.json")
    for n in range(5):
        store.put(AKeptCheckpoint.of(f"c{n}", {"n": n}, trigger=ATrigger.TOOL_RAN))

    splats = lockdep_report().get("splats") or []
    assert not [
        one for one in splats
        if "checkpoints_in_json" in str(one)
    ], splats


def test_a_missing_file_is_an_empty_store(tmp_path):
    assert InJson(tmp_path / "never-written.json").names() == []
