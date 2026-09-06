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


# --------------------------------------------------- sync and async pairs


@pytest.mark.parametrize("which", ["json", "sqlite"])
def test_every_store_offers_an_awaitable_pair(tmp_path, which):
    """Writing a checkpoint fsyncs, and an fsync on the loop froze this
    runtime for twenty minutes once."""
    import asyncio

    from core.state.where_checkpoints_are_kept import AnAsyncStore

    store = (
        InJson(tmp_path / f"{which}.json")
        if which == "json"
        else InSqlite(tmp_path / f"{which}.db")
    )
    assert isinstance(store, AnAsyncStore)

    one = AKeptCheckpoint.of("a", {"x": 1}, trigger=ATrigger.TURN_ENDED)

    async def go():
        await store.put_async(one)
        assert (await store.get_async("a")).state == {"x": 1}
        assert await store.names_async() == ["a"]
        assert await store.forget_async("a") is True
        assert await store.get_async("a") is None

    asyncio.run(go())


def test_the_async_pair_is_the_same_store_and_not_a_second_one():
    """Two implementations of one contract drift.

    The async pair drifting from the sync one is the drift nobody notices
    until a checkpoint is missing.
    """
    from core.state.where_checkpoints_are_kept import AnAsyncStore

    for name in ("put", "get", "names", "forget"):
        assert hasattr(AnAsyncStore, f"{name}_async")
        assert not hasattr(AnAsyncStore, name), (
            f"AnAsyncStore defines {name} itself instead of using the store's"
        )


# ------------------------------------------------------ how a state is written


def test_the_default_writer_is_json_a_person_can_read():
    """A checkpoint nobody can read is a resume point nobody can check."""
    from core.state.where_checkpoints_are_kept import THE_DEFAULT_WRITER

    written = THE_DEFAULT_WRITER.dumps({"b": 2, "a": 1})
    assert written == '{"a":1,"b":2}'
    assert THE_DEFAULT_WRITER.loads(written) == {"a": 1, "b": 2}


def test_the_writer_sorts_so_one_state_has_one_digest():
    from core.state.where_checkpoints_are_kept import THE_DEFAULT_WRITER

    assert THE_DEFAULT_WRITER.dumps({"a": 1, "b": 2}) == THE_DEFAULT_WRITER.dumps(
        {"b": 2, "a": 1}
    )


def test_a_writer_is_anything_that_answers_the_two_calls():
    from core.state.where_checkpoints_are_kept import HowAStateIsWritten

    class Backwards:
        def dumps(self, state):
            return repr(state)[::-1]

        def loads(self, raw):
            return eval(raw[::-1])  # noqa: S307 — a test's own round trip

    assert isinstance(Backwards(), HowAStateIsWritten)
    assert Backwards().loads(Backwards().dumps({"a": 1})) == {"a": 1}


# --- Lineage ------------------------------------------------------------------


def _three_on_two_branches(store):
    from core.state.where_checkpoints_are_kept import AKeptCheckpoint, ATrigger

    first = AKeptCheckpoint.of("first", {"n": 1}, trigger=ATrigger.TURN_ENDED)
    second = AKeptCheckpoint.of(
        "second", {"n": 2}, trigger=ATrigger.TOOL_RAN, after="first"
    )
    retried = AKeptCheckpoint.of(
        "retried", {"n": 3}, trigger=ATrigger.TOOL_RAN, after="first", branch="a retry"
    )
    for one in (first, second, retried):
        store.put(one)
    return first, second, retried


def test_a_retry_is_a_branch_rather_than_a_replacement(tmp_path):
    """The first attempt is still there, which is what makes the second a claim."""
    from core.state.where_checkpoints_are_kept import InJson, the_branches

    store = InJson(tmp_path / "c.json")
    _three_on_two_branches(store)
    assert the_branches(store) == {"a retry": 1, "main": 2}
    assert store.get("second").state == {"n": 2}
    assert store.get("retried").state == {"n": 3}


def test_the_line_back_walks_the_parents(tmp_path):
    from core.state.where_checkpoints_are_kept import InJson, the_line_back_from

    store = InJson(tmp_path / "c.json")
    _three_on_two_branches(store)
    assert [one.name for one in the_line_back_from(store, "second")] == [
        "second",
        "first",
    ]
    assert [one.name for one in the_line_back_from(store, "retried")] == [
        "retried",
        "first",
    ]


def test_a_row_naming_itself_as_its_parent_does_not_hang_the_walk(tmp_path):
    """A store is a file other processes write, so that row can arrive."""
    from core.state.where_checkpoints_are_kept import (
        AKeptCheckpoint,
        ATrigger,
        InJson,
        the_line_back_from,
    )

    store = InJson(tmp_path / "c.json")
    store.put(
        AKeptCheckpoint.of("loop", {"n": 1}, trigger=ATrigger.TURN_ENDED, after="loop")
    )
    assert [one.name for one in the_line_back_from(store, "loop")] == ["loop"]


def test_a_missing_parent_ends_the_walk_rather_than_raising(tmp_path):
    from core.state.where_checkpoints_are_kept import (
        AKeptCheckpoint,
        ATrigger,
        InJson,
        the_line_back_from,
    )

    store = InJson(tmp_path / "c.json")
    store.put(
        AKeptCheckpoint.of("child", {}, trigger=ATrigger.TURN_ENDED, after="gone")
    )
    assert [one.name for one in the_line_back_from(store, "child")] == ["child"]


def test_pruning_keeps_the_newest_on_each_branch_separately(tmp_path):
    """Pruning globally deletes a short branch to make room on a long one."""
    from core.state.where_checkpoints_are_kept import (
        AKeptCheckpoint,
        ATrigger,
        InJson,
        prune,
        what_is_on,
    )

    store = InJson(tmp_path / "c.json")
    for n in range(10):
        one = AKeptCheckpoint.of(f"m{n}", {"n": n}, trigger=ATrigger.TURN_ENDED)
        one = type(one)(**{**one.to_dict(), "trigger": one.trigger, "at": float(n)})
        store.put(one)
    kept_short = AKeptCheckpoint.of(
        "s0", {"n": 0}, trigger=ATrigger.TURN_ENDED, branch="a retry"
    )
    store.put(kept_short)

    gone = prune(store, keep=3)
    assert len(gone) == 7
    assert [one.name for one in what_is_on(store, "main")] == ["m7", "m8", "m9"]
    assert [one.name for one in what_is_on(store, "a retry")] == ["s0"], (
        "the short branch was pruned to make room on the long one"
    )


def test_pruning_never_removes_a_checkpoint_something_still_comes_after(tmp_path):
    """A chain with a hole restores to a history that stops mid-sentence."""
    from core.state.where_checkpoints_are_kept import (
        AKeptCheckpoint,
        ATrigger,
        InJson,
        prune,
    )

    store = InJson(tmp_path / "c.json")
    for n in range(6):
        one = AKeptCheckpoint(
            name=f"m{n}",
            trigger=ATrigger.TURN_ENDED,
            state={"n": n},
            at=float(n),
            after=f"m{n - 1}" if n else "",
        )
        store.put(one)
    prune(store, keep=2)
    assert store.get("m0") is not None, "m1 still names it"
    assert store.get("m4") is not None and store.get("m5") is not None


def test_a_row_written_before_branches_existed_is_on_the_main_line(tmp_path):
    from core.state.where_checkpoints_are_kept import AKeptCheckpoint, ATrigger

    old = AKeptCheckpoint.from_dict(
        {"name": "x", "trigger": str(ATrigger.TURN_ENDED), "state": {}, "at": 1.0}
    )
    assert old.branch == "main"
    assert old.after == ""


def test_both_providers_keep_the_lineage_promise(tmp_path):
    import itertools

    from core.state.where_checkpoints_are_kept import (
        InJson,
        InSqlite,
        what_a_checkpoint_store_promises,
    )

    n = itertools.count()
    for make, label in (
        (lambda: InJson(tmp_path / f"j{next(n)}.json"), "InJson"),
        (lambda: InSqlite(tmp_path / f"s{next(n)}.db"), "InSqlite"),
    ):
        kept = what_a_checkpoint_store_promises(make, called=label)
        broken = {k: v for k, v in kept.items() if v != "kept"}
        assert broken == {}, f"{label}: {broken}"
        assert "a parent and a branch survive the round trip" in kept


def test_an_async_put_waits_for_the_writes_that_produced_it(tmp_path):
    """A checkpoint on disk before its own writes restores a state that never was."""
    import asyncio
    import threading
    import time

    from core.state.nothing_lands_before_its_writes import (
        a_write_in_flight,
        forget_everything,
    )
    from core.state.where_checkpoints_are_kept import AKeptCheckpoint, ATrigger, InJson

    forget_everything()
    try:
        store = InJson(tmp_path / "c.json")
        landed: list[str] = []

        def slow_write():
            with a_write_in_flight("turn-4", "state.json"):
                time.sleep(0.15)
                landed.append("state.json")

        async def go():
            worker = threading.Thread(target=slow_write)
            worker.start()
            await asyncio.sleep(0.02)
            await store.put_async(
                AKeptCheckpoint.of("turn-4", {"n": 1}, trigger=ATrigger.TURN_ENDED)
            )
            worker.join()
            return landed

        assert asyncio.run(go()) == ["state.json"]
        assert store.get("turn-4") is not None
    finally:
        forget_everything()
