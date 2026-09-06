"""The same suite, whatever is underneath — and it catches a real difference.

LangGraph has one checkpointer interface and a conformance suite every backend
passes unchanged, so swapping a backend is a semantic contract rather than a
hope. Aura tests each concrete store against itself, which catches what that
store gets wrong and cannot catch what two stores disagree about — and the
disagreement is what breaks whatever swaps them.

Two stores here are real: a dict-backed one and one backed by the write
gateway on disk. They are different code answering the same questions, which
is the only arrangement in which "they agree" means anything.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.runtime.what_a_store_promises import (
    THE_PROMISES,
    AKeyedStore,
    what_this_store_keeps,
)


class InMemory:
    """The reference: what everything else has to behave like."""

    def __init__(self) -> None:
        self._held: dict[str, object] = {}

    def put(self, key: str, value: object) -> None:
        self._held[str(key)] = value

    def get(self, key: str, default: object = None) -> object:
        return self._held.get(str(key), default)

    def delete(self, key: str) -> None:
        self._held.pop(str(key), None)

    def keys(self):
        return list(self._held)


class OnDisk:
    """Different code, on a real file, answering the same questions."""

    def __init__(self, where: Path) -> None:
        self._where = where
        if not self._where.exists():
            self._where.write_text("{}", encoding="utf-8")

    def _read(self) -> dict:
        try:
            return json.loads(self._where.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, body: dict) -> None:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().write_text(
            self._where, json.dumps(body), source="a conformance test"
        )

    def put(self, key: str, value: object) -> None:
        body = self._read()
        body[str(key)] = value
        self._write(body)

    def get(self, key: str, default: object = None) -> object:
        return self._read().get(str(key), default)

    def delete(self, key: str) -> None:
        body = self._read()
        body.pop(str(key), None)
        self._write(body)

    def keys(self):
        return list(self._read())


@pytest.fixture()
def on_disk_factory():
    with tempfile.TemporaryDirectory() as where:
        at = Path(where)
        count = {"n": 0}

        def make() -> OnDisk:
            count["n"] += 1
            return OnDisk(at / f"store-{count['n']}.json")

        yield make


def test_the_promises_are_few_enough_to_mean_something():
    """A suite that tests everything tests nothing: no backend passes it."""

    assert 4 <= len(THE_PROMISES) <= 12, len(THE_PROMISES)
    assert len({name for name, _ in THE_PROMISES}) == len(THE_PROMISES)


def test_the_reference_store_keeps_every_promise():
    went = what_this_store_keeps(InMemory, called="in memory")
    assert went.conforms, went.broke
    assert not went.could_not_answer
    assert len(went.kept) == len(THE_PROMISES)


def test_a_store_on_disk_keeps_the_same_promises(on_disk_factory):
    went = what_this_store_keeps(on_disk_factory, called="on disk")
    assert went.conforms, went.broke
    assert len(went.kept) == len(THE_PROMISES)


def test_two_backends_agree_promise_for_promise(on_disk_factory):
    """The question a per-store test cannot ask."""

    memory = what_this_store_keeps(InMemory, called="in memory")
    disk = what_this_store_keeps(on_disk_factory, called="on disk")
    assert sorted(memory.kept) == sorted(disk.kept), (
        f"the backends disagree: {set(memory.kept) ^ set(disk.kept)}"
    )


def test_the_suite_catches_a_backend_that_merges_instead_of_replacing():
    """The suite has to be able to fail, or passing it says nothing."""

    class Merges(InMemory):
        def put(self, key: str, value: object) -> None:
            held = self._held.get(str(key))
            if isinstance(held, dict) and isinstance(value, dict):
                held.update(value)
            else:
                self._held[str(key)] = value

    went = what_this_store_keeps(Merges, called="one that merges")
    assert not went.conforms
    assert any("overwrite" in one for one in went.broke), went.broke


def test_a_thing_that_is_not_a_store_says_so_rather_than_failing_an_assertion():
    class NotAStore:
        pass

    went = what_this_store_keeps(NotAStore, called="not a store")
    assert not went.kept
    assert len(went.could_not_answer) == len(THE_PROMISES)
    assert went.conforms, "a thing that cannot answer is not a thing that broke a promise"


def test_each_promise_gets_a_fresh_store(on_disk_factory):
    """A suite whose fourth question depends on its third tests an order."""

    seen: list[int] = []

    class CountsItsBirths(InMemory):
        def __init__(self) -> None:
            super().__init__()
            seen.append(1)

    what_this_store_keeps(CountsItsBirths, called="counting")
    assert len(seen) == len(THE_PROMISES)


def test_the_protocol_is_checkable_at_runtime():
    assert isinstance(InMemory(), AKeyedStore)
    assert not isinstance(object(), AKeyedStore)
