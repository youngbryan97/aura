"""core/runtime/what_a_store_promises.py — the same suite, whatever is underneath.

LangGraph has one ``BaseCheckpointSaver`` interface and a conformance suite
every backend passes unchanged, so swapping a backend is a semantic contract
rather than a hope. CrewAI does the same for providers: golden message, tool,
stream and error cases, so a provider quirk cannot leak upward. Aura tests each
concrete store against itself, which catches what that store gets wrong and
cannot catch what two stores disagree about.

The disagreement is the thing. Two stores that both pass their own tests can
differ on what a read after a write returns, whether a delete of an absent key
raises, whether keys are ordered, whether an overwrite keeps or replaces — and
every one of those differences is a bug in whatever swaps them.

So: a protocol that says what any store of this kind promises, and a suite
that asks. The suite takes a factory and runs the same questions; a store that
cannot answer one says so in its own words rather than failing an assertion
about a method it never had.

Deliberately small. A conformance suite that tests everything tests nothing,
because no backend passes it and the failures stop meaning anything. These are
the promises something swapping backends actually relies on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("Aura.WhatAStorePromises")

__all__ = [
    "AKeyedStore",
    "HowItWent",
    "THE_PROMISES",
    "what_this_store_keeps",
]


@runtime_checkable
class AKeyedStore(Protocol):
    """The smallest thing every store here turns out to be.

    Named for what it is rather than for a backend. A metagraph, a state
    repository and a memory store are all this underneath, and the promises
    below are what something swapping them relies on.
    """

    def put(self, key: str, value: Any) -> Any: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def delete(self, key: str) -> Any: ...

    def keys(self) -> Iterable[str]: ...


@dataclass
class HowItWent:
    """What the suite found, promise by promise."""

    store: str
    kept: list[str] = field(default_factory=list)
    broke: list[str] = field(default_factory=list)
    could_not_answer: list[str] = field(default_factory=list)

    @property
    def conforms(self) -> bool:
        return not self.broke

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "conforms": self.conforms,
            "kept": list(self.kept),
            "broke": list(self.broke),
            "could_not_answer": list(self.could_not_answer),
        }


def _a_read_returns_what_was_written(store: Any) -> None:
    store.put("a", {"n": 1})
    assert store.get("a") == {"n": 1}, "a read did not return what was written"


def _an_absent_key_is_the_default(store: Any) -> None:
    assert store.get("nothing was put here", "the default") == "the default"


def _an_overwrite_replaces_rather_than_merges(store: Any) -> None:
    store.put("b", {"one": 1})
    store.put("b", {"two": 2})
    assert store.get("b") == {"two": 2}, "an overwrite merged instead of replacing"


def _a_delete_removes_it(store: Any) -> None:
    store.put("c", 1)
    store.delete("c")
    assert store.get("c") is None, "a deleted key came back"


def _deleting_what_is_not_there_does_not_raise(store: Any) -> None:
    store.delete("never existed")


def _keys_are_what_was_put(store: Any) -> None:
    for one in ("x", "y", "z"):
        store.put(one, one)
    found = set(store.keys())
    assert {"x", "y", "z"} <= found, f"keys() lost something: {sorted(found)}"


def _a_key_is_a_string_and_not_a_type(store: Any) -> None:
    """1 and "1" are different keys, or a numeric id collides with a name."""

    store.put("1", "a string")
    assert store.get("1") == "a string"


#: The promises, in the order they are asked. Small on purpose: a suite that
#: tests everything tests nothing, because no backend passes and the failures
#: stop meaning anything.
THE_PROMISES: tuple[tuple[str, Callable[[Any], None]], ...] = (
    ("a read returns what was written", _a_read_returns_what_was_written),
    ("an absent key is the default", _an_absent_key_is_the_default),
    ("an overwrite replaces rather than merges", _an_overwrite_replaces_rather_than_merges),
    ("a delete removes it", _a_delete_removes_it),
    ("deleting what is not there does not raise", _deleting_what_is_not_there_does_not_raise),
    ("keys are what was put", _keys_are_what_was_put),
    ("a key is a string and not a type", _a_key_is_a_string_and_not_a_type),
)


def what_this_store_keeps(make: Callable[[], Any], *, called: str = "") -> HowItWent:
    """Ask one store every promise. A fresh store per promise, from the factory.

    Fresh per promise because a suite whose fourth question depends on its
    third is testing an order rather than a contract, and the order is not
    what anybody swapping backends relies on.
    """

    went = HowItWent(store=called or getattr(make, "__name__", "a store"))
    for name, ask in THE_PROMISES:
        try:
            store = make()
        except Exception as exc:  # noqa: BLE001 — a store that will not build is a result
            went.could_not_answer.append(f"{name}: could not build ({exc})")
            continue
        if not isinstance(store, AKeyedStore):
            went.could_not_answer.append(f"{name}: not a keyed store")
            continue
        try:
            ask(store)
        except AssertionError as exc:
            went.broke.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — an unexpected raise is a break
            went.broke.append(f"{name}: raised {type(exc).__name__}: {exc}")
        else:
            went.kept.append(name)
    return went
