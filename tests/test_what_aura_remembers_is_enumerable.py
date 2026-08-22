"""One place answers what Aura remembers and where it lives.

`core/memory` is 104 modules, 31 of which reach a storage backend directly:
SQLite, JSON flat files, in-process TF-IDF, vector stores, an encrypted vault.
MemoryFacade coordinates nine of those and exposed each as a property — and
there was no way to ask it what the nine ARE. Nothing enumerated them, nothing
said which survive a restart, and nothing noticed when one was absent.

The cost is specific: a store that failed to register answered nothing while
every other store answered normally, so the surface read as healthy while one
kind of thing was quietly not being remembered.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from core.memory.memory_inventory import (
    MEMORY_INVENTORY_SCHEMA,
    MEMORY_STORES,
    collect_memory_inventory,
    memory_inventory_report,
)

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "core" / "memory" / "memory_facade.py"


class _Facade:
    """Only the attributes the register reads."""

    def __init__(self, **stores):
        for name, _key, _durable in MEMORY_STORES:
            setattr(self, name, stores.get(name))


class _Sqlite:
    db_path = "/data/episodes.db"

    @staticmethod
    def count() -> int:
        return 4127


class _Volatile:
    backend = "in-process deque"

    def __len__(self) -> int:
        return 12


class _Opaque:
    pass


class _Hostile:
    @staticmethod
    def count():
        raise RuntimeError("index corrupt")


class _BlockingCount:
    calls = 0

    @classmethod
    def count(cls) -> int:
        cls.calls += 1
        time.sleep(0.06)
        return 1


# ── the register describes the facade, and cannot drift from it ────────────


def test_every_declared_store_is_a_facade_property():
    """A register naming a store the facade does not have would report on
    nothing while looking complete."""
    tree = ast.parse(FACADE.read_text("utf-8"))
    properties = {
        node.name
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "MemoryFacade"
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        )
    }

    declared = {name for name, _key, _durable in MEMORY_STORES}
    assert declared <= properties, f"register names non-stores: {declared - properties}"


def test_no_facade_store_is_missing_from_the_register():
    """A tenth store cannot be added without appearing here. This is the
    property that keeps the register from becoming a stale document."""
    source = FACADE.read_text("utf-8")
    tree = ast.parse(source)
    resolved = {
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in {"get", "peek"}
        and node.targets[0].attr.startswith("_")
    }
    resolved = {name.lstrip("_") for name in resolved}
    declared = {name for name, _key, _durable in MEMORY_STORES}

    assert resolved, "found no store resolution in the facade; the check is vacuous"
    assert resolved <= declared, f"facade resolves stores the register omits: {resolved - declared}"


def test_the_service_keys_match_the_facade():
    source = FACADE.read_text("utf-8")

    for _name, key, _durable in MEMORY_STORES:
        assert f'"{key}"' in source, key


# ── what it reports ───────────────────────────────────────────────────────


def test_an_absent_store_is_named_not_omitted():
    inventory = collect_memory_inventory(_Facade())

    assert inventory.present_count == 0
    assert len(inventory.stores) == len(MEMORY_STORES)
    assert set(inventory.missing) == {n for n, _k, _d in MEMORY_STORES}


def test_a_missing_durable_store_is_separated_from_a_missing_volatile_one():
    """A missing volatile store loses a conversation's working set. A missing
    durable store means something the runtime believes it recorded was never
    written anywhere."""
    every_durable = {
        name: _Sqlite() for name, _k, durable in MEMORY_STORES if durable
    }
    inventory = collect_memory_inventory(_Facade(**every_durable))

    assert inventory.durable_missing == ()
    assert "short_term" in inventory.missing


def test_a_store_that_can_count_reports_its_size():
    inventory = collect_memory_inventory(_Facade(episodic=_Sqlite()))
    episodic = next(s for s in inventory.stores if s.name == "episodic")

    assert episodic.present is True
    assert episodic.item_count == 4127
    assert episodic.count_available is True
    assert episodic.backend == "/data/episodes.db"


def test_dunder_len_counts_too():
    inventory = collect_memory_inventory(_Facade(short_term=_Volatile()))
    short_term = next(s for s in inventory.stores if s.name == "short_term")

    assert short_term.item_count == 12
    assert short_term.backend == "in-process deque"
    assert short_term.durable is False


def test_a_store_that_cannot_count_says_so_rather_than_reporting_zero():
    """Zero would make an unreadable store indistinguishable from an empty one,
    which is the reading that turns a broken store into "nothing was
    remembered"."""
    inventory = collect_memory_inventory(_Facade(vector=_Opaque()))
    vector = next(s for s in inventory.stores if s.name == "vector")

    assert vector.present is True
    assert vector.item_count is None
    assert vector.count_available is False
    assert "no cheap count" in vector.detail


def test_a_store_whose_count_raises_does_not_take_the_probe_down():
    inventory = collect_memory_inventory(_Facade(episodic=_Hostile()))
    episodic = next(s for s in inventory.stores if s.name == "episodic")

    assert episodic.present is True
    assert episodic.count_available is False
    assert "RuntimeError" in episodic.detail


@pytest.mark.asyncio
async def test_health_inventory_does_not_call_undeclared_blocking_count_on_loop():
    _BlockingCount.calls = 0

    inventory = collect_memory_inventory(_Facade(cold=_BlockingCount()))
    cold = next(s for s in inventory.stores if s.name == "cold")

    assert _BlockingCount.calls == 0
    assert cold.item_count is None
    assert cold.count_available is False
    assert "nonblocking" in cold.detail


def test_the_backend_is_read_from_the_object_not_guessed_from_the_name():
    inventory = collect_memory_inventory(_Facade(graph=_Opaque()))
    graph = next(s for s in inventory.stores if s.name == "graph")

    assert graph.backend == "_Opaque"


def test_the_report_is_serialisable():
    import json

    payload = collect_memory_inventory(_Facade(episodic=_Sqlite())).to_dict()

    assert json.loads(json.dumps(payload))["schema"] == MEMORY_INVENTORY_SCHEMA


# ── wired into the live surface ───────────────────────────────────────────


def test_the_report_is_honest_when_no_facade_is_registered():
    """"Unavailable" and "nine stores, all empty" must not look the same."""
    from core.container import ServiceContainer

    ServiceContainer.clear()
    try:
        report = memory_inventory_report()
    finally:
        ServiceContainer.clear()

    assert report["available"] is False
    assert "no memory_facade" in report["reason"]


def test_the_health_report_carries_the_inventory():
    from core.runtime.health_contract import runtime_health_report

    report = runtime_health_report()

    assert "memory_inventory" in report
    assert report["memory_inventory"]["schema"] == MEMORY_INVENTORY_SCHEMA


def test_a_live_facade_produces_a_full_register():
    from core.container import ServiceContainer
    from core.memory.memory_facade import MemoryFacade
    from core.runtime.health_contract import runtime_health_report

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance("episodic_memory", _Sqlite(), required=False)
        ServiceContainer.register_instance("short_term_memory", _Volatile(), required=False)
        facade = MemoryFacade()
        facade.setup()
        ServiceContainer.register_instance("memory_facade", facade, required=False)

        inventory = runtime_health_report()["memory_inventory"]
    finally:
        ServiceContainer.clear()

    assert inventory["available"] is True
    assert inventory["declared"] == len(MEMORY_STORES)
    assert inventory["present"] == 2
    episodic = next(s for s in inventory["stores"] if s["name"] == "episodic")
    assert episodic["item_count"] == 4127
    assert "semantic" in inventory["durable_missing"]


@pytest.mark.parametrize("name,_key,_durable", MEMORY_STORES)
def test_each_store_is_reported_by_name(name, _key, _durable):
    reported = {s.name for s in collect_memory_inventory(_Facade()).stores}

    assert name in reported
