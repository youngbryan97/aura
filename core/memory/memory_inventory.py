"""One place that answers what Aura remembers, and where it lives.

`core/memory` is 104 modules, 31 of which reach a storage backend directly:
SQLite, JSON flat files, in-process TF-IDF, vector stores, an encrypted vault,
conceptual gravitation that physically moves embeddings. `MemoryFacade`
coordinates nine of those stores and exposes each as a property — and there was
no way to ask it what the nine ARE. Nothing enumerated them, nothing said which
survive a restart, and nothing noticed when one was absent.

That absence has a specific cost. "Is this remembered?" had no answerable form:
a caller had to know which store to ask, and a store that failed to register
answered nothing while every other store answered normally, so the system read
as healthy and quietly forgot one kind of thing.

This is a register, not a new store. It holds no memories and owns no data. It
reports, for each store the facade coordinates:

* whether it resolved at all — an absent store is the failure that hides;
* its backend, read from the object rather than assumed from its name;
* whether it is durable, because "remembered" and "remembered until restart"
  are different answers and were indistinguishable;
* how many items it holds, when the store can say cheaply — and a stated
  refusal when it cannot, rather than a zero that reads as empty.

The register is derived from the facade's own attributes, so a tenth store
cannot be added without appearing here; a test asserts that.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("Memory.Inventory")

MEMORY_INVENTORY_SCHEMA = "aura.memory.inventory.v1"

#: The stores MemoryFacade coordinates, as (attribute, container key, durable).
#:
#: Durability is declared here because it is a property of the STORE, not
#: something an object can be asked at runtime — a JSON file and an in-process
#: dict present the same interface and answer "is this remembered" differently.
#: The attribute names are checked against the facade, so this list cannot
#: drift from the code it describes.
MEMORY_STORES: tuple[tuple[str, str, bool], ...] = (
    ("episodic", "episodic_memory", True),
    ("semantic", "semantic_memory", True),
    ("vector", "vector_memory", True),
    ("ledger", "knowledge_ledger", True),
    ("graph", "knowledge_graph", True),
    ("short_term", "short_term_memory", False),
    ("goals", "goal_memory", True),
    ("vault", "blackhole_vault", True),
    ("cold", "cold_store", True),
)

#: Methods a store may expose to report its own size, cheapest first. A store
#: that has none is not interrogated: walking a store to count it is exactly the
#: kind of work a health probe must not do.
_COUNT_METHODS = ("count", "size", "__len__", "item_count", "total_records")

# Health is often collected on the event-loop thread. An arbitrary ``count``
# method is not evidence that the operation is in-memory: ColdStore's method
# opened SQLite, and a routine health poll froze the loop long enough to taint
# the runtime. Stores can publish this scalar only when they maintain it as
# part of their own commit path.
_NONBLOCKING_COUNT_ATTRIBUTE = "health_item_count"

#: Attributes that name a store's backing technology. Read from the object so
#: the register reports what a store IS rather than what its name suggests.
_BACKEND_ATTRIBUTES = ("backend", "backend_name", "db_path", "path", "store_path")


@dataclass(frozen=True)
class MemoryStoreStatus:
    """What is known about one store right now."""

    name: str
    service_key: str
    durable: bool
    present: bool
    backend: str = "unresolved"
    item_count: int | None = None
    count_available: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryInventory:
    """Every store, and what the set of them adds up to."""

    stores: tuple[MemoryStoreStatus, ...]
    schema: str = MEMORY_INVENTORY_SCHEMA
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def present_count(self) -> int:
        return sum(1 for store in self.stores if store.present)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(store.name for store in self.stores if not store.present)

    @property
    def durable_missing(self) -> tuple[str, ...]:
        """Absent stores that were supposed to survive a restart.

        This is the set worth alarming on: a missing volatile store loses a
        conversation's working set, a missing durable store means something the
        runtime believes it recorded was never written anywhere.
        """
        return tuple(
            store.name for store in self.stores if not store.present and store.durable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stores": [store.to_dict() for store in self.stores],
            "present": self.present_count,
            "declared": len(self.stores),
            "missing": list(self.missing),
            "durable_missing": list(self.durable_missing),
            "notes": list(self.notes),
        }


def _describe_backend(store: Any) -> str:
    for attribute in _BACKEND_ATTRIBUTES:
        value = getattr(store, attribute, None)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null"}:
            return text[:160]
    return type(store).__name__


def _count_items(store: Any) -> tuple[int | None, bool, str]:
    """How many items the store holds, when it can answer cheaply.

    A store that cannot answer returns ``(None, False, reason)``. Reporting an
    unknown size as zero would make an unreadable store indistinguishable from
    an empty one, which is the reading that turns a broken store into "nothing
    was remembered".
    """
    cached_value = getattr(store, _NONBLOCKING_COUNT_ATTRIBUTE, None)
    if not isinstance(cached_value, bool) and isinstance(cached_value, int):
        if cached_value >= 0:
            return cached_value, True, f"via {_NONBLOCKING_COUNT_ATTRIBUTE}"

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        return None, False, "store exposes no declared nonblocking count"

    for method_name in _COUNT_METHODS:
        method = getattr(store, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except Exception as exc:  # noqa: BLE001 — a probe must not take the caller down
            return None, False, f"{method_name} raised {type(exc).__name__}"
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value, True, f"via {method_name}()"
    return None, False, "store exposes no cheap count"


def collect_memory_inventory(facade: Any) -> MemoryInventory:
    """Read the register off a live facade.

    Never raises: an inventory that can fail is one a health surface will wrap
    in a broad except and report as "no data", which is the same silence this
    exists to remove.
    """
    statuses: list[MemoryStoreStatus] = []
    notes: list[str] = []

    for name, service_key, durable in MEMORY_STORES:
        store = getattr(facade, name, None)
        if store is None:
            statuses.append(
                MemoryStoreStatus(
                    name=name,
                    service_key=service_key,
                    durable=durable,
                    present=False,
                    detail="not registered in the service container",
                )
            )
            continue

        try:
            backend = _describe_backend(store)
        except Exception as exc:  # noqa: BLE001
            backend = "undescribable"
            notes.append(f"{name}: backend probe raised {type(exc).__name__}")

        count, available, detail = _count_items(store)
        statuses.append(
            MemoryStoreStatus(
                name=name,
                service_key=service_key,
                durable=durable,
                present=True,
                backend=backend,
                item_count=count,
                count_available=available,
                detail=detail,
            )
        )

    inventory = MemoryInventory(stores=tuple(statuses), notes=tuple(notes))
    if inventory.durable_missing:
        # A durable store that never registered means something the runtime
        # believes it recorded was written nowhere. That is worth saying out
        # loud rather than leaving in a dictionary nobody reads.
        logger.warning(
            "🧠 Memory inventory: %d durable store(s) absent: %s",
            len(inventory.durable_missing),
            ", ".join(inventory.durable_missing),
        )
    return inventory


def memory_inventory_report() -> dict[str, Any]:
    """The register for the live runtime, for a health surface to publish."""
    try:
        from core.container import ServiceContainer

        facade = ServiceContainer.get("memory_facade", default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "schema": MEMORY_INVENTORY_SCHEMA,
            "available": False,
            "reason": f"container unavailable: {type(exc).__name__}",
        }

    if facade is None:
        return {
            "schema": MEMORY_INVENTORY_SCHEMA,
            "available": False,
            "reason": "no memory_facade registered",
        }

    report = collect_memory_inventory(facade).to_dict()
    report["available"] = True
    return report


__all__ = [
    "MEMORY_INVENTORY_SCHEMA",
    "MEMORY_STORES",
    "MemoryInventory",
    "MemoryStoreStatus",
    "collect_memory_inventory",
    "register_memory_health_fragment",
    "memory_inventory_report",
]


def register_memory_health_fragment() -> None:
    """Publish this register to the runtime health surface.

    Public and idempotent, and called by `MemoryFacade._refresh_subsystems`
    rather than left as an import side effect: a module is imported once per
    process, so an import-time registration cannot be re-established after a
    reset or a hot reload. A health fragment that silently stops publishing is
    the failure this register was built to remove.
    """
    try:
        from core.runtime.health_fragments import register_health_fragment

        register_health_fragment("memory_inventory", memory_inventory_report)
    except (ImportError, AttributeError):
        logger.debug("health fragment registry unavailable; inventory not published")


register_memory_health_fragment()
