"""A memory store registered and not started yet is not a store that is missing.

The memory facade keeps the handles it resolved when it was made, and early in
a boot that can be before the stores exist. Read off the facade alone, eight
durable stores were "absent" (LIVE 2026-09-24, at boot) while every one of them
was registered and about to start: a warning that says something recorded was
written nowhere, when nothing had been recorded yet.
"""
from __future__ import annotations

import pytest

from core.container import ServiceContainer
from core.memory.memory_inventory import MEMORY_STORES, collect_memory_inventory


class _Early:
    """A facade made before any store was built: every handle is None."""

    def __getattr__(self, name):
        return None


class _Store:
    def count(self) -> int:
        return 7


@pytest.fixture
def empty_container():
    ServiceContainer.clear()
    yield ServiceContainer
    ServiceContainer.clear()


def test_a_registered_store_the_facade_has_not_seen_is_not_missing(empty_container):
    for _name, key, _durable in MEMORY_STORES:
        empty_container.register(key, _Store, required=False)
    inventory = collect_memory_inventory(_Early())
    assert inventory.durable_missing == ()
    assert all("not started yet" in store.detail for store in inventory.stores)


def test_a_built_store_the_facade_has_not_seen_is_read_from_the_container(empty_container):
    empty_container.register_instance("episodic_memory", _Store(), required=False)
    inventory = collect_memory_inventory(_Early())
    episodic = next(store for store in inventory.stores if store.name == "episodic")
    assert episodic.present and episodic.item_count == 7


def test_a_store_nobody_registered_is_still_missing(empty_container):
    inventory = collect_memory_inventory(_Early())
    assert set(inventory.durable_missing) == {name for name, _k, durable in MEMORY_STORES if durable}
