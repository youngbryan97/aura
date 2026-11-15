"""SQLAlchemy's ORM is read in a thread, not by whoever first asks for the audit log.

`persistent_state` is a lazily-built singleton, so the first caller pays for
importing `core.db.orm`, and that pulls in SQLAlchemy's ORM: hundreds of modules
to find, read and compile. Four of the 120 newest stall dumps caught the event
loop inside `sqlalchemy.orm`'s own module bodies, reached through
`create_persistent_state`.

An import is idempotent and `sys.modules` is shared, so reading it in a thread at
registration leaves the factory alone and lets its first call find the work done.
The import stays inside the factory on purpose: hoisted to module scope, a test
patching `core.db.orm.PersistentState` would stop being seen.
"""

from __future__ import annotations

import sys

import pytest

from core.providers import memory_provider

pytestmark = pytest.mark.unit


class _Container:
    """Enough of a service container to register memory services into and ask one back."""

    def __init__(self) -> None:
        self.factories: dict[str, object] = {}

    def register(self, name, factory, **_kwargs) -> None:
        self.factories[name] = factory

    def get(self, name, default=None):
        factory = self.factories.get(name)
        return factory() if callable(factory) else default


def test_registration_warms_it(monkeypatch):
    warmed: list[bool] = []
    monkeypatch.setattr(memory_provider, "_warm_the_orm_off_the_loop", lambda: warmed.append(True))
    memory_provider.register_memory_services(_Container())
    assert warmed == [True]


def test_the_warming_happens_in_a_thread(monkeypatch):
    """The ORM is imported on a daemon thread of its own, not on whoever registered the services."""
    import threading

    seen = threading.Event()
    where: list[threading.Thread] = []
    original = __import__

    def noted(name, *args, **kwargs):
        if name == "core.db.orm":
            where.append(threading.current_thread())
            seen.set()
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", noted)
    memory_provider._warm_the_orm_off_the_loop()
    assert seen.wait(30.0)
    assert where[0] is not threading.current_thread() and where[0].daemon


def test_the_factory_keeps_its_own_import(monkeypatch):
    """A patch of PersistentState is what the registered factory builds: its import is read when it is called."""
    pytest.importorskip("sqlalchemy")
    import core.db.orm as orm

    class Planted:
        pass

    monkeypatch.setattr(orm, "PersistentState", Planted)
    monkeypatch.setattr(memory_provider, "_warm_the_orm_off_the_loop", lambda: None)
    container = _Container()
    memory_provider.register_memory_services(container)
    assert isinstance(container.get("persistent_state"), Planted)


def test_a_missing_sqlalchemy_does_not_raise(monkeypatch):
    def refuse(name, *args, **kwargs):
        if name.startswith("core.db.orm"):
            raise ImportError("no sqlalchemy")
        return original(name, *args, **kwargs)

    original = __import__
    monkeypatch.setattr("builtins.__import__", refuse)
    # The warming thread swallows it; the factory's own path answers it.
    memory_provider._warm_the_orm_off_the_loop()


def test_it_leaves_the_module_importable_afterwards():
    memory_provider._warm_the_orm_off_the_loop()
    # Whether SQLAlchemy is present or not, nothing is left half-imported.
    assert "core.db.orm" not in sys.modules or sys.modules["core.db.orm"] is not None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
