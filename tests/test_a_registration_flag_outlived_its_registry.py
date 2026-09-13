"""The flag said registered; the container was empty.

`register_all_services` marks `_full_run` and returns early on every later
call. `ServiceContainer.clear()` resets that flag — but it is the only thing
that does, so any other route that empties the registry left the flag
standing, and every subsequent call handed back a container with nothing in
it.

The symptom was domain W vanishing from the subject battery about half the
time under random ordering: `unified_world_model` was never re-registered,
the organ read nothing, and `live_domains()` came back one short. A flake
that depends on which file ran first is a fact about the early return, not
about the battery.
"""

from __future__ import annotations

import pytest

from core.container import ServiceContainer
from core.service_registration import _REGISTRATION_WITNESS, register_all_services


@pytest.fixture(autouse=True)
def _pristine():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def test_a_standing_flag_over_an_emptied_registry_registers_again():
    register_all_services()
    assert ServiceContainer.has(_REGISTRATION_WITNESS)

    # Emptied WITHOUT clear(), which is the route that left the flag standing.
    with ServiceContainer._lock:
        ServiceContainer._services.clear()
        ServiceContainer._aliases.clear()
    assert getattr(register_all_services, "_full_run", False) is True
    assert not ServiceContainer.has(_REGISTRATION_WITNESS)

    register_all_services()
    assert ServiceContainer.has(_REGISTRATION_WITNESS)


def test_the_world_model_comes_back_with_it():
    """The service whose absence took domain W off the battery."""
    register_all_services()
    with ServiceContainer._lock:
        ServiceContainer._services.clear()
        ServiceContainer._aliases.clear()

    register_all_services()
    assert ServiceContainer.has("unified_world_model")


def test_a_populated_registry_still_returns_early(monkeypatch):
    """The guard must not turn an idempotent call into a re-registration."""
    register_all_services()
    calls: list[int] = []
    import core.service_registration as mod

    original = mod._register_all_services_body
    monkeypatch.setattr(
        mod,
        "_register_all_services_body",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    register_all_services()
    assert calls == []
