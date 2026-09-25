"""A service asked for from a coroutine: a built one at once, a first build off the loop.

Building a service can import a library and open its files. LIVE 2026-09-24:
the first skill after a boot built the persistent-state store on the event
loop, importing sqlalchemy, and the loop stood still for 5.1 s.

Its own module rather than a method, because ServiceContainer is held to the
number of methods it has.
"""

from __future__ import annotations

from typing import Any

__all__ = ["service_off_the_loop"]

_ABSENT = "_SENTINEL"


async def service_off_the_loop(name: str, default: Any = _ABSENT) -> Any:
    """``ServiceContainer.get(name, default)``, never building on the loop."""
    from core.container import ServiceContainer
    from core.runtime.executors import off_the_loop

    missing = object()
    built = ServiceContainer.peek(name, default=missing)
    if built is not missing:
        return built
    return await off_the_loop(ServiceContainer.get, name, default=default)
