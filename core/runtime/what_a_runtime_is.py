"""What a runtime is, said once, so two of them can be compared.

Aura has two objects that act as its runtime: ``AuraKernel``, which owns the
cognitive tick, and ``RobustOrchestrator``, which owns the process. Before
this module they shared exactly one public method — ``stop`` — so nothing
could take one and substitute the other, and no test could ask either of
them the question a runtime is for.

The five things the ledger asks a runtime boundary to expose are lifecycle,
message, state, capability and subscription. Each is named here with a
``runtime_`` prefix and delegates to what the implementation already does:
this is a boundary drawn around existing behaviour, not new behaviour. The
prefix is not decoration — ``start``, ``stop`` and ``state`` already mean
different things on the two classes, and renaming either one on a live
runtime to make a protocol fit would be the protocol changing the system to
suit itself.

``the_services_it_can_address`` is the ownership half (Soar #3). One runtime
instance owning every authoritative service is the goal; what is measurable
today is how many of the declared names a given runtime actually resolves,
and which it does not. That number goes up.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "THE_OPERATIONS",
    "AuraRuntime",
    "TheRuntimeOver",
    "a_runtime_over",
    "the_services_it_can_address",
    "what_is_missing_from",
]

#: The runtime surface, by what each operation is for. A Protocol alone
#: reports "not an instance" and nothing else; this says which of the five is
#: absent, which is the difference between a failing test and a useful one.
THE_OPERATIONS: dict[str, str] = {
    "runtime_start": "lifecycle: bring the runtime up",
    "runtime_stop": "lifecycle: bring it down",
    "runtime_deliver": "message: hand it something to act on",
    "runtime_state": "state: the state it is acting on, or None",
    "runtime_service": "capability: resolve an authoritative service by name",
    "runtime_watch": "subscription: a queue carrying what a topic carries",
}


@runtime_checkable
class AuraRuntime(Protocol):
    """The stable surface. Anything that can be Aura's runtime offers these."""

    async def runtime_start(self) -> None:
        ...

    async def runtime_stop(self) -> None:
        ...

    async def runtime_deliver(self, message: Any, *, origin: str = "user") -> Any:
        ...

    def runtime_state(self) -> Any:
        ...

    def runtime_service(self, name: str, default: Any = None) -> Any:
        ...

    async def runtime_watch(self, topic: str) -> Any:
        ...


class TheRuntimeOver:
    """The boundary, as a thin adapter rather than twelve more methods.

    The first version put the six operations on ``AuraKernel`` and
    ``RobustOrchestrator`` directly. That satisfied the Protocol and grew two
    classes that were already at 48 and 60 methods — and the review's whole
    point was too many methods per class, so paying for a boundary that way
    would have been buying the fix with the defect.

    An adapter is also the more honest shape. The boundary is a thing in its
    own right; putting it on the classes made it look like something they
    always were.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def inner(self) -> Any:
        return self._inner

    async def runtime_start(self) -> None:
        """Lifecycle up. `boot` on the kernel, `start` on the orchestrator."""
        for name in ("boot", "start"):
            found = getattr(self._inner, name, None)
            if callable(found):
                await found()
                return
        raise AttributeError(f"{type(self._inner).__name__} cannot be started")

    async def runtime_stop(self) -> None:
        """Lifecycle down. The full shutdown, not the kernel's flag-clearing stop."""
        for name in ("shutdown", "stop"):
            found = getattr(self._inner, name, None)
            if callable(found):
                await found()
                return
        raise AttributeError(f"{type(self._inner).__name__} cannot be stopped")

    async def runtime_deliver(self, message: Any, *, origin: str = "user") -> Any:
        """Hand it something to act on.

        The kernel thinks about it; the orchestrator processes it. A
        user-facing origin is a priority tick, which is the distinction `tick`
        already draws — nothing new is decided here.
        """
        process = getattr(self._inner, "process_user_input", None)
        if callable(process):
            return await process(str(message), origin=origin)
        tick = getattr(self._inner, "tick", None)
        if callable(tick):
            is_user = getattr(self._inner, "_is_user_facing_origin", None)
            priority = bool(is_user(origin)) if callable(is_user) else origin == "user"
            return await tick(str(message), priority=priority)
        raise AttributeError(f"{type(self._inner).__name__} takes no messages")

    def runtime_state(self) -> Any:
        """The state it is acting on.

        The orchestrator holds none of its own; asking it to keep one would
        create the second authority this boundary exists to rule out, so it
        goes to the kernel that owns it.
        """
        held = getattr(self._inner, "state", None)
        if held is not None:
            return held
        from core.container import ServiceContainer

        interface = ServiceContainer.get("kernel_interface", default=None)
        return getattr(getattr(interface, "kernel", None), "state", None)

    def runtime_service(self, name: str, default: Any = None) -> Any:
        """Resolve an authoritative service by name."""
        get = getattr(self._inner, "get", None)
        if callable(get):
            return get(name, default=default)
        resolve = getattr(self._inner, "_get_service", None)
        if callable(resolve):
            found = resolve(name)
            return default if found is None else found
        return default

    async def runtime_watch(self, topic: str) -> Any:
        """A queue carrying what the topic carries."""
        from core.event_bus import get_event_bus

        return await get_event_bus().subscribe(topic)


def a_runtime_over(inner: Any) -> TheRuntimeOver:
    """Wrap a kernel or an orchestrator in the boundary."""
    return TheRuntimeOver(inner)


def what_is_missing_from(candidate: Any) -> list[str]:
    """Which of the six operations this object does not offer.

    Empty means it satisfies the protocol. Named rather than counted, because
    "your runtime is missing something" is not an actionable sentence.
    """
    missing = []
    for operation in THE_OPERATIONS:
        attribute = getattr(candidate, operation, None)
        if not callable(attribute):
            missing.append(operation)
    return missing


def the_services_it_can_address(runtime: Any) -> dict[str, Any]:
    """How much of the declared service spine this runtime actually resolves.

    The spine is ``core/service_names.py``. A name it cannot resolve is a
    service some other holder owns — a module singleton, a boot-local
    variable — which is the ownership debt Soar #3 names. Reported as the
    names, not only the count: a count that goes down tells you nothing about
    what went missing.
    """
    from core.service_names import ServiceNames

    declared = sorted(
        value
        for key, value in vars(ServiceNames).items()
        if not key.startswith("_") and isinstance(value, str)
    )
    resolve = getattr(runtime, "runtime_service", None)
    if not callable(resolve):
        return {
            "declared": len(declared),
            "addressable": 0,
            "unaddressable": declared,
            "error": "runtime has no runtime_service",
        }
    unaddressable = []
    for name in declared:
        try:
            found = resolve(name, None)
        except Exception:  # noqa: BLE001 — an unresolvable name is the answer
            found = None
        if found is None:
            unaddressable.append(name)
    return {
        "declared": len(declared),
        "addressable": len(declared) - len(unaddressable),
        "unaddressable": unaddressable,
    }
