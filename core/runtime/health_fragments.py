"""Subsystems publish their health fragment; the runtime reads the register.

`core/runtime` is the foundation layer and its DEPS file bans imports of
cognition, agency and learning — for a stated reason: an import from an upper
module into the foundation makes the foundation un-bootable without the thing
it is supposed to be able to run WITHOUT, which is how a health surface ends up
unable to report on a mind that failed to start.

Wiring two new snapshots into `health_contract` broke that ban immediately, and
the layering gate caught it. Reaching down from the surface is the wrong
direction: the surface should not know what a sealed learning artifact is. So
the dependency is inverted — a subsystem registers a callable that returns its
own fragment, and the surface asks this register.

Absence is reported, never omitted. Each expected fragment is declared here by
name, so a subsystem that never registered appears as ``registered: False``
rather than silently vanishing from the report — a channel with no writer is
the defect this repository keeps rediscovering, and an unpublished fragment is
exactly that shape.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Runtime.HealthFragments")

HEALTH_FRAGMENTS_SCHEMA = "aura.runtime.health_fragments.v1"

#: Fragments the health surface expects to exist. Declared as names, which the
#: foundation may hold, rather than as imports, which it may not. A name here
#: with no registration is reported as missing.
EXPECTED_FRAGMENTS: tuple[str, ...] = (
    "cognitive_contracts",
    "cognitive_growth",
    "external_reach",
    "memory_inventory",
    "sealed_artifacts",
)

_LOCK = checked_lock("core.runtime.health_fragments", reentrant=True)
_PROVIDERS: dict[str, Callable[[], dict[str, Any]]] = {}


def register_health_fragment(name: str, provider: Callable[[], dict[str, Any]]) -> None:
    """Publish one subsystem's health fragment.

    Idempotent: re-registration replaces, because a module re-imported under a
    hot reload should not leave the previous closure holding stale state.
    """
    key = str(name or "").strip()
    if not key or not callable(provider):
        return
    with _LOCK:
        _PROVIDERS[key] = provider


def collect_health_fragments() -> dict[str, dict[str, Any]]:
    """Every expected fragment, present or explicitly absent.

    A provider that raises is reported as failing rather than allowed to take
    the health surface down with it: the surface exists to describe failures,
    so it must survive them.
    """
    with _LOCK:
        providers = dict(_PROVIDERS)

    fragments: dict[str, dict[str, Any]] = {}
    for name in sorted(set(EXPECTED_FRAGMENTS) | set(providers)):
        provider = providers.get(name)
        if provider is None:
            fragments[name] = {
                "registered": False,
                "reason": (
                    "no subsystem published this fragment; it is expected, so its "
                    "absence is reported rather than omitted"
                ),
            }
            continue
        try:
            value = provider()
        except Exception as exc:  # noqa: BLE001 — a health surface survives its inputs
            logger.warning("Health fragment %s failed: %s", name, exc)
            fragments[name] = {
                "registered": True,
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}"[:200],
            }
            continue
        fragments[name] = (
            {**value, "registered": True} if isinstance(value, dict) else {
                "registered": True,
                "available": False,
                "reason": "provider returned a non-mapping fragment",
            }
        )
    return fragments


def reset_health_fragments_for_test() -> dict[str, Callable[[], dict[str, Any]]]:
    """Empty the register and RETURN what was in it.

    Registration happens at module import, which happens once per process, so a
    reset that only clears is permanent for the rest of the session: every test
    file that runs afterwards finds an empty register and every expected
    fragment reports absent. That is how a fragment test passed alone and failed
    in a wide run.

    Returning the previous providers is what makes the reset reversible. Pass
    the result to :func:`restore_health_fragments_for_test`, or use
    :func:`health_fragments_reset` which does both.
    """
    with _LOCK:
        previous = dict(_PROVIDERS)
        _PROVIDERS.clear()
        return previous


def restore_health_fragments_for_test(
    providers: dict[str, Callable[[], dict[str, Any]]],
) -> None:
    """Put back what a reset took out."""
    with _LOCK:
        _PROVIDERS.clear()
        _PROVIDERS.update(providers)


@contextlib.contextmanager
def health_fragments_reset() -> Iterator[None]:
    """An empty register for the body, and everything back afterwards."""
    saved = reset_health_fragments_for_test()
    try:
        yield
    finally:
        restore_health_fragments_for_test(saved)


__all__ = [
    "EXPECTED_FRAGMENTS",
    "HEALTH_FRAGMENTS_SCHEMA",
    "collect_health_fragments",
    "health_fragments_reset",
    "restore_health_fragments_for_test",
    "register_health_fragment",
    "reset_health_fragments_for_test",
]
