"""Give an organ a rung on the OOM ladder in one line.

The shed ladder had one rung in the whole runtime. Not because nothing
holds memory — the tree carries hundreds of bounded histories, caches and
rings — but because ``shed_memory()`` was a method each organ had to
write by hand, and exactly one did. A ladder with one rung is a ladder
whose only real answer to memory pressure is a restart, which is the
outcome :mod:`core.runtime.oom_policy` exists to avoid.

So the method is written once, here, and an organ names what it holds::

    class PerceptionDaemon(CacheHolder):
        sheddable_caches = ("_medium_term_buffer",)
        oom_rationale = "recent perception, rebuilt by looking again"

Two properties this keeps that a hand-written hook usually loses. The
footprint is measured rather than declared, so a rung cannot claim room
it does not hold. And a shed reports what it *actually* freed, measured
across the clear — a rung that frees nothing says zero, and the ladder
moves on rather than asking it again.

What must never be named here is anything the organ cannot rebuild, and
the survey that came with this found that most of what looks sheddable is
not. The tree carries hundreds of bounded ``deque``s and the large ones
are evidence rather than cache: ``PhiCore`` keeps five 2,000-entry state
histories and the transition matrix Φ is measured from is *built* from
them, so shedding one destroys a measurement in progress rather than
freeing a cache. The same goes for the body's spend receipts and the
perception daemon's day. A rung has to be something the organ recomputes
on the next request and nobody is reading as a record — which is rarer
than the ring count suggests, and is why a real ladder is short.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger("Aura.Sheddable")

__all__ = ["CacheHolder", "retained_bytes"]

#: Deep sizing stops here. A history of dicts is three levels; past that
#: the cost of measuring outgrows what the measurement is worth.
_MAX_DEPTH = 4


def retained_bytes(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> int:
    """Bytes this container holds, counting each object once.

    Best effort and deliberately cheap. It is asked under memory
    pressure, where a precise number that takes a second to produce is
    worse than an approximate one that takes a millisecond.
    """

    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    try:
        total = sys.getsizeof(value)
    # not a failure: an object that cannot report its size contributes
    # nothing to the estimate, which is what 0 says here.
    except TypeError:
        return 0
    if depth >= _MAX_DEPTH:
        return total
    try:
        if isinstance(value, Mapping):
            for key, item in list(value.items()):
                total += retained_bytes(key, depth=depth + 1, seen=seen)
                total += retained_bytes(item, depth=depth + 1, seen=seen)
        elif isinstance(value, (str, bytes, bytearray)):
            return total
        elif isinstance(value, (Sequence, Iterable)) and not isinstance(value, type):
            for item in list(value):
                total += retained_bytes(item, depth=depth + 1, seen=seen)
    except (RuntimeError, TypeError, ValueError):
        # A structure that changed while being walked, or one that cannot
        # be iterated. Its own size is still a true lower bound.
        return total
    return total


class CacheHolder:
    """Mixin: names what this organ can drop, and drops it on request.

    Subclasses set ``sheddable_caches`` to attribute names. The container
    offers every singleton it builds to the OOM policy, so adopting this
    is the whole adoption — there is no registration call to forget.
    """

    #: Attributes holding rebuildable state. Nothing durable belongs here.
    sheddable_caches: tuple[str, ...] = ()

    #: Kernel semantics: higher is chosen sooner. A history nobody reads
    #: should go before a cache the next turn will want.
    oom_score_adj: int = 0
    oom_rationale: str = ""
    oom_recoverable: bool = True

    def memory_footprint_bytes(self) -> int:
        """How much the named caches hold right now."""

        total = 0
        for name in self.sheddable_caches:
            container = getattr(self, name, None)
            if container is None:
                continue
            total += retained_bytes(container)
        return total

    def shed_memory(self) -> int:
        """Empty the named caches; return bytes actually freed."""

        freed = 0
        for name in self.sheddable_caches:
            container = getattr(self, name, None)
            if container is None:
                continue
            before = retained_bytes(container)
            clear = getattr(container, "clear", None)
            if not callable(clear):
                continue
            try:
                clear()
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.debug("cache %r refused to clear: %s", name, exc)
                continue
            # Measured across the clear rather than assumed from the size
            # before it. A container that shares its contents with
            # something still holding them has freed less than it looks.
            freed += max(0, before - retained_bytes(container))
        return freed
