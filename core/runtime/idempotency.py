"""Idempotency keys for side-effecting work.

Why this exists
---------------
Aura produces a receipt for every consequential action, so you can always
see what happened. What a receipt cannot tell you is whether an action
happened *once*. A request that times out on the wire, a paired phone that
resends after a dropped socket, a client that retries a 502 — each arrives
as a fresh, fully authorized call, and the gate approves it again because
it is, on its own terms, a legitimate request. The mail gets sent twice.

OpenClaw's gateway requires an idempotency key on side-effecting methods
and keeps a short-lived dedupe cache so a retry is safe. Same idea here,
adapted to Aura's execution waist rather than a wire protocol.

What this is not
----------------
Not a cache for expensive work — the TTL is short and the store is bounded
and in-memory, so it survives a retry storm, not a restart. Losing it
degrades to today's behaviour (a retry re-executes), never to something
worse.

Single-flight matters as much as the cache
------------------------------------------
Two copies of the same request usually arrive *concurrently* — that is the
shape of a client retrying a call it thinks has stalled while the first
one is still running. A plain "have I seen this key?" check does nothing
there, because neither has finished. Callers past the first await on the
first one's outcome instead of starting their own.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_async_lock

# Long enough to cover a client retry cycle behind a slow action, short
# enough that a key is never mistaken for durable state. Aura's own
# conversation-lane timeout is ~335s, so a retry provoked by that timeout
# still lands inside the window.
DEFAULT_TTL_SECONDS = 900.0

# Bounded so a hostile or looping caller minting fresh keys cannot grow
# this without limit. Eviction is oldest-first.
MAX_ENTRIES = 2048


@dataclass
class _Entry:
    key: str
    created_at: float
    future: asyncio.Future = field(repr=False)
    replays: int = 0


@dataclass(frozen=True)
class IdempotentOutcome:
    """A result, plus whether this caller is the one that produced it."""

    value: Any
    replayed: bool
    key: str
    replays: int = 0


class IdempotencyLedger:
    """Keyed single-flight with a short-lived result cache."""

    def __init__(self, *, ttl_seconds: float = DEFAULT_TTL_SECONDS, max_entries: int = MAX_ENTRIES):
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: dict[str, _Entry] = {}
        self._lock = checked_async_lock("runtime.idempotency")

    def _evict_locked(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if now - entry.created_at > self._ttl]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self._max_entries:
            oldest = min(self._entries.values(), key=lambda item: item.created_at)
            self._entries.pop(oldest.key, None)

    async def run_once(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        *,
        ttl_seconds: float | None = None,
    ) -> IdempotentOutcome:
        """Run ``factory`` for this key, or return what it produced before.

        A caller that arrives while the first is still running waits for it
        rather than starting a second copy.
        """
        normalized = str(key or "").strip()
        if not normalized:
            # No key means no dedupe contract; do the work and say so.
            return IdempotentOutcome(value=await factory(), replayed=False, key="")

        ttl = float(ttl_seconds if ttl_seconds is not None else self._ttl)
        now = time.monotonic()

        async with self._lock:
            self._evict_locked(now)
            existing = self._entries.get(normalized)
            if existing is not None and now - existing.created_at <= ttl:
                existing.replays += 1
                replays = existing.replays
                future = existing.future
                owner = False
            else:
                future = asyncio.get_running_loop().create_future()
                self._entries[normalized] = _Entry(
                    key=normalized, created_at=now, future=future
                )
                replays = 0
                owner = True
                # Evict again after inserting, not only before: sweeping
                # first and then adding leaves the store at max_entries + 1,
                # so the bound would be one short of the number it states.
                self._evict_locked(now)

        if not owner:
            # Shielded so a waiter giving up (its own client hung up) cannot
            # cancel the in-flight action every other waiter is depending on.
            value = await asyncio.shield(future)
            return IdempotentOutcome(
                value=value, replayed=True, key=normalized, replays=replays
            )

        try:
            value = await factory()
        except BaseException as exc:
            # A failure is not a result. Drop the key so a retry is allowed
            # to actually retry — caching the exception would turn one
            # transient error into a permanently poisoned key.
            async with self._lock:
                entry = self._entries.pop(normalized, None)
            if entry is not None and not entry.future.done():
                entry.future.set_exception(exc)
            # Nobody may be waiting; consume it so asyncio does not warn.
            if entry is not None:
                entry.future.exception()
            raise

        async with self._lock:
            entry = self._entries.get(normalized)
        if entry is not None and not entry.future.done():
            entry.future.set_result(value)
        return IdempotentOutcome(value=value, replayed=False, key=normalized, replays=0)

    async def seen(self, key: str) -> bool:
        normalized = str(key or "").strip()
        if not normalized:
            return False
        async with self._lock:
            entry = self._entries.get(normalized)
            return entry is not None and time.monotonic() - entry.created_at <= self._ttl

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()


_ledger: IdempotencyLedger | None = None


def get_idempotency_ledger() -> IdempotencyLedger:
    global _ledger
    if _ledger is None:
        _ledger = IdempotencyLedger()
    return _ledger


def reset_idempotency_ledger_for_test() -> None:
    global _ledger
    _ledger = None


# Effect scopes that change nothing. A repeat costs nothing, so a key is
# neither required nor useful.
READ_ONLY_EFFECT_SCOPES = frozenset({"read_only", "status"})


def requires_idempotency_key(*, effect_scope: str, source: str) -> bool:
    """Whether this call must carry a key to be allowed to proceed.

    Deliberately narrow. Requiring a key everywhere would break every
    internal caller for no benefit — internal calls are not retried across
    a network, which is the only place duplicate delivery comes from. The
    rule is: work that changes something, arriving from somewhere that can
    resend it.
    """
    scope = str(effect_scope or "").strip().lower()
    if scope in READ_ONLY_EFFECT_SCOPES:
        return False
    origin = str(source or "").strip().lower()
    return origin.startswith(("paired_device", "remote", "webhook", "channel"))
